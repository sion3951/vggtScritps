import os
import glob
import time
import threading
from typing import List

import numpy as np
import torch
from tqdm.auto import tqdm
import viser
import viser.transforms as viser_tf
import cv2

from visual_util import segment_sky, download_file_from_url
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.geometry import closed_form_inverse_se3, unproject_depth_map_to_point_map
from vggt.utils.pose_enc import pose_encoding_to_extri_intri


def viser_wrapper(
    pred_dict: dict,
    port: int = 8080,
    init_conf_threshold: float = 50.0,
    use_point_map: bool = False,
    background_mode: bool = False,
    mask_sky: bool = False,
    image_folder: str = None,
):
    print(f"Starting viser server on port {port}")
    server = viser.ViserServer(host="0.0.0.0", port=port)
    server.gui.configure_theme(titlebar_content=None, control_layout="collapsible")

    images = pred_dict["images"]              # (S, 3, H, W) numpy
    world_points_map = pred_dict["world_points"]
    conf_map = pred_dict["world_points_conf"]
    depth_map = pred_dict["depth"]
    depth_conf = pred_dict["depth_conf"]
    extrinsics_cam = pred_dict["extrinsic"]   # (S, 3, 4) or (S, 4, 4) world->cam
    intrinsics_cam = pred_dict["intrinsic"]

    if not use_point_map:
        world_points = unproject_depth_map_to_point_map(depth_map, extrinsics_cam, intrinsics_cam)
        conf = depth_conf
    else:
        world_points = world_points_map
        conf = conf_map

    if mask_sky and image_folder is not None:
        # optional, not used here
        pass

    colors = images.transpose(0, 2, 3, 1)  # (S, H, W, 3)
    S, H, W, _ = world_points.shape
    points = world_points.reshape(-1, 3)
    colors_flat = (colors.reshape(-1, 3) * 255).astype(np.uint8)
    conf_flat = conf.reshape(-1)

    cam_to_world_mat = closed_form_inverse_se3(extrinsics_cam)  # (S, 4, 4)
    cam_to_world = cam_to_world_mat[:, :3, :]

    scene_center = np.mean(points, axis=0)
    points_centered = points - scene_center
    cam_to_world[..., -1] -= scene_center
    frame_indices = np.repeat(np.arange(S), H * W)

    gui_show_frames = server.gui.add_checkbox("Show Cameras", initial_value=True)
    gui_points_conf = server.gui.add_slider("Confidence Percent", min=0, max=100, step=0.1, initial_value=init_conf_threshold)
    gui_frame_selector = server.gui.add_dropdown("Show Points from Frames", options=["All"] + [str(i) for i in range(S)], initial_value="All")

    init_threshold_val = np.percentile(conf_flat, init_conf_threshold)
    init_conf_mask = (conf_flat >= init_threshold_val) & (conf_flat > 0.1)
    point_cloud = server.scene.add_point_cloud(
        name="viser_pcd",
        points=points_centered[init_conf_mask],
        colors=colors_flat[init_conf_mask],
        point_size=0.001,
        point_shape="circle",
    )

    frames: List[viser.FrameHandle] = []
    frustums: List[viser.CameraFrustumHandle] = []

    def visualize_frames(extrinsics_c2w_3x4: np.ndarray, images_: np.ndarray) -> None:
        for f in frames:
            f.remove()
        frames.clear()
        for fr in frustums:
            fr.remove()
        frustums.clear()

        def attach_callback(frustum: viser.CameraFrustumHandle, frame: viser.FrameHandle) -> None:
            @frustum.on_click
            def _(_) -> None:
                for client in server.get_clients().values():
                    client.camera.wxyz = frame.wxyz
                    client.camera.position = frame.position

        for img_id in tqdm(range(images_.shape[0])):
            cam2world_3x4 = extrinsics_c2w_3x4[img_id]
            T_world_camera = viser_tf.SE3.from_matrix(cam2world_3x4)
            frame_axis = server.scene.add_frame(
                f"frame_{img_id}",
                wxyz=T_world_camera.rotation().wxyz,
                position=T_world_camera.translation(),
                axes_length=0.05,
                axes_radius=0.002,
                origin_radius=0.002,
            )
            frames.append(frame_axis)

            img = images_[img_id]
            img = (img.transpose(1, 2, 0) * 255).astype(np.uint8)
            h, w = img.shape[:2]
            fy = 1.1 * h
            fov = 2 * np.arctan2(h / 2, fy)
            frustum_cam = server.scene.add_camera_frustum(
                f"frame_{img_id}/frustum", fov=fov, aspect=w / h, scale=0.05, image=img, line_width=1.0
            )
            frustums.append(frustum_cam)
            attach_callback(frustum_cam, frame_axis)

    def update_point_cloud() -> None:
        percentage = gui_points_conf.value
        threshold_val = np.percentile(conf_flat, percentage)
        conf_mask = (conf_flat >= threshold_val) & (conf_flat > 1e-5)
        if gui_frame_selector.value == "All":
            frame_mask = np.ones_like(conf_mask, dtype=bool)
        else:
            selected_idx = int(gui_frame_selector.value)
            frame_mask = frame_indices == selected_idx
        combined_mask = conf_mask & frame_mask
        point_cloud.points = points_centered[combined_mask]
        point_cloud.colors = colors_flat[combined_mask]

    @gui_points_conf.on_update
    def _(_e) -> None:
        update_point_cloud()

    @gui_frame_selector.on_update
    def _(_e) -> None:
        update_point_cloud()

    @gui_show_frames.on_update
    def _(_e) -> None:
        for f in frames:
            f.visible = gui_show_frames.value
        for fr in frustums:
            fr.visible = gui_show_frames.value

    visualize_frames(cam_to_world, images)

    def update_predictions(new_pred_dict: dict):
        images_n = new_pred_dict["images"]  # numpy (S,3,H,W)
        depth_map_n = new_pred_dict.get("depth")
        depth_conf_n = new_pred_dict.get("depth_conf")
        world_points_map_n = new_pred_dict.get("world_points")
        conf_map_n = new_pred_dict.get("world_points_conf")
        extrinsic_n = new_pred_dict["extrinsic"]
        intrinsic_n = new_pred_dict["intrinsic"]

        if not use_point_map and depth_map_n is not None:
            world_points_n = unproject_depth_map_to_point_map(depth_map_n, extrinsic_n, intrinsic_n)
            conf_n = depth_conf_n
        else:
            world_points_n = world_points_map_n
            conf_n = conf_map_n

        colors_n = images_n.transpose(0, 2, 3, 1)
        S_n, H_n, W_n, _ = world_points_n.shape
        points_n = world_points_n.reshape(-1, 3)
        colors_flat_n = (colors_n.reshape(-1, 3) * 255).astype(np.uint8)
        conf_flat_n = conf_n.reshape(-1)
        scene_center_n = np.mean(points_n, axis=0)
        points_centered_n = points_n - scene_center_n

        nonlocal frame_indices, conf_flat, points_centered, colors_flat
        frame_indices = np.repeat(np.arange(S_n), H_n * W_n)
        conf_flat = conf_flat_n
        points_centered = points_centered_n
        colors_flat = colors_flat_n

        percentage = gui_points_conf.value
        threshold_val = np.percentile(conf_flat_n, percentage)
        conf_mask = (conf_flat_n >= threshold_val) & (conf_flat_n > 1e-5)
        if gui_frame_selector.value == "All":
            frame_mask = np.ones_like(conf_mask, dtype=bool)
        else:
            selected_idx = int(gui_frame_selector.value)
            frame_mask = frame_indices == selected_idx
        combined_mask = conf_mask & frame_mask
        point_cloud.points = points_centered_n[combined_mask]
        point_cloud.colors = colors_flat_n[combined_mask]

        cam_to_world_n = closed_form_inverse_se3(extrinsic_n)[:, :3, :]
        cam_to_world_n[..., -1] -= scene_center_n
        visualize_frames(cam_to_world_n, images_n)

    if background_mode:
        def server_loop():
            while True:
                time.sleep(0.01)
        thread = threading.Thread(target=server_loop, daemon=True)
        thread.start()
        return {"server": server, "update": update_predictions}
    else:
        while True:
            time.sleep(0.01)
        return server


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print("Initializing and loading VGGT model...")
    model = VGGT()
    _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
    model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))
    model.eval()
    model = model.to(device)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        exit()

    temp_dir = "tmp"
    os.makedirs(temp_dir, exist_ok=True)
    temp_filename = os.path.join(temp_dir, "temp_frame.png")

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    cv2.imwrite(temp_filename, frame)
    files = sorted(glob.glob(os.path.join(temp_dir, "*.png")))
    images = load_and_preprocess_images(files).to(device)
    os.remove(temp_filename)

    if device == "cuda":
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    else:
        dtype = torch.float32

    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=(device == "cuda"), dtype=dtype if device == "cuda" else None):
            predictions = model(images)

    print("Converting pose encoding to extrinsic and intrinsic matrices...")
    extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
    predictions["extrinsic"] = extrinsic
    predictions["intrinsic"] = intrinsic
    print("Processing model outputs...")
    for key in list(predictions.keys()):
        if isinstance(predictions[key], torch.Tensor):
            predictions[key] = predictions[key].cpu().numpy().squeeze(0)

    pred_dict = {
        "images": images.cpu().numpy(),
        "world_points": predictions.get("world_points"),
        "world_points_conf": predictions.get("world_points_conf"),
        "depth": predictions.get("depth"),
        "depth_conf": predictions.get("depth_conf"),
        "extrinsic": predictions["extrinsic"],
        "intrinsic": predictions["intrinsic"],
    }

    print("Starting viser visualization (background mode)...")
    viser_handle = viser_wrapper(
        pred_dict,
        port=8080,
        init_conf_threshold=1,
        use_point_map=False,
        background_mode=True,
        mask_sky=False,
        image_folder=None,
    )
    print("Visualizer started. Entering capture loop...")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.1)
                continue

            #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            cv2.imwrite(temp_filename, frame)
            files = [temp_filename]
            images_loop = load_and_preprocess_images(files).to(device)
            os.remove(temp_filename)

            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=(device == "cuda"), dtype=dtype if device == "cuda" else None):
                    preds = model(images_loop)

            extrinsic, intrinsic = pose_encoding_to_extri_intri(preds["pose_enc"], images_loop.shape[-2:])
            preds["extrinsic"] = extrinsic
            preds["intrinsic"] = intrinsic
            for k in list(preds.keys()):
                if isinstance(preds[k], torch.Tensor):
                    preds[k] = preds[k].cpu().numpy().squeeze(0)

            pred_dict_single = {
                "images": images_loop.cpu().numpy(),
                "depth": preds.get("depth"),
                "depth_conf": preds.get("depth_conf"),
                "world_points": preds.get("world_points"),
                "world_points_conf": preds.get("world_points_conf"),
                "extrinsic": preds["extrinsic"],
                "intrinsic": preds["intrinsic"],
            }

            viser_handle["update"](pred_dict_single)
            time.sleep(0.02)
    except KeyboardInterrupt:
        print("Stopping capture loop.")
    finally:
        cap.release()


if __name__ == "__main__":
    main()
