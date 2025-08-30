# realtime.py
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import os
import glob
import time
import threading
import argparse
import tempfile
import signal

from typing import List, Optional

import numpy as np
import torch
from tqdm.auto import tqdm
import viser
import viser.transforms as viser_tf
import cv2

try:
    import onnxruntime
except Exception:
    onnxruntime = None
    print("onnxruntime not found. Sky segmentation may not work.")

from visual_util import segment_sky, download_file_from_url
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.geometry import closed_form_inverse_se3, unproject_depth_map_to_point_map
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

# fixed resolution requested
vggt_fixed_resolution = 518

def ensure_leading_dim(x):
    if x is None:
        return None
    x = np.asarray(x)
    if x.ndim <= 3:
        return x[np.newaxis, ...]
    return x


def unproject_batched_depth(depth_map, extrinsic_in, intrinsic_in):
    depth_map = None if depth_map is None else np.asarray(depth_map)
    extrinsic_in = None if extrinsic_in is None else np.asarray(extrinsic_in)
    intrinsic_in = None if intrinsic_in is None else np.asarray(intrinsic_in)

    if depth_map is None:
        return None

    if depth_map.ndim == 3:  # (S,H,W) -> (S,H,W,1)
        depth_map = depth_map[..., np.newaxis]
    S = depth_map.shape[0]

    if extrinsic_in.ndim <= 2:
        extrinsic_in = ensure_leading_dim(extrinsic_in)
    if intrinsic_in.ndim <= 2:
        intrinsic_in = ensure_leading_dim(intrinsic_in)

    if extrinsic_in is None or intrinsic_in is None:
        raise ValueError("extrinsic and intrinsic must be provided when using depth_map")

    if extrinsic_in.shape[0] == 1 and S > 1:
        extrinsic_in = np.tile(extrinsic_in, (S, 1, 1))
    if intrinsic_in.shape[0] == 1 and S > 1:
        intrinsic_in = np.tile(intrinsic_in, (S, 1, 1))

    world_points_list = []
    for i in range(S):
        # Handle depth map dimensions properly
        dm = depth_map[i]
        if dm.ndim == 3 and dm.shape[-1] == 1:
            dm = dm[..., 0]  # Remove the last dimension if it's 1
        elif dm.ndim == 3 and dm.shape[0] == 1:
            dm = dm[0]  # Remove the first dimension if it's 1
            
        ext = extrinsic_in[i]
        intr = intrinsic_in[i]
        
        # Handle potential batch dimension in intrinsic matrix
        if intr.ndim == 3 and intr.shape[0] == 1:
            intr = intr[0]
        if intr.shape != (3, 3):
            raise AssertionError(f"Intrinsic for frame {i} has wrong shape {intr.shape}; expected (3,3)")
            
        # Ensure depth map is 2D before passing to unproject function
        if dm.ndim != 2:
            dm = dm.squeeze()
            if dm.ndim != 2:
                raise ValueError(f"Depth map for frame {i} must be 2D after squeezing, but got shape {dm.shape}")
                
        wp = unproject_depth_map_to_point_map(dm, ext, intr)
        world_points_list.append(wp)

    return np.stack(world_points_list, axis=0)


class ViserStreamer:
    def __init__(
        self,
        pred_template: dict,
        port: int = 8080,
        init_conf_threshold: float = 25.0,
        use_point_map: bool = False,
        mask_sky: bool = False,
        image_folder: Optional[str] = None,
    ):
        self.port = port
        self.use_point_map = use_point_map
        self.mask_sky = mask_sky
        self.image_folder = image_folder
        self.init_conf_threshold = init_conf_threshold

        self.server = viser.ViserServer(host="0.0.0.0", port=port)
        self.server.gui.configure_theme(titlebar_content=None, control_layout="collapsible")

        self.gui_show_frames = self.server.gui.add_checkbox("Show Cameras", initial_value=True)
        self.gui_points_conf = self.server.gui.add_slider(
            "Confidence Percent", min=0, max=100, step=0.1, initial_value=init_conf_threshold
        )
        self.gui_frame_selector = self.server.gui.add_dropdown("Show Points from Frames", options=["All"], initial_value="All")

        self.point_cloud = None
        self.frames: List[viser.FrameHandle] = []
        self.frustums: List[viser.CameraFrustumHandle] = []
        self.current_points_centered = np.zeros((0, 3), dtype=np.float32)
        self.current_colors_flat = np.zeros((0, 3), dtype=np.uint8)
        self.current_conf_flat = np.zeros((0,), dtype=np.float32)
        self.frame_indices = np.zeros((0,), dtype=int)

        @self.gui_points_conf.on_update
        def _(_):
            self._update_point_cloud_mask()

        @self.gui_frame_selector.on_update
        def _(_):
            self._update_point_cloud_mask()

        @self.gui_show_frames.on_update
        def _(_):
            for f in self.frames:
                f.visible = self.gui_show_frames.value
            for fr in self.frustums:
                fr.visible = self.gui_show_frames.value

        def server_loop():
            while True:
                time.sleep(0.1)

        t = threading.Thread(target=server_loop, daemon=True)
        t.start()

    def _clear_frames(self):
        for f in self.frames:
            try:
                f.remove()
            except Exception:
                pass
        self.frames.clear()
        for fr in self.frustums:
            try:
                fr.remove()
            except Exception:
                pass
        self.frustums.clear()

    def _update_point_cloud_mask(self):
        if self.point_cloud is None:
            return
        current_percentage = self.gui_points_conf.value
        if self.current_conf_flat.size == 0:
            threshold_val = 0.0
        else:
            threshold_val = np.percentile(self.current_conf_flat, current_percentage)
        conf_mask = (self.current_conf_flat >= threshold_val) & (self.current_conf_flat > 1e-5)

        if self.gui_frame_selector.value == "All":
            frame_mask = np.ones_like(conf_mask, dtype=bool)
        else:
            selected_idx = int(self.gui_frame_selector.value)
            frame_mask = self.frame_indices == selected_idx

        combined_mask = conf_mask & frame_mask
        self.point_cloud.points = self.current_points_centered[combined_mask]
        self.point_cloud.colors = self.current_colors_flat[combined_mask]

    def update(self, pred_dict: dict):
        images = np.asarray(pred_dict["images"])
        if images.ndim == 2:
            images = np.stack([images] * 3, axis=-1)
        if images.ndim == 3:
            if images.shape[0] == 3:
                images = images[np.newaxis, ...]
            elif images.shape[-1] == 3:
                images = images.transpose(2, 0, 1)[np.newaxis, ...]
            else:
                images = images[np.newaxis, ...]
        elif images.ndim == 4:
            if images.shape[1] != 3 and images.shape[-1] == 3:
                images = images.transpose(0, 3, 1, 2)

        pred_dict["images"] = images
        S, _, H, W = images.shape

        if self.use_point_map:
            world_points = ensure_leading_dim(pred_dict.get("world_points"))
            conf_map = ensure_leading_dim(pred_dict.get("world_points_conf"))
        else:
            depth_map = ensure_leading_dim(pred_dict.get("depth"))
            conf_map = ensure_leading_dim(pred_dict.get("depth_conf"))

            # Normalize depth to (S,H,W,1)
            if depth_map is not None:
                if depth_map.ndim == 3:
                    depth_map = depth_map[..., np.newaxis]

            # Data from main() is already correctly batched.
            extrinsic_in = pred_dict.get("extrinsic")
            intrinsic_in = pred_dict.get("intrinsic")

            if depth_map is not None:
                world_points = unproject_batched_depth(depth_map, extrinsic_in, intrinsic_in)
            else:
                world_points = ensure_leading_dim(pred_dict.get("world_points"))

        if conf_map is not None and conf_map.ndim == 4 and conf_map.shape[-1] == 1:
            conf_map = conf_map[..., 0]

        if self.mask_sky and self.image_folder is not None and conf_map is not None:
            conf_map = apply_sky_segmentation(conf_map, self.image_folder)

        colors = images.transpose(0, 2, 3, 1)  # (S,H,W,3)
        points = world_points.reshape(-1, 3)
        colors_flat = (colors.reshape(-1, 3) * 255).astype(np.uint8)
        conf_flat = conf_map.reshape(-1) if conf_map is not None else np.zeros((points.shape[0],), dtype=np.float32)
        frame_indices = np.repeat(np.arange(S), H * W)

        cam_to_world_mat = closed_form_inverse_se3(ensure_leading_dim(pred_dict["extrinsic"]))  # (S,4,4)
        cam_to_world = cam_to_world_mat[:, :3, :]

        scene_center = np.mean(points, axis=0)
        points_centered = points - scene_center
        cam_to_world[..., -1] -= scene_center

        self.current_points_centered = points_centered
        self.current_colors_flat = colors_flat
        self.current_conf_flat = conf_flat
        self.frame_indices = frame_indices

        init_threshold_val = np.percentile(conf_flat, self.init_conf_threshold) if conf_flat.size else 0.0
        init_conf_mask = (conf_flat >= init_threshold_val) & (conf_flat > 1e-5)
        pts = points_centered[init_conf_mask]
        cols = colors_flat[init_conf_mask]

        if self.point_cloud is None:
            self.point_cloud = self.server.scene.add_point_cloud(
                name="viser_pcd",
                points=pts,
                colors=cols,
                point_size=0.001,
                point_shape="circle",
            )
        else:
            self.point_cloud.points = pts
            self.point_cloud.colors = cols

        options = ["All"] + [str(i) for i in range(S)]
        if self.gui_frame_selector.options != options:
            self.gui_frame_selector.options = options
            self.gui_frame_selector.value = "All"

        self._clear_frames()

        for img_id in range(S):
            cam2world_3x4 = cam_to_world[img_id]
            T_world_camera = viser_tf.SE3.from_matrix(cam2world_3x4)

            frame_axis = self.server.scene.add_frame(
                f"frame_{img_id}",
                wxyz=T_world_camera.rotation().wxyz,
                position=T_world_camera.translation(),
                axes_length=0.05,
                axes_radius=0.002,
                origin_radius=0.002,
            )
            self.frames.append(frame_axis)

            img = images[img_id]
            img_vis = (img.transpose(1, 2, 0) * 255).astype(np.uint8)
            h, w = img_vis.shape[:2]

            fy = 1.1 * h
            fov = 2 * np.arctan2(h / 2, fy)

            frustum_cam = self.server.scene.add_camera_frustum(
                f"frame_{img_id}/frustum", fov=fov, aspect=w / h, scale=0.05, image=img_vis, line_width=1.0
            )
            self.frustums.append(frustum_cam)

            def attach_callback(frustum, frame):
                @frustum.on_click
                def _(_):
                    for client in self.server.get_clients().values():
                        client.camera.wxyz = frame.wxyz
                        client.camera.position = frame.position

            attach_callback(frustum_cam, frame_axis)

        for f in self.frames:
            f.visible = self.gui_show_frames.value
        for fr in self.frustums:
            fr.visible = self.gui_show_frames.value

        self._update_point_cloud_mask()


def apply_sky_segmentation(conf: np.ndarray, image_folder: str) -> np.ndarray:
    S, H, W = conf.shape
    sky_masks_dir = image_folder.rstrip("/") + "_sky_masks"
    os.makedirs(sky_masks_dir, exist_ok=True)

    if onnxruntime is None:
        print("onnxruntime not available - skipping sky segmentation")
        return conf

    if not os.path.exists("skyseg.onnx"):
        print("Downloading skyseg.onnx...")
        download_file_from_url("https://huggingface.co/JianyuanWang/skyseg/resolve/main/skyseg.onnx", "skyseg.onnx")

    skyseg_session = onnxruntime.InferenceSession("skyseg.onnx")
    image_files = sorted(glob.glob(os.path.join(image_folder, "*")))
    sky_mask_list = []

    print("Generating sky masks...")
    for i, image_path in enumerate(tqdm(image_files[:S])):
        image_name = os.path.basename(image_path)
        mask_filepath = os.path.join(sky_masks_dir, image_name)

        if os.path.exists(mask_filepath):
            sky_mask = cv2.imread(mask_filepath, cv2.IMREAD_GRAYSCALE)
        else:
            sky_mask = segment_sky(image_path, skyseg_session, mask_filepath)

        if sky_mask.shape[0] != H or sky_mask.shape[1] != W:
            sky_mask = cv2.resize(sky_mask, (W, H))

        sky_mask_list.append(sky_mask)

    sky_mask_array = np.array(sky_mask_list)
    sky_mask_binary = (sky_mask_array > 0.1).astype(np.float32)
    conf = conf * sky_mask_binary
    print("Sky segmentation applied successfully")
    return conf


def capture_and_write_frame(cap, resolution, tmp_path):
    ret, frame = cap.read()
    if not ret:
        return None
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (resolution, resolution))
    cv2.imwrite(tmp_path, cv2.cvtColor(frame_resized, cv2.COLOR_RGB2BGR))
    return tmp_path


def safe_device_and_dtype():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        cuda_cap = torch.cuda.get_device_capability()[0] if device == "cuda" else 0
    except Exception:
        cuda_cap = 0
    dtype = torch.bfloat16 if (device == "cuda" and cuda_cap >= 8) else (torch.float16 if device == "cuda" else torch.float32)
    return device, dtype


def main():
    parser = argparse.ArgumentParser(description="VGGT webcam streaming with viser")
    parser.add_argument("--webcam-device", type=int, default=0, help="OpenCV webcam device id")
    parser.add_argument("--port", type=int, default=8080, help="Port for viser server")
    parser.add_argument("--conf_threshold", type=float, default=25.0, help="Initial confidence percentage")
    parser.add_argument("--use_point_map", action="store_true", help="Use point map instead of depth")
    parser.add_argument("--mask_sky", action="store_true", help="Apply sky segmentation")
    parser.add_argument("--image-folder", type=str, default=None, help="Folder used for sky mask lookup (optional)")
    args = parser.parse_args()

    device, dtype = safe_device_and_dtype()
    print(f"Using device: {device}, dtype: {dtype}")

    print("Initializing and loading VGGT model...")
    model = VGGT()
    _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
    try:
        model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))
    except Exception as e:
        print("Failed to download model automatically. Ensure you have model weights available locally or have internet. Error:", e)
        raise

    model.eval()
    model = model.to(device)

    cap = cv2.VideoCapture(args.webcam_device)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open webcam device {args.webcam_device}")

    dummy_img = np.zeros((1, 3, vggt_fixed_resolution, vggt_fixed_resolution), dtype=np.float32)
    dummy_depth = np.zeros((1, vggt_fixed_resolution, vggt_fixed_resolution, 1), dtype=np.float32)
    dummy_conf = np.ones((1, vggt_fixed_resolution, vggt_fixed_resolution), dtype=np.float32) * 1e-3
    dummy_extrinsic = np.tile(np.eye(3, 4, dtype=np.float32), (1, 1, 1))
    dummy_intrinsic = np.tile(np.eye(3, dtype=np.float32), (1, 1, 1))

    pred_template = {
        "images": dummy_img,
        "depth": dummy_depth,
        "depth_conf": dummy_conf,
        "extrinsic": dummy_extrinsic,
        "intrinsic": dummy_intrinsic,
    }

    streamer = ViserStreamer(
        pred_template,
        port=args.port,
        init_conf_threshold=args.conf_threshold,
        use_point_map=args.use_point_map,
        mask_sky=args.mask_sky,
        image_folder=args.image_folder,
    )

    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".jpg")
    os.close(tmp_fd)

    running = True

    def handle_sigint(sig, frame):
        nonlocal running
        running = False
        print("\nStopping...")

    signal.signal(signal.SIGINT, handle_sigint)
    signal.signal(signal.SIGTERM, handle_sigint)

    print("Starting webcam stream. Press Ctrl+C to stop.")
    try:
        while running:
            written = capture_and_write_frame(cap, vggt_fixed_resolution, tmp_path)
            if written is None:
                time.sleep(0.01)
                continue

            try:
                images_tensor = load_and_preprocess_images([tmp_path]).to(device)
            except Exception as e:
                print("load_and_preprocess_images failed:", e)
                img_bgr = cv2.imread(tmp_path)
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                img_norm = img_rgb.astype(np.float32) / 255.0
                images_tensor = torch.from_numpy(img_norm.transpose(2, 0, 1)).unsqueeze(0).to(device)

            with torch.no_grad():
                with torch.amp.autocast(device_type=device, dtype=dtype, enabled=(device=="cuda")):
                    predictions = model(images_tensor)

            extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images_tensor.shape[-2:])
            predictions["extrinsic"] = extrinsic
            predictions["intrinsic"] = intrinsic

            for key in list(predictions.keys()):
                val = predictions[key]
                if isinstance(val, torch.Tensor):
                    predictions[key] = val.cpu().numpy().squeeze(0)

            if predictions.get("images") is None:
                img_np = images_tensor.cpu().numpy().squeeze(0)
                predictions["images"] = img_np
            else:
                img_val = predictions["images"]
                if img_val.ndim == 4 and img_val.shape[0] == 1:
                    predictions["images"] = img_val.squeeze(0)
                if predictions["images"].ndim == 3 and predictions["images"].shape[0] == 3:
                    pass
                else:
                    if predictions["images"].ndim == 3 and predictions["images"].shape[-1] == 3:
                        predictions["images"] = np.transpose(predictions["images"], (2, 0, 1))
                        predictions["images"] = predictions["images"][np.newaxis, ...]
                    elif predictions["images"].ndim == 4 and predictions["images"].shape[-1] == 3:
                        predictions["images"] = np.transpose(predictions["images"], (0, 3, 1, 2))

            # Handle potential batch dimension in intrinsic and extrinsic
            if "intrinsic" in predictions:
                intrinsic_val = predictions["intrinsic"]
                if intrinsic_val.ndim == 3 and intrinsic_val.shape[0] == 1:
                    predictions["intrinsic"] = intrinsic_val[0]
            
            if "extrinsic" in predictions:
                extrinsic_val = predictions["extrinsic"]
                if extrinsic_val.ndim == 3 and extrinsic_val.shape[0] == 1:
                    predictions["extrinsic"] = extrinsic_val[0]

        # Handle depth map dimensions
        if "depth" in predictions:
            depth_val = predictions["depth"]
            if depth_val.ndim == 3 and depth_val.shape[0] == 1:
                predictions["depth"] = depth_val[0]
            elif depth_val.ndim == 2:
                predictions["depth"] = depth_val[np.newaxis, ...]
            # Ensure depth map is 2D (H, W)
            if predictions["depth"].ndim == 3 and predictions["depth"].shape[0] == 1:
                predictions["depth"] = predictions["depth"][0]

            if args.use_point_map and "world_points" not in predictions:
                if "depth" in predictions:
                    d = predictions["depth"]
                    d_in = d if d.ndim >= 3 else d[np.newaxis, ...]
                    predictions["world_points"] = unproject_batched_depth(d_in, predictions["extrinsic"], predictions["intrinsic"])

            for k in ["images", "depth", "depth_conf", "world_points", "world_points_conf", "extrinsic", "intrinsic"]:
                if k in predictions:
                    val = predictions[k]
                    if isinstance(val, np.ndarray) and val.ndim <= 3:
                        predictions[k] = ensure_leading_dim(val)

            streamer.update(predictions)
            time.sleep(0.01)

    finally:
        cap.release()
        try:
            os.remove(tmp_path)
        except Exception:
            pass
        print("Stopped streaming.")


if __name__ == "__main__":
    main()