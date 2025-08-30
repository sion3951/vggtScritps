import argparse
import time

import numpy as np
import torch
import cv2
import viser
import viser.transforms as viser_tf
from vggt.models.vggt import VGGT
from vggt.utils.geometry import closed_form_inverse_se3, unproject_depth_map_to_point_map
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

def preprocess_frame(frame, target_size=(224, 224)):
    frame_resized = cv2.resize(frame, target_size)
    frame_tensor = torch.from_numpy(frame_resized).float() / 255.0
    frame_tensor = frame_tensor.permute(2, 0, 1)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    frame_tensor = (frame_tensor - mean) / std
    return frame_tensor.unsqueeze(0)


class RealTimeViserWrapper:
    def __init__(self, port=8080, init_conf_threshold=50.0, use_point_map=False, mask_sky=False):
        self.port = port
        self.init_conf_threshold = init_conf_threshold
        self.use_point_map = use_point_map
        self.mask_sky = mask_sky

        print(f"Starting viser server on port {port}")
        self.server = viser.ViserServer(host="0.0.0.0", port=port)
        self.server.gui.configure_theme(titlebar_content=None, control_layout="collapsible")

        self.point_cloud = self.server.scene.add_point_cloud(
            name="real_time_pcd",
            points=np.zeros((0, 3)),
            colors=np.zeros((0, 3), dtype=np.uint8),
            point_size=0.001,
            point_shape="circle",
        )

        self.camera_frustum = None
        self.camera_frame = None

        self.latest_points = None
        self.latest_colors = None
        self.latest_conf = None
        self.latest_extrinsics = None
        self.latest_intrinsics = None
        self.latest_image = None  # full-res viz image (3, H, W) in [0,1]

        self.gui_show_frames = self.server.gui.add_checkbox("Show Camera", initial_value=True)
        self.gui_points_conf = self.server.gui.add_slider(
            "Confidence Percent", min=0, max=100, step=0.1, initial_value=init_conf_threshold
        )

        @self.gui_points_conf.on_update
        def _(_):
            self.update_point_cloud()

        @self.gui_show_frames.on_update
        def _(_):
            self.update_camera_visibility()

    def update_data(self, pred_dict):

        # Prefer explicit viz image (0..1 float); fallback to model-provided images if available.
        images_viz = pred_dict.get("images_for_viz", pred_dict.get("images", None))
        if images_viz is None:
            raise ValueError("No visualization image found in predictions (images_for_viz or images)")

        world_points = pred_dict["world_points"]      # (1, H_d, W_d, 3)
        conf_map = pred_dict["world_points_conf"]     # (1, H_d, W_d)
        depth_map = pred_dict.get("depth", None)
        depth_conf = pred_dict.get("depth_conf", None)
        extrinsics_cam = pred_dict["extrinsic"]      # (1, 3, 4) - already camera-to-world
        intrinsics_cam = pred_dict["intrinsic"]      # (1, 3, 3)

        # If not using point_map, optionally re-compute from depth (kept for compatibility)
        if not self.use_point_map and depth_map is not None:
            world_points = unproject_depth_map_to_point_map(depth_map, extrinsics_cam, intrinsics_cam)
            conf = depth_conf if depth_conf is not None else conf_map
        else:
            conf = conf_map

        # Depth/model resolution
        _, H_d, W_d, _ = world_points.shape

        # images_viz shape: (1, C, H_v, W_v) - convert/ensure dtype float32 in [0,1]
        images_viz = images_viz.astype(np.float32)
        _, C, H_v, W_v = images_viz.shape

        # Resize viz image to depth resolution so per-point colors match world_points
        if (H_v, W_v) != (H_d, W_d):
            resized = np.zeros((1, C, H_d, W_d), dtype=images_viz.dtype)
            for c in range(C):
                # cv2.resize expects (W, H)
                resized[0, c] = cv2.resize(images_viz[0, c], (W_d, H_d), interpolation=cv2.INTER_LINEAR)
            images_for_colors = resized
        else:
            images_for_colors = images_viz

        # Convert to (1, H_d, W_d, C)
        colors = images_for_colors.transpose(0, 2, 3, 1)

        S, H, W, _ = world_points.shape
        points = world_points.reshape(-1, 3)
        colors_flat = (np.clip(colors.reshape(-1, 3), 0.0, 1.0) * 255.0).astype(np.uint8)
        conf_flat = conf.reshape(-1)

        # CORRECTED: The model already outputs camera-to-world extrinsics
        # No need to invert them again
        cam_to_world = extrinsics_cam  # (1, 3, 4) - already camera-to-world
        
        # Create 4x4 transformation matrix for visualization
        cam_to_world_mat = np.eye(4)
        cam_to_world_mat[:3, :] = cam_to_world[0]

        # Recenter scene
        scene_center = np.mean(points, axis=0)
        points_centered = points - scene_center
        cam_to_world_mat[:3, 3] -= scene_center

        # Store: latest_colors corresponds to the flattened per-point colors (matching world_points)
        self.latest_points = points_centered
        self.latest_colors = colors_flat
        self.latest_conf = conf_flat
        self.latest_extrinsics = cam_to_world_mat[:3, :]  # Store as 3x4 for consistency
        self.latest_intrinsics = intrinsics_cam

        # Keep a separate full-resolution image for the frustum if provided; otherwise upsample
        # (we'll reuse the original images_viz if it was full-res)
        if (H_v, W_v) != (H_d, W_d):
            # if original viz image differs from depth resolution, keep original for frustum by upsampling
            full_for_frustum = np.zeros((C, H_v, W_v), dtype=images_viz.dtype)
            full_for_frustum = images_viz[0]
        else:
            full_for_frustum = images_viz[0]

        # store as (3, H, W) in [0,1]
        self.latest_image = full_for_frustum

        self.update_point_cloud()
        self.update_camera_frustum()

    def update_point_cloud(self):
        if self.latest_points is None or self.latest_colors is None or self.latest_conf is None:
            return
        current_percentage = self.gui_points_conf.value
        threshold_val = np.percentile(self.latest_conf, current_percentage)
        conf_mask = (self.latest_conf >= threshold_val) & (self.latest_conf > 1e-12)
        # apply mask
        self.point_cloud.points = self.latest_points[conf_mask]
        self.point_cloud.colors = self.latest_colors[conf_mask]

    def update_camera_frustum(self):
        if (self.latest_extrinsics is None or self.latest_intrinsics is None or
                self.latest_image is None):
            return

        if self.camera_frustum is not None:
            self.camera_frustum.remove()
        if self.camera_frame is not None:
            self.camera_frame.remove()

        # Create 4x4 transformation matrix from 3x4 extrinsics
        cam2world_4x4 = np.eye(4)
        cam2world_4x4[:3, :] = self.latest_extrinsics
        T_world_camera = viser_tf.SE3.from_matrix(cam2world_4x4)

        # latest_image is (3, H, W) in [0,1] — convert to HWC uint8
        img = (np.clip(self.latest_image.transpose(1, 2, 0), 0.0, 1.0) * 255.0).astype(np.uint8)
        h, w = img.shape[:2]

        fx = self.latest_intrinsics[0, 0, 0]
        fov = 2 * np.arctan2(w / 2, fx)

        self.camera_frustum = self.server.scene.add_camera_frustum(
            "latest_frame/frustum",
            fov=fov,
            aspect=w / h,
            scale=0.05,
            image=img,
            line_width=1.0
        )

        self.camera_frame = self.server.scene.add_frame(
            "latest_frame",
            wxyz=T_world_camera.rotation().wxyz,
            position=T_world_camera.translation(),
            axes_length=0.05,
            axes_radius=0.002,
            origin_radius=0.002,
        )

        self.update_camera_visibility()

    def update_camera_visibility(self):
        if self.camera_frustum is not None:
            self.camera_frustum.visible = self.gui_show_frames.value
        if self.camera_frame is not None:
            self.camera_frame.visible = self.gui_show_frames.value


def main():
    parser = argparse.ArgumentParser(description="VGGT real-time webcam demo with viser for 3D visualization")
    parser.add_argument("--use_point_map", action="store_true")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--conf_threshold", type=float, default=25.0)
    parser.add_argument("--fps", type=float, default=5.0)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("Initializing and loading VGGT model...")
    model = VGGT()
    _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
    model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))
    model.eval()
    model = model.to(device)

    visualizer = RealTimeViserWrapper(
        port=args.port,
        init_conf_threshold=args.conf_threshold,
        use_point_map=args.use_point_map,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise ValueError("Could not open webcam")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("Starting real-time webcam processing...")
    print("Press Ctrl+C to exit")

    frame_interval = 1.0 / args.fps
    last_frame_time = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame")
                break

            current_time = time.time()
            if current_time - last_frame_time >= frame_interval:
                last_frame_time = current_time

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # full-res viz image in [0,1], CHW, batched
                vis_images = (frame_rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)[None, ...]

                # model input
                images = preprocess_frame(frame_rgb).to(device)

                # pick dtype safely
                if device == "cuda":
                    capability = torch.cuda.get_device_capability()
                    if capability[0] >= 8:
                        amp_dtype = torch.bfloat16
                    else:
                        amp_dtype = torch.float16
                else:
                    amp_dtype = torch.float32

                with torch.no_grad():
                    if device == "cuda":
                        # use new API
                        with torch.amp.autocast(device_type="cuda", dtype=amp_dtype):
                            predictions = model(images)
                    else:
                        predictions = model(images)

                extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
                predictions["extrinsic"] = extrinsic
                predictions["intrinsic"] = intrinsic

                # move tensors to cpu numpy and squeeze batch dim where appropriate
                for key in list(predictions.keys()):
                    if isinstance(predictions[key], torch.Tensor):
                        predictions[key] = predictions[key].cpu().numpy().squeeze(0)

                # attach the full-res visualization image (0..1 float)
                predictions["images_for_viz"] = vis_images

                visualizer.update_data(predictions)

            time.sleep(0.001)

    except KeyboardInterrupt:
        print("Exiting...")
    finally:
        cap.release()
        print("Webcam released")


if __name__ == "__main__":
    main()
