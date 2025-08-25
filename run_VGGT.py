import numpy as np
import torch
import torch.nn.functional as F

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

from vggt.utils.pose_enc import pose_encoding_to_extri_intri

def run_VGGT(model, images, dtype, resolution=518, batch_size=6):
    
    num_images = len(images)
    all_extrinsic = []
    all_intrinsic = []
    all_depth_map = []
    all_depth_conf = []

    print(f"Processing {num_images} images in batches of {batch_size}")

    for i in range(0, num_images, batch_size):
        end_idx = min(i + batch_size, num_images)
        batch_images = images[i:end_idx]

        print(f"Processing batch {i//batch_size + 1}/{(num_images-1)//batch_size + 1}: images {i}-{end_idx-1}")

        # Process this batch
        batch_images = F.interpolate(batch_images, size=(resolution, resolution), mode="bilinear", align_corners=False)

        device = "cuda" if torch.cuda.is_available() else "cpu"

        with torch.no_grad():
            with torch.amp.autocast(device_type=device, dtype=dtype):
                batch_images = batch_images[None]  # add batch dimension
                aggregated_tokens_list, ps_idx = model.aggregator(batch_images)

            # Predict Cameras
            pose_enc = model.camera_head(aggregated_tokens_list)[-1]
            extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, batch_images.shape[-2:])

            # Predict Depth Maps
            depth_map, depth_conf = model.depth_head(aggregated_tokens_list, batch_images, ps_idx)

        # Convert to numpy and store
        all_extrinsic.append(extrinsic.squeeze(0).cpu().numpy())
        all_intrinsic.append(intrinsic.squeeze(0).cpu().numpy())
        all_depth_map.append(depth_map.squeeze(0).cpu().numpy())
        all_depth_conf.append(depth_conf.squeeze(0).cpu().numpy())

        # Clear GPU cache
        torch.cuda.empty_cache()

    # Concatenate all results
    extrinsic = np.concatenate(all_extrinsic, axis=0)
    intrinsic = np.concatenate(all_intrinsic, axis=0)
    depth_map = np.concatenate(all_depth_map, axis=0)
    depth_conf = np.concatenate(all_depth_conf, axis=0)

    return extrinsic, intrinsic, depth_map, depth_conf