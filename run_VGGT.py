import random
import numpy as np
import glob
import os
import copy
import torch
import torch.nn.functional as F

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

import argparse
from pathlib import Path
import trimesh
import pycolmap

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images_square
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map
from vggt.utils.helper import create_pixel_coordinate_grid, randomly_limit_trues
from vggt.dependency.track_predict import predict_tracks
from vggt.dependency.np_to_pycolmap import batch_np_matrix_to_pycolmap, batch_np_matrix_to_pycolmap_wo_track

def run_VGGT(model, images, dtype, resolution=518, batch_size=90):
    """
    Process images in batches to avoid GPU memory issues
    """
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

        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=dtype):
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