# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Dict
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor
from torch.utils.tensorboard import SummaryWriter

from model.x_mobility.utils import pack_sequence_dim, unpack_sequence_dim

SEMANTIC_COLORS = np.array(
    [
        [128, 128, 128],  # Background
        [0, 255, 0],  # NavigableSurface
        [255, 165, 0],  # Forklift
        [0, 0, 255],  # Pallet
        [255, 255, 0],  # Cone
        [255, 0, 255],  # Sign
        [255, 0, 0]  # Fence
    ],
    dtype=np.uint8)

writer = SummaryWriter(log_dir='runs/x_mobility_tensorboard')

def visualise_semantic(batch: Dict, output: Dict) -> torch.Tensor:
    target = batch['semantic_label'][:, :, 0]
    pred = torch.argmax(output['semantic_segmentation_1'].detach(), dim=-3)

    color_map = torch.tensor(SEMANTIC_COLORS, dtype=torch.uint8, device=pred.device)
    target = color_map[target.int()]
    pred = color_map[pred.int()]

    target = target.permute(0, 1, 4, 2, 3)
    pred = pred.permute(0, 1, 4, 2, 3)
    viz_video = torch.cat([target, pred], dim=-1).detach()

    b, s = viz_video.shape[:2]
    viz_video_2d = viz_video.reshape(b * s, 3, *viz_video.shape[-2:])
    writer.add_images("Semantic_GT_vs_Pred", viz_video_2d, dataformats='NCHW')

    return viz_video

def visualise_rgb(batch: Dict, output: Dict) -> torch.Tensor:
    target = torch.clamp(torch.round(batch['image'] * 255.0), 0, 255).to(torch.uint8)
    pred = torch.clamp(torch.round(output['rgb_1'].detach() * 255.0), 0, 255).to(torch.uint8)
    viz_video = torch.cat([target, pred], dim=-1).detach()

    b, s = viz_video.shape[:2]
    viz_video_2d = viz_video.reshape(b * s, 3, *viz_video.shape[-2:])
    writer.add_images("RGB_GT_vs_Pred", viz_video_2d, dataformats='NCHW')

    return viz_video

def visualise_depth(batch: Dict, output: Dict) -> torch.Tensor:
    target = torch.clamp(torch.round(batch['image'] * 255.0), 0, 255).to(torch.uint8)
    b, s = output['depth'].shape[:2]

    depth_pred = pack_sequence_dim(output['depth'].detach())
    depth_viz = torch.nn.functional.interpolate(
        depth_pred.unsqueeze(1),
        size=batch['image'].shape[-2:],
        mode="bicubic",
        align_corners=False,
    )
    depth_viz = unpack_sequence_dim(depth_viz, b, s)
    depth_viz = torch.clamp(
        torch.round(depth_viz * 255.0 / torch.max(depth_viz)), 0, 255).to(torch.uint8)
    depth_viz = depth_viz.repeat(1, 1, 3, 1, 1)

    viz_video = torch.cat([target, depth_viz], dim=-1).detach()
    b, s = viz_video.shape[:2]
    viz_video_2d = viz_video.reshape(b * s, 3, *viz_video.shape[-2:])
    writer.add_images("RGB_vs_Depth", viz_video_2d, dataformats='NCHW')

    return viz_video

def visualise_attention(batch: Dict, output: Dict):
    image = np.clip(
        batch['image'][:, 0].permute(0, 2, 3, 1).cpu().numpy() * 255, 0,
        255).astype(np.uint8)
    attention_pred = output['image_attentions'].detach()[:, 0].cpu().numpy()
    attention_pred = (attention_pred - np.min(attention_pred)) / (
        np.max(attention_pred) - np.min(attention_pred))
    attention_pred = np.kron(attention_pred, np.ones((14, 14)))

    for i, (img, att) in enumerate(zip(image, attention_pred)):
        cmap = plt.get_cmap('jet')
        att_viz = cmap(att)
        att_viz = Image.fromarray(
            (att_viz[:, :, :3] * 255).astype(np.uint8)).convert('RGBA')

        img_viz = Image.fromarray(img).convert('RGBA')
        img_viz = img_viz.resize(att_viz.size[:2], resample=Image.BICUBIC)

        overlayed = Image.blend(img_viz, att_viz, alpha=0.5)
        overlayed_tensor = ToTensor()(overlayed)
        writer.add_image(f'AttentionOverlay/{i}', overlayed_tensor)

    return []
