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

import gin
import pytorch_lightning as pl
import torch
from torch import nn

from model.loss.losses import XMobilityLoss
from model.x_mobility.x_mobility import XMobility
from model.eval.x_mobility_metrics import XMobilityMetrics
from model.visualization import visualise_rgb, visualise_semantic, visualise_depth, visualise_attention


@gin.configurable
class XMobilityTrainer(pl.LightningModule):
    '''Pytorch Lightning module of x-mobility for training (TensorBoard).'''
    def __init__(self, weight_decay: float, lr: float, scheduler_pct_start: float):
        super().__init__()
        self.weight_decay = weight_decay
        self.lr = lr
        self.scheduler_pct_start = scheduler_pct_start
        self.save_hyperparameters()

        self.model = XMobility()
        self.loss = XMobilityLoss()
        self.metrics = XMobilityMetrics()

    def forward(self, batch):
        return self.model(batch)

    def inference_prediction(self, batch, enable_semantic_inference=True, enable_rgb_inference=False):
        return self.model.inference_prediction(batch, enable_semantic_inference, enable_rgb_inference)

    def inference(self, batch, enable_semantic, enable_rgb, enable_depth):
        return self.model.inference(batch, enable_semantic, enable_rgb, enable_depth)

    def shared_step(self, batch):
        output = self.forward(batch)
        losses = self.loss(output, batch)
        return losses, output

    def training_step(self, batch, batch_idx):
        losses, output = self.shared_step(batch)
        self.log_and_visualize(batch, output, losses, batch_idx, prefix='train')
        return self.loss_reducing(losses)

    def validation_step(self, batch, batch_idx):
        losses, output = self.shared_step(batch)
        self.log_and_visualize(batch, output, losses, batch_idx, prefix='validation')
        val_loss = self.loss_reducing(losses)
        return {'val_loss': val_loss}

    def test_step(self, batch, batch_idx):
        losses, output = self.shared_step(batch)
        self.log_and_visualize(batch, output, losses, batch_idx, prefix='test')

    def log_and_visualize(self, batch, output, losses, batch_idx, prefix='train'):
        # Log losses
        for key, value in losses.items():
            self.log(f'{prefix}/losses/{key}', value, sync_dist=True)

        # Log metrics
        metrics = self.metrics.evaluate(batch, output, prefix)
        self.log_dict(metrics, sync_dist=True, on_epoch=True, on_step=False)

        # Only log video/images on validation set's first batch
        if prefix == 'validation':
            self.log('val_loss', self.loss_reducing(losses), sync_dist=True, on_epoch=True)
            if batch_idx == 0:
                self.visualise(batch, output, batch_idx, prefix=prefix)

    def log_video(self, name, viz):
        # TensorBoard requires 5D tensor [B, C, T, H, W]
        if isinstance(viz, torch.Tensor) and viz.dim() == 4:
            viz = viz.unsqueeze(0)
        if self.logger and hasattr(self.logger.experiment, "add_video"):
            self.logger.experiment.add_video(name, viz, global_step=self.global_step, fps=4)

    def log_image(self, name, img):
        if self.logger and hasattr(self.logger.experiment, "add_image"):
            if isinstance(img, list):
                for i, im in enumerate(img):
                    if isinstance(im, torch.Tensor) and im.dim() == 3:
                        self.logger.experiment.add_image(f"{name}_{i}", im, global_step=self.global_step)
            elif isinstance(img, torch.Tensor) and img.dim() == 3:
                self.logger.experiment.add_image(name, img, global_step=self.global_step)

    def visualise(self, batch, output, batch_idx, prefix='train'):
        epoch = self.current_epoch

        if 'semantic_segmentation_1' in output:
            semantic_viz = visualise_semantic(batch, output)
            self.log_video(f'{prefix}/semantic/e{epoch}_b{batch_idx}', semantic_viz)

        if 'rgb_1' in output:
            rgb_viz = visualise_rgb(batch, output)
            self.log_video(f'{prefix}/rgb/e{epoch}_b{batch_idx}', rgb_viz)

        if 'depth' in output:
            depth_viz = visualise_depth(batch, output)
            self.log_video(f'{prefix}/depth/e{epoch}_b{batch_idx}', depth_viz)

        if 'image_attentions' in output:
            att_viz = visualise_attention(batch, output)
            self.log_image(f'{prefix}/attention/e{epoch}_b{batch_idx}', att_viz)

    def loss_reducing(self, loss: torch.Tensor):
        return sum(loss.values())

    def configure_optimizers(self):
        parameters = self._add_weight_decay(
            self.model,
            self.weight_decay,
            skip_list=['relative_position_bias_table'],
        )
        optimizer = torch.optim.AdamW(parameters, lr=self.lr, weight_decay=0.0)
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.lr,
            total_steps=self.trainer.estimated_stepping_batches,
            pct_start=self.scheduler_pct_start,
        )
        return [optimizer], [{'scheduler': lr_scheduler, 'interval': 'step'}]

    def _add_weight_decay(self, model: nn.Module, weight_decay: float = 0.01, skip_list: list = None):
        no_decay, decay = [], []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if len(param.shape) == 1 or (skip_list and any(x in name for x in skip_list)):
                no_decay.append(param)
            else:
                decay.append(param)
        return [
            {'params': no_decay, 'weight_decay': 0.},
            {'params': decay, 'weight_decay': weight_decay}
        ]
