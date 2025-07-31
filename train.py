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
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.S
# See the License for the specific language governing permissions and
# limitations under the License.

import os

import gin
import pytorch_lightning as pl
#from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.loggers import TensorBoardLogger
#from torch.utils.tensorboard import SummaryWriter

from arg_parser import parse_arguments, TaskMode
from model.dataset.isaac_sim_dataset import XMobilityIsaacSimDataModule  # pylint: disable=unused-import
from model.trainer import XMobilityTrainer  # pylint: disable=unused-import


@gin.configurable
def train(dataset_path, output_dir, ckpt_path, wandb_entity_name,
          wandb_project_name, wandb_run_name, precision, epochs, data_module,
          model_trainer):
    # Create output directory if not exists.
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    data = data_module(dataset_path=dataset_path)
    if ckpt_path:
        model = model_trainer.load_from_checkpoint(checkpoint_path=ckpt_path,
                                                   strict=False)
    else:
        model = model_trainer()

    # W&B logger 대신 TensorBoardLogger 사용
    tb_logger = TensorBoardLogger(
        save_dir=output_dir,
        name="tensorboard_logs",
        version=wandb_run_name
    )

    callbacks = [
        pl.callbacks.ModelSummary(-1),
        pl.callbacks.LearningRateMonitor(),
        pl.callbacks.ModelCheckpoint(
            dirpath=os.path.join(output_dir, 'checkpoints'),
            save_top_k=3,
            monitor='val_loss',
            mode='min',
            save_last=True,
        ),
    ]
    trainer = pl.Trainer(
        max_epochs=epochs,
        precision=precision,
        accelerator='gpu',
        devices=2,
        strategy='ddp_find_unused_parameters_true',  
        sync_batchnorm=True,
        callbacks=callbacks,
        logger=tb_logger  
        )

    trainer.fit(model, datamodule=data)

    trainer.test(ckpt_path="last", datamodule=data)

    return tb_logger  # 반환값도 변경


def log_gin_config(logger: TensorBoardLogger, output_dir):
    # gin config를 output_dir에 저장만 하고 끝냄 (W&B Artifact 제거)
    gin_config_str = gin.operative_config_str()
    config_path = os.path.join(output_dir, "gin_config.txt")
    with open(config_path, "w", encoding='UTF-8') as f:
        f.write(gin_config_str)

    # 텐서보드에 직접 텍스트 로그는 여기서는 생략 (필요시 별도 구현 가능)


def main():
    args = parse_arguments(TaskMode.TRAIN)

    for config_file in args.config_files:
        gin.parse_config_file(config_file, skip_unknown=True)

    # train 함수 호출 시 W&B 관련 인자 이름 그대로 넘겨도 무방함
    tb_logger = train(
        args.dataset_path,
        args.output_dir,
        args.checkpoint_path,
        args.wandb_entity_name,
        args.wandb_project_name,
        args.wandb_run_name,
    )

    log_gin_config(tb_logger, args.output_dir)

    # wandb.finish() 삭제


if __name__ == '__main__':
    main()
