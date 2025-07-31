import numpy as np

import gin
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from arg_parser import parse_arguments, TaskMode
from model.dataset.isaac_sim_dataset import XMobilityIsaacSimDataModule
from model.trainer import XMobilityTrainer
from model.eval.prediction_evaluator import PredictionEvaulator


@gin.configurable
def evaluate_observation(dataset_path, checkpoint_path, wandb_entity_name,
                         wandb_project_name, wandb_run_name, num_gpus,
                         precision):
    data = XMobilityIsaacSimDataModule(dataset_path=dataset_path)
    model = XMobilityTrainer.load_from_checkpoint(checkpoint_path)
    model.eval()

    # TensorBoardLogger 생성 (version에 run_name 활용)
    tb_logger = TensorBoardLogger(
        save_dir="logs",
        name="observation_evaluation",
        version=wandb_run_name
    )

    trainer = pl.Trainer(num_nodes=num_gpus,
                         precision=precision,
                         logger=tb_logger,
                         strategy='ddp')
    trainer.test(model, datamodule=data)

    # wandb.finish() 제거


@gin.configurable
def evaluate_prediction(dataset_path,
                        checkpoint_path,
                        wandb_entity_name,
                        wandb_project_name,
                        wandb_run_name,
                        max_history_length=2,
                        max_future_length=[1, 3, 6],
                        use_trained_policy=False):
    data_module = XMobilityIsaacSimDataModule(
        dataset_path=dataset_path,
        sequence_length=max_history_length + np.max(max_future_length))
    model = XMobilityTrainer.load_from_checkpoint(checkpoint_path,
                                                  strict=False)
    model.eval()

    # TensorBoardLogger 생성 (version에 run_name 활용)
    tb_logger = TensorBoardLogger(
        save_dir="logs",
        name="prediction_evaluation",
        version=wandb_run_name
    )

    evaulator = PredictionEvaulator(model, data_module, tb_logger,
                                    max_history_length, max_future_length,
                                    use_trained_policy)
    evaulator.compute()

    # wandb.finish() 제거


def main():
    args = parse_arguments(TaskMode.EVAL)

    for config_file in args.config_files:
        gin.parse_config_file(config_file, skip_unknown=True)

    if args.eval_target == 'observation':
        # Run the evaluation loop.
        evaluate_observation(args.dataset_path, args.checkpoint_path,
                             args.wandb_entity_name, args.wandb_project_name,
                             args.wandb_run_name, args.num_gpus, args.precision)
    elif args.eval_target == 'imagination':
        evaluate_prediction(args.dataset_path, args.checkpoint_path,
                            args.wandb_entity_name, args.wandb_project_name,
                            args.wandb_run_name)
    else:
        raise ValueError('Unsupported eval target.')


if __name__ == '__main__':
    main()
