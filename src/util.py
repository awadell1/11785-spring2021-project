import argparse
from pathlib import Path
from torch import tensor
import wandb
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl
import boto3
from git import Repo


class ModelArtifact(ModelCheckpoint):
    def _save_model(self, filepath: str, trainer, pl_module):
        super()._save_model(filepath, trainer, pl_module)

        # Log artifact at most once
        run = trainer.logger.experiment
        artifact = wandb.Artifact(run.id, type="checkpoint")
        if not artifact.logged_by():
            artifact.add_reference(filepath, checksum=False)
            run.log_artifact(artifact)


class NNModule(pl.LightningModule):
    """ NN Module with some extra spice """

    input_size = None  # Expected Input size for the model

    @classmethod
    def default_args(cls):
        parser = cls.add_argparse_args()
        return parser.parse_args([])

    @classmethod
    def add_argparse_args(cls, parent=None):
        if parent is not None:
            parser = argparse.ArgumentParser(parents=[parent], add_help=False)
        else:
            parser = argparse.ArgumentParser()
        return parser

    def __init__(self, conf=None):
        super().__init__()
        if conf is not None:
            self.save_hyperparameters(conf)
        else:
            self.save_hyperparameters(self.default_args())


def default_tags():
    tags = []
    try:
        tags.append(Repo(".").head.reference.name)
    except:
        pass
    return tags


def fetch_model(Net, args):
    s3 = boto3.client("s3")
    ckpt = f"runs/{args.resume_from_checkpoint}/checkpoint.ckpt"
    Path(ckpt).parent.mkdir(exist_ok=True, parents=True)
    s3.download_file("11785-spring2021-hw3p2", ckpt, ckpt)
    return Net.load_from_checkpoint(ckpt, **vars(args))


def dice(y: torch.Tensor, y_hat: torch.Tensor):
    y_cls = torch.argmax(y, dim=1)
    intersect = torch.sum(y_cls * y_hat, dim=[1, 2])
    union = y_hat[0].numel() + y_cls[0].numel()
    return 2 * intersect / union
