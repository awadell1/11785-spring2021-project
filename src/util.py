import argparse
from pathlib import Path
import torch
from torch import nn
import pytorch_lightning as pl
import boto3
from git import Repo


class NNModule(pl.LightningModule):
    """NN Module with some extra spice"""

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


def fetch_model(Net, project, args):
    s3 = boto3.client("s3")
    ckpt = f"{project}/runs/{args.resume_from_checkpoint}/checkpoint.ckpt"
    Path(ckpt).parent.mkdir(exist_ok=True, parents=True)
    s3.download_file("11785-spring2021-project", ckpt, ckpt)
    return Net.load_from_checkpoint(ckpt, **vars(args))


class SparseDiceLoss(nn.Module):
    def __init__(self, cls_id) -> None:
        super().__init__()
        self.cls_id = cls_id

    def forward(self, y_pred, y_ref):
        """Compute Dice Loss using:
        y_pred - One-Hot Encoding of Segmentation from Model
        y_ref - Categorical Encoding of Reference Segmentation
        """
        # Mask the reference to just the target class
        y_ref_mask = y_ref == self.cls_id
        y_pred_mask = y_pred[:, self.cls_id]

        # Sum over all but batch dimension
        sum_dim = list(range(1, len(y_pred.shape) - 1))
        intersect = torch.sum(y_ref_mask * y_pred_mask, dim=sum_dim)
        union = torch.sum(y_ref_mask + y_pred_mask, dim=sum_dim)
        return torch.mean(2 * intersect / union)
