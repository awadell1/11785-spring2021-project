import argparse
import wandb
from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl


class ModelArtifact(ModelCheckpoint):
    def _save_model(self, filepath: str, trainer, pl_module):
        super()._save_model(filepath, trainer, pl_module)

        # Log artifact at most once
        run = trainer.logger.experiment
        artifact = wandb.Artifact(run.id, type="checkpoint")
        if not artifact.logged_by:
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
