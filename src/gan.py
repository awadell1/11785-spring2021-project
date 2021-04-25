""" Implimentation of GAN for Brain Tumor Segmentation """
# Standard imports
import logging
import argparse
from typing import List

# Import torch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from psutil import cpu_count

# Load code from disk
from . import util
from .dataset import Brats2017

# Start Logging
logging.getLogger().setLevel(logging.DEBUG)


class LiBrainTumorSegGan(util.NNModule):
    """
    Pytorch Implementation of GAN presented by:
    Z. Li, Y. Wang, and J. Yu, “Brain Tumor Segmentation Using an Adversarial Network,”
    in Brainlesion: Glioma, Multiple Sclerosis, Stroke and Traumatic Brain Injuries,
    vol. 10670, A. Crimi, S. Bakas, H. Kuijf, B. Menze, and M. Reyes, Eds. Cham:
    Springer International Publishing, 2018, pp. 123–132.
    """

    @classmethod
    def add_argparse_args(cls, parent=None):
        parser = super().add_argparse_args(parent=parent)

        # Segmenter
        parser.add_argument("--dropout_1", type=float, default=0.25)
        parser.add_argument("--dropout_2", type=float, default=0.25)
        parser.add_argument("--seg_lr", type=float, default=1e-3)
        parser.add_argument("--seg_beta1", type=float, default=0.9)
        parser.add_argument("--seg_beta2", type=float, default=0.999)
        parser.add_argument("--seg_gamma", type=float, default=0.96)

        # Adversary
        parser.add_argument("--adv_lr", type=float, default=1e-4)
        parser.add_argument("--adv_beta1", type=float, default=0.9)
        parser.add_argument("--adv_beta2", type=float, default=0.999)
        parser.add_argument("--adv_gamma", type=float, default=0.96)

        return parser

    def __init__(self, conf):
        super().__init__(conf=conf)

        # Segmenter
        self.segmenter = LiBrainTumorSegGen(
            dropout1=self.hparams["dropout_1"], dropout2=self.hparams["dropout_2"]
        )

        # Adversary
        self.adversary = LiBrainTumorSegAdv()

    def forward(self, x):
        """ Predict Segmentation from input patch """
        return self.segmenter(x)

    def training_step(self, batch, batch_idx, optimizer_idx):
        """ Train GAN on single batch """
        patch, label = batch

        # Run through segmenter
        gan_out, gen_segment = self.segmenter(patch)

        # Downsample label from 31x31 -> 15x15 and convert to one_hot encoding
        label_downsample = label[:, 1::2, 1::2]
        label_onehot = nn.functional.one_hot(label_downsample, num_classes=5)
        label_onehot = label_onehot.type(patch.dtype).transpose(-1, 1)
        label_onehot = label_onehot.to(self.device)

        # Run through adversary
        batch_size = label.shape[0]
        adv_real = self.adversary.forward(patch, label_onehot)
        real_labels = torch.ones(batch_size, 15, 15).to(self.device)
        real_loss = self.adversary.loss(adv_real, real_labels)

        # Training Segmenter
        if optimizer_idx == 0:
            loss = self.segmenter.loss(gan_out, label_downsample) + real_loss
            dsc = util.dice(gan_out, label_downsample).mean()
            self.log_dict({"gen_loss": loss, "train_dice": dsc})

        # Training Adversary
        elif optimizer_idx == 1:
            # Compute Fake Loss -> Loss
            adv_fake = self.adversary.forward(patch, gen_segment)
            fake_labels = torch.ones(batch_size, 15, 15).to(self.device)
            fake_loss = self.adversary.loss(adv_fake, fake_labels)
            loss = (real_loss + fake_loss) / 2
            self.log_dict({"adv_loss": loss})

        return loss

    def validation_step(self, batch, batch_idx):
        # Compute Segmentation Loss
        patch, label = batch
        x, _ = self.segmenter(patch)
        return util.dice(x, label[:, 1::2, 1::2])

    def validation_epoch_end(self, outputs: List[torch.Tensor]) -> None:
        avg_dice = torch.stack(outputs).mean()
        self.log_dict({"val_dice": avg_dice})
        return super().validation_epoch_end(outputs)

    def configure_optimizers(self):

        # Configure Segmenter Optimizers
        opt_seg = Adam(
            self.segmenter.parameters(),
            lr=self.hparams["seg_lr"],
            betas=(self.hparams["seg_beta1"], self.hparams["seg_beta2"]),
        )
        seg_sched = ExponentialLR(opt_seg, gamma=self.hparams["seg_gamma"])

        # Configure Adversary Optimizer
        opt_adv = Adam(
            self.segmenter.parameters(),
            lr=self.hparams["adv_lr"],
            betas=(self.hparams["adv_beta1"], self.hparams["adv_beta2"]),
        )
        adv_sched = ExponentialLR(opt_adv, gamma=self.hparams["adv_gamma"])

        return [opt_seg, opt_adv], [seg_sched, adv_sched]


class LiBrainTumorSegGen(nn.Module):
    """
    Pytorch Implementation of Generative network from:
    Z. Li, Y. Wang, and J. Yu, “Brain Tumor Segmentation Using an Adversarial Network,”
    in Brainlesion: Glioma, Multiple Sclerosis, Stroke and Traumatic Brain Injuries,
    vol. 10670, A. Crimi, S. Bakas, H. Kuijf, B. Menze, and M. Reyes, Eds. Cham:
    Springer International Publishing, 2018, pp. 123–132.
    """

    def __init__(self, dropout1=0.5, dropout2=0.5):
        super().__init__()

        # Feature Extraction Layers
        self.layers = nn.Sequential(
            CNNBlock(4, 64, kernel=3, pad=0),
            *[CNNBlock(64, 64, kernel=3, pad=0) for _ in range(3)],
            CNNBlock(64, 128, kernel=3, pad=0),
            *[CNNBlock(128, 128, kernel=3, pad=0) for _ in range(3)],
            nn.Dropout2d(dropout1),
            CNNBlock(128, 256, kernel=1),
            nn.Dropout2d(dropout2),
            nn.Conv2d(256, 5, kernel_size=1),
        )

        # Loss and log logits
        self.loss = nn.CrossEntropyLoss()
        self.logits = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.layers(x)
        return x, self.logits(x)


class LiBrainTumorSegAdv(nn.Module):
    """
    Pytorch Implementation of Adversarial network from:
    Z. Li, Y. Wang, and J. Yu, “Brain Tumor Segmentation Using an Adversarial Network,”
    in Brainlesion: Glioma, Multiple Sclerosis, Stroke and Traumatic Brain Injuries,
    vol. 10670, A. Crimi, S. Bakas, H. Kuijf, B. Menze, and M. Reyes, Eds. Cham:
    Springer International Publishing, 2018, pp. 123–132.
    """

    def __init__(self):
        super().__init__()

        self.mri_features = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(16, 64, kernel_size=3, padding=0, stride=2),
            nn.LeakyReLU(),
        )

        self.seg_features = nn.Sequential(
            nn.Conv2d(5, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(),
        )

        self.discriminator = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 1, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, mri_patch, mri_segment):
        # Predicts 1 iff mri_segment is real
        patch_feat = self.mri_features(mri_patch)
        seg_feat = self.seg_features(mri_segment)
        x = torch.cat((patch_feat, seg_feat), dim=1)
        return self.discriminator(x).squeeze(1)


class CNNBlock(nn.Module):
    def __init__(self, c_in, c_out, kernel=3, pad=None):
        super().__init__()
        pad = int((kernel - 1) / 2) if pad is None else pad
        self.layers = nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel_size=kernel, padding=pad),
            nn.BatchNorm2d(c_out),
            nn.ReLU(),
        )

        # Init Weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.layers(x)


def train(args):
    wandb_logger = WandbLogger(
        project="LiBrainTumorGAN", entity="idl-gan-brain-tumors", tags=args.tags
    )
    trainer = pl.Trainer(
        gpus=1 if torch.has_cuda else 0,
        precision=16 if torch.has_cuda else 32,
        log_every_n_steps=5,
        flush_logs_every_n_steps=10,
        max_epochs=args.max_epochs,
        callbacks=[
            ModelCheckpoint(
                dirpath=f"s3://11785-spring2021-hw3p2/LiBrainTumorGAN/runs/{wandb_logger.experiment.id}",
                filename="checkpoint",
                monitor="gen_loss",
                mode="min",
                save_top_k=1,
                verbose=True,
            ),
            LearningRateMonitor(log_momentum=False),
        ],
        logger=wandb_logger,
    )

    # Create Model
    if args.resume_from_checkpoint:
        model = util.fetch_model(LiBrainTumorSegGan, args)
    else:
        model = LiBrainTumorSegGan(args)

    print(model.hparams)

    # Watch Model gradients
    wandb_logger.watch(model, log_freq=1000)

    # Get train/val dataloaders
    train_ds, val_ds, _ = Brats2017.split_dataset(
        direction="axial",
        patch_size=31,
        patch_depth=1,
        n_samples=30,
    )
    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        pin_memory=True,
        shuffle=True,
        num_workers=cpu_count() if torch.has_cuda else 0,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=32,
        pin_memory=True,
        num_workers=cpu_count() if torch.has_cuda else 0,
    )

    # Kick off training
    trainer.fit(model, train_dl, val_dl)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--max_epochs", type=int, default=None)
    parser.add_argument(
        "--tags", type=str, action="append", default=util.default_tags()
    )
    parser = LiBrainTumorSegGan.add_argparse_args(parser)
    args = parser.parse_args()

    # Get Trainer
    train(args)
