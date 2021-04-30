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

    input_size = (4, 15, 15)  # (MRI, H, W)
    output_size = (5, 15, 15)  # (Class, H, W)

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

        # Overall
        parser.add_argument("--gan_epoch", type=int, default=500)
        parser.add_argument("--no_adversary", action="store_true")

        return parser

    def __init__(self, conf):
        super().__init__(conf=conf)

        # Segmenter
        self.segmenter = LiBrainTumorSegGen(
            dropout1=self.hparams["dropout_1"], dropout2=self.hparams["dropout_2"]
        )

        # Adversary
        self.adversary = LiBrainTumorSegAdv()
        self.use_adversary = not self.hparams["no_adversary"]

        # Get Dice Loss for each Class
        dice_losses = dict()
        for cls_id, cls_name in enumerate(
            ["empty", "brain", "edema", "nonenhance", "enhance"]
        ):
            dice_losses[cls_name] = util.SparseDiceLoss(cls_id)
        self.dice_losses = dice_losses

        # Enable Manual Optimization
        self.automatic_optimization = False

    def forward(self, x):
        """Predict Segmentation from input patch"""
        return self.segmenter(x)

    def one_hot(self, label, dtype):
        """One-Hot Encoding of Segmenter's results"""
        label_onehot = nn.functional.one_hot(label, num_classes=self.output_size[0])
        label_onehot.requires_grad_(False)
        label_onehot = label_onehot.type(dtype).transpose_(-1, 1)
        label_onehot = label_onehot.to(device=self.device)
        return label_onehot

    def training_step(self, batch, batch_idx, optimizer_idx):
        """Train GAN on single batch"""
        patch, label = batch
        batch_size = patch.shape[0]

        # Zero out gradients
        seg_opt, adv_opt = self.optimizers()
        seg_opt.zero_grad()
        adv_opt.zero_grad()

        # Run through segmenter
        gan_out, gen_segment = self.segmenter(patch)

        # Downsample label from 31x31 -> 15x15 and convert to one_hot encoding
        label_downsample = label[:, 1::2, 1::2]
        label_onehot = self.one_hot(label_downsample, patch.dtype)

        # Run through adversary
        adv_real = self.adversary.forward(patch, label_onehot)
        real_labels = self.adversary.real_label(batch_size, adv_real.dtype, self.device)
        real_loss = self.adversary.loss(adv_real, real_labels)

        # Plot results once an epoch
        if batch_idx == 0:
            fig = util.plot_model(patch[0], label[0], gan_out[0].argmax(dim=0))
            self.logger.experiment.log({"train_predictions": fig})

        # Training Segmenter
        gan_epoch = self.hparams["gan_epoch"]
        if (self.global_step % (2 * gan_epoch)) > gan_epoch or not self.use_adversary:
            loss = self.segmenter.loss(gan_out, label_downsample)
            if self.use_adversary:
                loss += real_loss
            log_dict = {"gen_loss": loss}

            # Log dice losses
            for cls_name, cls_dsc in self.dice_losses.items():
                log_dict[f"train_{cls_name}_dsc"] = cls_dsc(gan_out, label_downsample)

            self.log_dict(log_dict)

            # Update Segmenter's Weights
            self.manual_backward(loss)
            seg_opt.step()

        # Training Adversary
        else:
            # Compute Fake Loss -> Loss
            adv_fake = self.adversary.forward(patch, gen_segment)
            fake_labels = self.adversary.fake_label(
                adv_fake.shape[0], adv_fake.dtype, self.device
            )
            fake_loss = self.adversary.loss(adv_fake, fake_labels)
            loss = (real_loss + fake_loss) / 2
            self.log_dict(
                {"adv_loss": loss},
                prog_bar=True,
                on_step=True,
            )
            # Train Adversary
            self.manual_backward(loss)
            adv_opt.step()

        return loss

    def validation_step(self, batch, batch_idx):
        # Compute Segmentation Loss
        patch, label = batch
        x, _ = self.segmenter(patch)

        # Log dice losses
        log_dict = dict()
        label_downsample = label[:, 1::2, 1::2]
        for cls_name, cls_dsc in self.dice_losses.items():
            log_dict[f"val_{cls_name}_dsc"] = cls_dsc(x, label_downsample)

        self.log_dict(log_dict)

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

        # Placeholders for real/fake labels
        self._fake_label = None
        self._real_label = None

    def forward(self, mri_patch, mri_segment):
        # Predicts 1 iff mri_segment is real
        patch_feat = self.mri_features(mri_patch)
        seg_feat = self.seg_features(mri_segment)
        x = torch.cat((patch_feat, seg_feat), dim=1)
        return self.discriminator(x).squeeze(1)

    def real_label(self, batch_size, dtype, device):
        """Return a real label for use in training"""
        label = self._get_label(self._real_label, 1, batch_size, dtype, device)
        self._real_label = label
        return label

    def fake_label(self, batch_size, dtype, device):
        """Return a fake label for use in training"""
        label = self._get_label(self._fake_label, 0, batch_size, dtype, device)
        self._fake_label = label
        return label

    def _get_label(
        self, label: torch.Tensor, fill_value, batch_size, dtype, device
    ) -> torch.Tensor:

        """Return the real_labels size"""
        if label is None or batch_size != label.shape[0]:
            return torch.full(
                (batch_size, 15, 15),
                fill_value,
                requires_grad=False,
                dtype=dtype,
                device=device,
            )
        else:
            return label


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
                dirpath=f"s3://11785-spring2021-project/{wandb_logger.experiment.project}/runs/{wandb_logger.experiment.id}",
                filename="checkpoint",
                monitor="val_edema_dsc",
                mode="max",
                save_top_k=1,
                verbose=True,
            ),
            LearningRateMonitor(log_momentum=False),
        ],
        logger=wandb_logger,
    )

    # Create Model
    if args.resume_from_checkpoint:
        model = util.fetch_model(
            LiBrainTumorSegGan, wandb_logger.experiment.project, args
        )
    else:
        model = LiBrainTumorSegGan(args)

    print(model.hparams)

    # Watch Model gradients
    wandb_logger.watch(model, log_freq=1000)

    # Get train/val dataloaders
    train_ds, val_ds, _ = Brats2017.split_dataset(load_ds=False)
    dl_args = {
        "direction": "axial",
        "patch_size": 31,
        "patch_depth": 1,
        "n_samples": args.limit_patients,
    }
    train_ds = Brats2017(train_ds, **dl_args)
    val_ds = Brats2017(val_ds, **dl_args)
    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        pin_memory=True,
        sampler=Brats2017.train_sampler(train_ds),
        num_workers=cpu_count() if torch.has_cuda else 0,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=32,
        pin_memory=True,
        sampler=Brats2017.test_sampler(val_ds),
        num_workers=cpu_count() if torch.has_cuda else 0,
    )

    # Kick off training
    trainer.fit(model, train_dl, val_dl)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--max_epochs", type=int, default=None)
    parser.add_argument("--limit_patients", type=int, default=20)
    parser.add_argument(
        "--tags", type=str, action="append", default=util.default_tags()
    )
    parser = LiBrainTumorSegGan.add_argparse_args(parser)
    args = parser.parse_args()

    # Get Trainer
    train(args)
