""" Implimentation of GAN for Brain Tumor Segmentation """
# Standard imports
import logging
import argparse
from typing import List

# Import torch
import torch
import torch.nn as nn
from torch.tensor import Tensor
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

    Predicted Classes
        Output Label                    Brats2017 Class
        Label 0: Non-Tumor              Label 0
        Label 1: Core Tumor Regions     Label 1
        Label 2: Whole Tumor            Label 2
        Label 3: Enhancing Tumor        Label 4
    """

    input_size = (4, 15, 15)  # (MRI, H, W)
    output_size = (4, 15, 15)  # (Class, H, W)

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
        parser.add_argument("--no_adversary", action="store_true")
        parser.add_argument("--adversary_weight", type=float, default=1.0)

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
        self.dice = util.SparseDiceLoss(input_type="log_softmax")

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
        label[label == 4] = 3  # Remap Brats2017 Label 4 -> 3
        label_downsample = label[:, 1::2, 1::2]

        # Get loss
        if optimizer_idx == 0:
            loss, gan_out = self._seg_train(patch, label_downsample)

            # Plot results once an epoch
            if batch_idx == 0:
                fig = util.plot_model(patch[0], label[0], gan_out[0].argmax(dim=0))
                self.logger.experiment.log({"train_predictions": fig})
            return loss

        return self._adv_train(patch, label_downsample)

    def _seg_train(self, patch, label_downsample):
        # Run through segmenter
        gan_out, gen_segment = self.segmenter(patch)
        loss = self.segmenter.loss(gan_out, label_downsample)

        # Log Dice Loss
        self.log("train_dice", self.dice(gen_segment, label_downsample))

        if self.use_adversary:
            # Run Prediction through Adversary
            adv_fake = self.adversary(patch, gen_segment)
            real_labels = self.adversary.real_label(adv_fake)
            adv_tricked_loss = self.adversary.loss(adv_fake, real_labels)
            self.log("adv_tricked_loss", adv_tricked_loss)
            loss += adv_tricked_loss * self.hparams["adversary_weight"]

        self.log("train_seg_loss", loss)
        return loss, gan_out

    def _adv_train(self, patch, label_downsample):
        # Loss for Classifying Segmenter as Real
        gen_segment = self.segmenter(patch)[1]
        adv_fake = self.adversary(patch, gen_segment)
        fake_labels = self.adversary.fake_label(adv_fake)
        adv_fake_loss = self.adversary.loss(adv_fake, fake_labels)
        self.log("adv_fake_loss", adv_fake_loss)

        # Loss for Classifying Expert as Fake
        label_onehot = self.one_hot(label_downsample, patch.dtype)
        adv_real = self.adversary(patch, label_onehot)
        real_labels = self.adversary.real_label(adv_real)
        adv_real_loss = self.adversary.loss(adv_real, real_labels)
        self.log("adv_real_loss", adv_real_loss)

        # Adversary Loss
        loss = (adv_real_loss + adv_fake_loss) / 2
        self.log("train_adv_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        # Compute Segmentation Loss
        patch, label = batch
        x, logits = self.segmenter(patch)

        # Plot results once an epoch
        if batch_idx == 0:
            fig = util.plot_model(patch[0], label[0], x[0].argmax(dim=0))
            self.logger.experiment.log({"val_predictions": fig})

        # Log dice losses
        self.log("val_dice", self.dice(logits, label[:, 1::2, 1::2]))

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
            self.adversary.parameters(),
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
            nn.Conv2d(256, 4, kernel_size=1),
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
            nn.Conv2d(4, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(),
        )

        self.discriminator = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=0),
            nn.LeakyReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=0),
            nn.LeakyReLU(),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 1, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.Flatten(0),
        )
        self.loss = nn.BCEWithLogitsLoss()

        # Placeholders for real/fake labels
        self._fake_label = None
        self._real_label = None

        # Init Weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight,
                    mode="fan_out",
                    nonlinearity="leaky_relu",
                )
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0)

    def forward(self, mri_patch, mri_segment):
        # Predicts 1 iff mri_segment is real
        patch_feat = self.mri_features(mri_patch)
        seg_feat = self.seg_features(mri_segment)
        x = torch.cat((patch_feat, seg_feat), dim=1)
        return self.discriminator(x)

    def real_label(self, ref):
        """Return a real label for use in training"""
        label = self._get_label(self._real_label, ref, 1)
        self._real_label = label
        return label

    def fake_label(self, ref):
        """Return a fake label for use in training"""
        label = self._get_label(self._fake_label, ref, 0)
        self._fake_label = label
        return label

    def _get_label(
        self, label: torch.Tensor, ref: torch.Tensor, fill_value
    ) -> torch.Tensor:

        """Return the real_labels size"""
        batch_size = ref.shape[0]
        if label is None or batch_size != label.shape[0]:
            return torch.full(
                (batch_size,),
                fill_value,
                requires_grad=False,
                dtype=ref.dtype,
                device=ref.device,
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
                nn.init.normal_(m.weight.data, 1.0, 0.02)
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
        gradient_clip_val=0.2,
        callbacks=[
            ModelCheckpoint(
                dirpath=f"s3://11785-spring2021-project/{wandb_logger.experiment.project}/runs/{wandb_logger.experiment.id}",
                filename="checkpoint",
                monitor="val_dice",
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
