""" Implimentation of GAN for Brain Tumor Segmentation """
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from psutil import cpu_count
from .util import NNModule


class BrainTumorSegGan(NNModule):
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

        # Optimizer
        parser.add_argument("--lr_init", type=float, default=2e-3)
        parser.add_argument("--weight_decay", type=float, default=5e-6)
        parser.add_argument("--lr_patience", type=float, default=1)
        parser.add_argument("--lr_factor", type=float, default=0.5)

        return parser

    def __init__(self, conf):
        super().__init__(conf=conf)


class BrainTumorSegGen(nn.Module):
    """
    Pytorch Implementation of Generative network from:
    Z. Li, Y. Wang, and J. Yu, “Brain Tumor Segmentation Using an Adversarial Network,”
    in Brainlesion: Glioma, Multiple Sclerosis, Stroke and Traumatic Brain Injuries,
    vol. 10670, A. Crimi, S. Bakas, H. Kuijf, B. Menze, and M. Reyes, Eds. Cham:
    Springer International Publishing, 2018, pp. 123–132.
    """

    def __init__(self, dropout1=0.5, dropout2=0.5):
        # Feature Extraction Layers
        self.layers = nn.Sequential(
            CNNBlock(4, 64, kernel=3) * [CNNBlock(64, 64, kernel=3) for _ in range(3)],
            CNNBlock(64, 128, kernel=3)
            * [CNNBlock(64, 64, kernel=3) for _ in range(3)],
            nn.Dropout2d(dropout1),
            CNNBlock(256, 256, kernel=1),
            nn.Dropout2d(dropout2),
            nn.Conv2d(256, 5, kernel_size=1),
        )

        # Loss and log logits
        self.loss = nn.CrossEntropyLoss()
        self.logits = nn.LogSoftmax()

    def forward(self, x):
        x = self.layers(x)
        return x, self.logits(x)


class BrainTumorSegAdv(nn.Module):
    """
    Pytorch Implementation of Adversarial network from:
    Z. Li, Y. Wang, and J. Yu, “Brain Tumor Segmentation Using an Adversarial Network,”
    in Brainlesion: Glioma, Multiple Sclerosis, Stroke and Traumatic Brain Injuries,
    vol. 10670, A. Crimi, S. Bakas, H. Kuijf, B. Menze, and M. Reyes, Eds. Cham:
    Springer International Publishing, 2018, pp. 123–132.
    """

    def __init__(self):
        super().__init__()

        self.image_features = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(16, 64, kernel_size=3, padding=1, stride=2),
            nn.LeakyReLU(),
        )

        self.label_features = nn.Sequential(
            nn.Conv2d(5, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(),
        )

        self.discriminator = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 2, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.loss = nn.CrossEntropyLoss()

    def forward(self, img, labels):
        img_feat = self.image_features(img)
        label_feat = self.label_features(labels)
        x = torch.stack((img_feat, label_feat), dim=1)
        return self.discriminator(x)


class CNNBlock(nn.Module):
    def __init__(self, c_in, c_out, kernel=3):
        pad = int((kernel - 1) / 2)
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
