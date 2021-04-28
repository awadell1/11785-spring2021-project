from logging import log
import torch
from src.gan import LiBrainTumorSegAdv, LiBrainTumorSegGan, LiBrainTumorSegGen


def test_segmentor():
    seg = LiBrainTumorSegGen()
    input = torch.rand((1, 4, 31, 31))
    out, logits = seg(input)

    # Check shape
    assert list(out.shape) == [1, 5, 15, 15]
    assert list(logits.shape) == [1, 5, 15, 15]

    # Check logits sum to 1
    logit_sums = torch.sum(torch.exp(logits), dim=1)
    expected = torch.ones((out.shape))
    all(torch.isclose(logit_sums, expected).flatten())


def test_adversary():
    adv = LiBrainTumorSegAdv()
    mri_patch = torch.rand((1, 4, 31, 31))
    labels = torch.rand((1, 5, 15, 15))

    out = adv(mri_patch, labels)
    assert list(out.shape) == [1, 15, 15]
