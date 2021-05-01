from logging import log
import torch
from src.gan import LiBrainTumorSegAdv, LiBrainTumorSegGan, LiBrainTumorSegGen
from src.util import SparseDiceLoss


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
    assert all(torch.isclose(logit_sums, expected).flatten())


def test_adversary():
    adv = LiBrainTumorSegAdv()
    mri_patch = torch.rand((1, 4, 31, 31))
    labels = torch.rand((1, 5, 15, 15))

    out = adv(mri_patch, labels)
    assert list(out.shape) == [1, 15, 15]


def test_dice():
    loss = SparseDiceLoss(input_type="log_softmax")
    y_pred = torch.tensor([[0.7, 0.3, 0.6], [0.0, 0.8, 0.8]])
    y_pred = torch.stack((y_pred, 1 - y_pred), dim=1).log()
    y_ref = torch.tensor([[1, 0, 1], [0, 1, 1]])
    l1 = loss(y_pred, y_ref)
    l2 = loss(y_pred, (1 - y_ref).abs())
    assert 0 <= l1 and l1 <= 1
    assert 0 <= l2 and l2 <= 1
    assert torch.isclose(l1 + l2, torch.tensor(1.0))
