from logging import log
import torch
from src.gan import LiBrainTumorSegAdv, LiBrainTumorSegGan, LiBrainTumorSegGen
from src.dataset import Brats2017


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


def test_segmentor_loss():
    # Get Dataset
    batch_size = 2
    seg = LiBrainTumorSegGen()
    ds = Brats2017(
        "data/Brats17TrainingData",
        n_samples=1,
        patch_depth=1,
        patch_size=31,
    )
    for batch in torch.utils.data.DataLoader(ds, batch_size=batch_size):
        patch, label = batch
        out, logits = seg(patch)
        assert out.shape[0] == batch_size
        assert out.shape[1] == 5  # Channel is dim 1
        assert out.shape[2] == 15
        assert out.shape[3] == 15
        loss = seg.loss(out, label[:, 1::2, 1::2])
        assert len(loss.shape) == 0


def test_adversary():
    adv = LiBrainTumorSegAdv()
    mri_patch = torch.rand((1, 4, 31, 31))
    labels = torch.rand((1, 5, 15, 15))

    out = adv(mri_patch, labels)
    assert list(out.shape) == [1, 15, 15]
