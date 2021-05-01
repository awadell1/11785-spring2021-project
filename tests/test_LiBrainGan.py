from logging import log
import torch
from src.gan import LiBrainTumorSegAdv, LiBrainTumorSegGan, LiBrainTumorSegGen
from src.util import SparseDiceLoss


def test_segmentor():
    seg = LiBrainTumorSegGen()
    input = torch.rand((1, 4, 31, 31), requires_grad=False)
    out, logits = seg(input)

    # Check shape
    assert list(out.shape) == [1, 5, 15, 15]
    assert list(logits.shape) == [1, 5, 15, 15]

    # Check logits sum to 1
    logit_sums = torch.sum(torch.exp(logits), dim=1)
    expected = torch.ones((out.shape))
    assert all(torch.isclose(logit_sums, expected).flatten())

    # Check loss gradient
    target = torch.randint(0, 5, (1, 15, 15), requires_grad=False)
    loss = seg.loss(out, target)
    loss.backward()
    total_norm = 0
    for p in seg.parameters():
        total_norm += p.grad.norm() ** 2
    assert total_norm.sqrt() > 0.1


def test_adversary():
    adv = LiBrainTumorSegAdv()
    mri_patch = torch.rand((3, 4, 31, 31), requires_grad=True)
    labels = torch.rand((3, 5, 15, 15), requires_grad=False)

    out = adv(mri_patch, labels)
    adv_label = torch.tensor([0.0, 0.0, 0.0], requires_grad=False)
    loss = adv.loss(out, adv_label)
    assert list(out.shape) == [3]

    # Check loss gradient
    loss.backward()
    total_norm = 0
    for p in adv.parameters():
        total_norm += p.grad.norm() ** 2
    assert total_norm.sqrt() > 0.1
    assert mri_patch.grad.norm() > 0.1


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
