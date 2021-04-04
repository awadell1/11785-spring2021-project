from os import name
import torch
from src.dataset import Brats2017, patch_indices


def check_type_and_shape(ds: Brats2017, data: torch.Tensor, label: torch.Tensor):
    """ Confirm the type and shape of a returned sample """

    # Check data
    assert isinstance(data, torch.Tensor)
    data_shape = data.shape
    assert len(data_shape) == 4
    assert data_shape[-1] == len(ds.modality_postfix)
    assert data_shape[0:3] == ds.patch_shape
    assert data.dtype == torch.float32

    # Check label
    assert isinstance(label, torch.Tensor)
    label_shape = label.shape
    assert len(label_shape) == 3

    # Cross-check
    assert label_shape == data_shape[0:3]


def test_getitem_simple():
    # Default Dataset
    ds = Brats2017("data/Brats17TrainingData")

    # Try getting an item
    data, label = ds[0]
    check_type_and_shape(ds, data, label)


def test_patch_indices():
    input_size = (240, 240, 155)
    output_size = (144, 144, 19)
    out = patch_indices(input_size, output_size)
    assert isinstance(out, list)
    assert len(out) == 4 * 8
