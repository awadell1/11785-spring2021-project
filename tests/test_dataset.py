import os
from pathlib import Path
from tempfile import TemporaryDirectory
import torch
from torch.utils.data.dataloader import DataLoader
from src.dataset import Brats2017, patch_indices
from pytest import approx


def check_type_and_shape(
    ds: Brats2017, data: torch.Tensor, label: torch.Tensor, ndims=4
):
    """ Confirm the type and shape of a returned sample """

    # Check data
    assert isinstance(data, torch.Tensor)
    data_shape = data.shape
    assert len(data_shape) == ndims
    assert data_shape[0] == len(ds.modality_postfix)
    assert data_shape[1:] == ds.patch_shape[0 : ndims - 1]
    assert data.dtype == torch.float32

    # Check label
    assert isinstance(label, torch.Tensor)
    label_shape = label.shape
    assert len(label_shape) == ndims - 1

    # Cross-check
    assert label_shape == data_shape[1:]


def test_getitem_simple():
    # Default Dataset
    ds = Brats2017("data/Brats17TrainingData", n_samples=10)

    # Try getting an item
    data, label = ds[0]
    check_type_and_shape(ds, data, label)
    assert data.dtype == ds.data_type
    assert label.dtype == ds.label_type


def test_flat_patch():
    ds = Brats2017("data/Brats17TrainingData", patch_depth=1, n_samples=10)
    data, label = ds[0]
    check_type_and_shape(ds, data, label, ndims=3)
    assert data.dtype == ds.data_type
    assert label.dtype == ds.label_type


def test_patch_indices():
    input_size = (240, 240, 155)
    output_size = (144, 144, 19)
    out = patch_indices(input_size, output_size)
    assert isinstance(out, list)
    assert len(out) == 4 * 8


def test_dataloader():
    ds = Brats2017(["data/Brats17TrainingData/HGG/Brats17_2013_2_1"])
    batch_size = 8
    dl = DataLoader(ds, batch_size=batch_size)
    n_samples = 0
    for batch in dl:
        assert len(batch) == 2
        data, label = batch
        assert data.shape[0] == batch_size
        assert label.shape[0] == batch_size

        # Exit if running on full dataset (ie not on github)
        if len(ds) > 100:
            return
        else:
            n_samples += len(label)

    assert n_samples == len(ds)


def test_split():
    # Split Training set into a consistent train / val / test split

    # Construct Fake Dataset
    patients, _ = Brats2017.get_patient_dirs("data/Brats17TrainingData")
    patients = patients[:2]
    with TemporaryDirectory() as fake_dataset:
        root = Path(fake_dataset)
        for grp in ["HGG", "LGG"]:
            grp_dir = root.joinpath(grp)
            grp_dir.mkdir()

            # Populate with simlinked data
            for idx in range(10):
                for pdx, p in enumerate(patients):
                    dst = grp_dir.joinpath(f"Brats17_{idx}_{pdx}")
                    os.symlink(p.resolve(), dst, target_is_directory=True)

        # Split dataset
        train, val, test = Brats2017.split_dataset(root, load_ds=False)

        # Check size
        train_len = len(train)
        val_len = len(val)
        test_len = len(test)
        total_len = train_len + test_len + val_len
        assert train_len / total_len == approx(0.7, rel=0.01)
        assert val_len / total_len == approx(0.2, rel=0.01)
        assert test_len / total_len == approx(0.1, rel=0.01)
