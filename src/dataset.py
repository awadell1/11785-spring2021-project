import gzip
import shutil
from pathlib import Path
from itertools import product
import numpy as np
from numpy.core.numeric import indices
from numpy.lib.function_base import append
import torch
from torch._C import dtype
from torch.utils.data import Dataset
import nibabel as nib


class Brats2017(Dataset):
    def __init__(
        self,
        root: Path,
        direction="axial",
        patch_size=144,
        patch_depth=19,
        data_type=np.float32,
    ) -> None:
        super().__init__()

        # Parse Arguments
        self.root = Path(root)
        self.modality_postfix = ["flair", "t1", "t1ce", "t2"]
        self.label_postfix = "seg"
        self.direction = direction
        self.patch_shape = (patch_size, patch_size, patch_depth)
        self.data_type = data_type

        # Get list of patient folders
        patients = []
        for grp in self.root.iterdir():
            if not grp.is_dir():
                continue
            # Iterate over patient folders
            for patient in grp.iterdir():
                patients.append(patient)

        self.patients = patients

        # Get Patch Slices
        input_shape = (240, 240, 155)
        self.patches = patch_indices(input_shape, self.patch_shape)
        self.n_patches = len(self.patches)

    def __getitem__(self, index):
        # Get Patient Index
        pdx, patch_idx = divmod(index, self.n_patches)
        patient_dir = self.patients[pdx]
        patch = self.patches[patch_idx]

        # Get patient
        data, label = self.get_patient(patient_dir, patch)

        return data, label

    def get_patient(self, patient_dir: Path, patch_idx):
        """ Load Patient Files from disk and stack modalities """
        nii_volumes = []
        for mod in self.modality_postfix + [self.label_postfix]:
            # Get nii image
            filename = patient_dir.joinpath(f"{patient_dir.name}_{mod}.nii.gz")
            nii_data = nib.load(filename)

            # Slice to patch
            patch = nii_data.slicer[patch_idx].get_fdata(dtype=self.data_type)

            # Convert to Tensor
            nii_volumes.append(torch.from_numpy(patch))

        data = torch.stack(nii_volumes[0:-1], dim=3)
        labels = nii_volumes[-1]

        return data, labels


def patch_indices(input_size, output_size):

    # Number of slices
    dim_slices = []
    for dim in range(2):
        dim_slices.append(_slice_dim(input_size[dim], output_size[dim], shift_end=True))

    # Don't shift to get final depth slice
    dim_slices.append(_slice_dim(input_size[2], output_size[2], shift_end=False))
    print(dim_slices)

    # Iterate over slices
    indices = []
    for h, w, d in product(*dim_slices):
        indices.append((h, w, d))
    return indices


def _slice_dim(in_size, out_size, shift_end=False):
    for s in range(0, in_size, out_size):
        if s + out_size <= in_size:
            yield slice(s, s + out_size)
        elif shift_end:
            yield slice(in_size - out_size, in_size)


def bounding_box(vol, margin=0):
    nzero_idx = np.nonzero(vol)
    bb_min = []
    bb_max = []

    # Get Bounding Box
    for dim_nzeros in nzero_idx:
        bb_min.append(dim_nzeros.min())
        bb_max.append(dim_nzeros.max())

    # Inflate by margin
    for idx, (bmin, bmax, l) in enumerate(zip(bb_min, bb_max, vol.shape)):
        bb_min[idx] = max(bmin - margin, 0)
        bb_max[idx] = min(bmax + margin, l)

    return bb_min, bb_max


def transpose_vol(vol: np.ndarray, direction) -> np.ndarray:
    if direction == "axial":
        return vol
    if direction == "sagittal":
        return vol.transpose(2, 0, 1)
    if direction == "coronal":
        return vol.transpose(1, 0, 2)
    else:
        raise RuntimeError(f"Undefined direction: {direction}")
