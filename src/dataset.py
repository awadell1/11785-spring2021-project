from pathlib import Path
from itertools import product
import numpy as np
import torch
from torch.utils.data import Dataset
import nibabel as nib
from sklearn.model_selection import train_test_split


class Brats2017(Dataset):
    def __init__(
        self,
        patients: Path,
        direction="axial",
        patch_size=144,
        patch_depth=19,
        data_type=torch.float32,
        label_type=torch.long,
        flat_patch=None,
        n_samples=None,
    ) -> None:
        super().__init__()

        # Parse Arguments
        self.modality_postfix = ["flair", "t1", "t1ce", "t2"]
        self.label_postfix = "seg"
        self.direction = direction
        self.patch_shape = (patch_size, patch_size, patch_depth)
        self.data_type = data_type
        self.label_type = label_type
        self.flat_patch = flat_patch or patch_depth == 1

        # Get list of patient folders
        if isinstance(patients, list):
            # Input is the list of patient folders to use
            self.patients_dirs = [Path(p) for p in patients]
        else:
            # Input is root of folder structure : root/group/patient_dir
            self.patients_dirs, _ = Brats2017.get_patient_dirs(patients)

        if n_samples is not None:
            self.patients_dirs = self.patients_dirs[:n_samples]
        # Load scans into memore
        scans = []
        labels = []
        for idx, p_dir in enumerate(self.patients_dirs):
            print(f"Loading patient {idx} from {p_dir}")
            scan, label = self.load_patient(p_dir)
            scans.append(scan)
            labels.append(label)
        self.mri_scans = torch.stack(scans, dim=0)
        self.mri_labels = torch.stack(labels, dim=0)

        # Get Patch Slices
        self.patches = patch_indices(self.mri_scans.shape[2:], self.patch_shape)
        self.n_patches = len(self.patches)

        # Record length
        self.length = self.n_patches * len(self.patients_dirs)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        # Get Patient Index
        pdx, patch_idx = divmod(index, self.n_patches)
        patch = self.patches[patch_idx]

        # Get patient
        scan = self.mri_scans[pdx].__getitem__(patch)
        label = self.mri_labels[pdx].__getitem__(patch[1:])

        # Squeeze Depth iff flat_patch
        if self.flat_patch:
            scan = scan.squeeze(dim=3)
            label = label.squeeze(dim=2)

        return scan, label

    def load_patient(self, patient_dir: Path):
        """Load Patient Files from disk and stack modalities

        Returns
            data: torch.Tensor of shape (C, H, W, D) where
                C : 4 One for each MRI Scan
                H, W, D : Volumetric patch from the MRI scan
                    Note: Meaning of H, W, D varies with the patch direction
            labels: torch.Tensor of shape (L, P, H, W, D) where
                L - Semantic Segmentation of the Brain (Ie. Is this voxel a tumor?)
                H, W, D  Volumetic Patch of the brain
        """
        nii_volumes = []
        for mod in self.modality_postfix + [self.label_postfix]:
            # Get nii image
            filename = patient_dir.joinpath(f"{patient_dir.name}_{mod}.nii.gz")
            nii_data = nib.load(filename)

            # Slice to patch
            if mod in self.modality_postfix:
                dtype = self.data_type
            else:
                dtype = self.label_type

            patch = nii_data.get_fdata()

            # Convert to Tensor
            nii_volumes.append(torch.from_numpy(patch).type(dtype))

        data = torch.stack(nii_volumes[0:-1], dim=0)
        labels = nii_volumes[-1]

        return data, labels

    @staticmethod
    def get_patient_dirs(root):
        patient_type = []
        patients = []
        for grp in Path(root).iterdir():
            if not grp.is_dir():
                continue
            # Iterate over patient folders
            for patient in grp.iterdir():
                # Check that directory is a patient dir
                if not patient.name.startswith("Brats17"):
                    continue

                # Add patient to list
                patients.append(patient)
                patient_type.append(patient.parent.name)

        return patients, patient_type

    @staticmethod
    def split_dataset(root="data/Brats17TrainingData", load_ds=True, **kwarg):
        """Splits the dataset in train, val, testing subsets using a 70-20-10
        stratified split.
            Inputs
                root: Path to folder where Brats17 Dataset is stored
                load_ds=True: Iff false will just return the paths to the patient dirs
                **kwargs: Additional optional arguments to pass to Brats2017 when creating
                the datasets
        """

        # Stratify patients based on HGG or LGG
        patients, patient_type = Brats2017.get_patient_dirs(root)

        # Spit into Train / Val+Test
        split_1 = train_test_split(
            patients,
            patient_type,
            train_size=0.7,
            random_state=0,
            stratify=patient_type,
        )

        # Split Val+Test into Val / Test
        split_2 = train_test_split(
            split_1[1],
            split_1[3],
            test_size=1 / 3,
            random_state=0,
            stratify=split_1[3],
        )

        #  Get train, val and test patients
        train, val, test = split_1[0], split_2[0], split_2[1]

        # Check if we just want the patient names
        if not load_ds:
            return train, val, test

        # Build datasets
        return (
            Brats2017(train, **kwarg),
            Brats2017(val, **kwarg),
            Brats2017(test, **kwarg),
        )


def patch_indices(input_size, output_size):

    # Number of slices
    dim_slices = []
    for dim in range(2):
        dim_slices.append(_slice_dim(input_size[dim], output_size[dim], shift_end=True))

    # Don't shift to get final depth slice
    dim_slices.append(_slice_dim(input_size[2], output_size[2], shift_end=False))

    # Iterate over slices
    indices = []
    n_mod = 4
    for h, w, d in product(*dim_slices):
        indices.append((slice(n_mod), h, w, d))
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
