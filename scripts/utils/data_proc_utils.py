"""Data processing utilities for merging LFADS outputs with original datasets.

Provides helper functions to:
- Extract train/valid indices from the chopped H5 file
- Combine training and validation LFADS outputs into full arrays
- Merge LFADS outputs back into the original NWBDataset
"""

import typing

import h5py
import numpy as np
import pandas as pd


def get_train_valid_inds(
    original_h5: str,
    torch_outputs: h5py._hl.files.File,
    lfads_torch_outputs_path: str,
) -> typing.Tuple[np.ndarray, np.ndarray]:
    """Extract train/valid indices and optionally add them to LFADS output file.

    Parameters
    ----------
    original_h5 : str
        Path to the original chopped H5 file containing train_inds/valid_inds.
    torch_outputs : h5py.File
        Open H5 file of LFADS torch outputs.
    lfads_torch_outputs_path : str
        Path to the LFADS output H5 file (for writing indices if missing).

    Returns
    -------
    tuple of np.ndarray
        (train_inds, valid_inds)
    """
    original_h5_data = h5py.File(original_h5)
    train_inds = original_h5_data["train_inds"][()]
    valid_inds = original_h5_data["valid_inds"][()]

    # Add indices to torch output file if not already present
    if "train_inds" not in torch_outputs.keys():
        with h5py.File(lfads_torch_outputs_path, "a") as torch_output_data:
            torch_output_data.create_dataset("train_inds", data=train_inds)
            torch_output_data.create_dataset("valid_inds", data=valid_inds)

    return train_inds, valid_inds


def combine_train_valid_outputs(
    torch_outputs: h5py._hl.files.File,
    train_inds: np.ndarray,
    valid_inds: np.ndarray,
    merge_config: typing.Dict[str, str],
) -> typing.Dict[str, np.ndarray]:
    """Combine separate train and valid LFADS outputs into full arrays.

    Parameters
    ----------
    torch_outputs : h5py.File
        Open H5 file containing train_* and valid_* datasets.
    train_inds : np.ndarray
        Indices of training samples.
    valid_inds : np.ndarray
        Indices of validation samples.
    merge_config : dict
        Maps LFADS output names to desired signal type names.

    Returns
    -------
    dict of np.ndarray
        Combined output arrays keyed by LFADS field names.
    """
    n_batch = train_inds.size + valid_inds.size
    data_dict = {}
    for torch_name, snel_name in merge_config.items():
        train_output = torch_outputs[f"train_{torch_name}"][()]
        valid_output = torch_outputs[f"valid_{torch_name}"][()]
        full_output = np.empty((n_batch, train_output.shape[1], train_output.shape[2]))
        full_output[train_inds, :, :] = train_output
        full_output[valid_inds, :, :] = valid_output
        data_dict[torch_name] = full_output
    return data_dict


def merge_with_original_df(merged_df: pd.DataFrame, dataset) -> None:
    """Merge LFADS output DataFrame into the original dataset.

    For each signal type in the merged DataFrame, either updates existing
    columns or adds new continuous data to the dataset.

    Parameters
    ----------
    merged_df : pd.DataFrame
        DataFrame with MultiIndex columns containing LFADS outputs.
    dataset : BaseDataset
        The original dataset object to merge into (modified in place).
    """
    for key in merged_df.columns.levels[0].to_list():
        if key == "lfads_rates":
            chan_names = dataset.data["spikes"].columns.values
        else:
            chan_names = np.arange(merged_df[key].shape[1])
        if key in dataset.data.keys():
            dataset.data[key] = merged_df[key]
        else:
            dataset.add_continuous_data(
                merged_df[key].values,
                key,
                chan_names=chan_names,
            )
