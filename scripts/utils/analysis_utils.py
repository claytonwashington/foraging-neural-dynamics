"""Analysis utilities for the foraging neural dynamics pipeline.

Provides helper functions for loading merged datasets.
"""

import os
import types

import dill

import sys
from pathlib import Path

# Add project root and configs to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "configs"))
sys.path.insert(0, str(PROJECT_ROOT))
from config import config

# ---------------------------------------------------------------------------
# Shim: register core classes under the snel_toolkit namespace so that
# pickled datasets created with the original snel_toolkit can be loaded
# without having snel_toolkit installed.
# ---------------------------------------------------------------------------
from core.nwb_dataset import NWBDataset
from core.base_dataset import BaseDataset

_snel = types.ModuleType("snel_toolkit")
_snel_datasets = types.ModuleType("snel_toolkit.datasets")
_snel_nwb = types.ModuleType("snel_toolkit.datasets.nwb")

_snel_nwb.NWBDataset = NWBDataset
_snel_nwb.BaseDataset = BaseDataset
_snel_datasets.nwb = _snel_nwb
_snel.datasets = _snel_datasets

sys.modules["snel_toolkit"] = _snel
sys.modules["snel_toolkit.datasets"] = _snel_datasets
sys.modules["snel_toolkit.datasets.nwb"] = _snel_nwb


def load_dataset():
    """Load the merged dataset (NWBDataset with LFADS outputs) from pickle.

    Uses the experiment_name, run_date, and run_idx from config to
    locate the correct merged output file.

    Returns
    -------
    BaseDataset
        The merged dataset object containing spikes, LFADS rates,
        LFADS factors, and trial info.
    """
    base_dir = config["base_dir"]
    rat_name = config["experiment_name"].split("_")[0][:-8]
    dataset_load_dir = os.path.join(base_dir, "merged_datasets")
    if not os.path.exists(dataset_load_dir):
        os.makedirs(dataset_load_dir)
    dataset_load_path = os.path.join(
        dataset_load_dir,
        "{}_{}_{}_full_merged_output.pkl".format(
            rat_name, config["run_date"], config["run_idx"]
        ),
    )
    with open(dataset_load_path, "rb") as f:
        dataset = dill.load(f)
    return dataset
