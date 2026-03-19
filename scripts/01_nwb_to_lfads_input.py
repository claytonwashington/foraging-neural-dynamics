"""Script 1: NWB → lfads-torch input

Pipeline steps:
1. Load an NWB file into NWBDataset (filtering for wakeful periods)
2. Drop cross-correlated channels
3. Chop continuous data with LFADSInterface
4. Convert chopped H5 to lfads-torch format
5. Save the dataset and interface objects for later merging

Usage:
    python scripts/01_nwb_to_lfads_input.py
"""

import os
import sys
import h5py
import logging
import pickle
import numpy as np
import pandas as pd
import dill
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.nwb_dataset import NWBDataset
from core.lfads_interface import LFADSInterface
from configs.config import config

# ============================================================
# Setup logging
# ============================================================
logger = logging.getLogger()
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

# ============================================================
# Configuration
# ============================================================
BIN_SIZE = config["bin_size"]
XCORR_THRESHOLD = config["xcorr_threshold"]

base_dir = config["base_dir"]
lfads_torch_prefix = config["lfads_torch_prefix"]
nwb_path = os.path.join(base_dir, "NWB", config["experiment_name"] + ".nwb")
dataset_save_dir = os.path.join(base_dir, "datasets")
os.makedirs(dataset_save_dir, exist_ok=True)

# ============================================================
# Step 1: Load NWB file into NWBDataset
# ============================================================
logger.info(f"Loading {config['experiment_name']} from NWB")
dataset = NWBDataset(nwb_path, split_heldout=False, bin_width=10)

# Convert timestamp columns to timedelta for trial info
dataset.trial_info["poke_in_ts"] = pd.to_timedelta(
    dataset.trial_info["poke_in_ts"], unit="s"
)
dataset.trial_info["poke_out_ts"] = pd.to_timedelta(
    dataset.trial_info["poke_out_ts"], unit="s"
)

# ============================================================
# Step 2: Filter to wakeful periods only
# ============================================================
wake_idx_bounds = [
    (
        dataset.data.index.get_loc(
            np.min(dataset.trial_info[dataset.trial_info.epoch == epoch].start_time),
            method="nearest",
        ),
        dataset.data.index.get_loc(
            np.max(dataset.trial_info[dataset.trial_info.epoch == epoch].end_time),
            method="nearest",
        ),
    )
    for epoch in dataset.trial_info.epoch.unique()
]
dataset.data = pd.concat(
    [dataset.data.iloc[start:end] for start, end in wake_idx_bounds]
)

# ============================================================
# Step 3: Cross-correlation channel rejection
# ============================================================
assert dataset.bin_width == 10, "Cross-correlation analysis requires 10ms bin width"
pair_xcorr, drop_spk_names = dataset.get_pair_xcorr(
    "spikes", threshold=XCORR_THRESHOLD, zero_chans=True
)
logger.info(f"Zeroed {len(drop_spk_names)} channels above xcorr threshold {XCORR_THRESHOLD}")

# ============================================================
# Step 4: Chop data with LFADSInterface
# ============================================================
chop_merge_params = {
    "CHOP_MARGINS": 0,
    "DATA_FIELDNAME": "spikes",
    "MAX_OFFSET": 0,
    "OVERLAP": 100,
    "RANDOM_SEED": 0,
    "WINDOW": 500,
}

chop_cfg = chop_merge_params
chop_fields_map = {chop_cfg["DATA_FIELDNAME"]: "data"}

interface = LFADSInterface(
    window=chop_cfg["WINDOW"],
    overlap=chop_cfg["OVERLAP"],
    max_offset=chop_cfg["MAX_OFFSET"],
    chop_margins=chop_cfg["CHOP_MARGINS"],
    random_seed=chop_cfg["RANDOM_SEED"],
    chop_fields_map=chop_fields_map,
)

chop_df = dataset.data.copy()
save_path_chop = os.path.join(
    dataset_save_dir,
    config["experiment_name"]
    + "_chopped_{}xcorr_{}ms.h5".format(XCORR_THRESHOLD, BIN_SIZE),
)
interface.chop_and_save(chop_df, save_path_chop, overwrite=True)

# ============================================================
# Step 5: Convert to lfads-torch format
# ============================================================
path_modifier_h5 = config["experiment_name"] + "_chopped_{}xcorr_{}ms.h5".format(
    XCORR_THRESHOLD, BIN_SIZE
)
tf2_torch_file = os.path.join(dataset_save_dir, path_modifier_h5)

with h5py.File(tf2_torch_file, "r") as h5_in:
    train_encod_data = h5_in["train_data"][:]
    valid_encod_data = h5_in["valid_data"][:]
    train_recon_data = h5_in["train_data"][:]
    valid_recon_data = h5_in["valid_data"][:]
    train_inds = h5_in["train_inds"][:]
    valid_inds = h5_in["valid_inds"][:]

torch_dataset_str = "TORCH_" + path_modifier_h5
torch_output_path = os.path.join(
    lfads_torch_prefix, "lfads-torch", "datasets", torch_dataset_str
)
os.makedirs(os.path.dirname(torch_output_path), exist_ok=True)

kwargs = dict(dtype="float32", compression="gzip")
with h5py.File(torch_output_path, "w") as h5f:
    h5f.create_dataset("train_encod_data", data=train_encod_data, **kwargs)
    h5f.create_dataset("valid_encod_data", data=valid_encod_data, **kwargs)
    h5f.create_dataset("train_recon_data", data=train_recon_data, **kwargs)
    h5f.create_dataset("valid_recon_data", data=valid_recon_data, **kwargs)
    h5f.create_dataset("train_inds", data=train_inds, **kwargs)
    h5f.create_dataset("valid_inds", data=valid_inds, **kwargs)
    logger.info(f"lfads-torch input saved: {torch_output_path}")

# ============================================================
# Step 6: Save dataset and interface for merging step
# ============================================================
dataset_model_save_path = os.path.join(
    dataset_save_dir,
    config["experiment_name"]
    + "_dataset_{}xcorr_{}ms.pkl".format(XCORR_THRESHOLD, BIN_SIZE),
)
with open(dataset_model_save_path, "wb") as outf:
    logger.info(f"Dataset saved to {dataset_model_save_path}")
    dill.settings["recurse"] = True
    dill.dump(dataset, outf, protocol=4)

interface_path = os.path.join(
    dataset_save_dir,
    config["experiment_name"]
    + "_interface_{}xcorr_{}ms.pkl".format(XCORR_THRESHOLD, BIN_SIZE),
)
with open(interface_path, "wb") as rfile:
    logger.info(f"Interface saved to {interface_path}")
    pickle.dump(interface, rfile)

logger.info("Pipeline step 1 complete: NWB → lfads-torch input")
