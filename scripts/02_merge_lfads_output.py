"""Script 2: lfads-torch output → merged analysis dataset

Pipeline steps:
1. Load the saved interface and dataset objects from Step 1
2. Load lfads-torch model outputs
3. Combine train/valid outputs and merge chops
4. Smooth spikes, rates, and factors
5. Save the merged dataset for analysis

Usage:
    python scripts/02_merge_lfads_output.py
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

from configs.config import config, merge_config
from scripts.utils.data_proc_utils import (
    get_train_valid_inds,
    combine_train_valid_outputs,
    merge_with_original_df,
)

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
# Path setup
# ============================================================
base_dir = config["base_dir"]
rat_name = config["experiment_name"].split("_")[0][:-8]

PROJECT_STR = f"{rat_name}_{config['run_idx']}"
DATASET_STR = "TORCH_{}_chopped_{}xcorr_{}ms".format(
    config["experiment_name"], config["xcorr_threshold"], config["bin_size"]
)
RUN_TAG = config["run_date"] + f"_{rat_name}_PBT"
RUN_DIR = Path(config["run_dir"]) / PROJECT_STR / DATASET_STR / RUN_TAG
BEST_MODEL_DIR = RUN_DIR / "best_model"

nwb_path = os.path.join(base_dir, "NWB", config["experiment_name"] + ".nwb")
dataset_save_dir = os.path.join(base_dir, "datasets")
merged_dataset_save_dir = os.path.join(base_dir, "merged_datasets")
os.makedirs(merged_dataset_save_dir, exist_ok=True)

cache_dataset = os.path.join(
    dataset_save_dir,
    config["experiment_name"]
    + "_dataset_{}xcorr_{}ms.pkl".format(config["xcorr_threshold"], config["bin_size"]),
)
path_modifier_h5 = config["experiment_name"] + "_chopped_{}xcorr_{}ms.h5".format(
    config["xcorr_threshold"], config["bin_size"]
)
tf2_torch_file = os.path.join(dataset_save_dir, path_modifier_h5)
interface_path = os.path.join(
    dataset_save_dir,
    config["experiment_name"]
    + "_interface_{}xcorr_{}ms.pkl".format(
        config["xcorr_threshold"], config["bin_size"]
    ),
)
full_merge_save_path = os.path.join(
    merged_dataset_save_dir,
    "{}_full_merged_output.pkl".format(config["experiment_name"]),
)
lfads_torch_outputs_path = os.path.join(
    BEST_MODEL_DIR, "lfads_output_{}.h5".format(DATASET_STR.split(".")[0])
)

# ============================================================
# Step 1: Load interface and dataset
# ============================================================
logger.info("Loading interface and dataset from pickle")
with open(interface_path, "rb") as inf:
    interface = pickle.load(inf)

interface.merge_fields_map = merge_config

with open(cache_dataset, "rb") as inf:
    dataset = pickle.load(inf)

# ============================================================
# Step 2: Load lfads-torch outputs
# ============================================================
torch_outputs = h5py.File(lfads_torch_outputs_path)

# ============================================================
# Step 3: Combine train/valid outputs and merge chops
# ============================================================
train_inds, valid_inds = get_train_valid_inds(
    tf2_torch_file, torch_outputs, lfads_torch_outputs_path
)

data_dict = combine_train_valid_outputs(
    torch_outputs, train_inds, valid_inds, merge_config
)
merged_df = interface.merge(data_dict, smooth_pwr=1)

# Handle length mismatches (can happen when not using full data)
if dataset.data.shape[0] != merged_df.shape[0]:
    merged_df = merged_df.iloc[: dataset.data.shape[0]]
    merged_df.index = dataset.data.index

# Merge with original dataset
merge_with_original_df(merged_df, dataset)

# Remove duplicate indices (from multi-chop overlap)
dataset.data = dataset.data[~dataset.data.index.duplicated(keep="first")]

# ============================================================
# Step 4: Smooth spikes, rates, and factors
# ============================================================
spike_smooth_width = 100
rate_smooth_width = 50
factor_smooth_width = 50

dataset.smooth_spk(
    gauss_width=spike_smooth_width,
    name=f"smooth_{spike_smooth_width}",
    overwrite=False,
)

dataset.smooth_spk(
    signal_type="lfads_rates",
    gauss_width=rate_smooth_width,
    name=f"smooth_{rate_smooth_width}",
    overwrite=False,
)
dataset.data[f"lfads_rates_smooth_{rate_smooth_width}"] = dataset.data[
    f"lfads_rates_smooth_{rate_smooth_width}"
].fillna(0)

dataset.smooth_spk(
    signal_type="lfads_factors",
    gauss_width=factor_smooth_width,
    name=f"smooth_{factor_smooth_width}",
    overwrite=False,
)
dataset.data[f"lfads_factors_smooth_{factor_smooth_width}"] = dataset.data[
    f"lfads_factors_smooth_{factor_smooth_width}"
].fillna(0)

# ============================================================
# Step 5: Save merged dataset
# ============================================================
logger.info(f"Saving merged dataset to {full_merge_save_path}")
with open(full_merge_save_path, "wb") as f:
    dill.dump(dataset, f, protocol=dill.HIGHEST_PROTOCOL, recurse=True)

logger.info("Pipeline step 2 complete: lfads-torch output → merged dataset")
