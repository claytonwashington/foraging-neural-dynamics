"""Script 2: lfads-torch output → merged analysis dataset

Pipeline steps:
1. Load the saved interface and dataset objects from Step 1
2. Load lfads-torch model outputs
3. Combine train/valid outputs and merge chops
4. Smooth spikes, rates, and factors
5. Save the merged dataset for analysis
6. (Optional) Sanity-check plot: poke-in aligned raster, smoothed spikes,
   and smoothed LFADS rates — set PLOT_SANITY_CHECK = True to enable.

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


# ============================================================
# Step 6 (optional): Sanity-check plot — poke-in aligned signals
# ============================================================
# Toggle this flag to True to generate the alignment sanity-check figure.
PLOT_SANITY_CHECK = True

# Window around poke-in time (ms).  Values are snapped to the nearest bin.
MS_BEFORE = 100
MS_AFTER = 300

if PLOT_SANITY_CHECK:
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.cm as colormap
    import matplotlib.gridspec as gridspec
    import random

    bin_size_ms = config["bin_size"]  # ms per time bin

    # ── Snap window to nearest bin boundary ──────────────────────────────────
    ms_before = round(MS_BEFORE / bin_size_ms) * bin_size_ms
    ms_after = round(MS_AFTER / bin_size_ms) * bin_size_ms

    pre_td = np.timedelta64(int(ms_before * 1e6), "ns")   # ms → ns
    post_td = np.timedelta64(int(ms_after * 1e6), "ns")

    time_index = dataset.data.index.to_numpy()             # timedelta64[ns]

    # Extract arrays once (shape: T × N_neurons)
    spikes_arr = dataset.data["spikes"].to_numpy()
    spikes_smooth_arr = dataset.data[f"spikes_smooth_{spike_smooth_width}"].to_numpy()
    lfads_rates_arr = dataset.data["lfads_rates"].to_numpy()

    # ── Pick a random trial with a valid poke-in timestamp ───────────────────
    valid_trials = dataset.trial_info.dropna(subset=["poke_in_ts"])
    if valid_trials.empty:
        logger.warning("No trials with valid poke_in_ts found — skipping sanity plot.")
    else:
        trial_row = valid_trials.iloc[random.randint(0, len(valid_trials) - 1)]
        poke_in_ts = np.timedelta64(trial_row["poke_in_ts"].value, "ns")

        start_ts = poke_in_ts - pre_td
        end_ts = poke_in_ts + post_td

        start_ix = int(np.searchsorted(time_index, start_ts, side="left"))
        end_ix = int(np.searchsorted(time_index, end_ts, side="left")) + 1

        # Time axis in ms relative to poke-in
        t_bins = (time_index[start_ix:end_ix] - poke_in_ts).astype("float64") / 1e6

        spikes_win = spikes_arr[start_ix:end_ix]             # (T_win, N)
        spikes_smooth_win = spikes_smooth_arr[start_ix:end_ix]
        lfads_rates_win = lfads_rates_arr[start_ix:end_ix]

        n_neurons = spikes_win.shape[1]
        
        # Determine sorting based on peak firing rate time
        sort_order = np.argsort(np.argmax(spikes_smooth_win, axis=0))
        
        # Shared limits for heatmaps
        vmax_spk = np.percentile(spikes_smooth_win, 99)
        vmax_rate = np.percentile(lfads_rates_win, 99)

        # Bin index of poke-in (for vlines on pcolor axes)
        poke_bin = int(ms_before / bin_size_ms)

        # ── Figure layout ─────────────────────────────────────────────────────
        fig, axs = plt.subplots(3, 1, figsize=(10, 12), dpi=100)
        plt.subplots_adjust(top=0.88)

        ax_spk, ax_smooth, ax_rates = axs

        # ── Shared vmax across smoothed spikes & LFADS rates ─────────────────
        vmax = max(spikes_smooth_win.max(), lfads_rates_win.max())
        max_scale = 0.5

        titles = [
            f"Original spikes  (bin={bin_size_ms} ms)",
            f"Smoothed spikes  (σ={spike_smooth_width} ms)",
            "LFADS inferred rates",
        ]
        data_slices = [
            spikes_win[:, sort_order].T,
            spikes_smooth_win[:, sort_order].T,
            lfads_rates_win[:, sort_order].T,
        ]
        vmaxs = [1, max_scale * vmax, max_scale * vmax]
        cmaps = [colormap.bone_r, "viridis", "viridis"]

        fig.suptitle(
            f"Poke-in aligned  |  trial {int(trial_row.name)}  |  "
            f"window: −{ms_before:.0f} / +{ms_after:.0f} ms  |  bin: {bin_size_ms} ms",
            x=0.1, y=1,
        )

        for i in range(3):
            axs[i].pcolor(data_slices[i], cmap=cmaps[i], vmin=0, vmax=vmaxs[i])
            axs[i].set_title(titles[i])
            axs[i].vlines(poke_bin, 0, n_neurons, color="r")
            try:
                axs[i].set_xticklabels(
                    t_bins[axs[i].get_xticks().astype(int)].astype(int)
                )
            except Exception:
                axs[i].set_xticklabels(
                    t_bins[axs[i].get_xticks().astype(int)[:-1]].astype(int)
                )
            axs[i].set_ylabel("Neurons (sorted)")
            axs[i].set_xlabel("Time relative to poke-in (ms)")
            axs[i].spines["right"].set_visible(False)
            axs[i].spines["top"].set_visible(False)

        plt.subplots_adjust(hspace=0.5)
        fig.tight_layout()
        plt.show()

        logger.info(
            "Step 6 sanity-check plot complete  "
            f"(trial {int(trial_row.name)}, window −{ms_before:.0f}/+{ms_after:.0f} ms)"
        )
