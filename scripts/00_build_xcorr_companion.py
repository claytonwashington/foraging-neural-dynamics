"""Script 00: Build xcorr companion pkl from NWB

Loads the original NWB file, filters to wake epochs (same as Step 1),
and computes pairwise cross-correlations on the ORIGINAL un-zeroed spikes.
Saves:
  - Original spike matrix (all channels, wake-filtered, 10ms bins)
  - Full pairwise xcorr list
  - Channel rejection masks at multiple thresholds
  - Time index (for alignment with merged dataset)

This companion pkl is used by 04_preprocessing_validation.py so we
don't need to reload the NWB every time.

Usage:
    python scripts/00_build_xcorr_companion.py
"""

import os
import sys
import copy
import logging
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.nwb_dataset import NWBDataset
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
THRESHOLDS = [0.1, 0.2, 0.3]  # xcorr thresholds to compute rejection masks for

base_dir = config["base_dir"]
nwb_path = os.path.join(base_dir, "NWB", config["experiment_name"] + ".nwb")
save_dir = os.path.join(base_dir, "datasets")
os.makedirs(save_dir, exist_ok=True)

companion_save_path = os.path.join(
    save_dir,
    f"{config['experiment_name']}_xcorr_companion.pkl",
)

# ============================================================
# Step 1: Load NWB (same as 01_nwb_to_lfads_input.py)
# ============================================================
logger.info(f"Loading {config['experiment_name']} from NWB: {nwb_path}")
dataset = NWBDataset(nwb_path, split_heldout=False, bin_width=BIN_SIZE)

# Convert timestamp columns to timedelta for trial info
dataset.trial_info["poke_in_ts"] = pd.to_timedelta(
    dataset.trial_info["poke_in_ts"], unit="s"
)
dataset.trial_info["poke_out_ts"] = pd.to_timedelta(
    dataset.trial_info["poke_out_ts"], unit="s"
)

# ============================================================
# Step 2: Filter to wakeful periods (same as 01)
# ============================================================
def nearest_idx(index, target):
    """Find the index of the nearest value in a sorted TimedeltaIndex."""
    pos = index.searchsorted(target)
    pos = min(pos, len(index) - 1)
    if pos > 0:
        before = abs(index[pos - 1] - target)
        after = abs(index[pos] - target)
        if before < after:
            return pos - 1
    return pos

wake_idx_bounds = [
    (
        nearest_idx(dataset.data.index,
            np.min(dataset.trial_info[dataset.trial_info.epoch == epoch].start_time)),
        nearest_idx(dataset.data.index,
            np.max(dataset.trial_info[dataset.trial_info.epoch == epoch].end_time)),
    )
    for epoch in dataset.trial_info.epoch.unique()
]
dataset.data = pd.concat(
    [dataset.data.iloc[start:end] for start, end in wake_idx_bounds]
)

logger.info(f"Wake-filtered data: {dataset.data.shape[0]} bins, "
            f"{dataset.data['spikes'].shape[1]} channels")

# ============================================================
# Step 3: Compute pairwise xcorr WITHOUT zeroing
# ============================================================
logger.info("Computing pairwise cross-correlations (no zeroing)...")
pair_xcorr, _ = dataset.get_pair_xcorr(
    "spikes", threshold=None, zero_chans=False
)
logger.info(f"Computed {len(pair_xcorr)} pairwise correlations")

# Build a full correlation matrix from the pair list
spikes_np = dataset.data["spikes"].values.astype(np.float64)
n_neurons = spikes_np.shape[1]
chan_names = dataset.data["spikes"].columns.tolist()

xcorr_matrix = np.zeros((n_neurons, n_neurons))
for (i, k), corr_val in pair_xcorr:
    xcorr_matrix[i, k] = corr_val
    xcorr_matrix[k, i] = corr_val
np.fill_diagonal(xcorr_matrix, 1.0)

# ============================================================
# Step 4: Compute rejection masks at each threshold
# ============================================================
rejection_masks = {}  # threshold → boolean mask (True = KEEP)
rejection_names = {}  # threshold → list of dropped channel names

for thresh in THRESHOLDS:
    # Re-run the rejection logic from get_pair_xcorr without modifying data
    pair_corr_tmp = copy.deepcopy(pair_xcorr)
    pair_corr_tmp.sort(key=lambda x: x[1], reverse=False)
    chan_names_to_drop = []

    while pair_corr_tmp:
        pair, corr = pair_corr_tmp.pop(-1)
        if corr > thresh:
            c1 = [p[1] for p in pair_corr_tmp if pair[0] in p[0]]
            c2 = [p[1] for p in pair_corr_tmp if pair[1] in p[0]]
            cnt1 = sum(1 for c in c1 if c > thresh)
            cnt2 = sum(1 for c in c2 if c > thresh)
            if cnt1 > cnt2:
                chan_dropp = pair[0]
            elif cnt1 < cnt2:
                chan_dropp = pair[1]
            else:
                if np.mean(c1) if c1 else 0 > np.mean(c2) if c2 else 0:
                    chan_dropp = pair[0]
                else:
                    chan_dropp = pair[1]
            pair_corr_tmp = [
                p for p in pair_corr_tmp if chan_dropp not in p[0]
            ]
            chan_names_to_drop.append(chan_names[chan_dropp])
        else:
            break  # sorted ascending, so everything below is < threshold

    # Build boolean mask: True = keep, False = reject
    drop_indices = set()
    for name in chan_names_to_drop:
        drop_indices.add(chan_names.index(name))
    keep_mask = np.ones(n_neurons, dtype=bool)
    keep_mask[list(drop_indices)] = False

    rejection_masks[thresh] = keep_mask
    rejection_names[thresh] = chan_names_to_drop
    logger.info(f"  Threshold {thresh}: {len(chan_names_to_drop)}/{n_neurons} "
                f"channels rejected, {int(keep_mask.sum())} kept")

# ============================================================
# Step 5: Save companion pkl
# ============================================================
companion = {
    "experiment_name": config["experiment_name"],
    "bin_size": BIN_SIZE,
    "n_neurons": n_neurons,
    "chan_names": chan_names,
    "spikes": spikes_np,                  # (T, N) — original un-zeroed
    "time_index": dataset.data.index.to_numpy(),  # timedelta64[ns]
    "xcorr_matrix": xcorr_matrix,         # (N, N) full symmetric
    "pair_xcorr": pair_xcorr,             # raw pair list
    "thresholds": THRESHOLDS,
    "rejection_masks": rejection_masks,   # {thresh: bool array, True=keep}
    "rejection_names": rejection_names,   # {thresh: list of dropped names}
}

logger.info(f"Saving companion pkl to {companion_save_path}")
with open(companion_save_path, "wb") as f:
    pickle.dump(companion, f, protocol=4)

logger.info("Done. Companion pkl contents:")
logger.info(f"  spikes shape: {spikes_np.shape}")
logger.info(f"  xcorr_matrix shape: {xcorr_matrix.shape}")
for thresh in THRESHOLDS:
    n_rej = len(rejection_names[thresh])
    logger.info(f"  threshold {thresh}: {n_rej} rejected, "
                f"{n_neurons - n_rej} kept")
