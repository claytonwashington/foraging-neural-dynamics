"""Script 04: Preprocessing validation plots for reviewer rebuttal.

Produces three families of figures for one session:
  1. Cross-correlation analysis — full xcorr matrix, histogram, covariance
     (using ORIGINAL un-zeroed spikes from the companion pkl)
  2. Boundary-artifact check — continuous time-series + PCA trajectory
  3. Threshold robustness — condition-averaged PCA trajectories across
     xcorr thresholds for Navigation, Pre-move, and Outcome subspaces

Prerequisites:
  - Run 00_build_xcorr_companion.py first to generate the companion pkl
  - Run 02_merge_lfads_output.py first to generate the merged dataset

Usage:
    python scripts/04_preprocessing_validation.py
    (or run interactively with # %% cell markers)
"""

# %% Imports
import sys, os, copy, logging, pickle
import numpy as np
import pandas as pd
import dill
import scipy.signal as sig
from pathlib import Path
from sklearn.decomposition import PCA
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as colormap
import matplotlib.gridspec as gridspec

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs.config import config, ss_analysis_config
from scripts.utils.analysis_utils import load_dataset

# %% Logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                    format="%(asctime)s %(levelname)s %(message)s")

# %% Paths & constants
BIN_SIZE = config["bin_size"]
XCORR_THRESHOLD = config["xcorr_threshold"]
base_dir = config["base_dir"]
_session_tag = f"{config['experiment_name']}_{config['run_date']}_{config['run_idx']}"
FIG_DIR = os.path.join(PROJECT_ROOT, "figures", "validation_metrics", _session_tag)
os.makedirs(FIG_DIR, exist_ok=True)

SPIKE_SMOOTH_WIDTH = 100
RATE_SMOOTH_WIDTH = 50

# Helper: save figure as PDF
def save_fig(fig, name):
    fig.savefig(os.path.join(FIG_DIR, f"{name}.pdf"), dpi=300, bbox_inches="tight")
    logger.info(f"Saved {name}.pdf")


# %% Load companion pkl (original un-zeroed spikes + xcorr data)
companion_path = os.path.join(
    base_dir, "datasets",
    f"{config['experiment_name']}_xcorr_companion.pkl",
)
logger.info(f"Loading companion pkl: {companion_path}")
with open(companion_path, "rb") as f:
    companion = pickle.load(f)

orig_spikes = companion["spikes"]           # (T, N) — ALL original channels
xcorr_matrix = companion["xcorr_matrix"]    # (N, N)
chan_names = companion["chan_names"]
n_neurons = companion["n_neurons"]
rejection_masks = companion["rejection_masks"]  # {thresh: bool[N], True=keep}
rejection_names = companion["rejection_names"]  # {thresh: [names]}
orig_time_index = companion["time_index"]       # timedelta64[ns]
thresholds = companion["thresholds"]

logger.info(f"Companion: {orig_spikes.shape[0]} bins × {n_neurons} channels")
for t in thresholds:
    n_rej = len(rejection_names[t])
    logger.info(f"  threshold {t}: {n_rej} rejected, {n_neurons - n_rej} kept")

# %% Load merged dataset
logger.info("Loading merged dataset...")
dataset = load_dataset()
logger.info(f"Merged: data shape={dataset.data.shape}, "
            f"trial_info rows={len(dataset.trial_info)}")

# Ensure smoothed columns exist
spk_smooth_col = f"spikes_smooth_{SPIKE_SMOOTH_WIDTH}"
rate_smooth_col = f"lfads_rates_smooth_{RATE_SMOOTH_WIDTH}"
if spk_smooth_col not in dataset.data.columns.get_level_values(0):
    dataset.smooth_spk(gauss_width=SPIKE_SMOOTH_WIDTH,
                       name=f"smooth_{SPIKE_SMOOTH_WIDTH}", overwrite=False)
if rate_smooth_col not in dataset.data.columns.get_level_values(0):
    dataset.smooth_spk(signal_type="lfads_rates", gauss_width=RATE_SMOOTH_WIDTH,
                       name=f"smooth_{RATE_SMOOTH_WIDTH}", overwrite=False)
    dataset.data[rate_smooth_col] = dataset.data[rate_smooth_col].fillna(0)


# ============================================================
# Helper: custom scaler (from 03)
# ============================================================
class CustomScaler:
    def __init__(self, epsilon=0):
        self.mean = self.std = None
        self.epsilon = epsilon
    def fit_transform(self, data):
        self.mean = np.mean(data, axis=0)
        self.std = np.std(data, axis=0)
        return (data - self.mean) / (self.std + self.epsilon)
    def transform(self, data):
        return (data - self.mean) / (self.std + self.epsilon)


# ============================================================
# SECTION 1 — Cross-correlation analysis (ALL original units)
# ============================================================
# %% 1a: Correlation matrix heatmap
logger.info("Section 1: cross-correlation analysis (original un-zeroed spikes)")

fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(xcorr_matrix, cmap="RdBu_r", vmin=-0.3, vmax=0.3,
               interpolation="none", aspect="equal")
plt.colorbar(im, ax=ax, label="Pairwise correlation", shrink=0.8)

# Mark rejected channels on the axes
rejected_ixs_01 = np.where(~rejection_masks[0.1])[0]
for ix in rejected_ixs_01:
    ax.axhline(ix, color="red", lw=0.3, alpha=0.5)
    ax.axvline(ix, color="red", lw=0.3, alpha=0.5)

n_rej_01 = len(rejection_names[0.1])
ax.set_title(f"Pairwise spike cross-correlation matrix\n"
             f"({n_neurons} total units, red lines = {n_rej_01} rejected "
             f"at threshold={XCORR_THRESHOLD})", fontsize=11)
ax.set_xlabel("Unit index")
ax.set_ylabel("Unit index")
fig.tight_layout()
save_fig(fig, "01a_xcorr_matrix")
plt.close(fig)

# %% 1b: Histogram of pairwise correlations
upper_tri = xcorr_matrix[np.triu_indices(n_neurons, k=1)]

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.hist(upper_tri, bins=120, color="steelblue", edgecolor="none", alpha=0.85)

colors_thresh = {0.1: "red", 0.2: "orange", 0.3: "green"}
for thresh in thresholds:
    n_above = int(np.sum(upper_tri > thresh))
    n_rej = len(rejection_names[thresh])
    ax.axvline(thresh, color=colors_thresh[thresh], lw=2, ls="--",
               label=f"threshold={thresh} → {n_rej} units removed ({n_above} pairs above)")

ax.set_title("Pairwise correlation distribution (all original units)", fontsize=12)
ax.set_xlabel("Correlation coefficient")
ax.set_ylabel("Count")
ax.legend(fontsize=8)
fig.tight_layout()
save_fig(fig, "01b_xcorr_histogram")
plt.close(fig)

# %% 1c: Covariance matrices — all units and each threshold
cov_all = np.cov(orig_spikes.T)
vmax = np.percentile(np.abs(cov_all), 99)

fig, axes = plt.subplots(1, 4, figsize=(26, 6))

im0 = axes[0].imshow(cov_all, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                      interpolation="none", aspect="equal")
axes[0].set_title(f"All {n_neurons} units\n(before rejection)")
plt.colorbar(im0, ax=axes[0], shrink=0.8)

for ax_i, thresh in enumerate(thresholds):
    mask = rejection_masks[thresh]
    cov_t = np.cov(orig_spikes[:, mask].T)
    n_kept = int(mask.sum())
    im = axes[ax_i + 1].imshow(cov_t, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                                interpolation="none", aspect="equal")
    axes[ax_i + 1].set_title(f"threshold={thresh}\n"
                              f"{n_kept} units kept, "
                              f"{len(rejection_names[thresh])} removed")
    plt.colorbar(im, ax=axes[ax_i + 1], shrink=0.8)

for ax in axes:
    ax.set_xlabel("Unit index")
    ax.set_ylabel("Unit index")
fig.suptitle("Spike covariance matrices across xcorr rejection thresholds",
             fontsize=13)
fig.tight_layout()
save_fig(fig, "01c_covariance_matrices")
plt.close(fig)


# ============================================================
# SECTION 2 — Boundary artifact check
# ============================================================
# %% 2: Extract continuous segment and mark chop boundaries
logger.info("Section 2: boundary artifact check")

WINDOW_MS = 500
OVERLAP_MS = 100
STRIDE_MS = WINDOW_MS - OVERLAP_MS  # 400 ms

time_index = dataset.data.index.to_numpy()

# Find contiguous runs (gaps > 2*BIN_SIZE indicate discontinuities)
diffs_ms = np.diff(time_index).astype("timedelta64[ms]").astype(float)
gap_thresh = 2 * BIN_SIZE
gap_ixs = np.where(diffs_ms > gap_thresh)[0]
run_starts = np.concatenate([[0], gap_ixs + 1])
run_ends = np.concatenate([gap_ixs + 1, [len(time_index)]])
run_lengths = run_ends - run_starts

longest_run = np.argmax(run_lengths)
run_s, run_e = run_starts[longest_run], run_ends[longest_run]
logger.info(f"Longest contiguous run: {run_lengths[longest_run]} bins "
            f"({run_lengths[longest_run] * BIN_SIZE / 1000:.1f}s)")

# Pick ~15s segment from center
SEG_HALF_BINS = int(7500 / BIN_SIZE)
run_center = (run_s + run_e) // 2
seg_s = max(run_s, run_center - SEG_HALF_BINS)
seg_e = min(run_e, run_center + SEG_HALF_BINS)

seg_time = time_index[seg_s:seg_e]
seg_t_ms = (seg_time - seg_time[0]).astype("float64") / 1e6

# Identify which channels are active in the merged dataset
merged_spikes = dataset.data["spikes"].values.astype(np.float64)
chan_energy = np.sum(np.abs(merged_spikes), axis=0)
active_mask = chan_energy > 0

seg_spikes = merged_spikes[seg_s:seg_e]
seg_smooth = dataset.data[spk_smooth_col].values[seg_s:seg_e]

# Chop boundary times within segment
session_start_ms = time_index[0].astype("timedelta64[ms]").astype(float)
seg_start_ms = seg_time[0].astype("timedelta64[ms]").astype(float)
seg_end_ms = seg_time[-1].astype("timedelta64[ms]").astype(float)

first_boundary = STRIDE_MS * np.ceil((seg_start_ms - session_start_ms) / STRIDE_MS) + session_start_ms
boundaries_ms_abs = np.arange(first_boundary, seg_end_ms, STRIDE_MS)
boundaries_ms_rel = boundaries_ms_abs - seg_start_ms

# Segment data for all active channels
seg_spikes_active = seg_spikes[:, active_mask]       # (T, N_active)
seg_smooth_active = seg_smooth[:, active_mask]
seg_rates_smooth = dataset.data[rate_smooth_col].values[seg_s:seg_e]
seg_rates_smooth_active = seg_rates_smooth[:, active_mask]

n_active = int(active_mask.sum())

# Sort all neurons by peak activity time (same as Step 6)
sort_order = np.argsort(np.argmax(seg_smooth_active, axis=0))

# Compute chop boundary bin positions
boundary_bins = [int(np.searchsorted(seg_t_ms, b))
                 for b in boundaries_ms_rel if 0 < b < seg_t_ms[-1]]

vmax_spk = np.percentile(seg_spikes_active, 99)
vmax_smooth = np.percentile(seg_smooth_active, 99)
vmax_rate = np.percentile(seg_rates_smooth_active, 99)
vmax_shared = max(vmax_smooth, vmax_rate)

# %% 2a: Pcolor heatmaps — all neurons, sorted by peak activity
fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)

titles = [
    f"Raw spike counts (bin={BIN_SIZE} ms)",
    f"Smoothed spikes (σ={SPIKE_SMOOTH_WIDTH} ms)",
    f"Smoothed LFADS inferred rates (σ={RATE_SMOOTH_WIDTH} ms)",
]
data_panels = [
    seg_spikes_active[:, sort_order].T,
    seg_smooth_active[:, sort_order].T,
    seg_rates_smooth_active[:, sort_order].T,
]
vmaxs = [vmax_spk, vmax_shared, vmax_shared]
cmaps = ["bone_r", "viridis", "viridis"]

# x-tick labels in ms
n_xticks = 8
tick_ixs = np.linspace(0, len(seg_t_ms) - 1, n_xticks).astype(int)

for i, (ax, panel, title, vmax_i, cmap_i) in enumerate(
        zip(axes, data_panels, titles, vmaxs, cmaps)):
    ax.pcolor(panel, cmap=cmap_i, vmin=0, vmax=vmax_i)
    for bb in boundary_bins:
        ax.axvline(bb, color="red", lw=0.8, ls="--", alpha=0.7)
    ax.set_title(title, fontsize=11)
    ax.set_ylabel("Neurons (sorted by peak)")
    ax.set_xticks(tick_ixs)
    ax.set_xticklabels(seg_t_ms[tick_ixs].astype(int))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

axes[-1].set_xlabel("Time (ms, relative to segment start)")
fig.suptitle(
    f"Boundary artifact check — all {n_active} active units, sorted by peak activity\n"
    "Red dashed lines = chop boundaries",
    fontsize=13)
plt.subplots_adjust(hspace=0.4)
fig.tight_layout()
save_fig(fig, "02a_boundary_timeseries")
plt.close(fig)

# %% 2b: 2D PCA trajectory (line) with boundary points highlighted
# Use smoothed LFADS rates for cleaner trajectory
pca_2d = PCA(n_components=2)
scaler_2d = CustomScaler(1e-10)
rates_scaled = scaler_2d.fit_transform(seg_rates_smooth_active)
rates_pc = pca_2d.fit_transform(rates_scaled)

boundary_bin_ixs = np.array([
    int(np.searchsorted(seg_t_ms, b))
    for b in boundaries_ms_rel if 0 < b < seg_t_ms[-1]
])

fig, ax = plt.subplots(figsize=(8, 6))

# Draw trajectory as colored line segments (color = time)
from matplotlib.collections import LineCollection
points = rates_pc[:, :2].reshape(-1, 1, 2)
segs = np.concatenate([points[:-1], points[1:]], axis=1)
lc = LineCollection(segs, cmap="viridis", alpha=0.8, linewidth=0.8)
lc.set_array(seg_t_ms[:-1])
ax.add_collection(lc)
plt.colorbar(lc, ax=ax, label="Time (ms)")
ax.autoscale()

# Mark boundary bins with red x
ax.scatter(rates_pc[boundary_bin_ixs, 0], rates_pc[boundary_bin_ixs, 1],
           c="red", s=40, zorder=5, marker="x", linewidths=1.5,
           label="Chop boundaries")

ax.set_xlabel("PC 1")
ax.set_ylabel("PC 2")
ax.set_title("LFADS rate trajectory — chop boundaries highlighted\n"
             "(no jumps = no stitching artifacts)")
ax.legend()
fig.tight_layout()
save_fig(fig, "02b_boundary_pca_trajectory")
plt.close(fig)


# ============================================================
# SECTION 3 — Threshold robustness across PCA subspaces
# ============================================================
# %% 3: Setup
logger.info("Section 3: threshold robustness across PCA subspaces")

# Trial info filtering (same as 03)
MIN_POKES = ss_analysis_config["min_pokes_per_bout"]
SWITCH_THRESH = ss_analysis_config["switch_incl_thresh"]
PC_PAD = ss_analysis_config["pc_padding"]

dataset.trial_info = dataset.trial_info.dropna(subset=["trial_id_before_switch"])
dataset.trial_info = dataset.trial_info[
    dataset.trial_info["pokes_before_switch"] >= MIN_POKES
]
dataset.trial_info = dataset.trial_info[
    dataset.trial_info.trial_id_before_switch > SWITCH_THRESH
]
pre_switch_conditions = tuple(dataset.trial_info.trial_id_before_switch.unique())
dataset.trial_info = dataset.trial_info.reset_index(drop=True)


# %% Condition-averaged data extraction (from 03)
def grab_data_for_PC_space(input_array, input_indices, time_before_ms,
                           time_after_ms, align_point="start_time",
                           conditions=None):
    if conditions is None:
        conditions = pre_switch_conditions

    assert time_before_ms % BIN_SIZE == 0
    assert time_after_ms % BIN_SIZE == 0

    trial_info = dataset.trial_info
    pre_td = np.timedelta64(pd.Timedelta(time_before_ms, unit="ms"), "ns")
    post_td = np.timedelta64(pd.Timedelta(time_after_ms, unit="ms"), "ns")

    num_bins = int((time_after_ms + time_before_ms) / BIN_SIZE + 1)
    num_channels = input_array.shape[1]
    data = np.zeros((len(conditions), num_bins, num_channels))

    for ci, cond in enumerate(conditions):
        ct = trial_info[trial_info.trial_id_before_switch == cond]
        count = 0
        for i in range(ct.shape[0]):
            at = np.timedelta64(ct.iloc[i][align_point], "ns")
            si = np.searchsorted(input_indices, at - pre_td, side="left")
            ei = np.searchsorted(input_indices, at + post_td, side="left") + 1
            sl = input_array[si:ei]
            try:
                data[ci] += sl
                count += 1
            except ValueError:
                continue
        if count > 0:
            data[ci] /= count

    return data


# %% Build smoothed spike arrays at each threshold from original spikes
# We smooth the original un-zeroed spikes, then mask channels per threshold
logger.info("Building smoothed spike arrays at each threshold...")

def gaussian_smooth(spikes_2d, gauss_width_ms, bin_size_ms):
    """Apply the same Gaussian smoothing as BaseDataset.smooth_spk."""
    gauss_bin_std = gauss_width_ms / bin_size_ms
    win_len = int(6 * gauss_bin_std)
    window = sig.gaussian(win_len, gauss_bin_std, sym=True)
    window /= np.sum(window)
    out = np.zeros_like(spikes_2d, dtype=np.float64)
    shift_len = len(window) // 2
    for ch in range(spikes_2d.shape[1]):
        y = sig.lfilter(window, 1.0, spikes_2d[:, ch].astype(np.float64))
        out[:, ch] = np.concatenate([y[shift_len:], np.full(shift_len, np.nan)])
    return out

# Smooth the original (un-zeroed) spikes once
orig_smooth = gaussian_smooth(orig_spikes, SPIKE_SMOOTH_WIDTH, BIN_SIZE)

# Align time indices between companion and merged dataset
# The companion has the same wake-filtered time index as the merged dataset
merged_time = dataset.data.index.to_numpy()

# Build signal arrays for each threshold variant
# Each is (T_merged, N) — aligned to the merged dataset's time index
signal_variants = {}

for thresh in thresholds:
    mask = rejection_masks[thresh]  # True = keep
    # Start from the original smoothed spikes, zero out rejected channels
    arr = orig_smooth.copy()
    arr[:, ~mask] = 0.0
    # Trim/align to merged dataset length (handle any minor mismatch)
    min_len = min(arr.shape[0], len(merged_time))
    signal_variants[f"smooth_spk_thresh_{thresh}"] = arr[:min_len]

# Also include LFADS rates from merged dataset (only exists at threshold=0.1)
lfads_rates = dataset.data["lfads_rates"].to_numpy()
signal_variants["lfads_rates"] = lfads_rates

logger.info(f"Signal variants: {list(signal_variants.keys())}")


# %% Define PCA subspaces
subspace_defs = [
    {
        "name": "Navigation",
        "t_before": 0, "t_after": 3000,
        "align": "start_time",
    },
    {
        "name": "Pre-move",
        "t_before": 800, "t_after": 200,
        "align": "start_time",
    },
    {
        "name": "Outcome",
        "t_before": 0, "t_after": 1000,
        "align": "poke_in_ts",
    },
    {
        "name": "Reward_Pre-move",
        "t_before": 5000, "t_after": 2500,
        "align": "start_time",
    },
    {
        "name": "Reward_Outcome",
        "t_before": 0, "t_after": 3000,
        "align": "poke_in_ts",
    },
]


# %% Generate comparison plots for each subspace
for ss_def in subspace_defs:
    ss_name = ss_def["name"]
    logger.info(f"  Subspace: {ss_name}")

    # Build color map for conditions
    n_conds = len(pre_switch_conditions)
    cmap_arr = colormap.viridis(np.linspace(0, 1, n_conds))

    # Create figure: one 3D subplot per signal variant
    n_variants = len(signal_variants)
    fig = plt.figure(figsize=(6 * n_variants, 6))

    variant_labels = {
        "smooth_spk_thresh_0.1": "Smoothed spikes\n(xcorr ≤ 0.1, 314 units)",
        "smooth_spk_thresh_0.2": "Smoothed spikes\n(xcorr ≤ 0.2, 375 units)",
        "smooth_spk_thresh_0.3": "Smoothed spikes\n(xcorr ≤ 0.3, 385 units)",
        "lfads_rates": "LFADS inferred rates\n(trained at xcorr ≤ 0.1)",
    }

    for vi, (var_key, var_arr) in enumerate(signal_variants.items()):
        ax = fig.add_subplot(1, n_variants, vi + 1, projection="3d")

        # Extract condition-averaged data for this variant
        cond_data = grab_data_for_PC_space(
            var_arr, merged_time[:var_arr.shape[0]],
            time_before_ms=ss_def["t_before"],
            time_after_ms=ss_def["t_after"],
            align_point=ss_def["align"],
        )

        # Fit an independent PCA+scaler for THIS variant
        pca_input = cond_data.reshape(-1, cond_data.shape[2])
        valid = ~np.isnan(pca_input).any(axis=1)
        pca_input_v = pca_input[valid]
        scaler_v = CustomScaler(PC_PAD)
        scaled_v = scaler_v.fit_transform(pca_input_v)
        pca_v = PCA(n_components=3)
        pca_v.fit(scaled_v)

        for ci, cond in enumerate(pre_switch_conditions):
            traj = cond_data[ci]
            if np.any(np.isnan(traj)):
                continue
            proj = pca_v.transform(scaler_v.transform(traj)).T
            col = cmap_arr[ci]
            ax.plot(proj[0], proj[1], proj[2], color=col, lw=1.2, alpha=0.8)
            ax.scatter(proj[0, 0], proj[1, 0], proj[2, 0],
                       color=col, s=20, edgecolor="k", linewidth=0.5, zorder=5)

        ax.set_xlabel("PC1", fontsize=9)
        ax.set_ylabel("PC2", fontsize=9)
        ax.set_zlabel("PC3", fontsize=9)
        ax.set_title(variant_labels.get(var_key, var_key), fontsize=10)
        ax.tick_params(labelsize=7)

    n_rej_01 = len(rejection_names[0.1])
    n_rej_02 = len(rejection_names[0.2])
    n_rej_03 = len(rejection_names[0.3])
    fig.suptitle(
        f"{ss_name} subspace — threshold robustness comparison\n"
        f"(Units rejected: {n_rej_01} at 0.1, {n_rej_02} at 0.2, "
        f"{n_rej_03} at 0.3 out of {n_neurons} total; each panel uses its own PCA)",
        fontsize=12)
    fig.tight_layout()
    safe_name = ss_name.lower().replace(" ", "_").replace("-", "")
    save_fig(fig, f"03_{safe_name}_threshold_comparison")
    plt.close(fig)


# %% Summary
logger.info(f"\nAll figures saved to: {FIG_DIR}")
logger.info("Done.")
