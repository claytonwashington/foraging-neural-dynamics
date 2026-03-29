"""Script 3: State space analysis — 3D neural trajectory visualization

Generates Plotly and Matplotlib 3D plots of neural state space trajectories,
including:
- Trial-averaged spirals in navigation and pre-move subspaces
- Condition-averaged trajectories split by reward vs. no-reward
- Single-trial trajectories colored by reward history
- Single-trial outcome effects analysis (initial-state shifts, cosine angles,
  bootstrapping, and polar angle histograms)

Usage:
    Run cells interactively in VS Code or Jupyter (file uses # %% cell markers).

    python scripts/03_state_space_analysis.py
"""

# %% Imports
import sys
import os
import math
import logging
import typing
import numpy as np
import pandas as pd
import dill
from tqdm import tqdm
from pathlib import Path
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import matplotlib.cm as colormap
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import plotly.graph_objects as go
import ipywidgets as widgets
from IPython.display import display

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs.config import config, ss_analysis_config
from scripts.utils.analysis_utils import load_dataset

# %% Configuration
BIN_SIZE = config["bin_size"]
MIN_POKES_PER_BOUT = ss_analysis_config["min_pokes_per_bout"]
PC_padding = ss_analysis_config["pc_padding"]
SWITCH_THRESH_TO_INCLUDE = ss_analysis_config["switch_incl_thresh"]
PLOT_ALL_TRIALS = ss_analysis_config["plot_all_trials"]
base_dir = config["base_dir"]
results_dir = os.path.join(base_dir, "results")
os.makedirs(results_dir, exist_ok=True)

# %% Setup logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

# %% Load merged dataset
dataset = load_dataset()

# %% Drop trials with NaN switches
dataset.trial_info = dataset.trial_info.dropna(subset=["trial_id_before_switch"])

# %% Preserve original trial info before filtering
trial_info_orig = dataset.trial_info.copy()

# %% Filter: require minimum pokes before switch
dataset.trial_info = dataset.trial_info[
    dataset.trial_info["pokes_before_switch"] >= MIN_POKES_PER_BOUT
]

# %% Helper: Cartesian product of condition arrays
def get_cartesian_product(array1, array2):
    """Compute cartesian product of two arrays as tuple of tuples."""
    grid1, grid2 = np.meshgrid(array1, array2)
    cartesian_product = np.vstack([grid1.ravel(), grid2.ravel()]).T
    return tuple(map(tuple, cartesian_product))


# %% Establish condition constants
dataset.trial_info = dataset.trial_info[
    dataset.trial_info.trial_id_before_switch > SWITCH_THRESH_TO_INCLUDE
]

pre_switch_conditions = tuple(dataset.trial_info.trial_id_before_switch.unique())
pre_switch_conditions_noswitch = (-5.0, -4.0, -3.0, -2.0, -1.0, -8.0, -7.0, -6.0)
pre_switch_conditions_orig = tuple(trial_info_orig.trial_id_before_switch.unique())

action_conditions = get_cartesian_product(
    np.array(["left", "right"]), pre_switch_conditions
)
action_conditions_orig = get_cartesian_product(
    np.array(["left", "right"]), pre_switch_conditions_orig
)
patch_id_conditions = get_cartesian_product(
    np.array(["A", "B", "C"]), pre_switch_conditions
)
patch_id_conditions_orig = get_cartesian_product(
    np.array(["A", "B", "C"]), pre_switch_conditions_orig
)
reward_conditions_noswitch = get_cartesian_product(
    np.array(["reward", "no_reward"]), pre_switch_conditions_noswitch
)
reward_conditions = get_cartesian_product(
    np.array(["reward", "no_reward"]), pre_switch_conditions
)

# %% Reset index after filtering
dataset.trial_info = dataset.trial_info.reset_index(drop=True)


# %% Core function: extract condition-averaged or single-trial data
def grab_data_for_PC_space(
    input_array,
    input_indices,
    time_before_ms,
    time_after_ms,
    align_point="start_time",
    conditions=None,
    single_trial=False,
    trial_info_override=None,
    reward_condition="all",
    trial_info_full=False,
):
    """Extract neural data aligned to events, grouped by conditions.

    Parameters
    ----------
    input_array : np.ndarray
        T × N array of neural features (e.g., smoothed LFADS rates).
    input_indices : np.ndarray
        TimeDelta index array matching input_array rows.
    time_before_ms : int
        Time before alignment point (must be multiple of BIN_SIZE).
    time_after_ms : int
        Time after alignment point (must be multiple of BIN_SIZE).
    align_point : str
        Trial info field name for temporal alignment.
    conditions : tuple, optional
        Conditions to group trials by. Defaults to pre_switch_conditions.
    single_trial : bool
        If True, return list of individual trial trajectories.
    trial_info_override : pd.DataFrame, optional
        Use this trial info instead of dataset.trial_info.
    reward_condition : str
        'all', 'reward', or 'no_reward' — filter trials by reward.
    trial_info_full : bool
        If True, use trial_info_orig (unfiltered).

    Returns
    -------
    np.ndarray or tuple
        If single_trial=False: (n_conditions × n_bins × n_channels) array.
        If single_trial=True: (trajectories, conditions, trial_indices) tuple.
    """
    if conditions is None:
        conditions = pre_switch_conditions

    assert time_before_ms % BIN_SIZE == 0, "time_before_ms must be a multiple of BIN_SIZE"
    assert time_before_ms >= 0, "time_before_ms must be >= 0"
    assert time_after_ms % BIN_SIZE == 0, "time_after_ms must be a multiple of BIN_SIZE"

    trial_info = (
        trial_info_override
        if trial_info_override is not None
        else (dataset.trial_info if not trial_info_full else trial_info_orig)
    )

    pre_align_td = np.timedelta64(pd.Timedelta(time_before_ms, unit="ms"), "ns")
    post_align_td = np.timedelta64(pd.Timedelta(time_after_ms, unit="ms"), "ns")

    # Filter by reward condition
    if reward_condition == "all":
        trial_df = trial_info
        reward_time_type = "Reward" if align_point == "poke_in_ts" else "RewardPrevious"
    else:
        if align_point == "poke_in_ts":
            reward_time_type = "Reward"
        else:
            reward_time_type = "RewardPrevious"
        if reward_condition == "no_reward":
            trial_df = trial_info[trial_info[reward_time_type] == False]
        elif reward_condition == "reward":
            trial_df = trial_info[trial_info[reward_time_type] == True]
        else:
            raise ValueError("Invalid reward condition")

    if not single_trial:
        num_bins = int((time_after_ms + time_before_ms) / BIN_SIZE + 1)
        num_conditions = len(conditions)
        num_channels = dataset.data.spikes.shape[1]
        data = np.zeros((num_conditions, num_bins, num_channels))
    else:
        trajectories = []
        trial_conditions = []
        trial_indices = []

    for cond_ix, condition in enumerate(conditions):
        # Select trials matching this condition
        if conditions in (patch_id_conditions, patch_id_conditions_orig):
            condition_trials = trial_df[trial_df.StartStem == condition[0]]
            condition_trials = condition_trials[
                condition_trials.trial_id_before_switch == float(condition[1])
            ]
        elif conditions in (action_conditions, action_conditions_orig):
            if condition[0] == "left":
                condition_trials = trial_df[trial_df.StartLeaf % 2 == 1]
            else:
                condition_trials = trial_df[trial_df.StartLeaf % 2 == 0]
            condition_trials = condition_trials[
                condition_trials.trial_id_before_switch == float(condition[1])
            ]
        elif conditions in (reward_conditions_noswitch, reward_conditions):
            if condition[0] == "reward":
                condition_trials = trial_df[trial_df[reward_time_type] == 1]
            else:
                condition_trials = trial_df[trial_df[reward_time_type] == 0]
            condition_trials = condition_trials[
                condition_trials.trial_id_before_switch == float(condition[1])
            ]
        else:
            condition_trials = trial_df[trial_df.trial_id_before_switch == condition]

        if single_trial:
            trial_indices.extend(condition_trials.index)
            for i in tqdm(range(condition_trials.shape[0])):
                align_time = np.timedelta64(condition_trials.iloc[i][align_point], "ns")
                start_time = np.timedelta64(align_time, "ns") - pre_align_td
                start_ix = np.searchsorted(input_indices, start_time, side="left")
                end_time = np.timedelta64(align_time, "ns") + post_align_td
                end_ix = np.searchsorted(input_indices, end_time, side="left") + 1
                data_slice = input_array[start_ix:end_ix]
                trajectories.append(data_slice)
                trial_conditions.append(condition)
        else:
            for i in tqdm(
                range(condition_trials.shape[0]),
                desc=f"Processing condition {condition}",
            ):
                align_time = np.timedelta64(condition_trials.iloc[i][align_point], "ns")
                start_time = np.timedelta64(align_time, "ns") - pre_align_td
                start_ix = np.searchsorted(input_indices, start_time, side="left")
                end_time = np.timedelta64(align_time, "ns") + post_align_td
                end_ix = np.searchsorted(input_indices, end_time, side="left") + 1
                data_slice = input_array[start_ix:end_ix]
                try:
                    data[cond_ix, :, :] += data_slice
                except ValueError:
                    continue
            data[cond_ix, :, :] /= condition_trials.shape[0]

    if single_trial:
        return (
            np.array(trajectories),
            np.array(trial_conditions, dtype=object),
            np.array(trial_indices),
        )
    else:
        return data


# %% Set data field and extract arrays for PCA
data_field = "lfads_rates_smooth_50"
df_to_slice = dataset.data[data_field].to_numpy()
index_array = dataset.data.index.to_numpy()

# %% Grab condition-averaged data for fitting PCA subspaces
nav_data = grab_data_for_PC_space(
    df_to_slice, index_array, time_before_ms=0, time_after_ms=3000, align_point="start_time"
)
pre_move_data = grab_data_for_PC_space(
    df_to_slice, index_array, time_before_ms=800, time_after_ms=200, align_point="start_time"
)
outcome_data = grab_data_for_PC_space(
    df_to_slice, index_array, time_before_ms=0, time_after_ms=1000, align_point="poke_in_ts"
)
pre_move_outcome_data = grab_data_for_PC_space(
    df_to_slice, index_array, time_before_ms=1000, time_after_ms=400,
    align_point="start_time", conditions=reward_conditions,
)
nav_action_trajectories = grab_data_for_PC_space(
    df_to_slice, index_array, time_before_ms=0, time_after_ms=3000,
    align_point="start_time", conditions=action_conditions,
)
nav_patch_trajectories = grab_data_for_PC_space(
    df_to_slice, index_array, time_before_ms=0, time_after_ms=3000,
    align_point="start_time", conditions=patch_id_conditions,
)


# %% Custom scaler (avoids division-by-zero for constant channels)
class CustomScaler:
    """StandardScaler with configurable epsilon for numerical stability."""

    def __init__(self, epsilon=0):
        self.mean = None
        self.std = None
        self.epsilon = epsilon

    def fit_transform(self, data):
        self.mean = np.mean(data, axis=0)
        self.std = np.std(data, axis=0)
        return (data - self.mean) / (self.std + self.epsilon)

    def transform(self, data):
        return (data - self.mean) / (self.std + self.epsilon)

    def inverse_transform(self, data):
        return data * (self.std + self.epsilon) + self.mean


# %% Reshape for PCA and fit
nav_pca_input = nav_data.reshape(-1, nav_data.shape[2])
pre_move_pca_input = pre_move_data.reshape(-1, pre_move_data.shape[2])
outcome_pca_input = outcome_data.reshape(-1, outcome_data.shape[2])

nav_action_pca_input = nav_action_trajectories.reshape(-1, nav_action_trajectories.shape[2])
nav_patch_pca_input = nav_patch_trajectories.reshape(-1, nav_patch_trajectories.shape[2])
pre_move_outcome_pca_input = pre_move_outcome_data.reshape(-1, pre_move_outcome_data.shape[2])

# Scale data
nav_scaler = CustomScaler(PC_padding)
pre_move_scaler = CustomScaler(PC_padding)
nav_action_scaler = CustomScaler(PC_padding)
nav_patch_scaler = CustomScaler(PC_padding)
outcome_scaler = CustomScaler(PC_padding)

pre_move_outcome_scaler = CustomScaler(PC_padding)

nav_pca_input_scaled = nav_scaler.fit_transform(nav_pca_input)
pre_move_pca_input_scaled = pre_move_scaler.fit_transform(pre_move_pca_input)
nav_action_pca_input_scaled = nav_action_scaler.fit_transform(nav_action_pca_input)
nav_patch_pca_input_scaled = nav_patch_scaler.fit_transform(nav_patch_pca_input)
outcome_pca_input_scaled = outcome_scaler.fit_transform(outcome_pca_input)


# Remove NaN rows (can occur from reward=-1 trials)
pre_move_outcome_pca_input = pre_move_outcome_pca_input[
    ~np.isnan(pre_move_outcome_pca_input).any(axis=1)
]
pre_move_outcome_pca_input_scaled = pre_move_outcome_scaler.fit_transform(
    pre_move_outcome_pca_input
)

# Fit PCA models
n_pca = PCA(n_components=5)
pm_pca = PCA(n_components=5)
na_pca = PCA(n_components=5)
np_pca = PCA(n_components=5)
o_pca = PCA(n_components=5)

pmr_pca = PCA(n_components=5)

nav_pca = n_pca.fit_transform(nav_pca_input_scaled)
pre_move_pca = pm_pca.fit_transform(pre_move_pca_input_scaled)
nav_action_pca = na_pca.fit_transform(nav_action_pca_input_scaled)
nav_patch_pca = np_pca.fit_transform(nav_patch_pca_input_scaled)
outcome_pca = o_pca.fit_transform(outcome_pca_input_scaled)

pre_move_outcome_pca = pmr_pca.fit_transform(pre_move_outcome_pca_input_scaled)

# %% Grab data for projection / plotting
PLOT_ALL_TRIALS = False  # When True, includes trials not used for PCA fitting
conds = pre_switch_conditions_orig if PLOT_ALL_TRIALS else pre_switch_conditions
conds_act = action_conditions_orig if PLOT_ALL_TRIALS else action_conditions
conds_patch = patch_id_conditions_orig if PLOT_ALL_TRIALS else patch_id_conditions
ti_full = True if PLOT_ALL_TRIALS else False

if ss_analysis_config["trial_condition"] == "all" and PLOT_ALL_TRIALS == False:
    pass
else:
    nav_data = grab_data_for_PC_space(
        df_to_slice, index_array, time_before_ms=0, time_after_ms=3000,
        align_point="start_time", reward_condition=ss_analysis_config["trial_condition"],
        conditions=conds, trial_info_full=ti_full,
    )
nav_trajectories_by_condition = {
    condition: nav_data[i, :, :] for i, condition in enumerate(conds)
}

pre_move_data_to_project = grab_data_for_PC_space(
    df_to_slice, index_array, time_before_ms=1000, time_after_ms=1500,
    align_point="start_time", reward_condition="all",
    conditions=conds, trial_info_full=ti_full,
)
pre_move_trajectories_by_condition = {
    condition: pre_move_data_to_project[i, :, :]
    for i, condition in enumerate(conds)
}

outcome_data_to_project = grab_data_for_PC_space(
    df_to_slice, index_array, time_before_ms=0, time_after_ms=3000,
    align_point="poke_in_ts", reward_condition="all",
    conditions=pre_switch_conditions_noswitch, trial_info_full=ti_full,
)
outcome_trajectories_by_condition = {
    condition: outcome_data_to_project[i, :, :]
    for i, condition in enumerate(pre_switch_conditions_noswitch)
}

# K99 figure data: both reward conditions in pre-move subspace
outcome_data_both_conditions_pre_move = grab_data_for_PC_space(
    df_to_slice, index_array, time_before_ms=5000, time_after_ms=2500,
    align_point="start_time", reward_condition="all",
    conditions=reward_conditions, trial_info_full=ti_full,
)
outcome_trajectories_both_conditions_by_condition_pre_move = {
    condition: outcome_data_both_conditions_pre_move[i, :, :]
    for i, condition in enumerate(reward_conditions)
}

# K99 figure data: both reward conditions in outcome subspace
outcome_data_both_conditions_outcome = grab_data_for_PC_space(
    df_to_slice, index_array, time_before_ms=0, time_after_ms=3000,
    align_point="poke_in_ts", reward_condition="all",
    conditions=reward_conditions_noswitch, trial_info_full=ti_full,
)
outcome_trajectories_both_conditions_by_condition_outcome = {
    condition: outcome_data_both_conditions_outcome[i, :, :]
    for i, condition in enumerate(reward_conditions_noswitch)
}

# Action and patch ID data
if ss_analysis_config["trial_condition"] == "all" and PLOT_ALL_TRIALS == False:
    pass
else:
    nav_action_trajectories = grab_data_for_PC_space(
        df_to_slice, index_array, time_before_ms=0, time_after_ms=3000,
        align_point="start_time", conditions=conds_act, trial_info_full=ti_full,
        reward_condition=ss_analysis_config["trial_condition"],
    )
nav_action_trajectories_by_condition = {
    condition: nav_action_trajectories[i, :, :]
    for i, condition in enumerate(conds_act)
}

if ss_analysis_config["trial_condition"] == "all" and PLOT_ALL_TRIALS == False:
    pass
else:
    nav_patch_trajectories = grab_data_for_PC_space(
        df_to_slice, index_array, time_before_ms=0, time_after_ms=3000,
        conditions=conds_patch, trial_info_full=ti_full,
        align_point="start_time", reward_condition=ss_analysis_config["trial_condition"],
    )
nav_patch_trajectories_by_condition = {
    condition: nav_patch_trajectories[i, :, :]
    for i, condition in enumerate(conds_patch)
}


# %% Build color maps
sample_space = np.linspace(0, 1, int(len(pre_switch_conditions)))
sample_space_part = np.linspace(0.25, 1, int(len(pre_switch_conditions)))
sample_space_noswitch = np.linspace(0, 1, len(pre_switch_conditions) - 1)
sample_space_orig = np.linspace(0, 1, len(pre_switch_conditions_orig))

colors = colormap.viridis(sample_space)
color_map = {condition: colors[i] for i, condition in enumerate(sorted(pre_switch_conditions))}

colors_noswitch = colormap.viridis(sample_space_noswitch)
color_map_noswitch = {
    condition: colors_noswitch[i]
    for i, condition in enumerate(sorted(pre_switch_conditions_noswitch))
}

colors_orig = colormap.viridis(sample_space_orig)
color_map_orig = {
    condition: colors_orig[i]
    for i, condition in enumerate(sorted(pre_switch_conditions_orig))
}

# Build reward/action/patch color maps
reds = colormap.Reds(sample_space)
blues = colormap.Blues(sample_space)
greens = colormap.Greens(sample_space)
reds_part = colormap.Reds(sample_space_part)
blues_part = colormap.Blues(sample_space_part)
reds_noswitch = colormap.Reds(sample_space_noswitch)
blues_noswitch = colormap.Blues(sample_space_noswitch)

# Action maps
reds_map = {c: reds[i] for i, c in enumerate(sorted([a for a in action_conditions if "left" in a]))}
blues_map = {c: blues[i] for i, c in enumerate(sorted([a for a in action_conditions if "right" in a]))}

# Patch maps
reds_map_2 = {c: reds[i] for i, c in enumerate(sorted([p for p in patch_id_conditions if "A" in p]))}
blues_map_2 = {c: blues[i] for i, c in enumerate(sorted([p for p in patch_id_conditions if "B" in p]))}
greens_map = {c: greens[i] for i, c in enumerate(sorted([p for p in patch_id_conditions if "C" in p]))}

# Reward maps
reds_map_3 = {c: reds_part[i] for i, c in enumerate(sorted([r for r in reward_conditions if "reward" in r and "no_" not in r]))}
blues_map_3 = {c: blues_part[i] for i, c in enumerate(sorted([r for r in reward_conditions if "no_reward" in r]))}
reds_map_noswitch = {c: reds_noswitch[i] for i, c in enumerate(sorted([r for r in reward_conditions_noswitch if "reward" in r and "no_" not in r]))}
blues_map_noswitch = {c: blues_noswitch[i] for i, c in enumerate(sorted([r for r in reward_conditions_noswitch if "no_reward" in r]))}

# Merge all into unified color map
color_map.update(reds_map)
color_map.update(reds_map_2)
color_map.update(reds_map_3)
color_map.update(blues_map)
color_map.update(blues_map_2)
color_map.update(blues_map_3)
color_map.update(greens_map)
color_map_noswitch.update(reds_map_noswitch)
color_map_noswitch.update(blues_map_noswitch)


# %% Helper for extracting camera angles from Plotly
def eye_to_elev_azim(eye):
    """Convert Plotly camera eye coordinates to elevation/azimuth angles."""
    x, y, z = float(eye.x), float(eye.y), float(eye.z)
    azim = np.degrees(np.arctan2(y, x))
    elev = np.degrees(np.arctan2(z, np.sqrt(x**2 + y**2)))
    return elev, azim


# %% Core plotting function: 3D neural trajectory visualization
def make_3d_plot(
    neural_trajectories,
    conditions,
    pca,
    scaler,
    title,
    pcs_to_include: typing.List,
    color_style="condition",
    zero_offset_dot=0,
    reward_trials=None,
    color_map=color_map,
    dot_size_mult=1,
    add_halo=True,
    skip_nan=False,
    non_reward_begin_offset=None,
    plot_style="plotly",
    azim=0,
    elev=0,
    save_fig=False,
    skip_conditions=None,
    set_mpl_axes_equal_to_plotly_scale=True,
    extra_traces=None,
    extra_names=None,
    extra_color=None,
    extra_linewidth=3.0,
    extra_alpha=0.6,
    extra_halo=False,
    show=True,
):
    """Plot 3D PCA-projected neural trajectories.

    Parameters
    ----------
    neural_trajectories : iterable of np.ndarray
        List of (time × channels) arrays, one per condition.
    conditions : iterable
        Condition labels matching trajectories.
    pca : PCA
        Fitted PCA object for projection.
    scaler : CustomScaler
        Fitted scaler for standardization before projection.
    title : str
        Plot title.
    pcs_to_include : list of int
        Which 3 PC indices to plot (e.g., [0, 1, 3]).
    color_style : str
        'condition' or 'reward' for coloring strategy.
    zero_offset_dot : int
        Time index at which to place the colored start dot.
    reward_trials : array-like, optional
        Trial IDs that received reward (for reward coloring).
    plot_style : str
        'plotly' for interactive, 'mpl' for publication-quality static.
    save_fig : bool
        Save the figure to {base_dir}/figures/state_space/.
    extra_traces : iterable, optional
        Additional trajectories to overlay.

    Returns
    -------
    tuple
        (fig, ax) where ax is None for plotly.
    """
    assert color_style in ["condition", "reward"]
    assert plot_style in ["plotly", "mpl"]

    fig = None
    ax = None

    if plot_style == "plotly":
        fig = go.FigureWidget()
        out = widgets.Output()
        read_btn = widgets.Button(description="Read camera (elev/azim)")

        def read_camera(_):
            cam = fig.layout.scene.camera
            elev, azim = eye_to_elev_azim(cam.eye)
            with out:
                out.clear_output()
                print(f"elev={elev:.2f}°, azim={azim:.2f}°  |  eye=({cam.eye.x:.3f}, {cam.eye.y:.3f}, {cam.eye.z:.3f})")

        read_btn.on_click(read_camera)
    elif plot_style == "mpl":
        fig = plt.figure(figsize=(8, 8), dpi=300)
        ax = fig.add_subplot(111, projection="3d")

    sample_space_local = np.linspace(
        0, 1, int(len(neural_trajectories) * 2) + 2
    )
    blues_local = colormap.Blues(sample_space_local)[2::2]
    reds_local = colormap.Reds(sample_space_local)[2::2]

    projected_allcond = []

    def _project(traj_2d):
        projected_ = pca.transform(scaler.transform(traj_2d)).T
        if non_reward_begin_offset is not None and color_style != "reward":
            projected_[:, :non_reward_begin_offset] = np.nan
        return projected_

    norm_conditions = []
    for c in conditions:
        if isinstance(c, np.ndarray):
            c = tuple(c.tolist())
        elif isinstance(c, list):
            c = tuple(c)
        norm_conditions.append(c)

    # Main trajectories
    for i, (traj, cond) in enumerate(zip(neural_trajectories, norm_conditions)):
        if skip_nan and np.sum(np.isnan(traj)) > 0:
            continue
        if skip_conditions is not None and cond in skip_conditions:
            continue

        projected = _project(traj)
        projected_allcond.append(projected)

        if color_style == "condition":
            plot_col = color_map[cond]
            if plot_style == "plotly":
                plot_col_rgb = f"rgba({plot_col[0]*255}, {plot_col[1]*255}, {plot_col[2]*255}, 1.0)"
            else:
                plot_col_rgb = tuple(plot_col)
            if type(cond) == tuple and "no_reward" in cond and non_reward_begin_offset is not None:
                projected[:, :non_reward_begin_offset] = np.nan
        elif color_style == "reward":
            trial_prox = -1 * int(cond)
            if cond in reward_trials:
                plot_col = reds_local[trial_prox]
            else:
                if non_reward_begin_offset is not None:
                    projected[:, :non_reward_begin_offset] = np.nan
                plot_col = blues_local[trial_prox]
            plot_col_rgb = (
                f"rgba({plot_col[0]*255}, {plot_col[1]*255}, {plot_col[2]*255}, 1.0)"
                if plot_style == "plotly"
                else tuple(plot_col)
            )

        if plot_style == "mpl":
            dsm = dot_size_mult * 1.5
            if add_halo:
                ax.plot(
                    projected[pcs_to_include[0], :],
                    projected[pcs_to_include[1], :],
                    -1 * projected[pcs_to_include[2], :],
                    color="black", linewidth=2, alpha=0.5,
                )
            ax.plot(
                projected[pcs_to_include[0], :],
                projected[pcs_to_include[1], :],
                -1 * projected[pcs_to_include[2], :],
                color=plot_col_rgb,
            )
            ax.scatter(
                projected[pcs_to_include[0], zero_offset_dot],
                projected[pcs_to_include[1], zero_offset_dot],
                -1 * projected[pcs_to_include[2], zero_offset_dot],
                color=plot_col_rgb, s=9 * dsm, edgecolor="black", linewidth=1.5,
            )
        else:
            if add_halo:
                fig.add_trace(go.Scatter3d(
                    x=projected[pcs_to_include[0], :],
                    y=projected[pcs_to_include[1], :],
                    z=-1 * projected[pcs_to_include[2], :],
                    mode="lines",
                    line=dict(color="rgba(0, 0, 0, 0.5)", width=8),
                    showlegend=False,
                ))
            fig.add_trace(go.Scatter3d(
                x=projected[pcs_to_include[0], :],
                y=projected[pcs_to_include[1], :],
                z=-1 * projected[pcs_to_include[2], :],
                mode="lines+markers",
                line=dict(color=plot_col_rgb, width=5),
                marker=dict(size=2),
                name=str(cond),
                showlegend=True,
            ))
            marker_dict = (
                dict(color=plot_col_rgb, size=9 * dot_size_mult, line=dict(color="black", width=5))
                if add_halo
                else dict(color=plot_col_rgb, size=9 * dot_size_mult)
            )
            fig.add_trace(go.Scatter3d(
                x=[projected[pcs_to_include[0], zero_offset_dot]],
                y=[projected[pcs_to_include[1], zero_offset_dot]],
                z=[-1 * projected[pcs_to_include[2], zero_offset_dot]],
                mode="markers",
                marker=marker_dict,
                name=f"{cond} start",
                showlegend=False,
            ))

    # Overlay extra traces if provided
    if extra_traces is not None:
        if extra_names is None:
            extra_names = [f"extra_{i}" for i, _ in enumerate(extra_traces)]
        for ix, (xt, name) in enumerate(zip(extra_traces, extra_names)):
            if xt is None:
                continue
            projected = _project(xt)
            if isinstance(extra_color, (tuple, list, np.ndarray)):
                col = tuple(extra_color[ix])
            else:
                col = extra_color
            if col is None:
                col = "rgba(100,100,100,0.8)" if plot_style == "plotly" else (0.4, 0.4, 0.4, extra_alpha)

            if plot_style == "mpl":
                lc = col if isinstance(col, tuple) and len(col) == 4 else (0.4, 0.4, 0.4, extra_alpha)
                if extra_halo:
                    ax.plot(
                        projected[pcs_to_include[0], :], projected[pcs_to_include[1], :],
                        -1 * projected[pcs_to_include[2], :],
                        color="black", linewidth=extra_linewidth + 1.5, alpha=0.35,
                    )
                ax.plot(
                    projected[pcs_to_include[0], :], projected[pcs_to_include[1], :],
                    -1 * projected[pcs_to_include[2], :],
                    color=lc, linewidth=extra_linewidth, alpha=extra_alpha, label=str(name),
                )
            else:
                if isinstance(col, tuple):
                    cstr = f"rgba({int(col[0]*255)},{int(col[1]*255)},{int(col[2]*255)},{col[3] if len(col)==4 else extra_alpha})"
                else:
                    cstr = col
                if extra_halo:
                    fig.add_trace(go.Scatter3d(
                        x=projected[pcs_to_include[0], :], y=projected[pcs_to_include[1], :],
                        z=-1 * projected[pcs_to_include[2], :],
                        mode="lines", line=dict(color="rgba(0,0,0,0.4)", width=extra_linewidth + 2),
                        showlegend=False,
                    ))
                fig.add_trace(go.Scatter3d(
                    x=projected[pcs_to_include[0], :], y=projected[pcs_to_include[1], :],
                    z=-1 * projected[pcs_to_include[2], :],
                    mode="lines", line=dict(color=cstr, width=extra_linewidth),
                    name=str(name), showlegend=True,
                ))

    # Final layout
    if plot_style == "mpl":
        fig.suptitle(title, y=0.95, fontsize=16)
        plt.tight_layout()
        ax.set_xlabel(f"PC {pcs_to_include[0] + 1}")
        ax.set_ylabel(f"PC {pcs_to_include[1] + 1}")
        ax.set_zlabel(f"PC {pcs_to_include[2] + 1}")
        ax.view_init(elev, azim)
        ax.set_axis_off()

        if set_mpl_axes_equal_to_plotly_scale:
            def set_axes_equal(ax):
                x_limits = ax.get_xlim3d()
                y_limits = ax.get_ylim3d()
                z_limits = ax.get_zlim3d()
                x_range = abs(x_limits[1] - x_limits[0])
                y_range = abs(y_limits[1] - y_limits[0])
                z_range = abs(z_limits[1] - z_limits[0])
                plot_radius = 0.5 * max([x_range, y_range, z_range])
                ax.set_xlim3d([np.mean(x_limits) - plot_radius, np.mean(x_limits) + plot_radius])
                ax.set_ylim3d([np.mean(y_limits) - plot_radius, np.mean(y_limits) + plot_radius])
                ax.set_zlim3d([np.mean(z_limits) - plot_radius, np.mean(z_limits) + plot_radius])
            set_axes_equal(ax)

        if save_fig:
            fig_dir = os.path.join(base_dir, "figures", "state_space")
            os.makedirs(fig_dir, exist_ok=True)
            logger.info(f"Saving {title}")
            plt.savefig(
                os.path.join(fig_dir, "_".join(title.split(" ")) + ".pdf"),
                dpi=600, format="pdf", transparent=True,
            )
        if show:
            plt.show()
    else:
        fig.update_layout(
            scene=dict(
                xaxis_title=f"PC{pcs_to_include[0] + 1}",
                yaxis_title=f"PC{pcs_to_include[1] + 1}",
                zaxis_title=f"PC{pcs_to_include[2] + 1}",
            ),
            title=title,
            height=500,
        )
        if show:
            display(widgets.VBox([fig, read_btn, out]))

    return fig, projected_allcond


# ===========================================================================
# K99 FIGURES
# ===========================================================================

# %% Choose color map based on PLOT_ALL_TRIALS
cm = color_map_orig if PLOT_ALL_TRIALS else color_map

# %% Figure 3A Top Panel: Navigation PC space
title = "Path progression: move init to move init +3s in Nav. PC space"
nav_PCs = [0, 1, 3]
_fig, _ax = make_3d_plot(
    nav_trajectories_by_condition.values(),
    nav_trajectories_by_condition.keys(),
    n_pca, nav_scaler, title, nav_PCs,
    color_map=cm, plot_style="plotly", elev=90, azim=67.52,
)

# %% Figure 3A Bottom Panel: Pre-move PC space
title = "-1s to +1.5s in Pre-move PC space: all reward conds"
pre_move_PCs = [0, 1, 3]
_fig, _ax = make_3d_plot(
    pre_move_trajectories_by_condition.values(),
    pre_move_trajectories_by_condition.keys(),
    pm_pca, pre_move_scaler, title, pre_move_PCs,
    color_map=cm, zero_offset_dot=100, add_halo=False,
)

# %% Outcome PC space (without switch trial)
outcome_trajectories_by_condition_sans_last = {
    k: v for k, v in outcome_trajectories_by_condition.items() if k != -1.0
}
title = "poke-in 0s to poke-in +3s in outcome PC space"
outcome_PCs = [0, 1, 3]
_fig, _ax = make_3d_plot(
    outcome_trajectories_by_condition_sans_last.values(),
    outcome_trajectories_by_condition_sans_last.keys(),
    o_pca, outcome_scaler, title, outcome_PCs,
    color_map=color_map_noswitch, add_halo=False, skip_nan=True,
    plot_style="plotly", elev=20, azim=150, save_fig=False,
)

# %% K99: Both outcome conditions in outcome PC space
elev = 20
azim = 150
title = "poke-in aligned outcome | all conds (0,3) | e{}a{}".format(elev, azim)
pmr_PCs = [0, 1, 3]
_fig, _ax = make_3d_plot(
    outcome_trajectories_both_conditions_by_condition_outcome.values(),
    outcome_trajectories_both_conditions_by_condition_outcome.keys(),
    o_pca, outcome_scaler, title, pmr_PCs,
    color_map=cm, add_halo=False, skip_nan=True,
    plot_style="plotly", non_reward_begin_offset=0,
    zero_offset_dot=0, elev=elev, azim=azim, save_fig=False,
)

# %% K99: Both reward conditions in pre-move PC space
elev = 30
azim = -30
title = "poke-out aligned pre-move | rew: (-5,2.5) | no rew: (-0.9,2.5) | e{}a{}".format(elev, azim)
blacklist = ["0.0"]
skip_conds = [
    key for key in outcome_trajectories_both_conditions_by_condition_pre_move.keys()
    if key[1] in blacklist
]
pmr_PCs = [0, 1, 2]
_fig, _ax = make_3d_plot(
    outcome_trajectories_both_conditions_by_condition_pre_move.values(),
    outcome_trajectories_both_conditions_by_condition_pre_move.keys(),
    pm_pca, pre_move_scaler, title, pmr_PCs,
    color_map=cm, add_halo=False, skip_nan=True,
    plot_style="plotly", non_reward_begin_offset=0,
    zero_offset_dot=0, skip_conditions=skip_conds,
    elev=elev, azim=azim, save_fig=False,
)

# %% Figure 3A Middle Panel: Action in Nav PC space
title = "Path progression + action in Nav. PC space"
PCs = [0, 1, 2]
_fig, _ax = make_3d_plot(
    nav_action_trajectories_by_condition.values(),
    nav_action_trajectories_by_condition.keys(),
    na_pca, nav_action_scaler, title, PCs, color_map=cm,
)

# %% Figure 3A Right Panel: Patch ID in Nav PC space
title = "Path progression + patch ID in Nav. PC space"
PCs = [0, 1, 2]
_fig, _ax = make_3d_plot(
    nav_patch_trajectories_by_condition.values(),
    nav_patch_trajectories_by_condition.keys(),
    np_pca, nav_patch_scaler, title, PCs, color_map=cm,
)

# %% Single-trial navigation trajectories
title = "SINGLE TRIAL Path progression: move init to move init +3s in Nav. PC space"
nav_PCs = [0, 1, 3]
nav_data_to_project, nav_conditions, _ = grab_data_for_PC_space(
    df_to_slice, index_array, time_before_ms=0, time_after_ms=3000,
    align_point="start_time", single_trial=True,
)
_fig, _ = make_3d_plot(
    nav_data_to_project, nav_conditions, n_pca, nav_scaler, title, nav_PCs,
)

# %% Single-trial pre-move trajectories
title = "SINGLE TRIAL Move init -1s to Move init +3s in Pre-move PC space"
pre_move_PCs = [0, 1, 3]
pre_move_trajectories, pre_move_conditions, _ = grab_data_for_PC_space(
    df_to_slice, index_array, time_before_ms=1000, time_after_ms=1000,
    align_point="start_time", single_trial=True,
)
_fig, _ = make_3d_plot(
    pre_move_trajectories, pre_move_conditions, pm_pca, pre_move_scaler, title, pre_move_PCs,
)


# ===========================================================================
# SINGLE-TRIAL OUTCOME EFFECTS ANALYSIS
# ===========================================================================


# %% Helper: visualize single trials within specific bouts
def single_trial_outcome_space(
    bout_start,
    num_bouts_to_include=1,
    reward_style="previous",
    trial_info_full=False,
    non_reward_begin_offset=None,
    plot_style="plotly",
    title=None,
    time_before_ms=200,
    zero_offset_dot=0,
    azim=0,
    elev=0,
    save_fig=False,
    add_halo=True,
):
    """Plot single-trial trajectories for specific bout(s) in pre-move PC space.

    Parameters
    ----------
    bout_start : int
        Index of the first bout to visualize (0-indexed).
    num_bouts_to_include : int
        How many consecutive bouts to include.
    reward_style : str
        'previous' or 'current' — determines which reward field to use for
        coloring trials as rewarded vs. non-rewarded.
    trial_info_full : bool
        If True, use the unfiltered trial_info_orig.
    non_reward_begin_offset : int or None
        If set, blank the first N samples of non-reward trajectories.
    plot_style : str
        'plotly' or 'mpl'.
    title : str or None
        Custom title; auto-generated if None.
    time_before_ms : int
        Time before alignment point in ms.
    zero_offset_dot : int
        Sample index for the colored start dot.
    azim, elev : float
        Camera angles for matplotlib.
    save_fig : bool
        Save the figure.
    add_halo : bool
        Draw black halo behind trajectories.

    Returns
    -------
    tuple
        (outcome_trajectories, outcome_conditions, projected_all,
         trial_indices, trial_ids)
    """
    if title is None:
        title = (
            f"Single trial state space: bout {bout_start} to "
            f"{bout_start + num_bouts_to_include - 1} in pre-move PC space"
        )

    trial_df = dataset.trial_info if not trial_info_full else trial_info_orig
    switch_trial_idxs = trial_df[trial_df.trial_id_before_switch == 0]

    bout_end = bout_start + num_bouts_to_include - 1
    begin = 0 if bout_start == 0 else switch_trial_idxs.index[bout_start - 1] + 1
    end = switch_trial_idxs.index[bout_end] + 1
    chosen_trials = list(range(begin, end))

    if reward_style == "current":
        reward_trials = (
            trial_df.loc[chosen_trials]
            .query("Reward == True")
            .trial_id_before_switch.values
        )
    elif reward_style == "previous":
        reward_trials = (
            trial_df.loc[chosen_trials]
            .query("RewardPrevious == True")
            .trial_id_before_switch.values
        )
    else:
        raise ValueError(f"Invalid reward_style: {reward_style}")

    pc_obj = o_pca
    scaler_obj = outcome_scaler
    align_p = "start_time"

    outcome_PCs = [0, 1, 2]

    outcome_trajectories, outcome_conditions, trial_indices = grab_data_for_PC_space(
        df_to_slice,
        index_array,
        time_before_ms=time_before_ms,
        time_after_ms=0,
        align_point=align_p,
        conditions=pre_switch_conditions_orig,
        trial_info_full=trial_info_full,
        single_trial=True,
    )

    trial_ids = np.where(np.isin(trial_indices, chosen_trials))

    _, projected_all = make_3d_plot(
        outcome_trajectories[trial_ids],
        outcome_conditions[trial_ids],
        pc_obj,
        scaler_obj,
        title,
        outcome_PCs,
        color_style="reward",
        reward_trials=reward_trials,
        dot_size_mult=1.5,
        add_halo=add_halo,
        zero_offset_dot=zero_offset_dot,
        non_reward_begin_offset=non_reward_begin_offset,
        plot_style=plot_style,
        azim=azim,
        elev=elev,
        save_fig=save_fig,
    )

    return outcome_trajectories, outcome_conditions, projected_all, trial_indices, trial_ids


# %% All single trials in pre-move PC space (outcome effects analysis)
title = "SINGLE TRIAL Move init -0.2s to Move init in pre-move PC space"
outcome_PCs = [0, 1, 2]
outcome_trajectories_st, outcome_conditions_st, trial_indices_st = grab_data_for_PC_space(
    df_to_slice,
    index_array,
    time_before_ms=200,
    time_after_ms=0,
    align_point="start_time",
    single_trial=True,
)
_, single_traj = make_3d_plot(
    outcome_trajectories_st, outcome_conditions_st, o_pca, outcome_scaler,
    title, outcome_PCs,
)

# %% Trial-averaged trajectories in pre-move PC space
outcome_PCs = [0, 1, 2]
title = f"-1s to +4s in Pre-move PC space (PCs {outcome_PCs}): all reward conds"
_, average_traj = make_3d_plot(
    outcome_trajectories_by_condition.values(),
    outcome_trajectories_by_condition.keys(),
    o_pca,
    outcome_scaler,
    title,
    outcome_PCs,
    color_map=cm,
    zero_offset_dot=100,
    add_halo=False,
)

# %% Reorder trials by behavioral order and extract initial states
expected_shape = single_traj[0].shape
assert all(
    arr.shape == expected_shape for arr in single_traj
), "Not all projected trajectories have the same shape."

order = np.argsort(trial_indices_st)
trial_indices_sorted = trial_indices_st[order]
single_traj_sorted = np.array(single_traj)[order]
single_conds_sorted = outcome_conditions_st[order]

# Initial states are the last timepoint of each trajectory
init_traj = [traj[:, -1] for traj in single_traj_sorted]
initial_states = pd.DataFrame(
    {"initial_states": init_traj}, index=dataset.trial_info.index
)
dataset.trial_info = pd.concat((dataset.trial_info, initial_states), axis=1)

# %% Compute trial-to-trial changes in initial states
init_state_diff = [np.nan]
for i in range(1, len(initial_states)):
    diff = initial_states.iloc[i, 0] - initial_states.iloc[i - 1, 0]
    init_state_diff.append(diff)

dataset.trial_info["init_state_diff"] = init_state_diff

# %% Separate changes for post-reward vs. post-omission trials
sessions = [2, 4, 6, 8]
trial_thres = 50
best_patch_only = False

base_mask = (
    np.array(dataset.trial_info.epoch.isin(sessions))
    & np.array(dataset.trial_info.trial_number_by_epoch > trial_thres)
    & np.array(dataset.trial_info.IsSwitchPrevious == 0)
)

if best_patch_only:
    patch_mask = np.array(
        dataset.trial_info.EndStem == dataset.trial_info.BestStem
    )
    base_mask = base_mask & patch_mask

rewarded_diff = dataset.trial_info[
    np.array(dataset.trial_info.RewardPrevious == 1) & base_mask
].init_state_diff

omission_diff = dataset.trial_info[
    np.array(dataset.trial_info.RewardPrevious == 0) & base_mask
].init_state_diff

# %% Compute cosine angles between outcome effect vectors
total_diff_r = rewarded_diff.sum(axis=0, skipna=True)
total_diff_o = omission_diff.sum(axis=0, skipna=True)
mean_diff_r = rewarded_diff.mean(axis=0, skipna=True)
mean_diff_o = omission_diff.mean(axis=0, skipna=True)

logger.info(f"Total diff (reward):  {total_diff_r}")
logger.info(f"Total diff (omission): {total_diff_o}")
logger.info(f"Mean diff (reward):  {mean_diff_r}")
logger.info(f"Mean diff (omission): {mean_diff_o}")


def cosine_angle(v1, v2):
    """Compute cosine of angle between two vectors."""
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    return np.dot(v1, v2) / denom if denom > 0 else np.nan


outcome_effect_all = cosine_angle(total_diff_r, total_diff_o)
outcome_effect_23 = cosine_angle(total_diff_r[[1, 2]], total_diff_o[[1, 2]])
outcome_effect_24 = cosine_angle(total_diff_r[[1, 3]], total_diff_o[[1, 3]])
outcome_effect_34 = cosine_angle(total_diff_r[[2, 3]], total_diff_o[[2, 3]])

logger.info(f"Cosine angle PCs 1-5: {outcome_effect_all:.4f}")
logger.info(f"Cosine angle PCs 2+3: {outcome_effect_23:.4f}")
logger.info(f"Cosine angle PCs 2+4: {outcome_effect_24:.4f}")
logger.info(f"Cosine angle PCs 3+4: {outcome_effect_34:.4f}")


# %% Bootstrap 95% CI for cosine angle
save_results = False
PC_indices = [1, 3]

rewarded_arr = np.vstack(rewarded_diff.dropna().to_numpy())
omission_arr = np.vstack(omission_diff.dropna().to_numpy())
n_boot = 10_000
rng = np.random.default_rng(seed=0)

n_r, n_pc = rewarded_arr.shape
n_o, _ = omission_arr.shape

boot_cos_pcs = np.zeros(n_boot)
for b in range(n_boot):
    idx_r = rng.integers(0, n_r, size=n_r)
    idx_o = rng.integers(0, n_o, size=n_o)
    total_r = rewarded_arr[idx_r].sum(axis=0)
    total_o = omission_arr[idx_o].sum(axis=0)
    boot_cos_pcs[b] = cosine_angle(total_r[PC_indices], total_o[PC_indices])

ci_cos_pcs = np.nanpercentile(boot_cos_pcs, [2.5, 97.5])
logger.info(
    f"Bootstrap cosine angle (PCs {PC_indices[0]+1},{PC_indices[1]+1}): "
    f"95% CI = [{ci_cos_pcs[0]:.4f}, {ci_cos_pcs[1]:.4f}]"
)

if save_results:
    file_name = dataset.trial_info.nwb_file_name.unique()[0]
    results_file = os.path.join(
        results_dir,
        f"{file_name[:-5]}_cosine_angle_reward_vs_omission_"
        f"PCs{PC_indices[0]+1}{PC_indices[1]+1}.npz",
    )
    np.savez(results_file, boot_cos_pcs=boot_cos_pcs, ci_cos_pcs=ci_cos_pcs)
    logger.info(f"Saved bootstrap results to {results_file}")

# %% Plot cosine angle mean + CI
cos_mean = np.nanmean(boot_cos_pcs)
ci_low, ci_high = ci_cos_pcs
yerr = np.array([[cos_mean - ci_low], [ci_high - cos_mean]])

fig_ci, ax_ci = plt.subplots(figsize=(3, 4))
ax_ci.errorbar(0, cos_mean, yerr=yerr, fmt="o", capsize=4)
ax_ci.axhline(0, linestyle="--", linewidth=1)
ax_ci.set_xlim(-0.5, 0.5)
ax_ci.set_ylabel(f"Cosine angle (PC{PC_indices[0]+1}–PC{PC_indices[1]+1})")
ax_ci.set_xticks([])
ax_ci.set_title("Outcome effect")
plt.tight_layout()
plt.show()


# %% Polar angle histogram of neural-state shifts
def polar_angle_histogram_two_groups(
    V1: np.ndarray,
    V2: np.ndarray,
    *,
    n_bins: int = 30,
    density: bool = True,
    color1: str = "tab:blue",
    color2: str = "tab:orange",
    alpha1: float = 0.6,
    alpha2: float = 0.6,
    label1: str = "Group 1",
    label2: str = "Group 2",
    r_max: float = 0.5,
    title: str = "",
) -> None:
    """Overlay two polar (rose) histograms of 2D vector angles.

    Parameters
    ----------
    V1, V2 : np.ndarray, shape (n, 2)
        2D vectors for each group.
    n_bins : int
        Number of angular bins.
    density : bool
        If True, normalize to probability density.
    color1, color2 : str
        Colors for the two groups.
    alpha1, alpha2 : float
        Transparency for the bars.
    label1, label2 : str
        Legend labels.
    r_max : float
        Maximum radial extent.
    title : str
        Plot title.
    """
    V1 = np.asarray(V1, dtype=float)
    V2 = np.asarray(V2, dtype=float)
    if V1.ndim != 2 or V1.shape[1] != 2:
        raise ValueError("V1 must have shape (n, 2)")
    if V2.ndim != 2 or V2.shape[1] != 2:
        raise ValueError("V2 must have shape (n, 2)")

    theta1 = np.arctan2(V1[:, 1], V1[:, 0])
    theta2 = np.arctan2(V2[:, 1], V2[:, 0])

    bins = np.linspace(-np.pi, np.pi, n_bins + 1)
    c1, edges = np.histogram(theta1, bins=bins, density=density)
    c2, _ = np.histogram(theta2, bins=bins, density=density)

    centers = 0.5 * (edges[:-1] + edges[1:])
    width = edges[1] - edges[0]

    fig_polar = plt.figure(figsize=(5, 5))
    ax_polar = fig_polar.add_subplot(111, projection="polar")
    ax_polar.bar(
        centers, c1, width=width, color=color1, alpha=alpha1,
        align="center", label=label1,
    )
    ax_polar.bar(
        centers, c2, width=width, color=color2, alpha=alpha2,
        align="center", label=label2,
    )
    ax_polar.set_rlim(0, r_max)
    ax_polar.set_yticks(np.arange(0.1, r_max + 1e-6, 0.1))
    ax_polar.set_title(title)
    ax_polar.legend(frameon=False, loc="upper right")
    plt.tight_layout()
    plt.show()


# %% Normalize state-shift vectors and plot polar distribution
X_rew = np.vstack(rewarded_diff.to_numpy())[:, PC_indices]
norms_r = np.linalg.norm(X_rew, axis=1, keepdims=True)
X_rew_norm = np.divide(X_rew, norms_r, where=norms_r != 0)

X_omi = np.vstack(omission_diff.to_numpy())[:, PC_indices]
norms_o = np.linalg.norm(X_omi, axis=1, keepdims=True)
X_omi_norm = np.divide(X_omi, norms_o, where=norms_o != 0)

polar_angle_histogram_two_groups(
    X_rew_norm,
    X_omi_norm,
    n_bins=36,
    color1="firebrick",
    color2="royalblue",
    label1="Reward",
    label2="Omission",
    r_max=0.5,
    title=f"{trial_info_orig.nwb_file_name.unique()[0][:-5]}",
)

outcome_effect_norm = cosine_angle(
    X_rew_norm.sum(axis=0), X_omi_norm.sum(axis=0)
)
logger.info(f"Cosine angle (normalized shift vectors): {outcome_effect_norm:.4f}")

# %%
