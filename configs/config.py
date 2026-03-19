"""Configuration for the foraging neural dynamics analysis pipeline.

Update the paths below to match your local setup before running scripts.
"""

# ============================================================
# USER: Update these paths to match your environment
# ============================================================

# Base directory containing NWB files, datasets, and merged outputs.
# Expected subdirectories: NWB/, datasets/, merged_datasets/
BASE_DIR = "/snel/share/share/data/xulu_mpfc"

# Directory one level above your lfads-torch clone.
# e.g., if lfads-torch is at /home/user/lfads-torch/, set to /home/user
LFADS_TORCH_PREFIX = "/home/cbwash2/lfads-torch-cuda12"

# Directory for lfads-torch run outputs
RUN_DIR = "/snel/share/runs"

# ============================================================
# Dataset / experiment configuration
# ============================================================

config = {
    "experiment_name": "wilbur20210407_wake",   # Experiment identifier
    "run_date": "250225",                        # Date tag for the LFADS run
    "run_idx": "0",                              # Run index (usually 0)
    "xcorr_threshold": 0.1,                      # Cross-correlation threshold for channel rejection
    "bin_size": 10,                               # Bin width in ms
    "model_num": 1,
    "base_dir": BASE_DIR,
    "lfads_torch_prefix": LFADS_TORCH_PREFIX,
    "run_dir": RUN_DIR,
}

# Mapping from lfads-torch output field names to dataset signal names
merge_config = {
    "output_params": "lfads_rates",
    "factors": "lfads_factors",
    "gen_inputs": "lfads_gen_inputs",
}

# State space analysis parameters
ss_analysis_config = {
    "min_pokes_per_bout": 6,        # Minimum pokes in a bout to include
    "pc_padding": 1e-10,            # Small constant for PCA scaler stability
    "switch_incl_thresh": -9,       # Include trials up to this many before switch
    "trial_condition": "all",       # 'all', 'reward', or 'no_reward'
    "plot_all_trials": True,        # Plot all trials including those not used for PCA fitting
}
