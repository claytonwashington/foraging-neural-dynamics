"""Configuration for the foraging neural dynamics analysis pipeline.

Update the paths below to match your local setup before running scripts.
"""

# ============================================================
# USER: Update these paths to match your environment
# ============================================================

# TODO: Update to your local data directory (containing NWB/, datasets/, merged_datasets/)
BASE_DIR = "/path/to/your/data" 

# TODO: Update to the directory one level above your lfads-torch clone
# e.g., if lfads-torch is at /home/user/lfads-torch/, set to /home/user
LFADS_TORCH_PREFIX = "/path/to/your/lfads-torch"

# TODO: Update to your lfads-torch run output directory
RUN_DIR = "/path/to/your/runs"

# ============================================================
# Dataset / experiment configuration
# ============================================================

config = {
    "experiment_name": "wilbur_210408",   # Experiment identifier (must match your NWB/merged dataset filename)
    "run_date": "250907",                        # Update to the date tag of your LFADS run (format: YYMMDD)
    "run_idx": "0",                              # Run index (usually 0)
    "xcorr_threshold": 0.1,                      # Cross-correlation threshold for channel rejection
    "bin_size": 10,                               # Bin width in ms
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
