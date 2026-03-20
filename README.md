# Foraging Neural Dynamics

Code for analyzing neural dynamics during a multi-patch foraging task. This repository implements a pipeline from raw neural recordings (NWB format) through latent dynamics estimation (via LFADS) to 3D neural state space visualization.

## System Requirements

### Software Dependencies

- **Python** ≥ 3.7.16
- **lfads-torch** — Latent Factor Analysis via Dynamical Systems ([repo](https://github.com/arsedler9/lfads-torch))

**Python packages** (see `requirements.txt`):

| Package | Version | Purpose |
|---|---|---|
| numpy | ≥ 1.21 | Array operations |
| pandas | ≥ 1.3 | Data management |
| scipy | ≥ 1.7 | Signal processing |
| h5py | ≥ 3.0 | HDF5 file I/O |
| pynwb | ≥ 2.0 | NWB file reading |
| dill | ≥ 0.3 | Object serialization |
| scikit-learn | ≥ 1.0 | PCA |
| plotly | ≥ 5.0 | Interactive 3D plots |
| matplotlib | ≥ 3.5 | Static publication figures |
| ipywidgets | ≥ 7.0 | Interactive widgets |
| tqdm | ≥ 4.0 | Progress bars |
| PyYAML | ≥ 6.0 | Config file parsing |

### Operating System

Tested on **Ubuntu 20.04** and **Ubuntu 22.04**. Expected to work on any Linux distribution, macOS, or Windows with Python 3.7+.

### Hardware Requirements

- **Standard desktop computer** is sufficient for data processing and analysis (Steps 1, 2, 3)
- **GPU** (NVIDIA, ≥8 GB VRAM recommended) required only for LFADS model training. Training was performed on NVIDIA A40 GPUs.

## Installation Guide

### 1. Clone this repository

```bash
git clone https://github.com/claytonwashington/foraging-neural-dynamics.git
cd foraging-neural-dynamics
```

### 2. Create and activate a conda environment

```bash
conda create -n foraging python=3.7.16 -y
conda activate foraging
```

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 4. Install lfads-torch (for LFADS model training only)

Follow the installation instructions at [github.com/arsedler9/lfads-torch](https://github.com/arsedler9/lfads-torch).

**Estimated install time**: ~5 minutes (excluding lfads-torch GPU dependencies).

## Demo

### Quick start: Reproduce state space analysis from saved data

If you already have a merged dataset (output from Step 2), you can directly run the state space analysis:

1. Update `configs/config.py` with your data paths:

```python
BASE_DIR = "/path/to/your/data"  # containing merged_datasets/ subdirectory
```

2. Run the analysis interactively in VS Code or Jupyter:

```bash
# VS Code: open scripts/03_state_space_analysis.py and run cells with # %%
# Jupyter: jupyter notebook, then open the script
```

**Note:** To open the .py scripts as a Jupyter notebook, you can use JupyterLab (right-click --> "Open With"), or convert to `.ipynb` files using `jupytext`:

```bash
jupytext --to notebook scripts/03_state_space_analysis.py
```

### Full pipeline

To run the complete pipeline from NWB data:

```bash
# Step 1: NWB → LFADS input
python scripts/01_nwb_to_lfads_input.py

# Step 2 (after LFADS training): Merge outputs into analysis dataset
python scripts/02_merge_lfads_output.py

# Step 3: State space analysis and figure generation
python scripts/03_state_space_analysis.py
```

## Instructions for Use

### Running on your own data

1. **Prepare NWB file**: Format your neural data as an NWB file with spike times and trial metadata. See [pynwb documentation](https://pynwb.readthedocs.io/) for formatting guidelines. Example NWB files are available at 
## TODO: Add link to example NWB files in Dropbox

2. **Update configuration**: Edit `configs/config.py`:
   - Set `BASE_DIR` to your data directory
   - Set `experiment_name` to match your NWB filename
   - Adjust `bin_size`, `xcorr_threshold` as needed

3. **Run the pipeline**: Execute scripts 01 → 02 → 03 in order. Between 01 and 02, you must train an LFADS model using `lfads-torch`.

### Directory structure

Your `BASE_DIR` should contain:
```
BASE_DIR/
├── NWB/                    # Input NWB files
│   └── <experiment>.nwb
├── datasets/               # Intermediate chopped datasets (created by Step 1)
├── merged_datasets/        # Merged analysis datasets (created by Step 2)
└── figures/                # Output figures (created by Step 3)
```

### LFADS model training

Between Steps 1 and 2, train an LFADS model:

1. Copy configs from `lfads_torch_configs/` to your `lfads-torch` installation
2. Adjust `encod_data_dim` in the model config to match your unit count
3. Create and run the PBT training script (see `lfads_torch_configs/scripts/run_pbt_wilbur.py`)

See the `lfads_torch_configs/` directory for example configuration files.

## Reproduction Instructions

<!-- To reproduce the K99 figures from the paper:

1. Obtain the NWB data file for the Wilbur session (day 4: `wilbur20210408_wake.nwb`)
2. Run Steps 1–3 with the default configuration
3. In `scripts/03_state_space_analysis.py`, look for `# %% Figure 3A` and subsequent `# %% K99:` cells

Key figures:
- **Figure 3A Top**: Navigation PC space (move init to move init + 3s)
- **Figure 3A Bottom**: Pre-move PC space (-1s to +1.5s around move initiation)
- **Figure 3A Middle**: Action-coded trajectories in navigation space
- **Figure 3A Right**: Patch-coded trajectories in navigation space
- **K99 Outcome plots**: Outcome and pre-move trajectories split by reward condition -->

## TODO: Add specific instructions 

## Additional Information

### Pipeline description

1. **NWB → LFADS input** (`01_nwb_to_lfads_input.py`): Loads spike-sorted neural data from NWB format, performs cross-correlation-based channel rejection, and chops the continuous recording into overlapping time windows suitable for LFADS training.

2. **Merge LFADS outputs** (`02_merge_lfads_output.py`): Reassembles LFADS-inferred firing rates, latent factors, and generator inputs from chopped segments back into continuous time series aligned with the original trial structure.

3. **State space analysis** (`03_state_space_analysis.py`): Performs PCA on condition-averaged LFADS rates in different task-aligned windows, then projects single-trial and condition-averaged trajectories into 3D subspaces for visualization.

### Core module

The `core/` directory contains data processing classes:
- `NWBDataset`: Load and preprocess NWB files
- `LFADSInterface`: Chop data for LFADS and merge outputs
- `BaseDataset`: Base class with resampling, smoothing, cross-correlation

### License

This project is released under the **MIT License**. See [LICENSE](LICENSE).

### External dependencies

- **lfads-torch**: [github.com/arsedler9/lfads-torch](https://github.com/arsedler9/lfads-torch) (MIT License)
