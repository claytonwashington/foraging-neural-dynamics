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

A pre-processed merged dataset for the sample session is available in the public [Dropbox folder](https://www.dropbox.com/scl/fi/ds2mfnc4r70njihqx4w34/NeuralDataSharing.zip?rlkey=gvb0dsvi9s2586a68ex3yu51x&st=rwxspizp&dl=0). Download the `.pkl` file from `merged_datasets/` and place it in your local `merged_datasets/` directory, then:

1. Update `configs/config.py` with the path to where you placed the data and the experiment name:

```python
BASE_DIR = "/path/to/your/data"       # directory containing merged_datasets/
experiment_name = "wilbur20210408_wake"  # must match the downloaded .pkl filename
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
# Step 0 (Optional): Build companion dataset for preprocessing validation
python scripts/00_build_xcorr_companion.py

# Step 1: NWB → LFADS input
python scripts/01_nwb_to_lfads_input.py

# Step 2 (after LFADS training): Merge outputs into analysis dataset
python scripts/02_merge_lfads_output.py

# Step 3: State space analysis and figure generation
python scripts/03_state_space_analysis.py

# Step 4 (Optional): Preprocessing validation and robustness checks
python scripts/04_preprocessing_validation.py
```

> **Note for Steps 0 and 4:** These scripts require the **raw NWB file** to be present at `{BASE_DIR}/NWB/{experiment_name}.nwb` (in addition to the merged `.pkl` from Step 2). They do not need to be run interactively — running `python scripts/04_preprocessing_validation.py` from the terminal will automatically save all validation figures as PDFs to `figures/validation_metrics/{experiment_name}_{run_date}_{run_idx}/`.

## Instructions for Use

### Running on your own data

1. **Prepare NWB file**: Format your neural data as an NWB file with spike times and trial metadata. See [pynwb documentation](https://pynwb.readthedocs.io/) for formatting guidelines.

### Sample data

We provide sample data in a public [Dropbox folder](https://www.dropbox.com/scl/fi/ds2mfnc4r70njihqx4w34/NeuralDataSharing.zip?rlkey=gvb0dsvi9s2586a68ex3yu51x&st=rwxspizp&dl=0). The folder includes:

- **Sample NWB dataset** — raw spike-sorted neural data for testing the full pipeline from Step 1 (NWB → LFADS input).
- **Sample LFADS merged output** — a pre-processed merged dataset (`.pkl`) that can be used to directly run the state space analysis in Step 3, without needing to train an LFADS model (which requires a GPU).

To use the sample merged output for a quick test:
1. Download the `.pkl` file from the Dropbox folder
2. Place it in `{BASE_DIR}/merged_datasets/`
3. Update `configs/config.py` to set `experiment_name` to match the downloaded file
4. Run `scripts/03_state_space_analysis.py` interactively

## Reproduction Instructions

To reproduce the figures from the paper:

1. Download the sample NWB and merged output data from the [Dropbox folder](https://www.dropbox.com/scl/fi/ds2mfnc4r70njihqx4w34/NeuralDataSharing.zip?rlkey=gvb0dsvi9s2586a68ex3yu51x&st=rwxspizp&dl=0)
2. Place the merged output `.pkl` file in `{BASE_DIR}/merged_datasets/`
3. Update `configs/config.py` with the correct `BASE_DIR` and `experiment_name`
4. Open `scripts/03_state_space_analysis.py` interactively and run the cells marked with `# %% Figure 3A` and the "Both conditions" cells (e.g., `# %% Both reward conditions in pre-move PC space`).

## Additional Information

### Pipeline description

Note: Steps 1 through 3 represent the standard, required data pipeline. Steps 0 and 4 are optional validation scripts specifically designed to verify the robustness of the cross-correlation unit rejection threshold and window-stitching logic.

0. (Optional) **Build companion dataset** (`00_build_xcorr_companion.py`): Reads the raw NWB file, computes pairwise cross-correlations on the original un-zeroed spikes, and saves a companion `.pkl` to `{BASE_DIR}/datasets/`. Required as a prerequisite for Step 4.

1. **NWB → LFADS input** (`01_nwb_to_lfads_input.py`): Loads spike-sorted neural data from NWB format, performs cross-correlation-based channel rejection, and chops the continuous recording into overlapping time windows suitable for LFADS training.

2. **Merge LFADS outputs** (`02_merge_lfads_output.py`): Reassembles LFADS-inferred firing rates, latent factors, and generator inputs from chopped segments back into continuous time series aligned with the original trial structure.

3. **State space analysis** (`03_state_space_analysis.py`): Performs PCA on condition-averaged LFADS rates in different task-aligned windows, then projects single-trial and condition-averaged trajectories into 3D subspaces for visualization.

4. (Optional) **Preprocessing validation** (`04_preprocessing_validation.py`): Requires the companion pkl from Step 0 and the merged dataset from Step 2. Produces validation figures (PDFs saved to `figures/validation_metrics/{experiment_name}_{run_date}_{run_idx}/`) including cross-correlation histograms, stitching boundary checks, and PCA subspace robustness across different unit rejection thresholds.

### Core module

The `core/` directory contains data processing classes:
- `NWBDataset`: Load and preprocess NWB files
- `LFADSInterface`: Chop data for LFADS and merge outputs
- `BaseDataset`: Base class with resampling, smoothing, cross-correlation

### License

This project is released under the **MIT License**. See [LICENSE](LICENSE).

### External dependencies

- **lfads-torch**: [github.com/arsedler9/lfads-torch](https://github.com/arsedler9/lfads-torch) (MIT License)
