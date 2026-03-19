"""Quick smoke test: load merged dataset and verify core imports work."""
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

print("1. Testing core imports...")
from core.utils import parmap, rgetattr
from core.base_dataset import BaseDataset
from core.nwb_dataset import NWBDataset
from core.lfads_interface import LFADSInterface, chop_data, merge_chops
print("   ✅ All core imports OK")

print("2. Loading config...")
from configs.config import config, ss_analysis_config
print(f"   ✅ Config loaded: experiment={config['experiment_name']}, base_dir={config['base_dir']}")

print("3. Loading merged dataset (this may take a minute)...")
from scripts.utils.analysis_utils import load_dataset
dataset = load_dataset()
print(f"   ✅ Dataset loaded: {dataset.data.shape[0]} rows × {dataset.data.shape[1]} columns")
print(f"   Trial info: {len(dataset.trial_info)} trials")
print(f"   Signal types: {list(dataset.data.columns.get_level_values(0).unique())}")

print("4. Testing PCA on LFADS rates...")
import numpy as np
from sklearn.decomposition import PCA
rates = dataset.data['lfads_rates_smooth_50'].to_numpy()
print(f"   Rates shape: {rates.shape}")
# Quick PCA on first 1000 timepoints
pca = PCA(n_components=5)
projected = pca.fit_transform(rates[:1000])
print(f"   ✅ PCA projected shape: {projected.shape}")
print(f"   Explained variance: {pca.explained_variance_ratio_[:3].round(3)}")

print("\n🎉 All smoke tests passed! The standalone repo works with your data.")
