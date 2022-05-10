r'''
# SpectraXAI

A library to easily develop XAI models for spectral datasets.

## Features

 - Easy setup with pip
 - Documented using pdoc following the numpydoc style 

# Quickstart

As an example, we quickly develop a PLS model for a dataset and quickly plot the feature importance.
```python
"""
A small `spectraxai` example.
"""

# Load the dataset

from spectraxai.utils.datasets import load_GR_SSL
dataset = load_GR_SSL()

# Apply SG1 on the spectra and train & test a PLS model
 
from spectraxai.models import Model, StandardModel
from spectraxai.spectra import SpectralPreprocessing
from spectraxai.dataset import DatasetSplit

idx_trn, idx_tst = dataset.train_test_split(DatasetSplit.KENNARD_STONE, 0.8)
pls = StandardModel(Model.PLS)

results = pls.train_and_test(
    dataset, 
    preprocess=(SpectralPreprocessing.SG1, {"window_length": 7, "polyorder": 3}), 
    idx_trn=idx_trn
)

# Plot the feature importance

from spectraxai.explain import PostHocAnalysis

xai = PostHocAnalysis(dataset)
xai.bar_plot_importance(results.iloc[0]["feature_importance"])
```
'''

from .dataset import Dataset, DatasetSplit, Scale
from .models import Model, StandardModel
from .spectra import Spectra, SpectralPreprocessing