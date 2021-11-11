# eXplainable Artificial Intelligence for spectral datasets

This repository contains python code to enable XAI analysis on spectral datasets (e.g. VIS–NIR–SWIR).

- [eXplainable Artificial Intelligence for spectral datasets](#explainable-artificial-intelligence-for-spectral-datasets)
  - [Features](#features)
  - [Installation](#installation)
  - [Running tests](#running-tests)

## Documentation

The documentation is accessible on [static.i-bec.org/spectraxai](https://static.i-bec.org/spectraxai/) which is currently accessible only for the 155.207.180.* and 155.207.185.* subdomains.

## Features

## Installation

On a new conda environment or virtualenv run:

```
git clone https://github.com/i-bec-org/spectra-xai
cd spectra-xai && pip install .
```

or run
```
python setup.py install
```

## Running tests

Tests may be run through command line by executing e.g.:

```
python -m unittest tests/testStandardModelClass.py
```

## Examples

Some examples of the usage of this library may be found under the [examples](examples) folder.

