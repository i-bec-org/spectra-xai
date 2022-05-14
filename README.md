# eXplainable Artificial Intelligence for spectral datasets

This repository contains python code to enable XAI analysis on spectral datasets (e.g. VIS–NIR–SWIR).

- [eXplainable Artificial Intelligence for spectral datasets](#explainable-artificial-intelligence-for-spectral-datasets)
  - [Features](#features)
  - [Installation](#installation)
  - [Running tests](#running-tests)

[![Build Status](https://dev.azure.com/i-bec/spectra-xai/_apis/build/status/i-bec-org.spectra-xai?branchName=main)](https://dev.azure.com/i-bec/spectra-xai/_build/latest?definitionId=1&branchName=main)

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

Note that the cubist dependency requires a C compiler; in a Debian-based OS one can install it via:
```
sudo apt install build-essential
```

To install the development packages use:
```
pip install -e .[dev]
```

## Update on jupyter.i-bec.org

To update the package in our jupyter you need to:

1. ssh into the VM as administrator
2. change user to jupyter-administrator ```sudo su jupyter-administrator``` 
3. activate the environment ```source /opt/tljh/user/bin/activate && source activate spectraxai```
4. ```cd ~/installation_packages/spectra-xai``` where the installation files are located
5. run ```git pull``` to update the installation files
6. run ```sudo pip install .``` to install the update

## Running tests

Tests may be run through command line by executing e.g.:

```
pytest --ignore=examples --doctest-modules --html=report.html --cov=. --cov-report=html
```

## Building documentation

Documentation may be generated via:

```
pdoc --docformat numpy --math -o docs spectraxai
```

## Examples

Some examples of the usage of this library may be found under the [examples](examples) folder.

