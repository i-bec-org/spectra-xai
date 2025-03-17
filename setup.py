# SPDX-FileCopyrightText: 2025 Nikos Tsakiridis <tsakirin@auth.gr>
#
# SPDX-License-Identifier: GPL-3.0-or-later

# Always prefer setuptools over distutils
from setuptools import setup, find_packages

# To use a consistent encoding
from codecs import open
from os import path

# The directory containing this file
HERE = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(HERE, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

# This call to setup() does all the work
setup(
    name="spectraxai",
    version="1.0.0",
    packages=find_packages(
        include=["spectraxai", "spectraxai.utils", "spectraxai.data"]
    ),
    python_requires=">=3.9",
    install_requires=[
        "pandas",
        "numpy<2.0.0",
        "scipy",
        "scikit-learn==1.5.2",
        "matplotlib",
        "cubist>=0.1.0",
        "seaborn",
        "sage-importance",
        "kennard-stone",
        "xgboost",
    ],
    extras_require={
        "dev": [
            "flake8",
            "pytest",
            "pytest-cov",
            "pytest-html",
            "black",
            "ipykernel",
            "pdoc",
            "docformatter",
            "pydocstyle",
            "pre-commit",
        ]
    },
    include_package_data=True,
)
