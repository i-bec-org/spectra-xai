# SPDX-FileCopyrightText: 2026 Nikos Tsakiridis <tsakirin@auth.gr>
#
# SPDX-License-Identifier: GPL-3.0-or-later

from pathlib import Path

from setuptools import find_packages, setup

HERE = Path(__file__).parent
long_description = (HERE / "README.md").read_text(encoding="utf-8")

# This call to setup() does all the work
setup(
    name="spectraxai",
    version="1.1.0",
    packages=find_packages(
        include=["spectraxai", "spectraxai.utils", "spectraxai.data"]
    ),
    python_requires=">=3.13",
    install_requires=[
        "pandas",
        "numpy",
        "scipy",
        "scikit-learn>=1.5.2",
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
    package_data={"spectraxai": ["py.typed"]},
    include_package_data=True,
)
