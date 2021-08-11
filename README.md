## Sberbank version of AutoWoE

![GitHub all releases](https://img.shields.io/github/downloads/sberbank-ai-lab/AutoMLWhitebox/total?color=green&logo=github&style=plastic)
![PyPI - Downloads](https://img.shields.io/pypi/dm/autowoe?color=green&label=PyPI%20downloads&logo=pypi&logoColor=orange&style=plastic)


This is the repository for **AutoWoE** library, developed by Sber AI Lab AutoML group. This library can be used for automatic creation of interpretable ML model based on feature binning, WoE features transformation, feature selection and Logistic Regression.

**Authors:** Vakhrushev Anton, Grigorii Penkin

**Library setup** can be done by one of three scenarios below:

1. Installation from PyPI:
```bash
pip install autowoe
```
2. Installation from source code

First of all you need to install [git](https://git-scm.com/downloads) and [poetry](https://python-poetry.org/docs/#installation).

```bash

# Load LAMA source code
git clone https://github.com/sberbank-ai-lab/AutoMLWhitebox.git

cd AutoMLWhiteBox/

# !!!Choose only one item!!!

# 1. Recommended: Create virtual environment inside your project directory
poetry config virtualenvs.in-project true

# 2. Global installation: Don't create virtual environment
poetry config virtualenvs.create false --local

# For more information read poetry docs

# Install WhiteBox
poetry install

```


**Usage tutorials** are in Jupyter notebooks in the repository root. For **parameters description** take a look at `parameters_info.md`.

**Bugs / Questions / Suggestions:**:
- Vakhrushev Anton (AGVakhrushev@sberbank.ru)
