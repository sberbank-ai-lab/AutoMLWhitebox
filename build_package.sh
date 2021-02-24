#!/bin/bash

python3 -m venv autowoe_venv
source ./autowoe_venv/bin/activate

pip install -U pip
pip install -U poetry

poetry lock
poetry install
poetry build