# Introduction

This project uses uv, can be installed with;

- `pip install uv`

Then env is initiated with;

- `uv venv`

The env is maintained with;

- `uv sync` - If new dependencies are not downloaded locally
- `uv lock` - If you add new dependencies

## Kaggle Link

https://www.kaggle.com/competitions/recodai-luc-scientific-image-forgery-detection

## Data

- Data is stored in data/ folder. Add it if not present.

## TIMELINE

By 13.02 - try to have baseline pipeline up and running.

## Distribution

Three parts:

1. Fixing metric bugs, implementing/changing model architectures - Emre

2. Sliding window prediction w/decoder - Max

3. Image operations on pixel map - Einar
