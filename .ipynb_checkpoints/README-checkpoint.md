# README: Aging MERFISH Brains Analysis Pipeline

Welcome to the codebase accompanying the manuscript **"Aging MERFISH Brains"**. This repository provides all scripts, data references, and environment specifications required to reproduce the analysis.

---

## Overview

To replicate the analysis and results:

1. **Download the data** from [Figshare](https://figshare.com/articles/dataset/Aging_MERFISH_Brains/27919227).
2. **Download the environment containers** from [Zenodo](https://zenodo.org/records/14611122).
3. Follow the folder and script execution order described below.

---

## Project Structure

- The repository is organized into **sequentially numbered folders** (e.g., `01_preprocessing`, `02_analysis`, etc.), indicating the recommended execution order.
- Each folder contains:
  - A dedicated `README.md` explaining the logic and usage of that step.
  - **Numbered scripts** (e.g., `01_load_data.py`, `02_clean_data.py`) to be run in order.
  - `.ipynb` notebooks that include pre-run outputs (unless restricted by file size constraints).

---

## Setup and Environment

Environment configuration files (e.g., `environment.yml`) and installation instructions are provided in the [`Environments/`](./Environments) directory. These match the containers available on Zenodo and ensure reproducibility.

---

## Notes

- The project relies on external data and containers, so ensure you download them **before running any scripts**.
- If you encounter issues with file sizes or output rendering in notebooks, refer to the associated folder `README.md` for alternative steps.

---

## Questions

For further questions or clarifications, please contact the authors of the manuscript or raise an issue in this repository.

---

## Citation

If you use this codebase, please cite the manuscript and provide links to the Figshare and Zenodo records.
