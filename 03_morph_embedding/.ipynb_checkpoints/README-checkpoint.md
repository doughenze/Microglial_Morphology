# Morphology Embedding and Clustering

This directory contains code and scripts for filtering, embedding, and clustering **microglia** based on morphological features derived from IBA1 immunostaining and image texture.

---

## Overview

The goal of this module is to:

1. Ensure accurate mapping between segmented IBA1-positive cells and spatial transcriptomic microglia.
2. Extract both high-dimensional deep embeddings and handcrafted morphological features.
3. Cluster microglia based on these features to identify distinct morphological states.

---

## Step-by-Step Workflow

### 1. Save Baysor Cell Boundaries

**Notebook:** `01_Save_Baysor_Boundaries.ipynb`

- Converts Baysor's segmentation outputs into standardized `.parquet` format.
- This format is commonly used in **Vizgen** outputs and ensures compatibility with downstream tools.

---

### 2. Microglia Mapping to IBA1 Signal

**Notebook:** `02_Mapping_Microglia_to_Baysor.ipynb`

- Matches each **Baysor-derived cell boundary** to a single IBA1-positive cell segmented in the **preprocessing** stage.
- Uses spatial overlap logic to confirm that boundaries map uniquely to single IBA1 cells.
- Helps eliminate ambiguous overlaps and reduce contamination from **border-associated macrophages (BAMs)**.

---

### 3. Morphological Embedding via Pretrained VGG19

**Notebook:** `03_Morphology_Embedding.ipynb`

- Uses the `texture` Conda environment.
- Applies a **pretrained VGG19 model** (via TensorFlow) to extract morphology embeddings.
- Since no model training is performed, **CPU execution is sufficient**.
- Outputs a **morphology vector** for each microglial cell for downstream clustering and analysis.

---

### 4. Morphological Feature Extraction

**Script:** `process_image.py`  
**Launcher:** `04_submit_jobs.sh` (SLURM batch array)

- Computes classical image-derived features (e.g., area, circularity, eccentricity).
- Job submission script parallelizes this step via SLURM.
- Outputs one `.csv` or `.parquet` file per image/batch.

After all jobs complete, run:

**Script:** `concatenate_feature_vectors.py`

- Merges all batch-specific feature tables into a single consolidated dataframe.
- Handles **SLURM array job dependencies** to ensure orderly merging.

---

### 5. Morphological Clustering

**Notebook:** `05_Leiden_Clustering_and_ordering.ipynb`

- Applies **UMAP** for dimensionality reduction and **Leiden clustering** on the VGG19 morphology embeddings.
- Identifies shape-based microglia subtypes.
- Outputs `Shape_500`, the clustered microglial population used in the next stage of the analysis pipeline.

---

## Environments and Resources

- TensorFlow-based embedding requires the `texture` Conda environment.
- No GPU is required as only inference (not training) is performed.
- SLURM job resources can be adjusted based on image size and batch count.

---

## Summary of Key Files

| Step                            | File(s) Used                                      | Output                      |
|----------------------------------|--------------------------------------------------|-----------------------------|
| Save Baysor Boundaries          | `01_Save_Baysor_Boundaries.ipynb`                | `.parquet` boundary files   |
| Map Microglia to IBA1 Segments  | `02_Mapping_Microglia_to_Baysor.ipynb`           | Cleaned microglia dataset   |
| Morphology Embedding            | `03_Morphology_Embedding.ipynb`                  | VGG19 feature vectors       |
| Classical Feature Extraction    | `process_image.py`, `04_submit_jobs.sh`          | Per-batch morphology tables |
| Concatenate Features            | `concatenate_feature_vectors.py`                 | Unified morphology dataframe|
| Clustering & Embedding          | `05_Leiden_Clustering_and_ordering.ipynb`        | `Shape_500` object          |

---

Ensure all jobs are run using the correct environments and dependencies as defined in the [`Environments/`](../Environments) directory.

