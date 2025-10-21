# Analysis and Figure Generation

This directory is the **core driver of the analysis**. All figures in the associated manuscript and supplementary materials can be reproduced using the code provided here.

---

## Overview

The analysis pipeline proceeds through a series of structured notebooks and SLURM-based scripts. These steps:

- Perform quality control and visualization.
- Generate count tables with spatial RNA locations.
- Conduct compartmentalization and colocalization analysis.
- Produce all **main and extended figures** for the manuscript.

---

## 1. Dataset Description and Cell Typing

**Notebook:** `01_Full_MERFISH_Figure1_SI_1.ipynb`

- Performs full cell-type annotation and dataset-wide quality control.
- Establishes the baseline for Figure 1 and related supplementary panels.

---

## 2. Count Table Generation (High-Memory)

**Script:** `02_run_count_table_generator.sh`  
**Driver:** `count_table_generation.py`

- Launch this step using SLURM:

```bash
sbatch 02_run_count_table_generator.sh
```

- By default, the script **preloads all images into memory** to accelerate performance. This requires **up to 600 GB of RAM** for the largest samples.
- To reduce memory usage, comment out the monkey patch and cache preloading in `count_table_generation.py`.

---

## 3. Figure 1 Generation

**Notebook:** `03_Figure1_and_SI.ipynb`

- Reproduces Figure 1 and extended data panels.
- Saves the working microglia object used across the rest of the analysis.

---

## 4. Figures 2 and 3

- `04_Figure_2.ipynb`  
- `05_SI_Fig_5.ipynb`  
- `06_Figure_3_and_SI-Mic_Specific.ipynb`

**Note:** Figure 3 requires loading the Allen Brain Cell Atlas to compute **tau statistics** across cell types and may require additional RAM.

---

## 5. Colocalization and Spatial Analysis (Figures 4 & 5)

### a. Radius Estimation

**Notebook:** `07_Colocalization_radius_calculation.ipynb`

- Computes pairwise RNA transcript distances to establish a **colocalization radius** for downstream spatial analyses.

---

### b. 2D Distance CDF from Soma

**Script:** `08_run_distance_cdf.sh`  
**Driver:** `morphology_cdf.py`

- Launch with SLURM:

```bash
sbatch 08_run_distance_cdf.sh
```

- Requires transcript location data generated in Step 2.

---

### c. Subcellular Clustering (High-Memory)

**Script:** `09_run_subcell_cluster.sh`  
**Driver:** `Clustering_analysis_3d.py`

- Adapted from [cbib/dypfish](https://github.com/cbib/dypfish).
- Applies a modified 3D clustering approach to assign transcripts to microglial processes.
- Requires **high memory** due to image preloading. You may disable caching for lower RAM usage by commenting out monkey patch and cacheing lines in .py script.

```bash
sbatch 10_run_subcell_coloc.sh
```

---

### d. Transcript-Transcript Colocalization

**Script:** `10_run_subcell_coloc.sh`  
**Driver:** `process_coloc_3d.py`

- Adapted from [Nature 2024 study](https://www.nature.com/articles/s41586-023-06808-9) for **3D transcript colocalization**.
- Runs via:

```bash
sbatch 10_run_subcell_coloc.sh
```

- Also memory-intensive; caching can be disabled with performance trade-offs. You may disable caching for lower RAM usage by commenting out monkey patch and cacheing lines in .py script.

---

### e. Figure 4 and 5 Generation

- `11_Figure_4_and_SI.ipynb`
- `12_Figure_5_and_SI.ipynb`

These notebooks assemble final figures based on clustering and colocalization analyses.

---

## 6. Supplementary Data Figure Notebooks and scripts

- `SI_1_IHC_max_projection.ipynb` – Max projection for immunohistochemistry
- `SI_2_IHC_Axl.ipynb` – IHC-specific analysis for Axl
- `SI_3_submit_plots.sh` – Visualization of random microglia from MERFISH

---

## Summary Table

| Step                          | File(s)                                      | Notes                              |
|-------------------------------|----------------------------------------------|-------------------------------------|
| Cell typing and QC            | `01_Full_MERFISH_Figure1_SI_1.ipynb`         | Figure 1 core dataset intro         |
| Count table generation        | `02_run_count_table_generator.sh`, `count_table_generation.py` | 600 GB RAM if caching enabled       |
| Figure 1 generation           | `03_Figure1_and_SI.ipynb`                    | Produces key object for reuse       |
| Figures 2 & 3                 | `04_Figure_2.ipynb`, `05_SI_Fig_5.ipynb`, `06_Figure_3_and_SI-Mic_Specific.ipynb` | Tau stat uses ABC Atlas             |
| Colocalization radius         | `07_Colocalization_radius_calculation.ipynb`| Preprocessing for spatial scripts   |
| 2D transcript distance        | `08_run_distance_cdf.sh`, `morphology_cdf.py`| CDF calculation                     |
| Subcell clustering            | `09_run_subcell_cluster.sh`, `Clustering_analysis_3d.py` | DypFISH-inspired 3D segmentation    |
| Subcell colocalization       | `10_run_subcell_coloc.sh`, `process_coloc_3d.py` | Adapted from Nature 2024            |
| Figures 4 & 5                | `11_Figure_4_and_SI.ipynb`, `12_Figure_5_and_SI.ipynb` | Requires all prior steps            |
| Supplementary figures        | `SI_*`                                 | For extended data visuals           |

---

## Environment and Resources

- Most scripts and notebooks require access to high-memory compute nodes.
- Where noted, monkey patches and caching can be disabled to reduce memory usage at the cost of speed.
- Refer to the [`Environments/`](../Environments) directory to install the correct environments before running analysis.

