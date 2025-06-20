# Annotation Pipeline

This directory contains all code related to **cell type and brain region annotation**, leveraging existing frameworks for label transfer and atlas integration.

The annotation workflow proceeds in multiple stages:

---

## 1. scVI/scANVI Label Transfer

We first apply **scVI and scANVI** for label transfer based on methods described in:

- [Life Science Alliance, 2023](https://www.life-science-alliance.org/content/6/1/e202201701)

The notebook `01_scVI_and_scANVI.ipynb` performs label transfer from the adolescent mouse brain dataset available here: [http://mousebrain.org/adolescent/downloads.html](http://mousebrain.org/adolescent/downloads.html). This step uses the dedicated `scVI` Conda environment.

---

## 2. Coordinate Cleanup for Visualization

To ensure uniform orientation across all brains, we perform basic spatial transformations (rotation, mirroring) in:

- `02_scVI_cleanup.ipynb`

This step standardizes coordinate systems and extracts a subset of microglia for downstream use.

---

## 3. Manual Brain Region Annotation

Interactive segmentation of brain regions is performed in:

- `03_Region_Annotation.ipynb`

This notebook uses **matplotlib interactive widgets** for manual labeling. Each session allows segmentation of **one region at a time**, and generates a separate `.h5ad` file. These region-specific files can then be concatenated for full-region integration in the next stage.

---

## 4. Cell Type Cleanup in Specific Regions

Region-specific label refinement is carried out in:

- `04_CB_celltype_cleanup.ipynb`

This notebook focuses on cleaning cell types in the **cerebellum**, based on **marker gene expression**.

---

## 5. Allen Brain Cell Atlas Integration

From this point, the pipeline transitions to methods based on the **Allen Brain Cell (ABC) Atlas**, as described in:

- [Nature, 2024](https://www.nature.com/articles/s41586-023-06808-9)

The following notebooks were adapted from the authors’ original repository:  
[Zhuang Lab GitHub Repository](https://github.com/ZhuangLab/whole_mouse_brain_MERFISH_atlas_scripts_2023/blob/main/scripts/integrate_MERFISH_with_scRNA-seq/)

- `05_ABC_Atlas_Download.ipynb` – Downloads reference data from the ABC Atlas  
- `06_ABC_Integration_Round1.ipynb` – Initial integration with MERFISH data  
- `07_ABC_Integration_Round2.ipynb` – Final integration and refinement

---

## 6. Creation of Broad Cell Classes

As the original codebase did not assign **broad cell classes**, we added this step to group fine-grained subtypes into interpretable categories:

- `08_ABC_class_creation.ipynb`

This notebook produces the final integrated object: `ABC_cleaned.h5ad`, which is used both in the **morphology embedding** stage and throughout downstream analyses.

---

## Summary of Key Files

| Notebook                        | Purpose                                                      |
|----------------------------------|--------------------------------------------------------------|
| `01_scVI_and_scANVI.ipynb`       | scVI-based label transfer using external datasets            |
| `02_scVI_cleanup.ipynb`          | Normalize spatial orientation and extract microglia          |
| `03_Region_Annotation.ipynb`     | Manual region segmentation using interactive widgets         |
| `04_CB_celltype_cleanup.ipynb`   | Cell-type refinement using marker gene expression            |
| `05–07_ABC_Integration*.ipynb`   | Allen Brain Cell Atlas integration (adapted from Zhuang Lab) |
| `08_ABC_class_creation.ipynb`    | Assign broader cell classes and save final annotated object  |

---

## Environment Note

Ensure that each notebook is run using the correct environment as specified in the [`Environments/`](../Environments) directory. The `scVI` environment is required for label transfer steps.

