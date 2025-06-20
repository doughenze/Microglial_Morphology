# Environment Setup Instructions

This directory contains Conda environment configuration files for all required tools used throughout the codebase. Due to the complexity of some packages and conflicting dependencies, each major component of the pipeline is provided with its own dedicated environment.

We begin by detailing how to install the environment for **Baysor**, followed by instructions for other tools.

---

## 1. Baysor Environment

We use Conda for environment management. To install the environment for running Baysor, ensure you are in the same directory as the `baysor.yml` file and run:

```bash
conda env create -f baysor.yml
```

---

## 2. Vizgen Environment

Similarly, to install the Vizgen environment, ensure that `Vizgen.yml` is in the current working directory and execute:

```bash
conda env create -f Vizgen.yml
```

---

## 3. scVI Environment

The scVI dependencies can be challenging to resolve and should **not** be installed into a general-purpose single-cell or spatial analysis environment. Instead, use a dedicated environment by running:

```bash
conda env create -f scvi.yml
```

---

## 4. Texture Code (VGG19 Embedding and Segmentation)

This environment has additional system-level dependencies. First, install GCC and related build tools:

```bash
apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    make \
    libz-dev
```

> **Note:** On high-performance computing (HPC) systems where you lack administrator privileges, use `module load` to access compiler toolchains instead.

After setting up system dependencies, create the Conda environment:

```bash
conda env create -f texture.yml
```

---

## 5. ABC Atlas Label Transfer

This environment includes dependencies that may conflict with typical single-cell analysis tools, so it is isolated in its own environment.

If you haven’t already completed the dependency installation described for the **Texture Code**, please do so now.

Then, install the environment:

```bash
conda env create -f ABC.yml
```

---

## 6. Using the Environments in Jupyter Notebooks

Once the environments are installed, you can run `.ipynb` notebooks using the appropriate environment listed at the top of each notebook. These environments can be selected directly within Jupyter if kernel switching is configured.

---

## 7. Using Pre-Built Containers (Advanced)

We have also provided pre-built containers via a separate [Zenodo repository](https://zenodo.org/records/14611122), as referenced in the top-level `README.md`. These containers are built directly from the above Conda environments.

> These are **not ideal for modifying or installing new packages**, but are excellent for reproducing the original analysis environment.

### Running Python Scripts with a Container

```bash
singularity exec my_python_container.sif python my_script.py arg1 arg2
```

### Launching a Jupyter Notebook in a Container

```bash
singularity run my_container.sif jupyter notebook --ip=0.0.0.0 --port=8888 --allow-root
```

This will open a Jupyter Notebook session inside the container where you can run any notebook files as originally executed.

---

## Summary

Each environment is purpose-built for a specific stage of the pipeline to ensure compatibility and reproducibility. Be sure to activate the correct environment before executing code, and consult the `README.md` in each directory for step-specific instructions.

