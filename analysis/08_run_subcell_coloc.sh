#!/bin/bash
#SBATCH --job-name=subcellular_coloc
#SBATCH --output=logs/experiment_%A_%a.out
#SBATCH --error=logs/experiment_%A_%a.err
#SBATCH --array=0-11
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=100G 
#SBATCH --time=24:00:00
#SBATCH --partition=cpu

# Define the experiment paths
EXPERIMENTS=(
    "3-mo-male-1/"
    "3-mo-male-2/"
    "3-mo-male-3-rev2/"
    "3-mo-female-1-rev2/"
    "3-mo-female-2/"
    "3-mo-female-3/"
    "24-mo-male-1/"
    "24-mo-male-2/"
    "24-mo-male-4-rev2/"
    "24-mo-female-1/"
    "24-mo-female-3/"
    "24-mo-female-5/"
)

# Get the experiment path for this array task
EXPERIMENT=${EXPERIMENTS[$SLURM_ARRAY_TASK_ID]}

# Run the Python script for this experiment
python 07_process_coloc.py $EXPERIMENT
