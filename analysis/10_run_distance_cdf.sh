#!/bin/bash
#SBATCH --job-name=gene_distance_cdf
#SBATCH --output=cdf_logs/experiment_%A_%a.out
#SBATCH --error=cdf_logs/experiment_%A_%a.err
#SBATCH --array=0-4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=100G 
#SBATCH --time=24:00:00
#SBATCH --partition=cpu
#SBATCH --mail-type=ALL
#SBATCH --mail-user=dhenze@stanford.edu

# Load necessary modules
module load anaconda
conda activate Vizgen_2


morphologies=("2" "3" "1" "0" "4")

# Get the morphology for this array task
morph=${morphologies[$SLURM_ARRAY_TASK_ID]}

# Run the script
python morphology_cdf.py --morphology $morph
