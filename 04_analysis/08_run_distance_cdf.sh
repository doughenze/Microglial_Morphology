#!/bin/bash
#SBATCH --job-name=gene_distance_cdf
#SBATCH --output=cdf_logs/experiment_%A_%a.out
#SBATCH --error=cdf_logs/experiment_%A_%a.err
#SBATCH --array=0-1
#SBATCH --mem=100G 
#SBATCH --time=24:00:00
#SBATCH --partition=owners,quake
#SBATCH --mail-type=ALL
#SBATCH --mail-user=dhenze@stanford.edu

# Load modules from Vizgen.sif container
source /oak/stanford/groups/quake/doug/resources/miniconda3/etc/profile.d/conda.sh
conda activate Vizgen_2

cd /oak/stanford/groups/quake/doug/bruno_transfer/Papers/Shapes/full_run/conflicts_correction/Microglial_Morphology/04_analysis

morphologies=("3" "4")

# Get the morphology for this array task
morph=${morphologies[$SLURM_ARRAY_TASK_ID]}

# Run the script
python morphology_cdf.py --morphology $morph
