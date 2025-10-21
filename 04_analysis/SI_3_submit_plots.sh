#!/bin/bash
#SBATCH --job-name=submit_plots
#SBATCH --output=example_images_log/experiment_%A_%a.out
#SBATCH --error=example_images_log/experiment_%A_%a.err
#SBATCH --array=0-14
#SBATCH --mem=100G 
#SBATCH --time=1:00:00
#SBATCH --partition=owners,quake
#SBATCH --mail-type=ALL
#SBATCH --mail-user=dhenze@stanford.edu

source /oak/stanford/groups/quake/doug/resources/miniconda3/etc/profile.d/conda.sh
conda activate Vizgen_2

cd /oak/stanford/groups/quake/doug/bruno_transfer/Papers/Shapes/full_run/conflicts_correction/Microglial_Morphology/04_analysis

python SI_3_submit_images.py --task-index "${SLURM_ARRAY_TASK_ID}"

