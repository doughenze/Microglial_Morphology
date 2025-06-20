#!/bin/bash
#SBATCH --job-name=subcellular_coloc
#SBATCH --output=coloc_logs/experiment_%A_%a.out
#SBATCH --error=coloc_logs/experiment_%A_%a.err
#SBATCH --array=0-23
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=250G 
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --mail-type=ALL
#SBATCH --mail-user=dhenze@stanford.edu

# Load modules from Vizgen.sif container
module load anaconda
conda activate Vizgen_2

cd /hpc/projects/group.quake/doug/Papers/Shapes/full_run/conflicts_correction/Microglial_Morphology/04_analysis

# Define the experiment paths
EXPERIMENTS=(
    "/hpc/projects/group.quake/doug/Shapes_Spatial/3-mo-male-1/"  # small mem jobs < 300GB
    "/hpc/projects/group.quake/doug/Shapes_Spatial/3-mo-male-2/"
    "/hpc/projects/group.quake/doug/Shapes_Spatial/3-mo-male-3-rev2/"
    "/hpc/projects/group.quake/doug/Shapes_Spatial/3-mo-female-1-rev2/"
    "/hpc/projects/group.quake/doug/Shapes_Spatial/3-mo-female-2/"
    "/hpc/projects/group.quake/doug/Shapes_Spatial/3-mo-female-3/"
    "/hpc/projects/group.quake/doug/Shapes_Spatial/24-mo-male-1/"
    "/hpc/projects/group.quake/doug/Shapes_Spatial/24-mo-male-2/"
    "/hpc/projects/group.quake/doug/Shapes_Spatial/24-mo-male-4-rev2/"
    "/hpc/projects/group.quake/doug/Shapes_Spatial/24-mo-female-1/"
    "/hpc/projects/group.quake/doug/Shapes_Spatial/24-mo-female-3/"
    "/hpc/projects/group.quake/doug/Shapes_Spatial/24-mo-female-5/"
)

# Define morph classes to analyze: Only use the most complex morphologies
MORPH_CLASSES=("4" "3")

# Calculate experiment and morph class indices
N_MORPH_CLASSES=${#MORPH_CLASSES[@]}
EXPERIMENT_INDEX=$((SLURM_ARRAY_TASK_ID / N_MORPH_CLASSES))
MORPH_CLASS_INDEX=$((SLURM_ARRAY_TASK_ID % N_MORPH_CLASSES))

# Get the experiment and morph class
EXPERIMENT=${EXPERIMENTS[$EXPERIMENT_INDEX]}
MORPH_CLASS=${MORPH_CLASSES[$MORPH_CLASS_INDEX]}

# Run the Python script for this experiment
python process_coloc_3d.py "$EXPERIMENT" "$MORPH_CLASS"
