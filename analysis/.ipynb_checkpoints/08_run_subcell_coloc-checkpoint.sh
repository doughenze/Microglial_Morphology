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
#SBATCH --mail-type=ALL
#SBATCH --mail-user=dhenze@stanford.edu

# Load necessary modules
module load anaconda
conda activate Vizgen_2

cd /hpc/mydata/doug.henze/MERFISH/Shapes

# Define the experiment paths
EXPERIMENTS=(
    "/hpc/projects/group.quake/doug/Shapes_Spatial/3-mo-male-1/"
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

# Get the experiment path for this array task
EXPERIMENT=${EXPERIMENTS[$SLURM_ARRAY_TASK_ID]}

# Run the Python script for this experiment
python process_coloc.py $EXPERIMENT
