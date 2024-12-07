#!/bin/bash
#SBATCH --job-name=cluster_subcell
#SBATCH --output=cluster_logs/experiment_%A_%a.out
#SBATCH --error=cluster_logs/experiment_%A_%a.err
#SBATCH --array=0-47%12
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


# Define the experiment paths
EXPERIMENTS=(
    "Shapes_Spatial/3-mo-male-1/"
    "Shapes_Spatial/3-mo-male-2/"
    "Shapes_Spatial/3-mo-male-3-rev2/"
    "Shapes_Spatial/3-mo-female-1-rev2/"
    "Shapes_Spatial/3-mo-female-2/"
    "Shapes_Spatial/3-mo-female-3/"
    "Shapes_Spatial/24-mo-male-1/"
    "Shapes_Spatial/24-mo-male-2/"
    "Shapes_Spatial/24-mo-male-4-rev2/"
    "Shapes_Spatial/24-mo-female-1/"
    "Shapes_Spatial/24-mo-female-3/"
    "Shapes_Spatial/24-mo-female-5/"
)

# Define morph classes
MORPH_CLASSES=("0" "1" "2" "3" "4")

# Calculate experiment and morph class indices
N_MORPH_CLASSES=${#MORPH_CLASSES[@]}
EXPERIMENT_INDEX=$((SLURM_ARRAY_TASK_ID / N_MORPH_CLASSES))
MORPH_CLASS_INDEX=$((SLURM_ARRAY_TASK_ID % N_MORPH_CLASSES))

# Get the experiment and morph class
EXPERIMENT=${EXPERIMENTS[$EXPERIMENT_INDEX]}
MORPH_CLASS=${MORPH_CLASSES[$MORPH_CLASS_INDEX]}

# Run the Python script
python Clustering_analysis.py "$EXPERIMENT" "$MORPH_CLASS"
