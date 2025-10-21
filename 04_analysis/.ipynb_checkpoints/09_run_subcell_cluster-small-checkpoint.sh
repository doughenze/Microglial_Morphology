#!/bin/bash
#SBATCH --job-name=cluster_subcell
#SBATCH --output=cluster_logs/experiment_%A_%a.out
#SBATCH --error=cluster_logs/experiment_%A_%a.err
#SBATCH --array=0-5
#SBATCH --mem=400G 
#SBATCH --time=24:00:00
#SBATCH --partition=owners,bigmem
#SBATCH --mail-type=ALL
#SBATCH --mail-user=dhenze@stanford.edu

# Load modules from Vizgen.sif container
source /oak/stanford/groups/quake/doug/resources/miniconda3/etc/profile.d/conda.sh
conda activate Vizgen_2

cd /oak/stanford/groups/quake/doug/bruno_transfer/Papers/Shapes/full_run/conflicts_correction/Microglial_Morphology/04_analysis

# Define the experiment paths
EXPERIMENTS=(
    "/oak/stanford/groups/quake/shared/Vizgen/dough/output/202405250811_3-mo-male-mouse-1-cerebellum-IHC_VMSC12602/region_1/"
    "/oak/stanford/groups/quake/shared/Vizgen/dough/output/202405311300_3month-female-2-IHC_VMSC12602/region_0/"
    "/oak/stanford/groups/quake/shared/Vizgen/dough/output/202406171409_3m-female-3-IHC_VMSC12602/region_0/"
    "/oak/stanford/groups/quake/shared/Vizgen/dough/output/202406141135_24m-male-2-IHC_VMSC12602/region_0/"
    "/oak/stanford/groups/quake/shared/Vizgen/dough/output/202407011057_24-month-male-4-rev2_VMSC11602/region_0/"
    "/oak/stanford/groups/quake/shared/Vizgen/dough/output/202406141019_24m-female-5-IHC_VMSC11602/region_0/"
)

# Define morph classes to run it on: only use the complex morphologies
MORPH_CLASSES=("3" "4")
# Calculate experiment and morph class indices
N_MORPH_CLASSES=${#MORPH_CLASSES[@]}
EXPERIMENT_INDEX=$((SLURM_ARRAY_TASK_ID / N_MORPH_CLASSES))
MORPH_CLASS_INDEX=$((SLURM_ARRAY_TASK_ID % N_MORPH_CLASSES))

# Get the experiment and morph class
EXPERIMENT=${EXPERIMENTS[$EXPERIMENT_INDEX]}
MORPH_CLASS=${MORPH_CLASSES[$MORPH_CLASS_INDEX]}

# Run the Python script
python Clustering_analysis_3d.py "$EXPERIMENT" "$MORPH_CLASS"
