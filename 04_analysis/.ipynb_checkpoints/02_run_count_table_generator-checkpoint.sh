#!/bin/bash
#SBATCH --job-name=sliceCnt
#SBATCH --output=logs/sliceCnt_%A_%a.out
#SBATCH --error=logs/sliceCnt_%A_%a.err
#SBATCH --array=0-11
#SBATCH --cpus-per-task=1
#SBATCH --mem=600G
#SBATCH --time=6:00:00
#SBATCH --partition=cpu
#SBATCH --mail-type=ALL
#SBATCH --mail-user=dhenze@stanford.edu

module load anaconda
conda activate Vizgen_2

cd /hpc/projects/group.quake/doug/Papers/Shapes/full_run/conflicts_correction/Microglial_Morphology/04_analysis 

# ---- GLOBAL VARIABLES -----------------------------------------------------
H5=../03_morph_embedding/Shape_500.h5ad
OUTDIR=transcript_out_slice_by_slice_v3
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
# -------------------------------------------------------------------

EXP=${EXPERIMENTS[$SLURM_ARRAY_TASK_ID]}
echo "SLURM task $SLURM_ARRAY_TASK_ID  ->  $EXP"

# running the relevant python script

python count_table_generation.py --exp "$EXP" --h5 "$H5" --outdir "$OUTDIR"
