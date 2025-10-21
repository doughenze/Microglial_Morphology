#!/bin/bash
#SBATCH --job-name=sliceCnt_small
#SBATCH --output=logs/sliceCnt_small_%A_%a.out
#SBATCH --error=logs/sliceCnt_small_%A_%a.err
#SBATCH --array=4-5
#SBATCH --mem=600G
#SBATCH --time=9:00:00
#SBATCH --partition=owners,bigmem
#SBATCH --mail-type=ALL
#SBATCH --mail-user=dhenze@stanford.edu

source /oak/stanford/groups/quake/doug/resources/miniconda3/etc/profile.d/conda.sh
conda activate Vizgen_2

cd /oak/stanford/groups/quake/doug/bruno_transfer/Papers/Shapes/full_run/conflicts_correction/Microglial_Morphology/04_analysis 

# ---- GLOBAL VARIABLES -----------------------------------------------------
H5=../03_morph_embedding/Shape_500.h5ad
OUTDIR=transcript_out_slice_by_slice_v5
#EXPERIMENTS=(
#    "/hpc/projects/group.quake/doug/Shapes_Spatial/3-mo-male-1/"
#    "/hpc/projects/group.quake/doug/Shapes_Spatial/3-mo-male-2/"
#    "/hpc/projects/group.quake/doug/Shapes_Spatial/3-mo-male-3-rev2/"
#    "/hpc/projects/group.quake/doug/Shapes_Spatial/3-mo-female-1-rev2/"
#    "/hpc/projects/group.quake/doug/Shapes_Spatial/3-mo-female-2/"
#    "/hpc/projects/group.quake/doug/Shapes_Spatial/3-mo-female-3/"
#    "/hpc/projects/group.quake/doug/Shapes_Spatial/24-mo-male-1/"
#    "/hpc/projects/group.quake/doug/Shapes_Spatial/24-mo-male-2/"
#    "/hpc/projects/group.quake/doug/Shapes_Spatial/24-mo-male-4-rev2/"
#    "/hpc/projects/group.quake/doug/Shapes_Spatial/24-mo-female-1/"
#    "/hpc/projects/group.quake/doug/Shapes_Spatial/24-mo-female-3/"
#    "/hpc/projects/group.quake/doug/Shapes_Spatial/24-mo-female-5/"
#)
#EXPERIMENTS=(
#    "/oak/stanford/groups/quake/shared/Vizgen/dough/output/202405250811_3-mo-male-mouse-1-cerebellum-IHC_VMSC12602/region_1/"
#    "/oak/stanford/groups/quake/shared/Vizgen/dough/output/202406171454_3m-male-2-IHC_VMSC11602/region_0/"
#    "/oak/stanford/groups/quake/shared/Vizgen/dough/output/202407021559_3-mo-male-3-rev2_VMSC12602/region_0/"
#    "/oak/stanford/groups/quake/shared/Vizgen/dough/output/202407010924_3-month-female-1-rev2_VMSC12602/region_0/"
#    "/oak/stanford/groups/quake/shared/Vizgen/dough/output/202405311300_3month-female-2-IHC_VMSC12602/region_0/"
#    "/oak/stanford/groups/quake/shared/Vizgen/dough/output/202406171409_3m-female-3-IHC_VMSC12602/region_0/"
#    "/oak/stanford/groups/quake/shared/Vizgen/dough/output/202406101010_24month-male-1-IHC_VMSC12602/region_0/"
#    "/oak/stanford/groups/quake/shared/Vizgen/dough/output/202406141135_24m-male-2-IHC_VMSC12602/region_0/"
#    "/oak/stanford/groups/quake/shared/Vizgen/dough/output/202407011057_24-month-male-4-rev2_VMSC11602/region_0/"
#    "/oak/stanford/groups/quake/shared/Vizgen/dough/output/202406071120_24m-female-1-IHC_VMSC11602/region_0/"
#    "/oak/stanford/groups/quake/shared/Vizgen/dough/output/202406071304_24m-female-3-IHC_VMSC12602/region_0/"
#    "/oak/stanford/groups/quake/shared/Vizgen/dough/output/202406141019_24m-female-5-IHC_VMSC11602/region_0/"
#)

#"/oak/stanford/groups/quake/shared/Vizgen/dough/output/202405250811_3-mo-male-mouse-1-cerebellum-IHC_VMSC12602/region_1/"
#"/oak/stanford/groups/quake/shared/Vizgen/dough/output/202405311300_3month-female-2-IHC_VMSC12602/region_0/"
#"/oak/stanford/groups/quake/shared/Vizgen/dough/output/202406171409_3m-female-3-IHC_VMSC12602/region_0/"
#"/oak/stanford/groups/quake/shared/Vizgen/dough/output/202406141019_24m-female-5-IHC_VMSC11602/region_0/"
#"/oak/stanford/groups/quake/shared/Vizgen/dough/output/202407011057_24-month-male-4-rev2_VMSC11602/region_0/"
EXPERIMENTS=(
    "/oak/stanford/groups/quake/shared/Vizgen/dough/output/202406141135_24m-male-2-IHC_VMSC12602/region_0/"
    "/oak/stanford/groups/quake/shared/Vizgen/dough/output/202405250811_3-mo-male-mouse-1-cerebellum-IHC_VMSC12602/region_1/"
    "/oak/stanford/groups/quake/shared/Vizgen/dough/output/202405311300_3month-female-2-IHC_VMSC12602/region_0/"
    "/oak/stanford/groups/quake/shared/Vizgen/dough/output/202406171409_3m-female-3-IHC_VMSC12602/region_0/"
    "/oak/stanford/groups/quake/shared/Vizgen/dough/output/202406141019_24m-female-5-IHC_VMSC11602/region_0/"
    "/oak/stanford/groups/quake/shared/Vizgen/dough/output/202407011057_24-month-male-4-rev2_VMSC11602/region_0/"
    )
# -------------------------------------------------------------------

EXP=${EXPERIMENTS[$SLURM_ARRAY_TASK_ID]}
echo "SLURM task $SLURM_ARRAY_TASK_ID  ->  $EXP"

# running the relevant python script

python count_table_generation.py --exp "$EXP" --h5 "$H5" --outdir "$OUTDIR"
