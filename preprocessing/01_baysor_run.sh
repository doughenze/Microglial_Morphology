#!/bin/bash
#SBATCH --time=7-00:00:00
#SBATCH --mem=250G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --job-name=baysor_run
#SBATCH --mail-type=ALL
#SBATCH --array=0-11

source /oak/stanford/groups/quake/doug/resources/miniconda3/etc/profile.d/conda.sh
export JULIA_NUM_THREADS=10

# Define an array of paths
paths=(
    "202405250811_3-mo-male-mouse-1-cerebellum-IHC_VMSC12602/region_1"
    "202405311300_3month-female-2-IHC_VMSC12602/region_0"
    "202406071120_24m-female-1-IHC_VMSC11602/region_0"
    "202406071304_24m-female-3-IHC_VMSC12602/region_0"
    "202406101010_24month-male-1-IHC_VMSC12602/region_0"
    "202406141019_24m-female-5-IHC_VMSC11602/region_0"
    "202406141135_24m-male-2-IHC_VMSC12602/region_0"
    "202406171409_3m-female-3-IHC_VMSC12602/region_0"
    "202406171454_3m-male-2-IHC_VMSC11602/region_0"
    "202407010924_3-month-female-1-rev2_VMSC12602/region_0"
    "202407011057_24-month-male-4-rev2_VMSC11602/region_0"
    "202407021559_3-mo-male-3-rev2_VMSC12602/region_0"
)

# Get the current path based on the SLURM array task ID
path=${paths[$SLURM_ARRAY_TASK_ID]}

# Change to the current path directory
cd $path

# Run Baysor
/oak/stanford/groups/quake/doug/resources/Baysor-0.6.2/bin/baysor/bin/baysor run -s 6.5 -x global_x -y global_y -g gene -m 50 -p \
-o "${path}_6-5_micron" \
--save-polygons=geojson \
--prior-segmentation-confidence 0 \
"$path/filtered_transcripts.csv" \
:cell_id
