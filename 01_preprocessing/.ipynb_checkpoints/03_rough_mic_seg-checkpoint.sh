#!/bin/bash
#SBATCH --time=36:00:00
#SBATCH --mem=250G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --job-name=segment
#SBATCH --array=0-11
#SBATCH --partition=owners,quake
#SBATCH --mail-type=ALL
#SBATCH --mail-user=dhenze@stanford.edu

#repos=('24-mo-female-1/'
#    '24-mo-female-3/'
#    '24-mo-female-5/'
#    '24-mo-male-1/'
#    '24-mo-male-2/'
#    '24-mo-male-4-rev2/'
#    '3-mo-male-1/'
#    '3-mo-male-2/'
#    '3-mo-male-3-rev2/'
#    '3-mo-female-1-rev2/'
#    '3-mo-female-2/'
#    '3-mo-female-3/')

repos=('202406071120_24m-female-1-IHC_VMSC11602/region_0/'
    '202406071304_24m-female-3-IHC_VMSC12602/region_0/'
    '202406141019_24m-female-5-IHC_VMSC11602/region_0/'
    '202406101010_24month-male-1-IHC_VMSC12602/region_0/'
    '202406141135_24m-male-2-IHC_VMSC12602/region_0/'
    '202407011057_24-month-male-4-rev2_VMSC11602/region_0/'
    '202405250811_3-mo-male-mouse-1-cerebellum-IHC_VMSC12602/region_1/'
    '202406171454_3m-male-2-IHC_VMSC11602/region_0/'
    '202407021559_3-mo-male-3-rev2_VMSC12602/region_0/'
    '202407010924_3-month-female-1-rev2_VMSC12602/region_0/'
    '202405311300_3month-female-2-IHC_VMSC12602/region_0/'
    '202406171409_3m-female-3-IHC_VMSC12602/region_0/')

#repos=('202406071120_24m-female-1-IHC_VMSC11602/region_0/'
#    '202406071304_24m-female-3-IHC_VMSC12602/region_0/'
#    '202407011057_24-month-male-4-rev2_VMSC11602/region_0/')

repo="/oak/stanford/groups/quake/shared/Vizgen/dough/output/${repos[$SLURM_ARRAY_TASK_ID]}/"

# If environment is not active run it as follows
# singularity exec segment.sif conda run -n segment_cells python microglia_segmentation.py "${repo}images" 'Anti-Rabbit' 4096 "${repo}" --min_size 1500
# If running from conda environments

source /oak/stanford/groups/quake/doug/resources/miniconda3/etc/profile.d/conda.sh
conda activate Vizgen_2

python microglia_segmentation.py "${repo}images" 'Anti-Chicken' 8192 "${repo}" --min_size 1500
