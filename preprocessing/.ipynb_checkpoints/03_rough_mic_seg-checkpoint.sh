#!/bin/bash
#SBATCH --time=4-00:00:00
#SBATCH --mem=250G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --job-name=segment
#SBATCH --array=0-11

repos=('24-mo-female-1/'
    '24-mo-female-3/'
    '24-mo-female-5/'
    '24-mo-male-1/'
    '24-mo-male-2/'
    '24-mo-male-4-rev2/'
    '3-mo-male-1/'
    '3-mo-male-2/'
    '3-mo-male-3-rev2/'
    '3-mo-female-1-rev2/'
    '3-mo-female-2/'
    '3-mo-female-3/')

repo="/hpc/projects/group.quake/doug/Shapes_Spatial/${repos[$SLURM_ARRAY_TASK_ID]}/"

python microglia_segmentation.py "${repo}images" 'Anti-Rabbit' 4096 "${repo}" --min_size 1500
