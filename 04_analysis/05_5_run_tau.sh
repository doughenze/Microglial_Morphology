#!/bin/bash
#SBATCH --job-name=tau_calc
#SBATCH --output=tau_calc_%j.out
#SBATCH --error=tau_calc_%j.err
#SBATCH --partition=owners        # adjust for your cluster
#SBATCH --mem=320G
#SBATCH --time=04:00:00

# --- User-editable paths -------------------------------------------
ABC_H5AD="/oak/stanford/groups/quake/doug/bruno_transfer/references/ABC/whole/AIT17.0.rawcount_logCPM_10Xv3/AIT17.0.rawcount_logCPM_10Xv3.h5ad"
MAPPING_TSV="/oak/stanford/groups/quake/doug/bruno_transfer/references/ABC/AIT17.0.cl.df.v6_lock_230504/AIT17.0.cl.df.v6_lock_230504.tsv"
FILTER_H5AD="/oak/stanford/groups/quake/doug/bruno_transfer/Papers/Shapes/full_run/conflicts_correction/Microglial_Morphology/04_analysis/Transciptomic_labels_and_morphology_labels_full.h5ad"
OUTDIR="/oak/stanford/groups/quake/doug/bruno_transfer/Papers/Shapes/full_run/conflicts_correction/Microglial_Morphology/04_analysis"

# --- Environment ----------------------------------------------------
source /oak/stanford/groups/quake/doug/resources/miniconda3/etc/profile.d/conda.sh
conda activate Vizgen_2

cd /oak/stanford/groups/quake/doug/bruno_transfer/Papers/Shapes/full_run/conflicts_correction/Microglial_Morphology/04_analysis
# --- Run ------------------------------------------------------------
python compute_tau.py \
    --abc_h5ad  "${ABC_H5AD}" \
    --mapping_tsv "${MAPPING_TSV}" \
    $( [ -n "$FILTER_H5AD" ] && echo --filter_h5ad "${FILTER_H5AD}" ) \
    --outdir "${OUTDIR}"
