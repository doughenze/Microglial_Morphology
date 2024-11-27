#!/bin/bash
#SBATCH --job-name=process_batches
#SBATCH --output=slurm-%x.%j.out
#SBATCH --error=slurm-%x.%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=200G

# Define the paths
ADATAPATH="Microglia_cuttlefish_500_block5_conv1_umap.h5ad"
BASEPATH="/"
OUTPUTDIR="/shape_outputs/"

# Create the output directory if it doesn't exist
mkdir -p $OUTPUTDIR

# Read batch IDs from the adata file
BATCH_IDS=$(python -c "import scanpy as sc; ad = sc.read_h5ad('${ADATAPATH}'); print(' '.join(ad.obs['batchID'].unique()))")

# Array to hold job IDs
JOB_IDS=()

# Submit a separate job for each batch_id
for BATCH_ID in $BATCH_IDS; do
    JOB_ID=$(sbatch --parsable --export=ALL,ADATAPATH=$ADATAPATH,BASEPATH=$BASEPATH,BATCH_ID=$BATCH_ID,OUTPUTDIR=$OUTPUTDIR process_batch.sh)
    JOB_IDS+=($JOB_ID)
done

# Create a comma-separated list of job IDs
JOB_IDS_STR=$(IFS=,; echo "${JOB_IDS[*]}")

# Submit the concatenation job with dependency on the completion of all batch jobs
sbatch --dependency=afterok:$JOB_IDS_STR concatenate_job.sh $OUTPUTDIR feature_vectors_texture.csv
