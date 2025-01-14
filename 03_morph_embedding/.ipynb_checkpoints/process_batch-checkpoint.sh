#!/bin/bash
#SBATCH --job-name=process_batch
#SBATCH --output=slurm-%x.%j.out
#SBATCH --error=slurm-%x.%j.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=200G

# Load modules from Vizgen.sif container
module load anaconda
conda activate Vizgen_2

# Run the Python script for processing batches
python process_images.py $ADATAPATH $BASEPATH $BATCH_ID $OUTPUTDIR
