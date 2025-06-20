This is the preprocessing step in our code pipeline. It can be broken up into a few separate parts, we first want to run Baysor on the transcript outputs from the MERSCOPE. This can be accomplished by running the following line on a SLURM computing cluster:

sbatch 01_baysor_run.sh

Following this we are left with a .loom file for every brain that was run on the scope. We can combine these together in 02_Load_and_concat_baysor.ipynb 

For our final step of preprocessing we want to perform an Otsu segmentation on a max projected microglia image. To do this we are going to want to run microglia_segmentation.py. This python file is driven by the 03_rough_mic_seg.sh and can be run by running the following code snippet on a SLURM computing cluster:

sbatch 03_rough_mic_seg.sh

Performing preprocessing memory requirements depends on the size of the samples. These jobs ask for 250GB which would be excessive for smaller sections, 16 core CPUs should be easily able to handle these code blocks.