#!/bin/bash -l

# Request 1 hour of wallclock time (format hours:minutes:seconds).
#$ -l h_rt=16:00:00

# Request 256 gigabyte of RAM (must be an integer followed by M, G, or T)
#$ -l mem=8G

# Request 10 gigabyte of TMPDIR space (default is 10 GB)
#$ -l tmpfs=10G

# For 1 GPU
#$ -l gpu=1

# Request 6 cores.
#$ -pe smp 10

# Set the name of the job.
#$ -N batch_spikesort_Orecchiette

# Set the working directory to somewhere in your scratch space.  
#  This is a necessary step as compute nodes cannot write to $HOME.
# Replace "<your_UCL_id>" with your UCL user ID :)
#$ -wd /home/skgtjml/Scratch/workspace

module unload compilers mpi
module load compilers/gnu/4.9.2
module load cuda/10.1.243/gnu-4.9.2
module load cudnn/7.6.5.32/cuda-10.1

module unload python
module load python/miniconda3/4.10.3
source $UCL_CONDA_PATH/etc/profile.d/conda.sh

nvidia-smi

# Activate python environment
conda activate ibl_pykil_ss

# Your work should be done in $TMPDIR 
cd $TMPDIR

python /home/skgtjml/code/spikesorting_scripts/scripts/spikesorting_single_NP.py /home/skgtjml/code/spikesorting_scripts/scripts/json_files/single_NP.json