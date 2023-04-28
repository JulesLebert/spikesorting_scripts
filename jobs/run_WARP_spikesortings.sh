#!/bin/bash -l

# Request 1 hour of wallclock time (format hours:minutes:seconds).
#$ -l h_rt=24:00:00

# Request 256 gigabyte of RAM (must be an integer followed by M, G, or T)
#$ -l mem=32G

# Request 10 gigabyte of TMPDIR space (default is 10 GB)
#$ -l tmpfs=10G

# Request 6 cores.
#$ -pe smp 12

# Set the name of the job.
#$ -N spikesorting_warp

# Set the working directory to somewhere in your scratch space.  
#  This is a necessary step as compute nodes cannot write to $HOME.
# Replace "<your_UCL_id>" with your UCL user ID :)
#$ -wd /home/skgtjml/Scratch/workspace

# module load xorg-utils/X11R7.7
# module load matlab/full/r2021a/9.10
# module load cuda/10.1.243/gnu-4.9.2
module load python3/3.8

# Activate python environment
source /home/skgtjml/envs/spikesorting_scritps/bin/activate

# Your work should be done in $TMPDIR 
cd $TMPDIR

python /home/skgtjml/code/spikesorting_scripts/scripts/spikesorting_concatenated_WARP.py /home/skgtjml/code/spikesorting_scripts/scripts/json_files/spikesorting_params_concatenated_WARP_7.json