#!/bin/bash -l

# SGE Options
#$ -l h_rt=24:00:00        # Set a hard time limit of 24 hours for the job
#$ -pe omp 8               # Request 8 CPU cores
#$ -N app_bl_cfcasf        # Set the job name
#$ -m ea                   # Send an email when the job ends or aborts
#$ -j y                    # Combine stdout and stderr into a single file
#$ -t 1-200   # Submit an array job with 10 tasks

# Load MATLAB module
module load matlab

# Execute MATLAB code
matlab -nodisplay -r "app_givasf_scc3($NSLOTS, $SGE_TASK_ID)"
