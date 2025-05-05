#!/bin/bash -l

# options
#$ -l h_rt=48:00:00   # Specify the hard time limit for the job
#$ -pe omp 16

module load matlab/2023a

# Execute a code
matlab -nodisplay -nosplash -r "main_ex2_comptime($NSLOTS);exit"  > localout_comptime

