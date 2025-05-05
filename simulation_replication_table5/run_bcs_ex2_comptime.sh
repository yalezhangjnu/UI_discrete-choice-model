#!/bin/bash -l

# options
#$ -l h_rt=48:00:00   # Specify the hard time limit for the job
#$ -pe omp 16

module load matlab/2023a

# Execute a code
matlab -nodisplay -nosplash -r "bcs_ex2_comptime($NSLOTS,$SGE_TASK_ID);exit"  > localout$SGE_TASK_ID

