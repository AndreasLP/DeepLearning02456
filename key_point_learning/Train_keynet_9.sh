#!/bin/sh 
### General options 
### -- specify queue -- 
#BSUB -q gpuv100
### -- set the job Name -- 
#BSUB -J KeyNet9
### -- ask for number of cores (default: 1) -- 
#BSUB -n 1
### -- specify that the cores must be on the same host -- 
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"
### -- specify that we need 2GB of memory per core/slot -- 
#BSUB -R "rusage[mem=24GB]"
### -- set walltime limit: hh:mm -- 
#BSUB -W 24:00
### -- set the email address -- 
#BSUB -u s174480@student.dtu.dk
### -- send notification at start -- 
###BSUB -B 
### -- send notification at completion -- 
###BSUB -N 
### -- Specify the output and error file. %J is the job-id -- 
### -- -o and -e mean append, -oo and -eo mean overwrite -- 
#BSUB -o Results/%J/Output_%J.out 
#BSUB -e Results/%J/Error_%J.err
mkdir Results/$LSB_JOBID
python3 keypoint_learning_colab_9.py $LSB_JOBID
