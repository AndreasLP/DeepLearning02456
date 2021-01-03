#!/bin/sh
#BSUB -q gpuv100
#BSUB -gpu "num=1"
#BSUB -J pretrain_keynet_9
#BSUB -n 1
#BSUB -W 24:00
#BSUB -R "rusage[mem=48GB]"
#BSUB -u s174505@student.dtu.dk
#BSUB -o outputs/output_%J.out
#BSUB -e errors/error_%J.err
# Load modules
module swap cuda/8.0
module swap cudnn/v7.0-prod-cuda8
# Edit environment variables
unset PYTHONHOME
unset PYTHONPATH
export MUJOCO_GL=egl
export PATH="$HOME/.local/bin:$PATH"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco200/bin
export IS_BSUB_EGL=1
cd /work1/s174505/share_DeepLearning/Parallel_keynets/pretrained_sac_keypoints_9/
python3 train.py \
    --domain_name cartpole \
    --task_name swingup \
    --work_dir "$LSB_JOBID" 
