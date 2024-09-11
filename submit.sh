#!/bin/sh
#BSUB -q gpua100
#BSUB -J train
#BSUB -n 16
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 72:00
#BSUB -R "rusage[mem=23GB]"
##BSUB -R "select[gpu80gb]" #options gpu40gb or gpu80gb
#BSUB -o gpu_logs/gpu_%J.out
#BSUB -e gpu_logs/gpu_%J.err
##BSUB -N
# -- end of LSF options --

nvidia-smi

source ../envs/renner/bin/activate
# source ../miniconda3/bin/activate
# conda activate renner

# Options
# Run main.py --help to get options


TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 python3 train_model.py --experiment-name imagefit_pancreas --data-path Task07_Pancreas --imagefit-mode --activation-function sine --model-lr 1e-5 --latent-lr 1e-3 --num-epochs 10000 --batch-size 10 --num-workers 16 --volume-sidelength 512 512 100 --latent-size 128 --num-hidden-features 512 --num-hidden-layers 6




