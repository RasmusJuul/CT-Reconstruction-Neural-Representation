#!/bin/sh
#BSUB -q gpua100
#BSUB -J train
#BSUB -n 16
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 08:00
#BSUB -R "rusage[mem=8GB]"
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

# TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 python3 train_model.py --experiment-name fiber_imagefit_siren --model-type mlp --activation-function sine --learning-rate 1e-4 --num-epochs 100000 --batch-size 50 --num-workers 16 --imagefit-mode

TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 python3 train_model.py --experiment-name fiber_synthetic --model-type mlp --activation-function sine --learning-rate 1e-4 --num-epochs 100000 --batch-size 10000 --num-workers 16 --compiled --noisy-points --num-points 512 --data-path data/synthetic_fibers_plenoptic/fiber_00 --regularization-weight 1e-1

# TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 python3 train_model.py --experiment-name fiber_detectorfit_siren --model-type mlp --activation-function sine --learning-rate 1e-4 --num-epochs 100000 --batch-size 10000 --num-workers 16 #--checkpoint-path fiber_imagefit-2024-06-19-2110/MLP-epoch=5536.ckpt

# TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 python3 train_model.py --experiment-name fiber_real --model-type mlp --activation-function sine --learning-rate 1e-4 --num-epochs 100000 --batch-size 5000 --num-workers 16 --compiled --noisy-points --data-path data/fiber1 --num-points 512 --num-hidden-features 512 --num-hidden-layers 6 --num-freq-bands 8
