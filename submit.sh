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

# Hashgrid encoder

# TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 python3 train_model.py --experiment-name walnut --data-path walnut_small_angle/walnut_small --model-type mlp --activation-function sine --learning-rate 1e-4 --num-epochs 1000 --batch-size 2048 --num-workers 16 --num-points 512 --noisy-points --regularization-weight 1e-3 --encoder hashgrid

# TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 python3 train_model.py --experiment-name walnut --data-path walnut_small_angle/walnut_small --model-type mlp --activation-function sine --learning-rate 1e-4 --num-epochs 1000 --batch-size 2048 --num-workers 16 --num-points 512 --noisy-points --regularization-weight 1e-3 --noise-level 0.03 --encoder hashgrid

# TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 python3 train_model.py --experiment-name walnut --data-path walnut_small_angle/walnut_small --model-type mlp --activation-function sine --learning-rate 1e-4 --num-epochs 1000 --batch-size 2048 --num-workers 16 --num-points 512 --noisy-points --regularization-weight 1e-2 --noise-level 0.03 --encoder hashgrid

# TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 python3 train_model.py --experiment-name walnut --data-path walnut_small_angle/walnut_small --model-type mlp --activation-function sine --learning-rate 1e-4 --num-epochs 1000 --batch-size 2048 --num-workers 16 --num-points 256 --noisy-points --regularization-weight 1e-3 --noise-level 0.03 --encoder hashgrid

# TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 python3 train_model.py --experiment-name walnut --data-path walnut_small_angle/walnut_small --model-type mlp --activation-function sine --learning-rate 1e-4 --num-epochs 1000 --batch-size 2048 --num-workers 16 --num-points 512 --noisy-points --regularization-weight 1e-3 --noise-level 0.06 --encoder hashgrid

# TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 python3 train_model.py --experiment-name walnut --data-path walnut_small_angle/walnut_small --model-type mlp --activation-function sine --learning-rate 1e-4 --num-epochs 1000 --batch-size 2048 --num-workers 16 --num-points 512 --noisy-points --regularization-weight 1e-2 --noise-level 0.06 --encoder hashgrid

# frequency encoder
# TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 python3 train_model.py --experiment-name walnut --data-path walnut_small_angle/walnut_small --model-type mlp --activation-function sine --learning-rate 1e-4 --num-epochs 1000 --batch-size 2048 --num-workers 16 --num-points 512 --noisy-points --regularization-weight 1e-3 --encoder frequency

# TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 python3 train_model.py --experiment-name walnut --data-path walnut_small_angle/walnut_small --model-type mlp --activation-function sine --learning-rate 1e-4 --num-epochs 1000 --batch-size 2048 --num-workers 16 --num-points 512 --noisy-points --regularization-weight 1e-3 --noise-level 0.03 --encoder frequency

# TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 python3 train_model.py --experiment-name walnut --data-path walnut_small_angle/walnut_small --model-type mlp --activation-function sine --learning-rate 1e-4 --num-epochs 1000 --batch-size 2048 --num-workers 16 --num-points 512 --noisy-points --regularization-weight 1e-2 --noise-level 0.03 --encoder frequency

# TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 python3 train_model.py --experiment-name walnut --data-path walnut_small_angle/walnut_small --model-type mlp --activation-function sine --learning-rate 1e-4 --num-epochs 1000 --batch-size 2048 --num-workers 16 --num-points 256 --noisy-points --regularization-weight 1e-3 --noise-level 0.03 --encoder frequency

# TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 python3 train_model.py --experiment-name walnut --data-path walnut_small_angle/walnut_small --model-type mlp --activation-function sine --learning-rate 1e-4 --num-epochs 1000 --batch-size 2048 --num-workers 16 --num-points 512 --noisy-points --regularization-weight 1e-3 --noise-level 0.06 --encoder frequency

# TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 python3 train_model.py --experiment-name walnut --data-path walnut_small_angle/walnut_small --model-type mlp --activation-function sine --learning-rate 1e-4 --num-epochs 1000 --batch-size 2048 --num-workers 16 --num-points 512 --noisy-points --regularization-weight 1e-2 --noise-level 0.06 --encoder frequency

# Big walnut

# TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 python3 train_model.py --experiment-name walnut --data-path walnut_angle/walnut --model-type mlp --activation-function sine --learning-rate 1e-4 --num-epochs 1000 --batch-size 2048 --num-workers 16 --num-points 512 --noisy-points --regularization-weight 1e-2
# TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 python3 train_model.py --experiment-name walnut --data-path walnut_angle/walnut --model-type mlp --activation-function sine --learning-rate 1e-4 --num-epochs 1000 --batch-size 2048 --num-workers 16 --num-points 512 --noisy-points --regularization-weight 5e-2 --noise-level 0.03
# TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 python3 train_model.py --experiment-name walnut --data-path walnut_angle/walnut --model-type mlp --activation-function sine --learning-rate 1e-4 --num-epochs 1000 --batch-size 2048 --num-workers 16 --num-points 512 --noisy-points --regularization-weight 5e-3 --noise-level 0.03
# TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 python3 train_model.py --experiment-name walnut --data-path walnut_angle/walnut --model-type mlp --activation-function sine --learning-rate 1e-4 --num-epochs 1000 --batch-size 2048 --num-workers 16 --num-points 512 --noisy-points --regularization-weight 1e-2 --noise-level 0.03
# TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 python3 train_model.py --experiment-name walnut --data-path walnut_angle/walnut --model-type mlp --activation-function sine --learning-rate 1e-4 --num-epochs 1000 --batch-size 2048 --num-workers 16 --num-points 256 --noisy-points --regularization-weight 1e-3 --noise-level 0.03
# TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 python3 train_model.py --experiment-name walnut --data-path walnut_angle/walnut --model-type mlp --activation-function sine --learning-rate 1e-4 --num-epochs 1000 --batch-size 2048 --num-workers 16 --num-points 512 --noisy-points --regularization-weight 1e-3 --noise-level 0.06
# TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 python3 train_model.py --experiment-name walnut --data-path walnut_angle/walnut --model-type mlp --activation-function sine --learning-rate 1e-4 --num-epochs 1000 --batch-size 2048 --num-workers 16 --num-points 512 --noisy-points --regularization-weight 1e-2 --noise-level 0.06


TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 python3 train_model.py --experiment-name imagefit --data-path synthetic_fibers --imagefit-mode --model-type mlp --activation-function sine --learning-rate 1e-4 --num-epochs 1000 --batch-size 30 --num-workers 16 --encoder hashgrid --volume-sidelength 300 --latent-size 256


