#!/bin/sh
#BSUB -q gpua100
#BSUB -J rg
#BSUB -n 8
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 01:00
#BSUB -R "rusage[mem=16GB]"
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

MYTMP=${__LSF_JOB_TMPDIR__}

#---------------Fibers------------------

# cp data/synthetic_fibers/combined_interpolated_points_16.hdf5 $MYTMP

# TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 python3 train_raygan.py --experiment-name raygan_fiber9 --data-path synthetic_fibers/test_fiber_16_projections_9 --activation-function relu --model-lr 1e-3 --d-lr 1e-4 --num-epochs 300 --batch-size 12500 --num-workers 8 --smoothness-weight 5e-3 --adversarial-weight 1e-3 --encoder hashgrid --ray-data-path $MYTMP/combined_interpolated_points_16.hdf5 --checkpoint-path raygan_fiber9_hashgrid_relu_regularization-weight-0.005_noise-level-None-2025-02-04-2208/last.ckpt

# TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 python3 train_model.py --experiment-name NeuralField_fiber9 --data-path synthetic_fibers/test_fiber_16_projections_9 --no-latent --encoder hashgrid --activation-function relu --model-lr 1e-4 --num-epochs 200 --batch-size 2500 --num-workers 16 --smoothness-weight 5e-3

#---------------BugNIST-SL------------------

# cp data/bugnist_256/SL_combined_interpolated_points_4.hdf5 $MYTMP

# TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 python3 train_raygan.py --experiment-name raygan_SL0 --data-path bugnist_256/SL_0 --activation-function relu --model-lr 1e-4 --d-lr 1e-4 --num-epochs 200 --batch-size 10000 --num-workers 16 --smoothness-weight 5e-3 --encoder hashgrid --adversarial-weight 1e-2 --ray-data-path $MYTMP/SL_combined_interpolated_points_4.hdf5

# TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 python3 train_model.py --experiment-name NeuralField_SL0 --data-path bugnist_256/SL_0 --no-latent --activation-function sine --model-lr 1e-4 --num-epochs 20 --batch-size 2500 --num-workers 16 --smoothness-weight 5e-3

#-----------------Pancreas------------------
# TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 python3 train_model.py --experiment-name NeuralField_Pancreas0 --data-path Task07_Pancreas/test_16proj_0 --no-latent --encoder hashgrid --activation-function relu --model-lr 1e-3 --num-points 512 --num-epochs 500 --batch-size 2048 --num-workers 16 --smoothness-weight 5e-4 --noisy-points

# TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 python3 volume_fit.py --experiment-name NeuralField_Pancreas0 --data-path Task07_Pancreas/fbp_0.tif --activation-function sine --model-lr 1e-4 --num-epochs 5 --batch-size 10 --num-workers 4 --imagefit-mode

cp data/Task07_Pancreas/combined_interpolated_points_16.hdf5 $MYTMP

TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 python3 train_raygan.py --experiment-name raygan_Pancreas0 --data-path Task07_Pancreas/test_16proj_0 --activation-function relu --num-points 512 --model-lr 1e-3 --d-lr 5e-5 --num-epochs 1000 --batch-size 3800 --num-workers 8 --smoothness-weight 5e-4 --encoder hashgrid --adversarial-weight 1e-3 --ray-data-path $MYTMP/combined_interpolated_points_16.hdf5 --midpoint 0.15 --checkpoint-path "raygan_Pancreas0_projection_shape_(16, 512, 1)-2025-02-05-1917/last.ckpt"

#---------------Citrus------------------

# TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 python3 train_model.py --experiment-name citrus_recon --data-path Citrus/volume_0 --no-latent --encoder hashgrid --activation-function relu --model-lr 1e-4 --num-points 256 --num-epochs 100 --batch-size 10000 --num-workers 16 --smoothness-weight 5e-2 --noisy-points

# TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 python3 train_model.py --experiment-name citrus_recon --data-path Citrus/volume_0 --no-latent --activation-function sine --model-lr 1e-4 --num-points 256 --num-epochs 100 --batch-size 10000 --num-workers 16 --smoothness-weight 5e-2 --noisy-points
