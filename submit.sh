#!/bin/sh
#BSUB -q gpua100
#BSUB -J rg
#BSUB -n 16
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 00:30
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

# TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 python3 train_raygan.py --experiment-name raygan_fiber9 --data-path synthetic_fibers/test_fiber_16_projections_9 --activation-function relu --model-lr 1e-4 --d-lr 1e-3 --num-epochs 150 --batch-size 2500 --num-workers 16 --regularization-weight 5e-3 --adversarial-weight 1e-1 --encoder hashgrid --ray-data-path $MYTMP/combined_interpolated_points_16.hdf5

# TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 python3 train_model.py --experiment-name NeuralField_fiber9 --data-path synthetic_fibers/test_fiber_16_projections_9 --no-latent --encoder hashgrid --activation-function relu --model-lr 1e-4 --num-epochs 200 --batch-size 2500 --num-workers 16 --regularization-weight 5e-3

# TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 python3 train_raygan.py --experiment-name raygan_fiber9 --data-path synthetic_fibers/test_fiber_16_projections_9 --activation-function sine --model-lr 1e-4 --d-lr 1e-3 --num-epochs 150 --batch-size 5000 --num-workers 16 --regularization-weight 5e-3 --adversarial-weight 1e-1 --ray-data-path $MYTMP/combined_interpolated_points_16.hdf5 --checkpoint-path NeuralField_fiber9_None_sine_regularization-weight-0.005_noise-level-None_latent-size-256-2025-01-27-2009/last.ckpt --weights-only

# TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 python3 train_model.py --experiment-name NeuralField_fiber9 --data-path synthetic_fibers/test_fiber_16_projections_9 --no-latent --activation-function sine --model-lr 1e-4 --num-epochs 200 --batch-size 2500 --num-workers 16 --regularization-weight 5e-3 --num-points 256

#---------------BugNIST-SL------------------

# cp data/bugnist_256/SL_combined_interpolated_points_4.hdf5 $MYTMP

# TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 python3 train_raygan.py --experiment-name raygan_SL0 --data-path bugnist_256/SL_0 --activation-function relu --model-lr 1e-4 --d-lr 1e-4 --num-epochs 200 --batch-size 10000 --num-workers 16 --regularization-weight 5e-3 --encoder hashgrid --adversarial-weight 1e-2 --ray-data-path $MYTMP/SL_combined_interpolated_points_4.hdf5

# TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 python3 train_model.py --experiment-name NeuralField_SL0 --data-path bugnist_256/SL_0 --no-latent --activation-function sine --model-lr 1e-4 --num-epochs 20 --batch-size 2500 --num-workers 16 --regularization-weight 5e-3

#-----------------Pancreas------------------
# TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 python3 train_model.py --experiment-name NeuralField_Pancreas0 --data-path Task07_Pancreas/test_0 --no-latent --encoder hashgrid --activation-function relu --model-lr 1e-3 --num-points 512 --num-epochs 100 --batch-size 2048 --num-workers 16 --regularization-weight 5e-4 --noisy-points

TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 python3 train_model.py --experiment-name NeuralField_Pancreas0 --data-path Task07_Pancreas/test_0 --no-latent --activation-function sine --model-lr 1e-3 --num-points 512 --num-epochs 1000 --batch-size 4096 --num-workers 16 --regularization-weight 5e-4 --noisy-points

# TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 python3 volume_fit.py --experiment-name NeuralField_Pancreas0 --data-path Task07_Pancreas/fbp_0.tif --activation-function sine --model-lr 1e-4 --num-epochs 5 --batch-size 10 --num-workers 4 --imagefit-mode

# cp data/Task07_Pancreas/combined_interpolated_points_50.hdf5 $MYTMP

# TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 python3 train_raygan.py --experiment-name raygan_Pancreas0 --data-path Task07_Pancreas/test_0 --activation-function relu --num-points 512 --model-lr 1e-4 --d-lr 1e-4 --num-epochs 200 --batch-size 10000 --num-workers 16 --regularization-weight 5e-3 --encoder hashgrid --adversarial-weight 1e-1 --ray-data-path $MYTMP/combined_interpolated_points_16.hdf5 

# TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 python3 train_raygan.py --experiment-name raygan_Pancreas0 --data-path Task07_Pancreas/test_0 --activation-function sine --num-points 512 --model-lr 1e-4 --d-lr 1e-4 --num-epochs 200 --batch-size 2500 --num-workers 16 --regularization-weight 5e-3 --adversarial-weight 1e-3 --ray-data-path $MYTMP/combined_interpolated_points_50.hdf5 --checkpoint-path NeuralField_Pancreas0_None_sine_regularization-weight-0.0_noise-level-None_latent-size-256-2025-01-20-1432/epoch=96.ckpt --weights-only

#---------------Citrus------------------

# TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 python3 train_model.py --experiment-name citrus_recon --data-path Citrus/volume_0 --no-latent --encoder hashgrid --activation-function relu --model-lr 1e-4 --num-points 256 --num-epochs 100 --batch-size 10000 --num-workers 16 --regularization-weight 5e-2 --noisy-points

# TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 python3 train_model.py --experiment-name citrus_recon --data-path Citrus/volume_0 --no-latent --activation-function sine --model-lr 1e-4 --num-points 256 --num-epochs 100 --batch-size 10000 --num-workers 16 --regularization-weight 5e-2 --noisy-points
