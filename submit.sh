#!/bin/sh
#BSUB -q gpua100
#BSUB -J rgSL0
#BSUB -n 16
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 12:00
#BSUB -R "rusage[mem=20GB]"
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

# TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 python3 train_raygan.py --experiment-name raygan_fiber9 --data-path synthetic_fibers/test_fiber_16_projections_9 --activation-function relu --model-lr 1e-4 --d-lr 1e-3 --num-epochs 200 --batch-size 2500 --num-workers 16 --regularization-weight 5e-3 --encoder hashgrid --ray-data-path $MYTMP/combined_interpolated_points_16.hdf5

# TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 python3 train_model.py --experiment-name NeuralField_fiber9 --data-path synthetic_fibers/test_fiber_16_projections_9 --no-latent --encoder hashgrid --activation-function relu --model-lr 1e-4 --num-epochs 200 --batch-size 2500 --num-workers 16 --volume-sidelength 256 256 256 --regularization-weight 5e-3

#---------------BugNIST-SL------------------

cp data/bugnist_256/SL_combined_interpolated_points.hdf5 $MYTMP

TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 python3 train_raygan.py --experiment-name raygan_SL0 --data-path bugnist_256/SL_cubed_clean/soldat_5_012 --activation-function relu --model-lr 1e-4 --d-lr 1e-3 --num-epochs 200 --batch-size 2500 --num-workers 16 --regularization-weight 5e-3 --encoder hashgrid --ray-data-path $MYTMP/SL_combined_interpolated_points.hdf5

# TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 python3 train_model.py --experiment-name NeuralField_SL0 --data-path bugnist_256/SL_cubed_clean/soldat_5_012 --no-latent --encoder hashgrid --activation-function relu --model-lr 1e-4 --num-epochs 200 --batch-size 2500 --num-workers 16 --volume-sidelength 256 256 256 --regularization-weight 5e-3


