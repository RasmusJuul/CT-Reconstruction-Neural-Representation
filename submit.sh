#!/bin/sh
#BSUB -q gpua10
#BSUB -J raygan
#BSUB -n 16
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 08:00
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
cp data/bugnist_256/SL_clean_combined_interpolated_points_2_angles.hdf5 $MYTMP

cp data/bugnist_256/SL_cubed_clean.hdf5 $MYTMP

# cp data/FiberDataset/filaments_volumes.hdf5 $MYTMP

# ls $MYTMP


# TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 python3 train_raygan.py --experiment-name raygan --data-path FiberDataset/filaments_volumes_100 --encoder hashgrid --activation-function relu --model-lr 1e-4 --d-lr 1e-3 --num-epochs 100000 --batch-size 7500 --num-workers 16 --regularization-weight 1e-2 --ray-data-path $MYTMP/combined_interpolated_points.hdf5

# TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 python3 train_raygan.py --experiment-name raygan_SL --data-path bugnist_256/SL_cubed_clean/soldat_16_000 --encoder hashgrid --activation-function leaky_relu --model-lr 5e-5 --d-lr 1e-4 --num-epochs 1000 --batch-size 5000 --num-workers 16 --regularization-weight 5e-3 --ray-data-path $MYTMP/SL_clean_combined_interpolated_points_2_angles.hdf5

# TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 python3 train_model.py --experiment-name NeuralField_SL --data-path bugnist_256/SL_cubed/soldat_16_000 --no-latent --encoder hashgrid --activation-function leaky_relu --model-lr 1e-4 --num-epochs 1000 --batch-size 7500 --num-workers 16 --volume-sidelength 256 256 256 --noisy-points --regularization-weight 1e-2


TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 python3 train_raygan.py --experiment-name neuralgan_SL --data-path bugnist_256/SL_cubed_clean/soldat_16_000 --encoder hashgrid --activation-function relu --model-lr 1e-4 --d-lr 1e-4 --num-epochs 10000 --batch-size 2500 --num-workers 16 --regularization-weight 1e-2 --ray-data-path $MYTMP/SL_clean_combined_interpolated_points_2_angles.hdf5