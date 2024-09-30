#!/bin/sh
#BSUB -q gpua100
#BSUB -J train
#BSUB -n 16
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 02:00
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

# MYTMP=${__LSF_JOB_TMPDIR__}

# cp data/FiberDataset/filaments_volumes.hdf5 $MYTMP

# ls $MYTMP

# TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 python3 train_model.py --experiment-name volumefit_filament --data-path $MYTMP/filaments_volumes.hdf5 --imagefit-mode --activation-function sine --model-lr 5e-5 --latent-lr 2e-3 --num-epochs 100000 --batch-size 10 --num-workers 16 --volume-sidelength 256 256 256 --latent-size 256 --num-hidden-features 512 --num-hidden-layers 6 --encoder spherical

# TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 python3 train_model.py --experiment-name imagefit_cube_adversarial --data-path cube_data.hdf5 --imagefit-mode --activation-function sine --model-lr 5e-5 --latent-lr 2e-3 --num-epochs 10000 --batch-size 250 --num-workers 16 --volume-sidelength 100 100 100 --latent-size 32 --num-hidden-features 64 --num-hidden-layers 5 --adversarial --weights-only --checkpoint-path imagefit_cubes_None_sine_latent-size-32-2024-09-16-1447/MLP-epoch=6133.ckpt

# TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 python3 train_model.py --experiment-name imagefit_cube_adversarial --data-path cube_data.hdf5 --imagefit-mode --activation-function sine --model-lr 5e-5 --latent-lr 2e-3 --num-epochs 10000 --batch-size 500 --num-workers 16 --volume-sidelength 100 100 100 --latent-size 32 --num-hidden-features 64 --num-hidden-layers 5 --adversarial

TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 python3 train_model.py --experiment-name CT-tiny-cuda-nn --data-path FiberDataset/filaments_volumes_0 --no-latent --encoder hashgrid --activation-function relu --model-lr 1e-3 --num-epochs 100000 --batch-size 1000 --num-workers 16 --volume-sidelength 256 256 256  --noisy-points --regularization-weight 1e-3 --noise-level 5e-2

