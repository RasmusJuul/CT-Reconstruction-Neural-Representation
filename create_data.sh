#!/bin/sh
#BSUB -q compute
#BSUB -J create
#BSUB -n 24
#BSUB -R "span[hosts=1]"
##BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 03:00
#BSUB -R "rusage[mem=10GB]"
##BSUB -R "select[gpu32gb]" #options gpu40gb or gpu80gb
#BSUB -o gpu_logs/gpu_%J.out
#BSUB -e gpu_logs/gpu_%J.err
##BSUB -N
# -- end of LSF options --

source ../envs/renner/bin/activate

# Options
# Run main.py --help to get options

python3 create_data.py

python3 create_ray_data.py