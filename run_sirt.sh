#!/bin/sh
#BSUB -q gpuv100
#BSUB -J sirt
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 12:00
#BSUB -R "rusage[mem=8GB]"
##BSUB -R "select[gpu80gb]" #options gpu40gb or gpu80gb
#BSUB -o gpu_logs/gpu_%J.out
#BSUB -e gpu_logs/gpu_%J.err
##BSUB -N
# -- end of LSF options --

nvidia-smi


source ../miniconda3/bin/activate
conda activate renner

# python3 sirt_test.py --mode plenoptic --projection-number 9
# python3 sirt_test.py --mode plenoptic --projection-number 16
# python3 sirt_test.py --mode plenoptic --projection-number 25
# python3 sirt_test.py --mode plenoptic --projection-number 36

# python3 sirt_test.py --mode rotation --projection-number 9
# python3 sirt_test.py --mode rotation --projection-number 16
# python3 sirt_test.py --mode rotation --projection-number 25
python3 sirt_test.py --mode rotation --projection-number 36