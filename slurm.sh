#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4 # request at least 4 cpus per gpu
#SBATCH --gpus-per-task=1
#SBATCH --mem=50gb # usually request more that 1G memory. default is 8gb per cpu
#SBATCH --time=00:10:00
#SBATCH --partition=hpg-ai
#SBATCH --output=rapidsai_test%A_%a.log
#SBATCH --array=1-1
#SBATCH --gres=gpu:1 


ml python
module purge
module load rapidsai/23.02


chmod a+x main.py

./main.py --test -i ssss
