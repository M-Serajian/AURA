#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4 # request at least 4 cpus per gpu
#SBATCH --gpus-per-task=1
#SBATCH --mail-type=ALL                 # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=m.serajian@ufl.edu  # Where to send mail
#SBATCH --mem=10gb # usually request more that 1G memory. default is 8gb per cpu
#SBATCH --time=10:10:00
#SBATCH --partition=hpg-ai
#SBATCH --output=rapidsai_test%A_%a.log
#SBATCH --array=1-1
#SBATCH --gres=gpu:1 


ml python;module purge;module load rapidsai/23.02

chmod a+x main.py
date
./main.py --test -i /home/m.serajian/share/MTB/gerbil_output/csv/1_1285_MTB_genomes.csv
date
