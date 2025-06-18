#!/bin/bash
#
#SBATCH --job-name=prompt_llama
#SBATCH --comment="Prompt the whole dataset."
#SBATCH --mail-type=ALL
#SBATCH --mail-user=j.grimm@campus.lmu.de
#SBATCH --chdir=/home/g/grimmj/LLAMCo
#SBATCH --output=/home/g/grimmj/LLAMCo/scripts/output/slurm.%j.%N.out
#SBATCH --ntasks=1

python3 -u prompt.py
