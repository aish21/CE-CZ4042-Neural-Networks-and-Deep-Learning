#!/bin/bash
#SBATCH --job-name=final
#SBATCH --output=Output.out
#SBATCH --error=Error.err
#SBATCH --nodes=1
#SBATCH --partition=SCSEGPU_UG
#SBATCH --mem=8000M
#SBATCH --gres=gpu:1

module load anaconda
source activate /apps/conda_env/CZ4042_v3
python main.py

