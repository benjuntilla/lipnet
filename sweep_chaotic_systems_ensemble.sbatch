#!/bin/bash
#SBATCH -N 19
#SBATCH -n 19
#SBATCH -c 1
#SBATCH -t 4-0
#SBATCH --mem=32G
#SBATCH -p general
#SBATCH -q public
#SBATCH -o slurm-logs/sweep-output-%j.log
#SBATCH -e slurm-logs/sweep-error-%j.log
#SBATCH --mail-type=ALL
#SBATCH --mail-user=%u@asu.edu

module load mamba/latest cuda-12.4.1-gcc-12.1.0

source activate /data/grp_klee263/envs/lipnet

cd /data/grp_klee263/src/lipnet

python sweep-controller.py --sweep_config_path sweep_configs/chaotic_systems_ensemble.yaml --project_name chaotic-systems
