#!/bin/bash
#SBATCH -N 38
#SBATCH -n 38
#SBATCH -c 2
#SBATCH -t 0-4
#SBATCH --mem=16G
#SBATCH -p general
#SBATCH -q public
#SBATCH -o slurm-logs/sweep-output-%j.log
#SBATCH -e slurm-logs/sweep-error-%j.log
#SBATCH --mail-type=ALL
#SBATCH --mail-user=%u@asu.edu

module load mamba/latest cuda-12.4.1-gcc-12.1.0

source activate /data/grp_klee263/envs/lipnet

cd /data/grp_klee263/src/lipnet

python sweep-controller.py --sweep_config_path sweep_configs/chaotic_systems.yaml --project_name chaotic-systems
