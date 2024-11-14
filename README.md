# lipnet

## installation on sol supercomputer
- ssh to sol
- clone this repo to `/data/grp_klee263/src/lipnet`
- move training data into the folder `data/` (via globus, ssh, or sol web interface)
- log into a compute node with `interactive`
- activate mamba with `module load mamba/latest`
- create mamba environment with appropriate dependencies
  - `mamba create -p /data/grp_klee263/envs/lipnet -c pytorch -c nvidia pytorch
    torchvision pytorch-cuda=12.4`
  - `source activate /data/grp_klee263/envs/lipnet`
  - `mamba install -c conda-forge pytest tqdm wandb torchdiffeq matplotlib`
- while you're in the environment, log into wandb with `wandb login`

## running it
- use `sbatch` on your chosen sbatch script, e.g., `sbatch sweep_chaotic_systems.sbatch`
- ???
- dance and celebrate!

# acknowledgements
- https://github.com/elyall/wandb_on_slurm
