# lipnet

## running on sol supercomputer

- ssh to sol
- clone this repo to `/data/grp_klee263/src/lipnet`
- move training data `data/` into this repo (via globus or sol web interface)
- log into a compute node with `interactive`
- activate mamba with `module load mamba/latest`
- create mamba environment named `lkan-cuda` with appropriate dependencies
  - `mamba create -p /data/grp_klee263/envs/lipnet -c pytorch -c nvidia pytorch
    torchvision pytorch-cuda=12.4`
  - `source activate /data/grp_klee263/envs/lipnet`
  - `mamba install -c conda-forge pytest tqdm wandb torchdiffeq matplotlib`
- while you're in the environment, log into wandb with `wandb login`
- use `sbatch` on your chosen sbatch script, e.g., `sbatch train_lkan.sbatch` or `sbatch sweep_lkan.sbatch`
- ???
- dance and celebrate!

# acknowledgements
- https://github.com/elyall/wandb_on_slurm
- https://github.com/Blealtan/efficient-kan
