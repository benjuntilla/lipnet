from utils import generate_sweep_sbatch, get_runs_from_sweep
import simplejson as json
import argparse
import math

parser = argparse.ArgumentParser(description='Run wandb sweep with specified config.')
parser.add_argument('--sweep_id', type=str, default="adxfqd90", help='id of existing sweep you want to resume')
parser.add_argument('--nepoch', type=int, default="500", help='number of epochs')
parser.add_argument('--project', type=str, default="chaotic-systems", help='wandb project')
args = parser.parse_args()

# get crashed runs
sweep_path = f"{args.project}/sweeps/{args.sweep_id}"
runs = get_runs_from_sweep(sweep_path)
crashed_runs = [run for run in runs if run.state == "crashed"]
print(f"{len(crashed_runs)=}")

# set tags before moving them
for cr in crashed_runs:
    tags = [f"sweep={cr.sweep_name}", f"project={args.project}"]
    cr.tags = tags
    cr.update()
print(f"added tags {tags} to the runs")

# get maximum projected walltime
def get_projected_walltime_days(run):
    rate = run.summary['_runtime'] / run.summary['epoch']  # runtime is in seconds
    projected_days = rate * args.nepoch / 60 / 60 / 24
    return math.ceil(projected_days)

projected_walltimes = [get_projected_walltime_days(crashed_run) for crashed_run in crashed_runs]
max_walltime = min(max(projected_walltimes), 7)
print(f"{projected_walltimes=}")

# generate sbatch template
template = generate_sweep_sbatch(len(crashed_runs), args.sweep_id, max_walltime)

# save it
sbatch_path = "./sweep_rerun.sbatch"
with open(sbatch_path, "w") as f:
    f.write(template)
print(f"saved the script to {sbatch_path}")
print(f"now all you have to do is move the crashed runs in the sweep to a safe place and run {sbatch_path}!")
