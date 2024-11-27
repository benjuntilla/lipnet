import wandb
import subprocess
import yaml
import os
import argparse

# fix for stupid bug when running wandb sweep on slurm
# https://github.com/wandb/wandb/issues/5272#issuecomment-1881950880
os.environ['WANDB_DISABLE_SERVICE'] = "True"

# Gather nodes allocated to current slurm job
result = subprocess.run(['scontrol', 'show', 'hostnames'], stdout=subprocess.PIPE)
node_list = result.stdout.decode('utf-8').split('\n')[:-1]


def run(sweep_config_path, project_name, entity_name, sweep_id=None):
    wandb.init(project=project_name)
    if not sweep_id:
        with open(sweep_config_path) as file:
            sweep_config = yaml.load(file, Loader=yaml.FullLoader)
        sweep_id = wandb.sweep(sweep_config, project=project_name)
    sp = []
    for node in node_list:
        sp.append(subprocess.Popen(['srun',
                        '--nodes=1',
                        '--ntasks=1',
                        '-w',
                        node,
                        'sweep-agent.sh',
                        sweep_id,
                        project_name,
                        entity_name]))
    exit_codes = [p.wait() for p in sp]  # wait for processes to finish
    return exit_codes

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run wandb sweep with specified config.')
    parser.add_argument('--sweep_config_path', type=str, default=os.path.join(os.getcwd(), 'src', 'sweep_configs', 'lkan.yaml'), help='Path to the sweep configuration YAML file')
    parser.add_argument('--project_name', type=str, default="lkan", help='wandb project name')
    parser.add_argument('--sweep_id', type=str, nargs="?", help='id of existing sweep if you want to resume it')
    parser.add_argument('--entity_name', type=str, default="benjuntilla-arizona-state-university", help='name of entity')
    args = parser.parse_args()

    run(args.sweep_config_path, args.project_name, args.entity_name, args.sweep_id)
