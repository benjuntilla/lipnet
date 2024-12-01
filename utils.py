import argparse
import ast
import os
import random
from random import SystemRandom
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchdiffeq import odeint
import requests
import simplejson as json
import urllib.request 
import os
import wandb
from pathlib import Path
import re


def compute_pointwise_se(time_series_list):
    # convert all time series to numpy arrays
    series_arrays = [np.array(ts) for ts in time_series_list]
    
    # check that all time series have the same length
    lengths = [len(ts) for ts in series_arrays]
    if len(set(lengths)) != 1:
        raise ValueError(f"All time series must have the same length. Found lengths: {lengths}")
    
    # stack time series into a 2D array (series x time)
    stacked_series = np.vstack(series_arrays)
    
    # compute standard error at each time point
    n_series = len(time_series_list)
    standard_error = np.std(stacked_series, axis=0, ddof=1) / np.sqrt(n_series)
    
    return standard_error

  
def compute_pointwise_mean(time_series_list):
    # convert all time series to numpy arrays
    series_arrays = [np.array(ts) for ts in time_series_list]
    
    # check that all time series have the same length
    lengths = [len(ts) for ts in series_arrays]
    if len(set(lengths)) != 1:
        raise ValueError(f"All time series must have the same length. Found lengths: {lengths}")
    
    # stack time series into a 2D array (series x time)
    stacked_series = np.vstack(series_arrays)
    
    # compute mean at each time point
    pointwise_mean = np.mean(stacked_series, axis=0)
    
    return pointwise_mean


def calculate_smape(actual, predicted):
  return 2 * 100 * np.mean(np.abs(actual - predicted) / (np.abs(actual) + np.abs(predicted)))


def get_dysts_metadata():
  dysts_metadata_url = "https://raw.githubusercontent.com/williamgilpin/dysts/refs/heads/master/dysts/data/chaotic_attractors.json"
  with urllib.request.urlopen(dysts_metadata_url) as url:
      return json.load(url)


def get_run_configs_from_sweep(sweep_path):
    wandb.login()
    api = wandb.Api()
    sweep = api.sweep(sweep_path)
    
    # create mapping of run config by checkpoint ID (string)
    run_configs = {str(run.config["ckpt"]):run.config for run in sweep.runs}

    return run_configs


def get_runs_from_sweep(sweep_path):
    wandb.login()
    api = wandb.Api()
    sweep = api.sweep(sweep_path)
    return sweep.runs


def get_valid_checkpoint_paths(run_ids):
    # recursively get ckpt files using rglob in pathlib
    experiments_path = Path("experiments")
    checkpoint_paths = [str(path) for path in list(experiments_path.rglob("*.ckpt"))]

    # get checkpoint path ids to obtain wandb run configs
    checkpoint_path_ids = {}
    for checkpoint_path in checkpoint_paths:
        checkpoint_id_match = re.search(r'experiment_(\d+)', checkpoint_path)
        if not checkpoint_id_match:
            raise Exception(f"Malformed ckpt file name found: {checkpoint_path}")
        checkpoint_id = str(checkpoint_id_match.group(1))

        checkpoint_path_ids[checkpoint_path] = checkpoint_id

    # filter checkpoint paths to the ones in the sweep
    valid_checkpoint_paths = [path for path in checkpoint_paths if checkpoint_path_ids[path] in run_ids]

    return valid_checkpoint_paths


def generate_sweep_sbatch(nodes, sweep_id, walltime_days, memory_gb=32):
  return f"""#!/bin/bash
#SBATCH -N {nodes}
#SBATCH -n {nodes}
#SBATCH -c 1
#SBATCH -t {walltime_days}-0
#SBATCH --mem={memory_gb}G
#SBATCH -p general
#SBATCH -q public
#SBATCH -o slurm-logs/sweep-output-%j.log
#SBATCH -e slurm-logs/sweep-error-%j.log
#SBATCH --mail-type=ALL
#SBATCH --mail-user=%u@asu.edu

module load mamba/latest cuda-12.4.1-gcc-12.1.0

source activate /data/grp_klee263/envs/lipnet

cd /data/grp_klee263/src/lipnet

python sweep-controller.py --project_name chaotic-systems --sweep_id {sweep_id}"""


######################################################## utils ########################################################
def str_to_activation(activation_name):
    if activation_name.lower() == "tanh":
        return nn.Tanh
    elif activation_name.lower() == "relu":
        return nn.ReLU
    elif activation_name.lower() == "sigmoid":
        return nn.Sigmoid
    else:
        raise ValueError(f"Unsupported activation function: {activation_name}")


def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def get_batch(
    data, t, batch_len=60, batch_size=100, device=torch.device("cpu"), reverse=False
):
    # data is of the shape (split_size * t * num_states)
    def generate_ids_tensor(length, size):
        # return a tensor of shape size*1, containing integer numbers in [0,length -1]
        return torch.from_numpy(
            np.random.choice(np.arange(length, dtype=np.int64), size, replace=False)
        )

    datapoint_ids = generate_ids_tensor(len(data), batch_size)
    starting_point_ids = generate_ids_tensor(len(t) - batch_len, batch_size)

    batch_y0 = data[
        datapoint_ids, starting_point_ids, :
    ]  # randomly choosing the continuous chunk of trajectories (batch_size * batch_size * num_states)
    batch_t = t[:batch_len]  # time coordinates --> T
    batch_y = torch.stack(
        [data[datapoint_ids, starting_point_ids + i, :] for i in range(batch_len)],
        dim=1,
    )  # (batch_size * batch_len * num_states)
    if reverse:
        batch_y0 = batch_y[:, -1, :]
        batch_t = batch_t.flip([0])
        batch_y = batch_y.flip([1])
    return batch_y0.to(device), batch_t.to(device), batch_y.to(device)


def create_net(n_inputs, n_outputs, n_layers=1, n_units=100, nonlinear=nn.Tanh):
    if n_layers == 0:
        layers = [nn.Linear(n_inputs, n_outputs)]
    else:
        layers = [nn.Linear(n_inputs, n_units)]
        for i in range(n_layers - 1):
            layers.append(nonlinear())
            layers.append(nn.Linear(n_units, n_units))

        layers.append(nonlinear())
        layers.append(nn.Linear(n_units, n_outputs))
    return nn.Sequential(*layers)


# Define LipschitzLinear with normalization
class LipschitzLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(LipschitzLinear, self).__init__(in_features, out_features, bias=bias)

        # Initialize a trainable Lipschitz constant per output unit (row in the weight matrix)
        self.lipschitz_constant = nn.Parameter(torch.ones(out_features))

    def forward(self, input):
        # Apply softplus to ensure the Lipschitz constant is positive
        softplus_ci = F.softplus(self.lipschitz_constant)

        # Normalize the weight matrix based on the Lipschitz constraint
        normalized_weight = self.normalize_weights(self.weight, softplus_ci)

        # Use the normalized weight for the forward pass
        return F.linear(input, normalized_weight, self.bias)

    def normalize_weights(self, W, softplus_ci):
        absrowsum = torch.sum(torch.abs(W), dim=1)
        scale = torch.minimum(torch.tensor(1.0), softplus_ci / absrowsum)
        return W * scale[:, None]  # Apply the scaling row-wise


# Now, modify the create_net function to use LipschitzLinear instead of nn.Linear
def create_net_lip(n_inputs, n_outputs, n_layers=1, n_units=100, nonlinear=nn.Tanh):
    if n_layers == 0:
        layers = [LipschitzLinear(n_inputs, n_outputs)]
    else:
        layers = [LipschitzLinear(n_inputs, n_units)]
        for i in range(n_layers - 1):
            layers.append(nonlinear())
            layers.append(LipschitzLinear(n_units, n_units))

        layers.append(nonlinear())
        layers.append(LipschitzLinear(n_units, n_outputs))

    return nn.Sequential(*layers)


######################################################## utils ########################################################

######################################################## model ########################################################


class ODEfunc_lip(nn.Module):
    def __init__(self, dim, nlayer, nunit, device=torch.device("cpu")):
        super(ODEfunc_lip, self).__init__()
        self.gradient_net = create_net_lip(
            n_inputs=dim,
            n_outputs=dim,
            n_layers=nlayer,
            n_units=nunit,
            nonlinear=nn.Tanh,
        ).to(device)
        self.NFE = 0

    def forward(self, t, y):
        output = self.gradient_net(y)
        return output


class ODEfunc(nn.Module):
    def __init__(self, dim, nlayer, nunit, device=torch.device("cpu")):
        super(ODEfunc, self).__init__()
        self.gradient_net = create_net(
            dim, dim, n_layers=nlayer, n_units=nunit, nonlinear=nn.Tanh
        ).to(device)
        self.NFE = 0

    def forward(self, t, y):
        output = self.gradient_net(y)
        return output


class ODEfunc_ensemble(nn.Module):
    def __init__(
        self,
        dim,
        nlayer,
        nunit,
        ensemble_combo="mlp_lip",
        nonlinear_combo="tanh_tanh",
        device=torch.device("cpu"),
    ):
        super(ODEfunc_ensemble, self).__init__()

        nonlinear1, nonlinear2 = nonlinear_combo.split("_")
        activation1 = str_to_activation(nonlinear1)
        activation2 = str_to_activation(nonlinear2)

        if ensemble_combo == "mlp_lip":
            self.gradient_net1 = create_net(
                dim, dim, n_layers=nlayer, n_units=nunit, nonlinear=activation1
            ).to(device)
            self.gradient_net2 = create_net_lip(
                n_inputs=dim,
                n_outputs=dim,
                n_layers=nlayer,
                n_units=nunit,
                nonlinear=activation2,
            ).to(device)
        elif ensemble_combo == "mlp_mlp":
            self.gradient_net1 = create_net(
                dim, dim, n_layers=nlayer, n_units=nunit, nonlinear=activation1
            ).to(device)
            self.gradient_net2 = create_net(
                dim, dim, n_layers=nlayer, n_units=nunit, nonlinear=activation2
            ).to(device)
        elif ensemble_combo == "lip_lip":
            self.gradient_net1 = create_net_lip(
                n_inputs=dim,
                n_outputs=dim,
                n_layers=nlayer,
                n_units=nunit,
                nonlinear=activation1,
            ).to(device)
            self.gradient_net2 = create_net_lip(
                n_inputs=dim,
                n_outputs=dim,
                n_layers=nlayer,
                n_units=nunit,
                nonlinear=activation2,
            ).to(device)

        self.NFE = 0

    def forward(self, t, y):
        output = self.gradient_net1(y) + self.gradient_net2(y)
        return output


######################################################## models ########################################################