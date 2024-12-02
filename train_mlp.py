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

import wandb

torch.set_default_dtype(torch.float64)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

######################################################## parse and init log ########################################################


def parse_kanlayer(value):
    # Convert string input to a list of lists
    return ast.literal_eval(value)


parser = argparse.ArgumentParser(
    description="This script is for training an interpretable NODE using KAN."
)

parser.add_argument("--project", type=str, default="mlp", help="wandb project name")

# system related
parser.add_argument("--r", type=int, default=10, help="random_seed")
parser.add_argument(
    "--dpath",
    type=str,
    default=os.path.join(os.path.dirname(__file__), "data/lorenz63_torch_-1_1.npz"),
    help="The file path of the training file.",
)
parser.add_argument("--odeint", type=str, default="dopri5", help="integrator")

# MLP related
parser.add_argument(
    "--nlayer",
    type=ast.literal_eval,
    default=4,
    help="hidden layers of MLP hidden layer",
)
parser.add_argument(
    "--nunit", type=int, default=100, help="number of neurons per hidden layer"
)

parser.add_argument(
    "--architecture",
    type=str,
    choices=["mlp_lip", "mlp_mlp", "lip_lip", "lip", "mlp"],
    default="mlp",
    help="Combination of networks to use in ODEfunc_ensemble (lip, mlp, mlp_lip, mlp_mlp, or lip_lip)",
)

parser.add_argument(
    "--nonlinear_combo",
    type=str,
    default="tanh_tanh",
    help="Combination of nonlinear functions for the two ensemble members (e.g., tanh_relu, sigmoid_tanh)",
)

# training related
parser.add_argument("--lr", type=float, default=0.01, help="learning rate")
parser.add_argument("--nepoch", type=int, default=500, help="max epochs")
parser.add_argument(
    "--niterbatch", type=int, default=100, help="max steps in one epoch"
)
parser.add_argument("--lMB", type=int, default=100, help="length of one seq")
parser.add_argument("--nMB", type=int, default=40, help="number of seqs in each batch")

parser.add_argument("--enable_early_stopping", action=argparse.BooleanOptionalAction)

args = parser.parse_args()


# load datasets
data_path = args.dpath
data = np.load(data_path)
num_states = int(data["num_states"])
dataset_name = str(data["data_name"])

# Get time steps for each split
train_time_steps = torch.tensor(data.get("train_time_steps", data["time_steps"])).to(
    device
)
val_time_steps = torch.tensor(data.get("val_time_steps", data["time_steps"])).to(device)
test_time_steps = torch.tensor(data.get("test_time_steps", data["time_steps"])).to(
    device
)

print(
    f"Number of timesteps - Train: {len(train_time_steps)}, Val: {len(val_time_steps)}, Test: {len(test_time_steps)}"
)

# set up seeds
seed = args.r
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# init_checkpoint_filepath
save_path = f"experiments/{dataset_name}/"
makedirs(save_path)
experimentID = int(SystemRandom().random() * 100000)
print(f"Checkpoint ID is: {experimentID}")
ckpt_path = os.path.join(save_path, "experiment_" + str(experimentID) + ".ckpt")
fig_save_path = os.path.join(save_path, "experiment_" + str(experimentID))
makedirs(fig_save_path)
print(ckpt_path)

wandb_config = {
    "architecture": args.architecture,
    "seed": args.r,
    "learning_rate": args.lr,
    "dataset": dataset_name,
    "epochs": args.nepoch,
    "mlp_layer": str([num_states, args.nlayer, num_states]),
    "ckpt": experimentID,
    "odeint": args.odeint,
    "nunits": args.nunit,
    "train_time_steps": len(train_time_steps),
    "val_time_steps": len(val_time_steps),
    "test_time_steps": len(test_time_steps),
}

if not (args.architecture in ("mlp", "lip")):
    wandb_config["nonlinear_functions"] = args.nonlinear_combo

if args.enable_early_stopping:
    wandb_config["enable_early_stopping"] = True

wandb.init(
    # set the wandb project where this run will be logged
    project=args.project,
    # track hyperparameters and run metadata
    config=wandb_config,
)
######################################################## parse and init log ########################################################

######################################################## main ########################################################

# set up dataset
t = torch.tensor(data["time_steps"])
train_data = torch.tensor(data["train_data"][:, :, :])
val_data = torch.utils.data.DataLoader(torch.tensor(data["val_data"]), batch_size=50)
test_data = torch.utils.data.DataLoader(torch.tensor(data["test_data"]), batch_size=50)

# model initialization

match args.architecture:
    case "mlp":
        odefunc = ODEfunc(
            nlayer=args.nlayer,
            nunit=args.nunit,
            dim=num_states,
        )
    case "lip":
        odefunc = ODEfunc_lip(
            nlayer=args.nlayer,
            nunit=args.nunit,
            dim=num_states,
        )
    case _:
        odefunc = ODEfunc_ensemble(
            nlayer=args.nlayer,
            nunit=args.nunit,
            dim=num_states,
            ensemble_combo=args.architecture,
            nonlinear_combo=args.nonlinear_combo,
        )

params = odefunc.parameters()
optimizer = optim.Adamax(params, lr=args.lr)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 0.9987)

# training loop
best_loss = 1e30
frame = 0
early_stop_patience = 100  # Number of epochs to wait for improvement
epochs_without_improvement = 0  # Counter for epochs without improvement

for itr in range(args.nepoch):
    # train
    print("=={0:d}==".format(itr))
    for i in range(args.niterbatch):
        optimizer.zero_grad()
        batch_y0, batch_t, batch_y = get_batch(
            train_data, train_time_steps, args.lMB, args.nMB
        )
        pred_y = (
            odeint(
                odefunc,
                batch_y0,
                batch_t,
                method=args.odeint,
            )
            .to(device)
            .transpose(0, 1)
        )
        loss = torch.mean(torch.abs(pred_y - batch_y))
        print(itr, i, loss.item(), end="\r")
        loss.backward()
        optimizer.step()
        wandb.log({"loss": loss.item(), "epoch": itr, "step": i})  # Log loss to wandb
    scheduler.step()

    # validation with plots
    with torch.no_grad():
        val_loss = 0
        for d in val_data:
            pred_y = (
                odeint(odefunc, d[:, 0, :], val_time_steps, method=args.odeint)
                .to(device)
                .transpose(0, 1)
            )
            val_loss += torch.mean(torch.abs(pred_y - d)).item()
        wandb.log({"val_loss": val_loss, "epoch": itr})  # Log validation loss to wandb

        if val_loss < best_loss:
            print("saving...", val_loss)
            torch.save(
                {
                    "state_dict": odefunc.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "epoch": itr,
                    "best_loss": best_loss,
                    "random_states": {
                        "torch": torch.get_rng_state(),
                        "numpy": np.random.get_state(),
                        "random": random.getstate(),
                    },
                    "NFE": odefunc.NFE,
                },
                ckpt_path,
            )
            best_loss = val_loss
            epochs_without_improvement = 0  # Reset counter if there's an improvement
        else:
            epochs_without_improvement += 1  # Increment the counter if no improvement

        # Early stopping condition
        if args.enable_early_stopping and epochs_without_improvement >= early_stop_patience:
            print(
                f"Early stopping at epoch {itr} due to no improvement in validation loss."
            )
            break

        plt.figure()
        plt.tight_layout()
        save_file = os.path.join(fig_save_path, "image_{:03d}.png".format(frame))
        fig = plt.figure(figsize=(12, 4))
        fig.suptitle("Validation split", fontsize=16)
        for i in range(num_states):
            subplot = fig.add_subplot(1, num_states, i + 1)
            subplot.set_title(f"State {i+1}")
            subplot.plot(
                val_time_steps.cpu().numpy(),
                d[0, :, i].detach().numpy(),
                lw=2,
                color="k",
            )
            subplot.plot(
                val_time_steps.cpu().numpy(),
                pred_y.detach().numpy()[0, :, i],
                lw=2,
                color="c",
                ls="--",
            )
            plt.savefig(save_file)
        plt.close(fig)
        plt.close("all")
        plt.clf()
        frame += 1

# testing on best ckpt based on best validation loss
ckpt = torch.load(ckpt_path, weights_only=False)
odefunc.load_state_dict(ckpt["state_dict"])

odefunc.NFE = 0
test_loss = 0
test_sol = np.zeros_like(data["test_data"])
batch_idx = 50
for i, d in enumerate(test_data):
    pred_y = (
        odeint(odefunc, d[:, 0, :], test_time_steps, method=args.odeint)
        .to(device)
        .transpose(0, 1)
    )
    test_sol[batch_idx * i : batch_idx * (i + 1), :, :] = pred_y.detach().numpy()
    test_loss += torch.mean(torch.abs(pred_y - d)).item()
wandb.log({"test_loss": test_loss})  # Log test loss to wandb

# plot best fitted curves
fig = plt.figure(figsize=(12, 4))
fig.suptitle("Test split", fontsize=16)
test_t = t[len(train_time_steps)+len(val_time_steps):]
for i in range(num_states):
    subplot = fig.add_subplot(1, num_states, i + 1)
    subplot.set_title(f"State {i+1}")
    subplot.plot(test_t, data["test_data"][0, :, i], lw=3, color="k")
    subplot.plot(test_t, test_sol[0, :, i], lw=2, color="c", ls="--")

save_file = os.path.join(fig_save_path, "image_best.png")
plt.savefig(save_file)

# Log the image to wandb
wandb.log({"best_fitted_curves": wandb.Image(save_file)})

wandb.finish()  # Finish the wandb run

######################################################## main ########################################################
