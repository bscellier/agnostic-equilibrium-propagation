import argparse
import torch

from datasets import load_dataset
from gui.gui import GUI
from hyperparameters import load_hyperparameters
from run import build_network

parser = argparse.ArgumentParser(description='Agnostic Eqprop')
parser.add_argument('--dataset', type = str, default = 'MNIST', help="The dataset used. Either `MNIST' or `FashionMNIST'")
parser.add_argument('--architecture', type = str, default = '1h', help="The model's architecture. Either '1h' or 'conv_avg'")

parser.add_argument('--method', type = str, default = 'centered', help="The method used to train the network: Optimistic Aeqprop, Pessimistic Aeqprop, Centered Aeqprop, ou AutoDiff")
parser.add_argument('--nudging', type = float, default = 0.5, help="The nudging value used during training")

parser.add_argument('--name', type = str, default = 'untitled', help="The name of the experiment")

args = parser.parse_args()


if __name__ == "__main__":

    dataset = args.dataset
    architecture = args.architecture

    hparams = load_hyperparameters(dataset, architecture)

    network = build_network(
        layer_shapes = hparams["layer_shapes"],
        edges = hparams["edges"],
        weight_types = hparams["weight_types"],
        gains = hparams["gains"],
        weight_shapes = hparams["weight_shapes"] if "weight_shapes" in hparams else None,
        paddings = hparams["paddings"] if "paddings" in hparams else None,
        min_intervals = hparams["min_intervals"],
        max_intervals = hparams["max_intervals"],
        learning_rates_biases = hparams["learning_rates_biases"],
        learning_rates_weights = hparams["learning_rates_weights"],
        weight_penalty = hparams["weight_penalty"],
        weight_decay = hparams["weight_decay"],
        )

    device = 'cpu'
    model_path = '/'.join(['runs', dataset, architecture, args.method, str(args.nudging), args.name, 'model.pt'])

    network.set_device(device)
    network.load(model_path)

    training_data, test_data = load_dataset(dataset)

    GUI(network, test_data).mainloop()
