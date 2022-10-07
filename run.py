import argparse
import os
import torch

from datasets import load_dataloaders
from hyperparameters import load_hyperparameters
from network.network import Network
from training.epoch import Trainer, Evaluator
from training.sgd import Aeqprop, AutoDiff
from training.monitor import Monitor

parser = argparse.ArgumentParser(description='Agnostic Eqprop')
parser.add_argument('--dataset', type = str, default = 'MNIST', help="The dataset used. Either `MNIST' or `FashionMNIST'")
parser.add_argument('--architecture', type = str, default = '1h', help="The model's architecture")

parser.add_argument('--method', type = str, default = 'centered', help="The method used to train the network: Optimistic Aeqprop, Pessimistic Aeqprop, Centered Aeqprop, ou AutoDiff")
parser.add_argument('--nudging', type = float, default = 0.5, help="The nudging value used during training")

parser.add_argument('--name', type = str, default = 'untitled', help="The name of the experiment")

parser.add_argument('--verbose', dest='verbose', action='store_true')
parser.add_argument('--not-verbose', dest='verbose', action='store_false')
parser.set_defaults(verbose=False)

args = parser.parse_args()


def build_network(layer_shapes, edges, weight_types, gains, weight_shapes, paddings, min_intervals, max_intervals, learning_rates_biases, learning_rates_weights, weight_penalty, weight_decay):
    """Builds a network, given its high level characteristics

    Args:
        layer_shapes (list of tuple of ints): the shapes of the tensors representing the layers of the network
        edges (list of tuple of ints): the indices of pre-synaptic and post-synaptic layers for each edge
        weight_types (list of str): the types of the weights for each edge. Either 'dense' or 'conv_avg_pool'
        gains (list of float32): the gains of the weights used at initialization
        weight_shapes (list of tuple of ints): the shapes of the weight tensors. Required for convolutional weights  # FIXME: this is not needed in some cases
        paddings (list of ints): the paddings of the convolutions, if applicable.  # FIXME: this is not needed in some cases
        min_intervals (list of float32): minimum values of the intervals for the layers' states
        max_intervals (list of float32): maximum values of the intervals for the layers' states
        learning_rates_biases (list of float32): learning rates for the biases
        learning_rates_weights (list of float32): learning rates for the weights
        weight_penalty (float32): the squared penalty for the parameters
        weight_decay (float32): the weight decay for the parameters

    Returns:
        Network: the network
    """
    
    # creates an instance of Network
    network = Network(layer_shapes[0])

    # creates the layers of the network
    for shape, min_interval, max_interval, learning_rate in zip(layer_shapes[1:], min_intervals, max_intervals, learning_rates_biases):
        network.add_layer(shape, min_interval=min_interval, max_interval=max_interval, learning_rate=learning_rate, bias_penalty=weight_penalty, bias_decay=weight_decay)

    if weight_shapes == None: weight_shapes = [None] * len(edges)  # FIXME. hack: weight shapes might not be provided as they are only required for convolutional weights
    if paddings == None: paddings = [0] * len(edges)  # FIXME. hack: paddings might not be provided as they are only required for convolutional weights

    # creates the edges (i.e. weights) of the network, between the layers
    for (idx_pre, idx_post), weight_type, gain, shape, padding, learning_rate in zip(edges, weight_types, gains, weight_shapes, paddings, learning_rates_weights):
        network.add_edge(idx_pre, idx_post, weight_type, gain, shape, padding, learning_rate=learning_rate, weight_penalty=weight_penalty, weight_decay=weight_decay)

    # tells the network which layer is the output layer and what cost function to use
    network.pack(linearized_cost=False)

    return network



if __name__ == "__main__":

    torch.backends.cudnn.deterministic = True

    dataset = args.dataset
    architecture = args.architecture

    hparams = load_hyperparameters(dataset, architecture)

    batch_size = hparams["batch_size"]
    training_loader, test_loader = load_dataloaders(dataset, batch_size)

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

    device = "cuda" if torch.cuda.is_available() else "cpu"
    network.set_device(device)

    threshold = hparams["threshold"]
    thresholds = [threshold] * network.num_layers
    network.thresholds = thresholds

    hparams["training_mode"] = args.method
    hparams["nudging"] = args.nudging

    if args.method == "autodiff":
        differentiator = AutoDiff(network)
        differentiator.lr_scaling = hparams["nudging"]
        differentiator.max_iterations = hparams["max_iterations_first_phase"]
    else:
        differentiator = Aeqprop(network)
        differentiator.training_mode = hparams["training_mode"]
        differentiator.nudging = hparams["nudging"]
        differentiator.max_iterations_first_phase = hparams["max_iterations_first_phase"]
        differentiator.max_iterations_second_phase = hparams["max_iterations_second_phase"]
        differentiator.max_iterations_params = hparams["max_iterations_params"]

    trainer = Trainer(network, training_loader, differentiator)
    trainer.use_soft_targets = hparams["use_soft_targets"]

    evaluator = Evaluator(network, test_loader)
    evaluator.max_iterations = hparams["max_iterations_first_phase"]

    path = '/'.join(['runs', dataset, architecture, args.method, str(args.nudging), args.name])

    monitor = Monitor(network, trainer, evaluator, path)

    hparams["dataset"] = dataset
    hparams["path"] = path
    hparams["device"] = device

    print('\n'.join(['{} = {}'.format(key, value) for key, value in hparams.items()]))  # logs the characteristics of the run
    print()

    num_epochs = hparams["num_epochs"]
    lr_decay = hparams["lr_decay"]
    verbose = args.verbose
    monitor.run(num_epochs, lr_decay=lr_decay, verbose=verbose)
