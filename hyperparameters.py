def load_hyperparameters(dataset, architecture):
    """Loads the hyperparameters for training a network of a given architecture on a given dataset.

    Args:
        dataset (str): The dataset used for training. Either 'MNIST' or 'FashionMNIST'
        architecture (str): The architecture of the network. Either '1h' (1-hidden layer Hopfield network) or 'conv_avg' (convolutional Hopfield network with average pooling)

    Returns:
        dictionary of hyperparameters

        Hyperparameters:
            'layer_shapes' (list of tuple of ints): the shapes of the tensors representing the layers of the network
            'min_intervals' (list of floats): minima of the intervals for the states of the hidden and output layers
            'max_intervals' (list of floats): maxima of the intervals for the states of the hidden and output layers
            'edges' (list of tuple of ints): the indices of pre-synaptic and post-synaptic layers for each edge
            'weight_types' (list of str): the types of the weights for each edge. Either 'dense' or 'conv_avg_pool'
            'gains' (list of float32): the gains of the weights used at initialization
            'threshold' (float): threshold value used as a convergence criterion for the layers
            'max_iterations_first_phase' (int): the maximum number of iterations allowed to converge to equilibrium in the first phase of Aeqprop
            'max_iterations_second_phase' (int): the maximum number of iterations allowed to converge to equilibrium in the second phase of Aeqprop
            'max_iterations_params' (int): the maximum number of iterations allowed for the parameters to converge to equilibrium in the second phase of Aeqprop
            'training_mode' (str): either Optimistic Aeqprop, Centered Aeqprop, or Pessimistic Aeqprop
            'nudging' (float): the nudging value used to train via Aeqprop
            'learning_rates_biases' (list of float32): learning rates for the biases
            'learning_rates_weights' (list of float32): learning rates for the weights
            'num_epochs' (int): number of epochs of training
            'batch_size' (int): size of the mini-batch, i.e. number of examples from the dataset processed simultaneously
            'use_soft_targets' (bool): if True, the "one-hot code" of y is of the form [0, ..., 0, 0.9, 0, ..., 0] instead of [0, ..., 0, 1, 0, ..., 0]
            'weight_penalty' (float32): the squared penalty for the parameters
            'weight_decay' (float32): the weight decay for the parameters
            'lr_decay' (float32): rate of decay of the learning rates at each epoch
    """

    if dataset == 'MNIST':

        if architecture == '1h': # 1-hidden-layer Hopfield network
            hparams = {
            'layer_shapes': [(1, 28, 28), (2048,), (10,)],
            'min_intervals': [0., -1.],
            'max_intervals': [1., 2.],
            'edges': [(0, 1), (1, 2)],
            'weight_types': ['dense', 'dense'],
            'gains': [0.8, 1.2],
            'threshold': 1e-3,
            'max_iterations_first_phase': 100,
            'max_iterations_second_phase': 100,
            'max_iterations_params': 1,
            'training_mode': 'centered',
            'nudging': 0.5,
            'learning_rates_weights': [0.1, 0.05],
            'learning_rates_biases': [0.02, 0.01],
            'num_epochs': 200,
            'batch_size': 32,
            'use_soft_targets': False,
            'weight_penalty': None,
            'weight_decay': None,
            'lr_decay': 0.99,
            }

        elif architecture == 'conv_avg': # convolutional Hopfield network with average pooling
            hparams = {
            'layer_shapes': [(1, 28, 28), (32, 12, 12), (64, 4, 4), (10,)],
            'min_intervals': [0., 0., -1.],
            'max_intervals': [1., 1., 2.],
            'edges': [(0, 1), (1, 2), (2, 3)],
            'weight_shapes': [(32, 1, 5, 5), (64, 32, 5, 5), (64, 4, 4, 10)],
            'weight_types': ['conv_avg_pool', 'conv_avg_pool', 'dense'],
            'gains': [0.6, 0.6, 1.5],
            'threshold': 1e-3,
            'max_iterations_first_phase': 100,
            'max_iterations_second_phase': 100,
            'max_iterations_params': 1,
            'training_mode': 'centered',
            'nudging': 0.2,
            'learning_rates_weights': [0.128, 0.032, 0.008],
            'learning_rates_biases': [0.032, 0.008, 0.002],
            'num_epochs': 200,
            'batch_size': 16,
            'use_soft_targets': False,
            'weight_penalty': None,
            'weight_decay': None,
            'lr_decay': 0.99,
            }

        else:
            raise ValueError("expected '1h' or 'conv_avg', but got {}".format(architecture))
    
    elif dataset == 'FashionMNIST':

        if architecture == '1h': # 1-hidden-layer Hopfield network
            hparams = {
            'layer_shapes': [(1, 28, 28), (2048,), (10,)],
            'min_intervals': [0., -1.],
            'max_intervals': [1., 2.],
            'edges': [(0, 1), (1, 2)],
            'weight_types': ['dense', 'dense'],
            'gains': [0.8, 1.2],
            'threshold': 1e-3,
            'max_iterations_first_phase': 100,
            'max_iterations_second_phase': 100,
            'max_iterations_params': 1,
            'training_mode': 'centered',
            'nudging': 0.2,
            'learning_rates_weights': [0.1, 0.05],
            'learning_rates_biases': [0.02, 0.01],
            'num_epochs': 200,
            'batch_size': 32,
            'use_soft_targets': False,
            'weight_penalty': None,
            'weight_decay': None,
            'lr_decay': 0.99,
            }

        elif architecture == 'conv_avg': # convolutional Hopfield network with average pooling
            hparams = {
            'layer_shapes': [(1, 28, 28), (32, 12, 12), (64, 4, 4), (10,)],
            'min_intervals': [0., 0., -1.],
            'max_intervals': [1., 1., 2.],
            'edges': [(0, 1), (1, 2), (2, 3)],
            'weight_shapes': [(32, 1, 5, 5), (64, 32, 5, 5), (64, 4, 4, 10)],
            'weight_types': ['conv_avg_pool', 'conv_avg_pool', 'dense'],
            'gains': [0.6, 0.6, 1.5],
            'threshold': 1e-3,
            'max_iterations_first_phase': 100,
            'max_iterations_second_phase': 100,
            'max_iterations_params': 1,
            'training_mode': 'centered',
            'nudging': 0.2,
            'learning_rates_weights': [0.128, 0.032, 0.008],
            'learning_rates_biases': [0.032, 0.008, 0.002],
            'num_epochs': 200,
            'batch_size': 16,
            'use_soft_targets': False,
            'weight_penalty': None,
            'weight_decay': None,
            'lr_decay': 0.99,
            }

        else:
            raise ValueError("expected '1h' or 'conv_avg', but got {}".format(architecture))

    else:
        raise ValueError("expected 'MNIST' or 'FashionMNIST' but got {}".format(dataset))

    return hparams
