def load_hyperparameters(dataset, architecture):

    if dataset == 'MNIST':

        if architecture == '1h':
            # 1 layer network
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

        elif architecture == 'conv_avg':
            # convolutional network
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

        return hparams
    
    elif dataset == 'FashionMNIST':

        if architecture == '1h':
            # 1 layer network
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

        elif architecture == 'conv_avg':
            # convolutional network
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

        return hparams

    else:
        raise ValueError("expected 'MNIST' or 'FashionMNIST' but got {}".format(dataset))
