import torch
import torch.nn.functional as F

from network.variable.layer import Layer
from network.variable.parameter import Bias, DenseWeight, ConvWeight
from network.variable.control import ControlKnob
from network.interaction import Penalty, DenseInteraction, BiasInteraction, ConvAvgPoolInteraction, ControlInteraction
from network.cost import SquaredError, LinearizedError, PenaltyCost



class Network:
    """
    Class used to implement layered Hopfield-like networks. This includes networks with fully connected layers, as well as convolutional layers.

    Attributes
    ----------
    layers_state (list of Tensors): the state of the layers of the network
    learning_rates (list of float32): the learning rates
    num_layers (int): the number of the layers in the network
    num_params (int): the number of parameter variables in the network
    thresholds (list of floats): list of threshold values used as convergence criterion for the layers
    input_layer (Layer): input layer of the network
    layers (list of Layer): list of ``hidden layers'' of the network, including the output layer
    params (list of Parameter): the parameters (weights and biases) of the network
    control_knobs (list of ControlKnobs): the control knobs of the network
    interactions (list of Interaction): list of all interactions of the network, except the control and cost interactions
    control_interactions (list of ControlInteraction): list of all control interactions of the network
    cost_interactions (list of CostInteraction): list of all cost interactions of the network
    batch_size (int): Size of the mini-batch, i.e. number of examples from the dataset processed simultaneously
    device (str): Either 'cpu' or 'cuda'
    output_layer (Layer): output layer of the network
    target_layer (Layer): target layer of the network
    _y (Tensor): label tensor. Shape is (batch size,). Type is int.

    Methods
    -------
    add_layer(shape, min_interval, max_interval, learning_rate, bias_penalty, bias_decay)
        Adds a layer to the network, as well as the bias of the layer, the associated control knob and the corresponding interactions
    add_edge(idx_pre, idx_post, weight_type, gain, shape, padding, learning_rate, weight_penalty, weight_decay)
        Adds an edge between two layers of the network
    pack(idx_output_layer, linearized_cost)
        Tells the network which layer will be the output layer
    params_state()
        Returns the current parameters (weights and biases) of the network
    params()
        Returns the parameter variables of the network
    layers()
        Returns the layers
    init_layers()
        Initializes the layers of the network to zero
    layers_shape()
        Returns the list of shapes of the layers in the network
    predict(without_clamping):
        Returns the prediction associated to a given configuration
    energy_fn()
        Returns the energy (Hopfield energy) of a given configuration
    cost_fn()
        Returns the cost value (squarred error between output layer and target) of a given configuration
    error_fn(without_clamping)
        Computes the error value for the current state configuration
    params_state()
        Returns the current parameter values (weights and biases) of the network
    gradient_step(step_size)
        Runs one step of gradient descent on the augmented energy in the state space
    homeostatic_relaxation(max_iterations, adjust_control_knobs):
        Let the network relax to equilibrium with homeostatic control on the parameters
    clamped_relaxation(max_iterations)
        Let the network relax to equilibrium with clamped control knobs
    set_data(x, y, reset, use_soft_targets):
        Set input x and target y
    set_device(self, device)
        Set the tensors of the network on a given device
    set_nudging(nudging)
        Set the nudging value
    save(model_path)
        Saves the model
    load(model_path)
        Loads the model parameters
    """

    def __init__(self, input_shape, device=None):
        """Creates an instance of Network.

        The network is initially `empty', i.e. without layers and/or connections

        Args:
            input_shape (tuple of int): shape of the input layer
            device (str, optional): The device where the tensors of the network are set. Either 'cpu' or 'cuda'. Default: None
        """

        self._batch_size = 1  # FIXME
        self._device = device

        self._input_layer = Layer(input_shape, batch_size=self._batch_size, device=self._device)

        self._layers = [self._input_layer]
        self._params = []
        self._control_knobs = []

        self._interactions = []
        self._control_interactions = []
        self._cost_interactions = []


    def add_layer(self, shape, min_interval=0., max_interval=1., learning_rate=1e-2, bias_penalty=None, bias_decay=None):
        """Adds a layer to the network, as well as the bias of the layer, the associated control knob and the corresponding interactions

        Args:
            shape (tuple of ints): shape of the layer (which is also the shape of the bias and control knob)
            min_interval (float32, optional): minimum of the interval for the layer's state. Default: 0.
            max_interval (float32, optional): maximum of the interval for the layer's state. Default: 1.
            learning_rate (float32, optional): learning rate for the bias. Default: 1e-2
            bias_penalty (float32, optional): the strength of the penalty for the bias (in the phase without nudging). Default: None
            bias_decay (float32, optional): the weight decay for the bias (difference of the bias penalty between the two phases). Default: None
        """

        layer = Layer(shape, min_interval=min_interval, max_interval=max_interval, batch_size=self._batch_size, device=self._device)
        self._layers.append(layer)

        layer_penalty = Penalty(layer)
        self._interactions.append(layer_penalty)

        shape = tuple([dim if idx==0 else 1 for idx, dim in enumerate(shape)])  # e.g. converts shape (32, 16, 16) into shape (32, 1, 1). Used for the biases of convolutional layers
        bias = Bias(shape, device=self._device)
        self._params.append(bias)

        bias_interaction = BiasInteraction(layer, bias)
        self._interactions.append(bias_interaction)

        if bias_penalty and bias_decay:
            bias_penalty = PenaltyCost(bias, penalty=bias_penalty, decay=bias_decay)
            self._cost_interactions.append(bias_penalty)

        control_knob = ControlKnob(bias)
        self._control_knobs.append(control_knob)

        control_interaction = ControlInteraction(control_knob, bias, learning_rate)
        self._control_interactions.append(control_interaction)

    def add_edge(self, idx_pre, idx_post, weight_type, gain, shape=None, padding=0, learning_rate=1e-2, weight_penalty=None, weight_decay=None):
        """Adds an edge between two layers of the network.

        Adding an edge consists in adding the weight as well as the control knob and associated interactions

        Args:
            idx_pre (int): index of layer_pre, the `pre-synaptic' layer
            idx_post (int): index of layer_post, the `post-synaptic' layer
            weight_type (str): either `dense', `conv_avg_pool' or `conv_max_pool'
            gain (float32): the gain (scaling factor) of the weight at initialization
            shape (tuple of ints, optional): the shape of the weight tensor. Required in the case of convolutional weights. Default: None
            padding (int, optional): the padding of the convolution, if applicable. Default: 0
            learning_rate (float32, optional): learning rate for the bias. Default: 1e-2
            weight_penalty (float32, optional): the strength of the penalty for the bias (in the phase without nudging). Default: None
            weight_decay (float32, optional): the weight decay for the weight tensor. Default: None
        """

        layer_pre = self._layers[idx_pre]
        layer_post = self._layers[idx_post]

        if weight_type == "dense":
            weight = DenseWeight(layer_pre.shape, layer_post.shape, gain, device=self._device)
            weight_interaction = DenseInteraction(layer_pre, layer_post, weight)
        else:  # weight_type == "conv_avg_pool":
            weight = ConvWeight(shape, gain, device=self._device)
            weight_interaction = ConvAvgPoolInteraction(layer_pre, layer_post, weight, padding)

        self._params.append(weight)
        self._interactions.append(weight_interaction)

        if weight_penalty and weight_decay:
            weight_penalty = PenaltyCost(weight, penalty=weight_penalty, decay=weight_decay)
            self._cost_interactions.append(weight_penalty)

        control_knob = ControlKnob(weight)
        self._control_knobs.append(control_knob)

        control_interaction = ControlInteraction(control_knob, weight, learning_rate)
        self._control_interactions.append(control_interaction)

    def pack(self, idx_output_layer=-1, linearized_cost=False):
        """Tells the network which layer will be the output layer

        Args:
            idx_output_layer (int): the index of the layer that plays the role of output layer
            linearized_cost (bool, optional): whether we use the standard squared error cost function (False) or the linearized version (True). Default: False
        """

        self._output_layer = self._layers[idx_output_layer]
        self._target_layer = Layer(self._output_layer.shape, batch_size=self._batch_size, device=self._device)

        if linearized_cost: cost_interaction = LinearizedError(self._output_layer, self._target_layer)
        else: cost_interaction = SquaredError(self._output_layer, self._target_layer)

        self._interactions.append(cost_interaction)
        self._cost_interactions.append(cost_interaction)

        del self._layers[0]  # remove the input layer from the list of `floating' layers (at inference, inputs are clamped and need not be relaxed)

    @property
    def layers_state(self):
        """Gets and sets the layers' states of the network"""

        return [layer.state for layer in self._layers]

    @layers_state.setter
    def layers_state(self, layers_state):

        for layer, state in zip(self._layers, layers_state): layer.state = state

    @property
    def learning_rates(self):
        """Gets and sets the learning rates for the parameters of the network"""

        return [interaction.learning_rate for interaction in self._control_interactions]

    @learning_rates.setter
    def learning_rates(self, learning_rates):

        # if len(learning_rates) != len(self._control_interactions): raise ValueError("expected length {} but got {}".format(len(self._control_interactions), len(learning_rates)))

        for interaction, learning_rate in zip(self._control_interactions, learning_rates): interaction.learning_rate = learning_rate

    @property
    def num_layers(self):
        """Returns the number of layers of the network"""

        return len(self._layers)

    @property
    def num_params(self):
        """Returns the number of parameters in the network"""

        return len(self._params)

    @property
    def thresholds(self):
        """Get and sets the threshold values of the layers of the network"""

        return [layer.threshold for layer in self._layers]

    @thresholds.setter
    def thresholds(self, thresholds):
        # TODO: raises an error if thresholds does not have the right length
        for layer, threshold in zip(self._layers, thresholds): layer.threshold = threshold

    def params_state(self):
        """Returns the current parameters (weights and biases) of the network

        Returns:
            params (list of Tensor): List of param tensors. Each tensor has type float32. Each weight tensor has shape (batch_size, weight_shape) and each bias tensor has shape (batch_size, layer_shape).
        """

        return [param.state for param in self._params]

    def params(self):
        """Returns the parameter variables of the network"""

        return self._params

    def layers(self):
        """Returns the layers"""

        return self._layers

    def layer_shapes(self):
        """Returns the list of shapes of the layers in the network"""

        input_shape = self._input_layer.shape

        return [input_shape] + [layer.shape for layer in self._layers]

    def init_layers(self):
        """Initialize the layers of the network to zero, i.e. s=0."""

        for layer in self._layers: layer.init_state(self._batch_size, self._device)

    def set_device(self, device):
        """Set the tensors of the network on a given device"""

        self._device = device

        for layer in self._layers: layer.state = layer.state.to(device)
        for param in self._params: param.state = param.state.to(device)
        for control_knob in self._control_knobs: control_knob.state = control_knob.state.to(device)

    def predict(self, without_clamping=False):
        """Prediction associated to a given configuration

        Args:
            without_clamping (bool, optional): If True, prediction is not clipped. Default: False.

        Returns:
            Vector of size (batch_size,) and of type int: each coordinate is the index of the predicted category for the corresponding example in the current mini-batch
        """

        if not without_clamping:
            output = self._output_layer.state  # state of output layer
        else:
            output = self._output_layer.compute_min(without_clamping)
        prediction = torch.argmax(output, dim=1)  # the predicted category is the index of the output unit that has the highest value
        return prediction

    def energy_fn(self):
        """Returns the energy of the current configuration.

        Returns:
            Tensor of shape (batch_size,) and type float32. Vector of energy values for each of the examples in the current mini-batch
        """

        return sum([interaction.energy_fn() for interaction in self._interactions])

    def cost_fn(self):
        """Returns the cost value of the current output configuration.

        Returns:
            Tensor of shape (batch_size,) and type float32. Vector of cost values for each of the examples in the current mini-batch
        """

        return sum([cost_interaction.cost_fn() for cost_interaction in self._cost_interactions])

    def error_fn(self, without_clamping=False):
        """Returns the error value for the current output configuration.

        Args:
            without_clamping (bool, optional): If True, prediction is not clipped. Default: False.

        Returns:
            Tensor of shape (batch_size,) and type bool. Vector of error values for each of the examples in the current mini-batch
        """

        prediction = self.predict(without_clamping)
        label = self._y
        return torch.ne(prediction, label)

    def set_data(self, x, y, reset=False, use_soft_targets=False):
        """Set input and target

        Args:
            x: input. Tensor of shape (batch_size, width, height). Type is float32.
            y: Tensor of shape (batch size,) and type int. Labels associated to the inputs in the batch.
            reset (bool, optional): if True, resets the state of the network to zero.
            use_soft_targets (bool, optional): if True, the "one-hot code" of y is of the form [0, ..., 0, 0.9, 0, ..., 0] instead of [0, ..., 0, 1, 0, ..., 0]. Default: False
        """

        x = x.to(self._device)
        y = y.to(self._device)

        self._input_layer.state = x

        self._y = y
        num_classes = self._target_layer.shape[0]  # number of categories in the classification task
        y_one_hot = F.one_hot(y, num_classes=num_classes).type(torch.float32)  # convert the label y into the one-hot code of y.
        if use_soft_targets: y_one_hot *= 0.9
        self._target_layer.state = y_one_hot

        batch_size = x.size(0)
        if reset or self._batch_size != batch_size:
            self._batch_size = batch_size
            self.init_layers()

    def set_nudging(self, nudging):
        """Set the nudging value

        Args:
            nudging (float32): Nudging value
        """

        for cost_interaction in self._cost_interactions: cost_interaction.nudging = nudging

    def homeostatic_relaxation(self, backward=False, max_iterations=100, use_criterion=True, adjust_control_knobs=True):
        """Let the layers of the network relax to equilibrium, a (local) minimum of the cost-augmented energy.

        Relaxes the network's layers for a maximum of max_iterations iterations. Stops if the convergence criterion is below a threshold.

        Args:
            backward (bool, optional): if False, relaxes the layers from inputs to outputs ; if True, proceeds from outputs to inputs. Default: False.
            max_iterations (int, optional): Maximum number of iterations allowed to converge to equilibrium. Default: 100.
            adjust_control_knobs (bool, optional): if False, the values of control knobs are not computed. This is useful to save time and computations
            if the values of the control knobs are not used, as is the case for inference at test time for example. Default: True

        Returns:
            int: Number of iterations performed to converge to equilibrium
        """

        for iteration in range(max_iterations):
            self._relax_layers(backward=backward)
            if use_criterion and all([layer.has_converged() for layer in self._layers]): break  # checks if the convergence criterion is satisfied

        if adjust_control_knobs:
            for control_knob in self._control_knobs: control_knob.adjust()

        num_iterations = iteration + 1
        return num_iterations

    def clamped_relaxation(self, max_iterations=1):
        """Let the network relax to equilibrium with clamped control knobs.

        Relaxes the network's layers and parameters for a maximum of max_iterations iterations. Stops if the convergence criterion is below a threshold.

        Args:
            max_iterations (int, optional): Maximum number of iterations allowed to converge to equilibrium. Default: 1

        Returns:
            int: Number of iterations performed to converge to equilibrium
        """

        for iteration in range(max_iterations):
            self._relax_layers(backward=True)  # relaxes the layers one by one, from the output layer backward to the input layer
            self._relax_params()

        num_iterations = iteration + 1
        return num_iterations

    def gradient_step(self, step_size):
        """Runs one step of gradient descent in the state space
        
        Args:
            step_size (float32): Rate of the gradient descent.
        """

        grads = [layer.get_energy_grad() for layer in self._layers]
        for layer, grad in zip(self._layers, grads):
            new_state = layer.clamp(layer.state - step_size * grad)
            layer.violation = torch.abs(new_state - layer.state).sum()
            layer.state = new_state

    def save(self, path):
        """Saves the model parameters

        Args:
            path (str): path where to save the network's parameters
        """

        torch.save(self.params_state(), path)

    def load(self, path):
        """Loads the model parameters

        Args:
            path (str): path where to load the network's parameters from
        """

        param_states = torch.load(path, map_location=torch.device(self._device))
        for param, state in zip(self._params, param_states): param.state = state

    def _relax_layers(self, backward=False, step_size=None):
        """Starting from the current configuration of the layers, relaxes all the layers one by one.

        The order of relaxation of the layers may be `forward' or `backward': e.g. with three layers, either [0, 1, 2] (forward) or [2, 1, 0] (backward)

        Args:
            backward (bool, optional): if False, relaxes the layers from inputs to outputs ; if True, proceeds from outputs to inputs. Default: False.
        """

        if step_size:
            self.gradient_step(step_size)
        else:
            list_layers = reversed(self._layers) if backward else self._layers

            for layer in list_layers: layer.relax()

    def _relax_params(self):
        """Computes the configuration of the parameters that minimizes the energy function, given the state of the layers fixed"""

        for param in self._params: param.relax()