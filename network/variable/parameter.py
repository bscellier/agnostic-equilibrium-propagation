# import math
from abc import ABC, abstractmethod
import numpy as np
import torch

from network.variable.layer import QuadraticFloatingVariable



class Parameter(QuadraticFloatingVariable, ABC):
    """Abstract class for parameter variables

    Methods
    -------
    get_energy_grad():
        Returns the gradient of the energy wrt the parameter, i.e. dE/dtheta
    gradient_step(step_size):
        One step of gradient descent wrt the energy
    """

    def __init__(self, shape, min_interval=None, max_interval=None, threshold = 1e-3, device=None):
        """Initializes an instance of Parameter

        Args:
            shape (tuple of ints): Shape of the tensor used to represent the parameter. Type is float32.
            min_interval (float32, optional): minimum of the interval for the layer's state. Default: None
            max_interval (float32, optional): maximum of the interval for the layer's state. Default: None
            device (str, optional): Either 'cpu' or 'cuda'. Default: None.
        """

        QuadraticFloatingVariable.__init__(self, shape, min_interval, max_interval, threshold)

        self._state = torch.empty(*shape, dtype=torch.float32, device=device)

    def compute_min(self):
        """Computes the parameter value that minimizes the energy function, given the state of the layers and other parameters fixed. 
        The global energy is of the form |u-theta|^2 / 2*epsilon + E + beta C.
        We assume that C is independent of theta and that E is of the form E = a * theta^2 + b * theta + c, where theta is the parameter.
        Then, the parameter value that minimizes the global energy is theta = (u - epsilon * b) / (1 + 2 epsilon * a)
        """

        linear_coef = sum([fn() for fn in self._linear_coefficients])
        quadratic_coef = sum([fn() for fn in self._quadratic_coefficients])
        learning_rate = self.learning_rate()

        min_state = (self.control_state() - learning_rate * linear_coef) / (1. + 2. * learning_rate * quadratic_coef)
        if self._min_interval or self._max_interval: min_state = torch.clamp(min_state, min=self._min_interval, max=self._max_interval)
        return min_state



class Bias(Parameter):
    """Class for biases

    Attributes
    ----------
    shape (tuple of int): Shape of the bias Tensor. Type is float32.
    state (Tensor): Tensor of shape layer_shape representing the bias. Type is float32.

    Methods
    -------
    init_state():
        Initializes the bias
    """

    _counter = 0

    def __init__(self, *args, **kwargs):
        """Initializes an instance of Bias

        Args:
            shape (tuple of ints): Shape of the bias Tensor. Type is float32.
        """

        Parameter.__init__(self, *args, **kwargs)

        self.init_state()

        self.name = 'Bias_{}'.format(Bias._counter)

        Bias._counter += 1

    def init_state(self):
        """Initializes the bias tensor to zero, i.e. b=0."""

        # TODO: implement recommended initialization schemes for biases, instead of zero

        torch.nn.init.constant_(self._state, 0.)



class DenseWeight(Parameter):
    """Class for dense ('fully connected') weights

    Attributes
    ----------
    shape (tuple of ints): shape of the weight tensor
    state (Tensor): Tensor of shape weight_shape representing the dense weights. Type is float32.

    Methods
    -------
    init_state(shape, gain, device=None):
        Initializes the weight tensor
    """

    _counter = 0

    def __init__(self, layer_pre_shape, layer_post_shape, gain, **kwargs):
        """Initializes an instance of DenseWeight

        Args:
            layer_pre_shape (tuple of ints): shape of the pre-synaptic layer
            layer_post_shape (tuple of ints): shape of the post-synaptic layer
            gain (float32): Number used to scale the weight tensor (~ proportional to the standard deviations of the weight)
        """

        shape = layer_pre_shape + layer_post_shape
        Parameter.__init__(self, shape, **kwargs)

        self._layer_pre_shape = layer_pre_shape
        self._layer_post_shape = layer_post_shape

        self.init_state(gain)

        self.name = 'DenseWeight_{}'.format(DenseWeight._counter)

        DenseWeight._counter += 1

    def init_state(self, gain, mode='xavier_uniform'):
        """Initializes the weight tensor according to a uniform or normal distribution.

        Args:
            gain (float32): Number used to scale the weight tensor (~ proportional to the standard deviations of the weight)
            mode (str, optional): method to initialize the weight tensor. Either 'xavier_uniform', 'xavier_normal', 'kaiming_uniform' or 'kaiming_normal'. Default: 'xavier_uniform'.
        """

        size_pre = 1
        for dim in self._layer_pre_shape: size_pre *= dim
        size_post = 1
        for dim in self._layer_post_shape: size_post *= dim
        
        if mode == 'xavier_uniform':
            # half xavier uniform
            scale = gain * 0.5 * np.sqrt(6. / (size_pre + size_post))
            torch.nn.init.uniform_(self._state, -scale, +scale)
        elif mode == 'xavier_normal':
            # half xavier normal
            scale = gain * 0.5 * np.sqrt(2. / (size_pre + size_post))
            torch.nn.init.normal_(self._state, std=scale)
        elif mode == 'kaiming_uniform':
            # half kaiming uniform
            scale = gain * 0.5 * np.sqrt(3. / size_pre)
            torch.nn.init.uniform_(self._state, -scale, +scale)
        else:  #  mode == 'kaiming_normal'
            # half kaiming normal
            scale = gain * 0.5 * np.sqrt(1. / size_pre)
            torch.nn.init.normal_(self._state, std=scale)



class ConvWeight(Parameter):
    """Class for convolutional weights

    Attributes
    ----------
    shape (tuple of ints): shape of the weight tensor. Shape is (out_channels, in_channels, height, width).
    state (Tensor): Tensor of shape weight_shape representing the convolutional weights. Type is float32.

    Methods
    -------
    init_state(shape, gain, device=None):
        Initializes the weight tensor
    """

    _counter = 0

    def __init__(self, shape, gain, **kwargs):
        """Initializes an instance of ConvWeight

        Args:
            shape (tuple of ints): shape of the convolutional weight tensor. Shape is (out_channels, in_channels, height, width).
            gain (float32): Number used to scale the weight tensor (~ proportional to the standard deviations of the weight)
        """

        Parameter.__init__(self, shape, **kwargs)

        self.init_state(gain)

        self.name = 'ConvWeight_{}'.format(ConvWeight._counter)

        ConvWeight._counter += 1

    def init_state(self, gain, mode='kaiming_normal'):
        """Initializes the weight tensor.

        Args:
            gain (float32): Number used to scale the weight tensor (~ proportional to the standard deviations of the weight)
            mode (str, optional): method to initialize the weight tensor. Either 'xavier_uniform', 'xavier_normal', 'kaiming_uniform' or 'kaiming_normal'. Default: 'kaiming_normal'.
        """

        (channels_out, channels_in, width, height) = self._shape
        size_pre = channels_in * width * height
        size_post = channels_out
        
        if mode == 'xavier_uniform':
            # half xavier uniform
            scale = gain * 0.5 * np.sqrt(6. / (size_pre + size_post))
            torch.nn.init.uniform_(self._state, -scale, +scale)
        elif mode == 'xavier_normal':
            # half xavier normal
            scale = gain * 0.5 * np.sqrt(2. / (size_pre + size_post))
            torch.nn.init.normal_(self._state, std=scale)
        elif mode == 'kaiming_uniform':
            # half kaiming uniform
            scale = gain * 0.5 * np.sqrt(3. / size_pre)
            torch.nn.init.uniform_(self._state, -scale, +scale)
        else:  #  mode == 'kaiming_normal'
            # half kaiming normal
            scale = gain * 0.5 * np.sqrt(1. / size_pre)
            torch.nn.init.normal_(self._state, std=scale)