from abc import ABC, abstractmethod
import torch



class FloatingVariable(ABC):
    """
    Abstract class for floating variables (layers and parameters).

    Attributes
    ----------
    shape (tuple of int): shape of the tensor used to represent the state of the variable
    state (Tensor): the state of the variable
    min_interval (flaot32): minimum of the interval for the variable's state
    max_interval (float32): maximum of the interval for the variable's state
    threshold (float32): Threshold value used as a criterion of convergence to equilibrium
    violation (float32): L1-norm between the variable's previous state and the new state
    energy_fns (list of functions): list of functions returning the energy terms of the variable's interactions
    grad_fns (list of functions): list of functions returning the energy gradients of the variable's interactions

    Methods
    -------
    add_interaction(interaction, linear_fn, quadratic_fn):
        adds an interaction the variable is involved in
    clamp(state):
        Returns the value of the variable's state clamped between min_interval and max_interval
    get_energy_grad():
        Gradient of the energy wrt the variable, i.e. dE/dz, where z is the variable
    has_converged():
        Checks if the convergence criterion is satisfied
    init_state()
        Initializes the state of the variable
    """

    def __init__(self, shape, min_interval, max_interval, threshold):
        """Initializes an instance of FloatingVariable

        Args:
            shape (tuple of int): shape of the tensor used to represent the variable's state
            min_interval (float32): minimum of the interval for the variable's state
            max_interval (float32): maximum of the interval for the variable's state
            threshold (float32): Threshold value used as a criterion of convergence to equilibrium
        """

        self._shape = shape
        self._min_interval = min_interval
        self._max_interval = max_interval
        self._threshold = threshold
        self._violation = threshold + 1.  # FIXME

        self._energy_fns = []
        self._grad_fns = []

    @property
    def min_interval(self):
        """Gets the minimum of the interval for the variable's state"""

        return self._min_interval

    @property
    def max_interval(self):
        """Gets the minimum of the interval for the variable's state"""

        return self._max_interval

    @property
    def shape(self):
        """Gets the shape of the variable (the tensor)"""

        return self._shape

    @property
    def state(self):
        """Gets and sets the current state of the variable"""

        return self._state

    @state.setter
    def state(self, state):
        self._state = state

    @property
    def threshold(self):
        """Gets and sets the threshold value of the variable"""

        return self._threshold

    @threshold.setter
    def threshold(self, threshold):
        # TODO: raise a ValueError if invalid argument
        if isinstance(threshold, float) and threshold > 0.:  # checks that the threshold entered is a positive real number
            self._threshold = threshold

    @property
    def violation(self):
        """Gets the violation of equilibrium of the last relaxation phase"""

        return self._violation

    @violation.setter
    def violation(self, violation):
        self._violation = violation

    def add_interaction(self, energy_fn, grad_fn):
        """Adds an interaction to the variable.

        Args:
            interaction (Interaction): the interaction to be added
        """

        # TODO: the method should check that the variable belongs to the interaction

        self._energy_fns.append(energy_fn)
        self._grad_fns.append(grad_fn)

    def has_converged(self):
        """Checks if the variable's state is at equilibrium, i.e. if the convergence criterion is satisfied

        Returns:
            bool: True if the violation is smaller than the variable's threshold
        """

        return self._violation < self._threshold

    @abstractmethod
    def init_state(self):
        """Initializes the variable"""
        pass

    def get_energy_grad(self):
        """Gradient of the energy wrt the variable, i.e. dE/dz, where z is the variable"""

        return sum([grad_fn() for grad_fn in self._grad_fns])

    def clamp(self, state):
        """Returns the value of the variable's state clamped between min_interval and max_interval"""

        return torch.clamp(state, min=self._min_interval, max=self._max_interval)

    def _local_energy(self):
        """Returns the part of the energy the variable is involved in"""

        return sum([energy_fn().mean() for energy_fn in self._energy_fns])


class QuadraticFloatingVariable(FloatingVariable, ABC):
    """Abstract class for floating variables whose local energy is quadratic.

    Attributes
    ----------
    _linear_coefficients (list of functions): list of functions returning the linear coefficients of the variable's interactions
    _quadratic_coefficients (list of functions): list of functions returning the quadratic coefficients of the variable's interactions

    Methods
    -------
    add_interaction(energy_fn, linear_fn, quadratic_fn):
        adds an interaction the variable is involved in
    compute_min():
        Computes the variable's configuration that minimizes the global energy, given the state of other variables fixed
    relax():
        Updates the state of the variable. Computes the configuration of the variable that minimizes the energy, given the state of other variables.
    get_energy_grad():
        Gradient of the energy wrt the variable
    """

    def __init__(self, shape, min_interval, max_interval, threshold):
        """Initializes an instance of Layer

        Args:
            shape (tuple of int): shape of the tensor used to represent the variable's state
            min_interval (float32): minimum of the interval for the variable's state
            max_interval (float32): maximum of the interval for the variable's state
            threshold (float32): Threshold value used as a criterion of convergence to equilibrium
        """

        self._linear_coefficients = []
        self._quadratic_coefficients = []

        FloatingVariable.__init__(self, shape, min_interval, max_interval, threshold)

    def add_interaction(self, energy_fn, linear_fn=None, quadratic_fn=None):
        """Adds an interaction to the state variable.

        Overrides the method of the class FloatingVariable

        Args:
            energy_fn (fn): energy function of the interaction added
            linear_fn (fn): linear coef function of the interaction added
            quadratic_fn (fn): quadratic coef function of the interaction added
        """

        # TODO: the method should check that the variable belongs to the interaction

        self._energy_fns.append(energy_fn)
        if linear_fn: self._linear_coefficients.append(linear_fn)
        if quadratic_fn: self._quadratic_coefficients.append(quadratic_fn)

    @abstractmethod
    def compute_min(self):
        """Computes the configuration of the variable that minimizes the global energy of the network, given the state of other variables fixed.

        Returns:
            Tensor: the configuration of the variable that minimizes the energy. Type is float32.
        """
        pass

    def relax(self):
        """Updates the state of the variable. Computes the configuration of the variable that minimizes the energy, given the state of other variables."""

        old_state = self._state
        self._state = self.compute_min()
        self._violation = torch.abs(old_state - self._state).sum()

    def get_energy_grad(self):
        """Gradient of the energy wrt the variable, i.e. dE/dz, where z is the variable

        Overrides the method of the class FloatingVariable

        We assume that E as a function of z is of the form E(z) = a * z^2 + b * z + c.
        Then, dE/dz = 2 a * z + b

        Returns:
            Tensor of shape variable_shape. Type is float32
        """

        linear_coef = sum([fn() for fn in self._linear_coefficients])
        quadratic_coef = sum([fn() for fn in self._quadratic_coefficients])
        energy_grad = 2. * quadratic_coef * self._state + linear_coef
        return energy_grad



class Layer(QuadraticFloatingVariable):
    """
    Class used to implement a layer of a network.

    The units of the layer are independent (i.e. not connected).

    Attributes
    ----------
    _counter (int): the number of Layers instanciated so far
    name (str): the layer's name (used e.g. to identify the layer in tensorboard)
    """

    _counter = 0

    def __init__(self, shape, min_interval = 0., max_interval = 1., threshold = 1e-3, batch_size=1, device=None):
        """Initializes an instance of Layer

        Args:
            shape (tuple of int): shape of the tensor used to represent the state of the layer
            min_interval (float32, optional): minimum of the interval for the layer's state. Default: 0.
            max_interval (float32, optional): maximum of the interval for the layer's state. Default: 1.
            threshold (float32, optional): Threshold value used as a criterion of convergence to equilibrium. Default: 1e-3
            batch_size (int, optional): the size of the current batch processed. Default: 1
            device (str, optional): the device on which to run the layer's tensor. Either `cuda' or `cpu'. Default: None
        """

        QuadraticFloatingVariable.__init__(self, shape, min_interval, max_interval, threshold)

        self.init_state(batch_size, device)

        self.name = 'Layer_{}'.format(Layer._counter)

        Layer._counter += 1

    def compute_min(self, without_clamping=False):
        """Computes the configuration of the layer that minimizes the global energy of the network, given the state of other variables fixed.

        It is assumed that all interactions the layer is involved in are quadratic in the layer's state, i.e. the interaction's energy is of the form
        E_i(z) = a_i z^2 + b_i z + c_i, where z is the layer's state.
        Thus, the global energy of the network is also quadratic in the layer's state, i.e. of the form E(z) = a * z^2 + b * z + c,
        with coefficients a = sum_i a_i and b = sum_i b_i.
        The minimum of this quadractic function is obtained at the point z = - a / 2*b

        Args:
            without_clamping (bool, optional): if True, computes the minimum in R^d. If False, computes the minimum in [min_interval, max_interval]^d. Default: False.

        Returns:
            Tensor: the configuration of the layer that minimizes the energy. Shape is (batch_size, layer_shape). Type is float32.
        """

        linear_coef = sum([fn() for fn in self._linear_coefficients])
        quadratic_coef = sum([fn() for fn in self._quadratic_coefficients])

        min_state = - linear_coef / (2. * quadratic_coef)  # At this point, min_state is the configuration that minimizes the energy in R^d, where d is the dimension of the layer.
        if not without_clamping: min_state = torch.clamp(min_state, min=self._min_interval, max=self._max_interval)  # Now, min_state achieves the minimum of the energy in [min, max]^d

        return min_state

    def init_state(self, batch_size, device):
        """Initializes the state of the layer to zero

        Args:
            batch_size (int): size of the mini-batch of examples
            device (str): Either 'cpu' or 'cuda'
        """

        shape = (batch_size,) + self._shape
        self._state = torch.zeros(shape, requires_grad=False, device=device)