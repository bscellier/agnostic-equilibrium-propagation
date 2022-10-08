from abc import ABC, abstractmethod
import torch
import torch.nn.functional as F

from network.interaction import MultiQuadraticInteraction



class CostInteraction(MultiQuadraticInteraction, ABC):
    """Abstract class for cost interactions

    Attributes
    ----------
    nudging (float32): Nudging value

    Methods
    -------
    cost_fn():
        Returns the cost value of the current configuration
    energy_fn():
        Returns the cost interaction's energy, which is energy = nudging * cost
    """

    def __init__(self, *variables):
        """Constructor of the CostInteraction class"""

        self._nudging = 0.

        MultiQuadraticInteraction.__init__(self, *variables)

    @property
    def nudging(self):
        """Gets and sets the nudging value"""

        return self._nudging

    @nudging.setter
    def nudging(self, nudging):
        if isinstance(nudging, float):  # checks that the nudging value entered is a real number
            self._nudging = nudging
        else: raise ValueError('nudging must be a float')

    @abstractmethod
    def cost_fn(self):
        """Returns the value of the cost function evaluated at the current state

        Returns:
            Vector of size (batch_size,) and of type float32. Each coordinate is the cost value of an example in the current mini-batch
        """
        pass

    def energy_fn(self):
        """Energy of the interaction. It is equal to nudging * cost

        Returns:
            Vector of size (batch_size,) and of type float32. Each value is the energy term of an example in the current mini-batch
        """

        return self._nudging * self.cost_fn()



class SquaredError(CostInteraction):
    """Class for the squared error cost function, between the output layer and the target layer

    Attributes
    ----------
    _output_layer (Layer): the output layer
    _target_layer (Layer): the target layer
    nudging (float32): Nudging value

    Methods
    -------
    cost_fn():
        Returns the squared error between the output layer and the target
    energy_fn():
        Returns the interaction's energy
    """

    def __init__(self, output_layer, target_layer):
        """Initializes an instance of SquaredError

        Args:
            output_layer (Layer): the output layer
            target_layer (Layer): the target layer
        """

        self._output_layer = output_layer
        self._target_layer = target_layer
        self._nudging = 0.

        CostInteraction.__init__(self, output_layer)

    def cost_fn(self):
        """Returns the cost value (the squared error) of the current state configuration.

        Returns:
            Tensor of shape (batch_size,) and type float32. Vector of cost values for each of the examples in the current mini-batch
        """

        output = self._output_layer.state  # state of output layer: shape is (batch_size, num_classes)
        target = self._target_layer.state  # desired output: shape is (batch_size, num_classes)
        return 0.5 * ((output - target) ** 2).sum(dim=1)  # Vector of shape (batch_size,)

    def linear_fns(self):
        """Returns the linear_coef function for the output layer"""
        return [self._linear_coef_output]

    def quadratic_fns(self):
        """Returns the quadratic_coef function for the output layer"""
        return [self._quadratic_coef_output]

    def _linear_coef_output(self):
        """Returns the linear coefficient of the interaction.

        The energy of the interaction is quadratic in the layer's state, i.e. of the form E_i(s) = alpha_i s^2 + beta_i s + gamma_i
        The quantity that is returned is beta_i

        Returns:
            Tensor of shape (batch_size, layer_shape) and type float32: the linear contribution
        """

        return - self._nudging * self._target_layer.state

    def _quadratic_coef_output(self):
        """Returns the quadratic coefficient of the interaction.

        The energy of the interaction is quadratic in the layer's state, i.e. of the form E_i(s) = alpha_i s^2 + beta_i s + gamma_i
        The quantity that is returned is alpha_i

        Returns:
            float32: the quadratic contribution
        """

        return 0.5 * self._nudging



class LinearizedError(CostInteraction):
    """Class for the squared error cost function between the output layer and the target layer, with linearized nudging.

    Attributes
    ----------
    _output_layer (Layer): the output layer
    _target_layer (Layer): the target layer
    nudging (float32): Nudging value

    Methods
    -------
    cost_fn():
        Returns the squared error between the output layer and the target
    energy_fn():
        Returns the nudging interaction's energy
    """

    # FIXME: some hacks due to the saved free state that may not have the same batch size as the current mini-batch (only occurs when nudging=0)

    def __init__(self, output_layer, target_layer):
        """Initializes an instance of LinearizedError

        Args:
            output_layer (Layer): the output layer
            target_layer (Layer): the target layer
        """

        self._output_layer = output_layer
        self._target_layer = target_layer
        self._nudging = 0.

        self._free_state = output_layer.state

        CostInteraction.__init__(self, output_layer)

    @property
    def nudging(self):
        """Gets and sets the nudging value"""

        return self._nudging

    @nudging.setter
    def nudging(self, nudging):
        if isinstance(nudging, float):  # checks that the nudging value entered is a real number
            if self._nudging == 0.: self._free_state = self._output_layer.state  # we save the free equilibrium state of output units 
            self._nudging = nudging
        else: raise ValueError('nudging must be a float')

    def cost_fn(self):
        """Returns the cost value (the squared error) of the current state configuration.

        Returns:
            Tensor of shape (batch_size,) and type float32. Vector of cost values for each of the examples in the current mini-batch
        """

        output = self._output_layer.state  # state of output layer: shape is (batch_size, num_classes)
        target = self._target_layer.state  # desired output: shape is (batch_size, num_classes)
        return 0.5 * ((output - target) ** 2).sum(dim=1)  # Vector of shape (batch_size,)

    def energy_fn(self):
        """Energy of the interaction. It is equal to nudging * cost

        Returns:
            Vector of size (batch_size,) and of type float32. Each value is the energy term of an example in the current mini-batch
        """

        output = self._output_layer.state
        target = self._target_layer.state
        free_state = self._free_state
        nudging = self._nudging

        if nudging == 0.: return torch.zeros(output.shape[0], device=output.device)
        else: return nudging * ((free_state - target) * (output - 0.5 * free_state - 0.5 * target)).sum(dim=1)

    def linear_fns(self):
        """Returns the linear_coef function for the output layer"""
        return [self._linear_coef_output]

    def quadratic_fns(self):
        """Returns the quadratic_coef function for the output layer"""
        return [None]

    def _linear_coef_output(self):
        """Returns the linear coefficient of the interaction.

        The energy of the interaction is quadratic in the layer's state, i.e. of the form E_i(s) = alpha_i s^2 + beta_i s + gamma_i
        The quantity that is returned is beta_i

        Returns:
            Tensor of shape (batch_size, layer_shape) and type float32: the linear contribution
        """

        nudging = self._nudging

        return nudging * (self._free_state - self._target_layer.state) if nudging else 0. * self._target_layer.state  # FIXME


class PenaltyCost(CostInteraction):
    """Class for penalty interaction terms. This is similar to weight decay.

    Attributes
    ----------
    _variable (FloatingVariable): the variable on which we apply a quadratic penalty
    _penalty (float32): the strength of the penalty
    _decay (float32): the decay

    Methods
    -------
    energy_fn():
        Returns the penalty's energy
    """

    def __init__(self, variable, penalty = 0.5, decay=1e-4):
        """Initializes an instance of PenaltyCost`

        Args:
            variable (FloatingVariable): the variable that has a quadratic penalty
            penalty (float32, optional): the strength of the penalty. Default: 0.5
            decay (float32, optional): the decay. Default: 1e-4
        """

        self._variable = variable
        self._penalty = penalty
        self._decay = decay

        CostInteraction.__init__(self, variable)

    def cost_fn(self):
        """Returns the variable's energy term"""

        return (self._penalty + self._nudging * self._decay) * (self._variable.state ** 2).sum()

    def linear_fns(self):
        """Returns the linear_coef function for the variable"""
        return [None]

    def quadratic_fns(self):
        """Returns the quadratic_coef function for the variable"""
        return [self._quadratic_coef]

    def _quadratic_coef(self):
        """Returns the quadratic coefficient of the interaction"""

        return self._penalty + self._nudging * self._decay