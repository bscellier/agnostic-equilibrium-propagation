from abc import ABC, abstractmethod
import sys
import torch



class TrainingProcedure(ABC):
    """
    Abstract class for performing inference and SGD (stochastic gradient descent)

    Methods
    -------
    inference()
        Let the network relax to equilibrium with nudging=0 for the current mini-batch of examples
    training_step()
        Performs one step of SGD on the current mini-batch
    """

    @abstractmethod
    def inference(self):
        """Performs inference, i.e. let the network relax to equilibrium with nudging=0 for the current mini-batch of examples

        Returns:
            int: the number of iterations performed to converge to equilibrium
        """
        pass

    @abstractmethod
    def training_step(self):
        """Performs one step of stochastic gradient descent on the current mini-batch"""
        pass



class Aeqprop(TrainingProcedure):
    """
    Class used to perform SGD (stochastic gradient descent) via Agnostic Equilibrium Propagation (Aeqprop)

    Attributes
    ----------
    network (Network): the network to train
    training_mode (str): either Optimistic Aeqprop, Centered Aeqprop, or Pessimistic Aeqprop
    nudging (float): the nudging value used to train via Aeqprop
    max_iterations_first_phase (int): the maximum number of iterations allowed to converge to equilibrium in the first phase of Aeqprop
    max_iterations_second_phase (int): the maximum number of iterations allowed to converge to equilibrium in the second phase of Aeqprop
    max_iterations_params (int): the maximum number of iterations allowed for the parameters to converge to equilibrium in the second phase of Aeqprop

    Methods
    -------
    inference()
        Let the network relax to equilibrium with nudging=0 for the current mini-batch of examples
    training_step()
        Performs one step of SGD via Aeqprop on the current mini-batch
    """

    def __init__(self, network, training_mode='centered', nudging=0.2, max_iterations_first_phase=100, max_iterations_second_phase=100, max_iterations_params=1):
        """Creates an instance of Aeqprop

        Args:
            network (Network): the network to optimize via Aeqprop
            training_mode (str, optional): either Optimistic Aeqprop, Centered Aeqprop, or Pessimistic Aeqprop. Default: 'centered'
            nudging (float, optional): the nudging value used to train via Aeqprop. Default: 0.2
            max_iterations_first_phase (int, optional): the maximum number of iterations allowed to converge to equilibrium in the first phase of Aeqprop. Default: 100
            max_iterations_second_phase (int, optional): the maximum number of iterations allowed to converge to equilibrium in the second phase of Aeqprop. Default: 100
            max_iterations_params (int, optional): the maximum number of iterations allowed for the parameters to converge to equilibrium in the second phase of Aeqprop. Default: 1
        """

        self._network = network

        self._nudging = nudging
        self._training_mode = training_mode
        self._set_nudgings()

        self.max_iterations_first_phase = max_iterations_first_phase
        self.max_iterations_second_phase = max_iterations_second_phase
        self.max_iterations_params = max_iterations_params

    @property
    def nudging(self):
        """Get and sets the nudging value used for training"""

        return self._nudging

    @nudging.setter
    def nudging(self, nudging):
        # TODO: raise an error if nudging is not a positive float

        self._nudging = nudging
        self._set_nudgings()

    @property
    def training_mode(self):
        """Get and sets the training mode"""

        return self._training_mode

    @training_mode.setter
    def training_mode(self, training_mode):
        # TODO: raise an error if training_mode is neither 'optimistic', 'pessimistic' or 'centered'

        self._training_mode = training_mode
        self._set_nudgings()

    def inference(self):
        """Performs inference, i.e. let the network relax to equilibrium with nudging=0 for the current mini-batch of examples

        Returns:
            int: the number of iterations performed to converge to equilibrium
        """

        self._network.set_nudging(0.)
        num_iterations = self._network.homeostatic_relaxation(backward=False, max_iterations=self.max_iterations_first_phase, adjust_control_knobs=False)

        return num_iterations

    def training_step(self):
        """Performs one step of stochastic gradient descent on the current mini-batch

        Training step depends on the attributes training_mode (Optimistic Aeqprop, Pessimistic Aeqprop, Centered Aeqprop) and nudging
        Tracks how much the variables change between the two phases

        Returns:
            list of float: the L1-norm between the first equilibrium configuration and the second equilibrium configuration
            list of float: the L1-norm between the first equilibrium params and the second equilibrium params
        """

        free_state = self._network.layers_state  # hack: we store the `free state' (i.e. the equilibrium state of the layers with nudging=0)

        # First phase (homeostatic phase)
        self._network.set_nudging(self._first_nudging)
        self._network.homeostatic_relaxation(backward=True, max_iterations=self.max_iterations_first_phase, adjust_control_knobs=True)
        layers = self._network.layers_state
        params = self._network.params_state()
        
        # Second phase (clamped phase)
        self._network.set_nudging(self._second_nudging)
        self._network.layers_state = free_state  # hack: we start the second phase from the `free state' again
        self._network.homeostatic_relaxation(backward=True, max_iterations=self.max_iterations_second_phase, adjust_control_knobs=False)  # first, we let the layers settle to equilibrium
        self._network.clamped_relaxation(max_iterations=self.max_iterations_params)  # then, we let the parameters (together with the layers) settle to equilibrium
        layers_new = self._network.layers_state
        params_new = self._network.params_state()

        # Monitor by how much the variables change between the two phases
        layers_change = [torch.abs(layer-layer_new).sum().item() for layer, layer_new in zip(layers, layers_new)]
        params_change = [torch.abs(param-param_new).sum().item() for param, param_new in zip(params, params_new)]

        return layers_change, params_change

    def _set_nudgings(self):
        """Sets the values of first_nudging and second_nudging, depending on the attributes training_mode and nudging

        first_nudging: nudging value in the first phase (homeostatic phase)
        second_nudging: nudging value in the second phase (clamped phase)
        """

        if self._training_mode == "optimistic":
            self._first_nudging = 0.
            self._second_nudging = self._nudging
        elif self._training_mode == "pessimistic":
            self._first_nudging = -self._nudging
            self._second_nudging = 0.
        else:  # self._training_mode == "centered"
            self._first_nudging = -self._nudging / 2.
            self._second_nudging = self._nudging / 2.



class AutoDiff(TrainingProcedure):
    """
    Class used to perform SGD (stochastic gradient descent) via automatic differentiation (AutoDiff)

    Attributes
    ----------
    network (Network): the network to train
    lr_scaling (float): an optional scaling number for the learning rates (to compensate for the nudging beta that scales the learning rates in Aeqprop)
    param_decay (float): 
    max_iterations (int): the maximum number of iterations allowed to converge to equilibrium (with nudging=0)

    Methods
    -------
    inference()
        Let the network relax to equilibrium with nudging=0 for the current mini-batch of examples
    training_step()
        Performs one step of SGD via automatic differentiation on the current mini-batch
    """

    def __init__(self, network, lr_scaling=1.0, param_decay=0., max_iterations=100):
        """Creates an instance of AutoDiff

        Args:
            network (Network): the network to train
            lr_scaling (float, optional): an optional scaling number for the learning rates. Default: 1.
            param_decay (float, optional): Default: 0.
            max_iterations (int, optional): the maximum number of iterations allowed to converge to equilibrium. Default: 100
        """

        self._network = network

        self.lr_scaling = lr_scaling
        self.param_decay = param_decay
        self.max_iterations = max_iterations

    def inference(self):
        """Performs inference, i.e. let the network relax to equilibrium with nudging=0 for the current mini-batch of examples

        Returns:
            int: the number of iterations performed to converge to equilibrium
        """

        for param in self._network.params(): param.state.requires_grad = True  # FIXME: this should be set to False at the end of this method

        self._network.init_layers()
        self._network.set_nudging(0.)
        num_iterations = self._network.homeostatic_relaxation(backward=False, max_iterations=self.max_iterations, use_criterion=True, adjust_control_knobs=False)

        return num_iterations

    def training_step(self):
        """Performs one step of SGD via automatic differentiation on the current mini-batch"""

        cost_mean = self._network.cost_fn().mean()

        param_grads = torch.autograd.grad(cost_mean, [param.state for param in self._network.params()])
        for param, lr, grad in zip(self._network.params(), self._network.learning_rates, param_grads):
            learning_rate = self.lr_scaling * lr
            param.state = (1.-self.param_decay) * param.state - learning_rate * grad

        for param in self._network.params(): param.state = param.state.detach()

        # TODO: the variables returned by the method are useless
        layers_change = [0. for _ in range(self._network.num_layers)]
        params_change = [0. for _ in range(self._network.num_params)]

        return layers_change, params_change