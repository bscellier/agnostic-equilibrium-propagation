import torch



class ControlKnob:
    """Class used to represent control knobs

    Each parameter of the network is linked to a control knob through a quadratic interaction.

    Attributes
    ----------
    shape (tuple of int): shape of the tensor used to represent the control knob
    state (Tensor): the clamped control values. Shape is the same as the weight tensor. Type is float32.
    param (Parameter): the parameter associated to the control knob

    Methods
    -------
    adjust_control_knob():
        Computes the value of the control knobs for which the parameter is at equilibrium
    """

    def __init__(self, param):
        """Creates an instance of ControlKnob.

        Args:
            param (Parameter): the parameter associated to the control knob
        """

        self._shape = param.shape

        self._param = param

        param.control_knob = self  # FIXME: hack

        self._state = self.init_state()

    @property
    def shape(self):
        """Gets the shape of the state variable (the tensor)"""

        return self._shape

    @property
    def state(self):
        """Gets and sets the value of the control knob.

        Returns:
            state (Tensor): Shape is param_shape. Type is float32.
        """

        return self._state

    @state.setter
    def state(self, state):
        self._state = state

    def init_state(self):
        """Initializes the control knob tensor to zero.
            
        Returns:
            Tensor of type float32. Control knob tensor.
        """

        return torch.zeros(*self._shape)

    def adjust(self):
        """Computes the value of the control knob for which the corresponding parameter is at equilibrium.

        The equilibrium value of the parameter is characterized by the first-order condition (theta-u)/epsilon + dE/dtheta = 0
        Therefore u = theta + epsilon * dE/dtheta
        """

        grad = self._param.get_energy_grad()
        self._state = self._param.state + self.learning_rate() * grad
