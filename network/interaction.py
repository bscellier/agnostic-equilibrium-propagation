from abc import ABC, abstractmethod
import torch
import torch.nn.functional as F



class Interaction(ABC):
    """Abstract class for interactions

    An interaction between variables is defined by its energy function

    Methods
    -------
    energy_fn():
        Returns the interaction's energy
    energy_grad_fns():
        Returns the list of gradient functions wrt the variables of the interaction
    """

    def __init__(self, *variables):
        """Constructor of Interaction

        Args:
            variables (FloatingVariable): the variables involved in the multi-quadratic interaction
        """

        for variable, grad_fn in zip(variables, self.energy_grad_fns()):
            variable.add_interaction(self.energy_fn, grad_fn)

        self._variables = variables

    @abstractmethod
    def energy_fn(self):
        """Returns the interaction's energy

        Returns:
            Vector of size (batch_size,) and of type float32. Each value is the interaction's energy term of an example in the current mini-batch
        """
        pass

    def energy_grad_fns(self):
        """Returns the list of gradient functions wrt the variables

        Returns:
            List of functions. Each function returns the energy gradient of the corresponding variable
        """

        return [self._energy_grad(variable) for variable in self._variables]


    def _energy_grad(self, variable):
        """Returns the variable's gradient wrt the energy function

        Default implementation, valid for any variable and any energy function

        Returns:
            Tensor of shape (batch_size, variable_shape) and type float32: the gradient of the energy function wrt the variable
        """

        variable.state.requires_grad = True

        energy_mean = torch.mean( self.energy_fn() )
        energy_grad = torch.autograd.grad(energy_mean, variable.state)[0]

        variable.state.requires_grad = False

        return energy_grad



class MultiQuadraticInteraction(Interaction, ABC):
    """Abstract class for multi-quadratic interactions

    A 'multi-quadratic' interaction is an interaction such that for each variable z involved in the interaction,
    the interaction's energy as a function of z is quadratic, i.e. of the form
    E(z) = a z^2 + b z + c
    for some coefficients a, b and c. We call a the 'quadratic coefficient' of z, and b is the 'linear coefficient' of z.

    Methods
    -------
    energy_fn():
        Returns the interaction's energy
    linear_fns():
        Returns the list of linear_coef functions for each of the variables
    quadratic_fns():
        Returns the list of quadratic_coef functions for each of the variables
    """

    def __init__(self, *variables):
        """Constructor of MultiQuadraticInteraction

        Args:
            variables (list of FloatingVariable): the variables involved in the multi-quadratic interaction
        """

        for variable, linear_fn, quadratic_fn in zip(variables, self.linear_fns(), self.quadratic_fns()):
            variable.add_interaction(self.energy_fn, linear_fn, quadratic_fn)

    @abstractmethod
    def linear_fns(self):
        """Returns the list of linear_coef functions for each of the variables

        The energy of the interaction is quadratic in each variable, i.e. of the form E(z) = a z^2 + b z + c
        The linear coefficient of z is b

        Returns:
            List of functions. Each function returns the linear coefficient of the corresponding variable
        """
        pass

    @abstractmethod
    def quadratic_fns(self):
        """Returns the list of quadratic_coef functions for each of the variables

        The energy of the interaction is quadratic in each variable, i.e. of the form E(z) = a z^2 + b z + c
        The linear coefficient of z is a

        Returns:
            List of functions. Each function returns the quadratic coefficient of the corresponding variable
        """
        pass



class Penalty(MultiQuadraticInteraction):
    """Class for penalty interaction terms

    The energy of a penalty is of the form E = coef * ||variable||^2

    Attributes
    ----------
    _variable (FloatingVariable): the variable on which we apply a quadratic penalty
    _coef (flaot32): the strength of the penalty

    Methods
    -------
    energy_fn():
        Returns the penalty's energy
    """

    def __init__(self, variable, coef=0.5):
        """Initializes an instance of Penalty

        Args:
            variable (FloatingVariable): the variable that has a quadratic penalty
            coef (float32, optional): the strength of the penalty. Default: 0.5
        """

        self._variable = variable
        self._coef = coef

        MultiQuadraticInteraction.__init__(self, variable)

    def energy_fn(self):
        """Returns the variable's energy term"""
        return self._coef * (self._variable.state ** 2).flatten(start_dim=1).sum(dim=1)

    def linear_fns(self):
        """Returns the linear function of the variable"""
        return [None]

    def quadratic_fns(self):
        """Returns the quadratic function of the variable"""
        return [lambda: self._coef]


class ControlInteraction(Interaction):
    """Class for control interactions

    A control interaction is defined between a parameter (theta) and its corresponding control knob (u).
    The energy term is of the form U = |u-theta|^2 / 2*eps

    Attributes
    ----------
    _control_knob (ControlKnob): the control knob coupled to the parameter
    _param (Parameter): the controled parameter
    learning_rate (float32): inverse of the coupling factor between the parameter and the control knob, i.e. eps in U = |u-theta|^2 / 2 * eps

    Methods
    -------
    energy_fn():
        Returns the energy term of the interaction, which is U = |u-theta|^2 / 2*eps
    """

    def __init__(self, control_knob, param, learning_rate=1e-2):
        """Initializes an instance of ControlInteraction

        Args:
            control_knob (ControlKnob): the control knob coupled to the parameter
            param (Parameter): the controled parameter
            learning_rate (float32, optional): inverse of the coupling factor between the parameter and the control knob. Default: 1e-2
        """

        self._control_knob = control_knob
        self._param = param
        self._learning_rate = learning_rate

        control_knob.learning_rate = lambda : self._learning_rate
        param.learning_rate = lambda : self._learning_rate
        param.control_state = lambda : control_knob.state

    @property
    def learning_rate(self):
        """Gets and sets the learning rate"""

        return self._learning_rate

    @learning_rate.setter
    def learning_rate(self, learning_rate):
        if isinstance(learning_rate, float) and learning_rate >= 0.:  # checks that the learning rate entered is a non-negative real number
            self._learning_rate = learning_rate
        else: raise ValueError('learning_rate must be a non-negative float')

    def energy_fn(self):
        """Energy of the control interaction. It is equal to ||u-theta||^2 / 2*epsilon

        Returns:
            Vector of size (batch_size,) and of type float32. Each value is the energy term of an example in the current mini-batch
        """

        control = self._control_knob.state  # state of control knob: shape is param_shape
        param = self._param.state  # state of param: shape is param_shape
        return ((control - param) ** 2).sum() / (2. * self._learning_rate)


class MultiLinearInteraction(MultiQuadraticInteraction, ABC):
    """Abstract class for multi-linear interactions.

    A multi-linear interaction is an interaction whose energy is linear in each of its variables.
    That is, for each variable z involved in the interaction, the interaction's energy as a function of z is linear, i.e. of the form
    E(z) = b z + c
    for some coefficients b and c. We call b is the 'linear coefficient' of z.
    A multi-linear interaction is thus a special instance of a multi-quadratic interaction.

    Attributes
    ----------
    variables (list of FloatingVariable): the list of variables involved in the interaction

    Methods
    -------
    energy_fn():
        Returns the interaction's energy term
    linear_fns():
        Returns the list of linear_coef functions for each of the variables
    quadratic_fns():
        Returns a list of None for each of the variables
    """

    def __init__(self, *variables):
        """Constructor of MultiLinearInteraction

        Args:
            variables (list of FloatingVariable): the list of variables involved in the interaction
        """

        self._variables = variables

        MultiQuadraticInteraction.__init__(self, *variables)

    def linear_fns(self):
        """Returns the linear functions of the variables"""
        return [lambda: self._linear_coef(variable) for variable in self._variables]

    def quadratic_fns(self):
        """Returns the quadratic functions of the variables"""
        return [None for _ in self._variables]

    def _linear_coef(self, variable):
        """Returns the interaction's linear influence on the pre-synaptic layer.

        Default implementation, valid for any multi-linear interaction

        Returns:
            Tensor of shape (batch_size, layer_pre_shape) and type float32: the linear contribution
        """

        variable.state.requires_grad = True

        energy_mean = torch.mean( self.energy_fn() )
        linear_coef = torch.autograd.grad(energy_mean, variable.state)[0]

        variable.state.requires_grad = False

        return linear_coef


class BiasInteraction(MultiLinearInteraction):
    """Interaction of the Bias

    A bias interaction is defined between a layer and its corresponding bias variable

    Attributes
    ----------
    _layer (Layer): the layer involved in the interaction
    _bias (Bias): the layer's bias

    Methods
    -------
    energy_fn():
        Returns the energy of the bias interaction
    """

    def __init__(self, layer, bias):
        """Initializes an instance of BiasInteraction

        Args:
            layer (Layer): the layer involved in the interaction
            bias (Bias): the layer's bias
        """

        self._layer = layer
        self._bias = bias

        MultiLinearInteraction.__init__(self, layer, bias)

    def energy_fn(self):
        """Energy term of the bias.

        Returns:
            Vector of size (batch_size,) and of type float32. Each value is the energy term of an example in the current mini-batch
        """

        return - self._layer.state.mul(self._bias.state).flatten(start_dim=1).sum(dim=1)

    def linear_fns(self):
        """Returns the linear functions of the layer and the bias. Overrides the default implementation of MultiLinearInteraction"""
        return [self._linear_coef_layer, self._linear_coef_bias]

    def _linear_coef_layer(self):
        """Returns the linear influence of the bias on the layer's state.

        Returns:
            Tensor of shape (batch_size, layer_shape) and type float32: the linear contribution
        """

        return - self._bias.state

    def _linear_coef_bias(self):
        """Returns the interaction's linear influence on the bias"""

        coef = - self._layer.state.mean(dim=0)

        if len(coef.shape) > 1:
            dims = tuple(range(1, len(coef.shape)))
            coef = coef.mean(dim=dims, keepdim=True)
            
        return coef


class DenseInteraction(MultiLinearInteraction):
    """Dense ('fully connected') interaction between two layers

    A dense interaction is defined between three variables: two adjacent layers, and the corresponding weight tensor between the two.

    Attributes
    ----------
    layer_pre (Layer): pre-synaptic layer. Tensor of shape (batch_size, layer_pre_shape). Type is float32.
    layer_post (Layer): post-synaptic layer. Tensor of shape (batch_size, layer_post_shape). Type is float32.
    weight (Parameter): weight tensor between layer_pre and layer_post. Tensor of shape (layer_pre_shape, layer_post_shape). Type is float32.

    Methods
    -------
    energy_fn():
        Returns the interaction's energy term
    """

    # TODO: batch_mean does not seem to converge

    def __init__(self, layer_pre, layer_post, dense_weight, batch_mean=False):
        """Initializes an instance of DenseInteraction

        Args:
            layer_pre (Layer): pre-synaptic layer
            layer_post (Layer): post-synaptic layer
            dense_weight (DenseWeight): the dense weights between the pre- and post-synaptic layer
            batch_mean (bool, optional): whether we use batch_mean or not. Default: False
        """

        self._layer_pre = layer_pre
        self._layer_post = layer_post
        self._weight = dense_weight

        self._batch_mean = batch_mean

        MultiLinearInteraction.__init__(self, layer_pre, layer_post, dense_weight)

    def energy_fn(self):
        """Returns the energy term of a dense interaction.

        Example:
            - layer_pre is of shape (16, 1, 28, 28), i.e. batch_size is 16, with 1 channel of 28 by 28 (e.g. input tensor for MNIST)
            - layer_post is of shape (16, 2048), i.e. batch_size is 16, with 2048 units
            - weight is of shape (1, 28, 28, 2048)
        pre * W is the tensor product of pre and W over the dimensions (1, 28, 28). The result is a tensor of shape (16, 2048).

        Returns:
            Vector of size (batch_size,) and of type float32. Each value is the energy term of an example in the current mini-batch
        """

        layer_pre = self._layer_pre.state
        layer_post = self._layer_post.state
        if self._batch_mean:
            layer_pre -= layer_pre.mean(dim=0, keepdim=True)
            layer_post -= layer_post.mean(dim=0, keepdim=True)
        dims = len(layer_pre.shape) - 1  # number of dimensions involved in the tensor product layer_pre * weight
        return - torch.tensordot(layer_pre, self._weight.state, dims=dims).mul(layer_post).sum(dim=1)  # Hebbian term: layer_pre * weight * layer_post

    def linear_fns(self):
        """Returns the linear functions of the layers and the weights. Overrides the default implementation of MultiLinearInteraction"""
        return [self._linear_coef_layer_pre, self._linear_coef_layer_post, self._linear_coef_weight]

    def _linear_coef_layer_pre(self):
        """Returns the interaction's linear influence on the pre-synaptic layer.

        Returns:
            Tensor of shape (batch_size, layer_pre_shape) and type float32: the linear contribution on layer_pre
        """

        layer_post = self._layer_post.state
        dims = len(layer_post.shape) - 1  # number of dimensions involved in the tensor product
        if self._batch_mean:
            layer_post -= layer_post.mean(dim=0, keepdim=True)
        dim_weight = len(self._weight.state.shape) - 1
        permutation = (dim_weight,)+tuple(range(dim_weight))
        linear_coef = - torch.tensordot(layer_post, self._weight.state.permute(permutation), dims=dims)  # TODO: check that this works when self._param has more than 2 dimensions.
        return linear_coef

    def _linear_coef_layer_post(self):
        """Returns the interaction's linear influence on the post-synaptic layer.

        Returns:
            Tensor of shape (batch_size, layer_post_shape) and type float32: the linear contribution on layer_post
        """

        layer_pre = self._layer_pre.state
        dims = len(layer_pre.shape) - 1  # number of dimensions involved in the tensor product
        if self._batch_mean:
            layer_pre -= layer_pre.mean(dim=0, keepdim=True)
        linear_coef = - torch.tensordot(layer_pre, self._weight.state, dims=dims)  # TODO: check that this works when self._param has more than 2 dimensions.
        return linear_coef

    def _linear_coef_weight(self):
        """Returns the interaction's linear influence on the weight, dE/dtheta = layer_pre^T * layer_post
        This is the usual Hebbian term.

        Returns:
            Tensor of shape weight_shape and type float32: the linear contribution on the weights
        """

        layer_pre = self._layer_pre.state
        layer_post = self._layer_post.state
        if self._batch_mean:
            layer_pre -= layer_pre.mean(dim=0, keepdim=True)
            layer_post -= layer_post.mean(dim=0, keepdim=True)
        batch_size = layer_pre.shape[0]
        linear_coef = - torch.tensordot(layer_pre, layer_post, dims=([0], [0])) / batch_size
        return linear_coef


class ConvAvgPoolInteraction(MultiLinearInteraction):
    """Convolutional interaction between two layers, with 2*2 average (or `mean') pooling.

    A convolutional interaction with average pooling is defined between three variables: two adjacent layers, and the corresponding convolutional weight tensor between the two.

    Attributes
    ----------
    layer_pre (Layer): pre-synaptic layer. Tensor of shape (batch_size, layer_pre_shape). Type is float32.
    layer_post (Layer): post-synaptic layer. Tensor of shape (batch_size, layer_post_shape). Type is float32.
    weight (Parameter): convolutional weight tensor between layer_pre and layer_post. Type is float32.
    padding (int): padding of the convolution.

    Methods
    -------
    energy_fn():
        Returns the interaction's energy term
    """

    def __init__(self, layer_pre, layer_post, conv_weight, padding=0):
        """Initializes an instance of ConvAvgPoolInteraction

        Args:
            layer_pre (Layer): pre-synaptic layer. Tensor of shape (batch_size, layer_shape). Type is float32.
            layer_post (Layer): post-synaptic layer. Tensor of shape (batch_size, layer_shape). Type is float32.
            conv_weight (ConvWeight): convolutional weights between layer_pre and layer_post. Type is float32.
            padding (int, optional): padding of the convolution. Default: 0
        """

        self._layer_pre = layer_pre
        self._layer_post = layer_post
        self._weight = conv_weight
        self._padding = padding

        MultiLinearInteraction.__init__(self, layer_pre, layer_post, conv_weight)

    def energy_fn(self):
        """Computes the energy term corresponding to this weight tensor.

        Returns:
            Vector of size (batch_size,) and of type float32. Each value is the energy term of an example in the current mini-batch
        """

        layer_pre = self._layer_pre.state
        layer_post = self._layer_post.state

        return - F.avg_pool2d(F.conv2d(layer_pre, self._weight.state, padding=self._padding), 2).mul(layer_post).sum(dim=(3,2,1))  # Hebbian term: layer_pre * weight * layer_post

    def linear_fns(self):
        """Returns the linear functions of the layers and the weights. Overrides the default implementation of MultiLinearInteraction"""
        return [self._linear_coef_layer_pre, self._linear_coef_layer_post, self._linear_coef_weight]

    def _linear_coef_layer_pre(self):
        """Returns the interaction's linear influence on the pre-synaptic layer.

        Returns:
            Tensor of shape (batch_size, layer_pre_shape) and type float32: the linear contribution
        """

        layer_post = self._layer_post.state
        layer_post_augmented = F.interpolate(layer_post, scale_factor=2) / 4.  # augment layer_post
        linear_coef = - F.conv_transpose2d(layer_post_augmented, self._weight.state, padding=self._padding)  # TODO: check that this works when self._param has several dimensions.
        return linear_coef

    def _linear_coef_layer_post(self):
        """Returns the interaction's linear influence on the post-synaptic layer.

        Returns:
            Tensor of shape (batch_size, layer_post_shape) and type float32: the linear contribution
        """

        layer_pre = self._layer_pre.state
        linear_coef = - F.avg_pool2d(F.conv2d(layer_pre, self._weight.state, padding=self._padding), 2)  # TODO: check that this works when self._param has several dimensions
        return linear_coef

    def _linear_coef_weight(self):
        """Returns the interaction's linear influence on the weight.

        Returns:
            Tensor of shape weight_shape and type float32: the linear contribution
        """

        layer_pre = self._layer_pre.state
        layer_post = self._layer_post.state
        batch_size = self._layer_pre.shape[0]

        layer_post = F.interpolate(layer_post, scale_factor=2) / 4.

        linear_coef = - F.conv2d(layer_pre.transpose(0, 1), layer_post.transpose(0, 1), padding=self._padding).transpose(0, 1) / batch_size

        return linear_coef