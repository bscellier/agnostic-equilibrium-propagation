from abc import ABC, abstractmethod
import torch



class Statistic(ABC):
    """Abstract class for statistics

    Attributes
    ----------

    Methods
    -------
    do_measurement()
        Do a measurement and adjusts the value of the statistic accordingly
    get()
        Returns the current value of the statistic
    reset()
        Resets the statistic to zero
    """

    @abstractmethod
    def do_measurement(self):
        """Do a measurement and adjusts the value of the statistic accordingly"""
        pass

    @abstractmethod
    def get(self):
        """Returns the current value of the statistic"""
        pass

    @abstractmethod
    def reset(self):
        """Resets the statistic to zero"""
        pass


class Counter(Statistic):
    """Class to count how many examples of the dataset have been processed

    Attributes
    ----------
    _num_examples (int): the number of examples processed so far
    _dataset_size (int): the number of examples in the dataset

    Methods
    -------
    _count_fn()
        Counts the number of examples in the current mini-batch
    """

    display = True
    name = None
    option = None
    display_name = None

    manual = False

    def __init__(self, network, dataset_size):
        """Initializes an instance of Counter

        Args:
            network (Network): the network where we count the number of examples processed
            dataset_size (int): the number of examples in the dataset
        """

        self._count_fn = lambda **kwargs: network.layers_state[0].shape[0]

        self._num_examples = 0

        self._dataset_size = dataset_size

    def do_measurement(self, **kwargs):
        """Measures the statistics and adds it to the sum"""
        self._num_examples += self._count_fn()

    def get(self):
        """Returns the number of examples processed so far"""
        return self._num_examples

    def reset(self):
        """Resets the number of examples processed to zero"""
        self._num_examples = 0

    def __str__(self):
        string = 'Example {:5d}/{}'.format(self._num_examples, self._dataset_size)
        return string
        

class ErrorFinder(Statistic):
    """
    Class used to find the examples in a dataset that a given network misclassifies

    Attributes
    ----------
    name (str): the name of the statistic
    _network (Network): the model to evaluate
    _list_indices (list of int): list of indices of misclassified images in the dataset
    writter (bool): whether or not this statistic is added the tensorboard summary writter  # FIXME
    display (bool): whether or not this statistic is displayed in the summeary logs of each epoch  # FIXME

    Methods
    -------
    do_measurement()
        Measures the statistics and adds it to the sum
    get()
        Returns the list of indices
    reset()
        Resets the list of indices to an empty list
    """

    display = False
    name = None  # 'Error_finder'
    option = None
    display_name = None

    manual = True

    def __init__(self, network):
        """Initializes an instance of ErrorFinder

        Args:
            network (Network): the network used to classify the examples of the dataset
        """

        self._network = network

        self._list_indices = []

    def do_measurement(self, idx, **kwargs):
        """Returns the indices of the misclassified images in idx"""

        mask = self._network.error_fn(without_clamping=True)
        indices = idx[mask].numpy()

        self._list_indices.extend(indices)

    def get(self):
        """Returns the list of indices"""
        return self._list_indices

    def reset(self):
        """Resets the list of indices to an empty list"""
        self._list_indices = []

    def __str__(self):
        num_mistakes = len(self._list_indices)
        return 'Number of mistakes = {}'.format(num_mistakes)




class MeanStat(Statistic):
    """
    Class used to accumulate statistics over the mini-batches to get statistics over the entire dataset.

    Attributes
    ----------
    name (str): the name of the statistics
    sum (float32): cumulative sum of the statistics
    num_steps (int): number of times the function "do_meaurement" has been called.
    percentage (bool): whether the statistics is played in percentage or not
    string (str): the string to be formatted.

    Methods
    -------
    do_measurement()
        Measures the statistics and adds it to the sum
    get()
        Returns the mean of the statistics
    reset()
        Resets sum and num_steps to zero
    """

    def __init__(self, measure_fn, display_name=None, name=None, precision=3, percentage=False, display=False):
        """
        Initializes an instance of Stat.
        
        Args:
            measure_fn: the function that measures the statistics in the model
            name (str): the name of the statistics
            precision (int, optional): the number of decimals to display. Default: 3.
            percentage (bool, optional): whether the statistics is displayed in percentage or not. Default: False.
        """

        self._measure_fn = measure_fn

        self.display_name = display_name

        self.name = name

        self._sum = 0.
        self._num_steps = 0

        self._display_string = display_name + ' = {:.' + str(precision) + 'f}'
        if percentage: self._display_string += '%'

        self._percentage = percentage

        self.display = display

    def do_measurement(self, *args, **kwargs):
        """Measures the statistics and adds it to the sum"""
        amount = self._measure_fn(*args, **kwargs)
        self._sum += amount
        self._num_steps += 1

    def get(self):
        """Returns the mean statistics"""
        mean = self._sum / self._num_steps
        if self._percentage: mean *= 100.
        return mean

    def reset(self):
        """Resets the variables to zero"""
        self._sum = 0.
        self._num_steps = 0

    def __str__(self):
        mean = self.get()
        return self._display_string.format(mean)


class EnergyStat(MeanStat):
    """
    Class used to measure the mean energy (when the network is at equilibrium) over the dataset.
    """

    manual = False

    option = None

    def __init__(self, network):
        """Initializes an instance of EnergyStat

        Args:
            network (Network): the network whose energy is measured at equilibrium
        """

        energy_fn = lambda **kwargs: network.energy_fn().mean().item()
        display_name = 'Energy'
        name = 'Energy'
        precision = 2
        percentage = False
        display = True

        MeanStat.__init__(self, energy_fn, display_name, name, precision, percentage, display)


class CostStat(MeanStat):
    """
    Class used to measure the mean cost (when the network is at equilibrium, i.e. the `loss') over the dataset.
    """

    manual = False

    option = None

    def __init__(self, network):
        """Initializes an instance of CostStat

        Args:
            network (Network): the network whose cost function is measured at equilibrium
        """

        cost_fn = lambda **kwargs: network.cost_fn().mean().item()
        display_name = 'Cost'
        name = 'Cost'
        precision = 5
        percentage = False
        display = True

        MeanStat.__init__(self, cost_fn, display_name, name, precision, percentage, display)


class ErrorStat(MeanStat):
    """
    Class used to measure the mean error rate over the dataset.
    """

    manual = False

    def __init__(self, network, without_clamping=False):
        """Initializes an instance of ErrorStat

        Args:
            network (Network): the network whose error rate is measured
            without_clamping (bool, optional): if True, the prediction of the network is not clipped. Default: False
        """

        self.option = 'no_clipping' if without_clamping else None

        error_fn = lambda **kwargs: network.error_fn(without_clamping).type(torch.float).mean().item()
        display_name = 'Error'
        name = 'Error'
        precision = 3
        percentage = True
        display = True

        MeanStat.__init__(self, error_fn, display_name, name, precision, percentage, display)


class ViolationStat(MeanStat):
    """
    Class used to measure the mean violation of the equilibrium condition (at the end of the relaxation phase to equilibrium) over the dataset.
    """

    manual = False

    def __init__(self, variable):
        """Creates and instance of ViolationStat

        Args:
            variable (FLoatingVariable): the variable whose equilibrium condition's violation we want to track
        """

        self.option = variable.name

        violation_fn = lambda **kwargs: variable.violation.item()
        display_name = 'Violation_{}'.format(variable.name)
        name = 'Violation'
        precision = 5
        percentage = False
        display = False

        MeanStat.__init__(self, violation_fn, display_name, name, precision, percentage, display)


class SaturationStat(MeanStat):
    """
    Class used to measure the mean saturation over the dataset.

    The `saturation' is the fraction of individual variables that are cliped to the min or the max of the variable's state interval
    """

    manual = False

    def __init__(self, variable, mode='min'):
        """Creates and instance of SaturationStat

        Args:
            variable (FLoatingVariable): the variable whose saturation we want to track
            mode (str, optional): either `min' (the minimum of the variable's interval) or `max' (the maximum of the variable's interval). Default: `min'
        """

        self.option = variable.name

        value = variable.min_interval if mode=='min' else variable.max_interval

        saturation_fn = lambda **kwargs: (variable.state == value).type(torch.float).mean().item()
        display_name = mode + 'Saturation_{}'.format(variable.name)
        name = mode + 'Saturation'
        precision = 1
        percentage = True
        display = False

        MeanStat.__init__(self, saturation_fn, display_name, name, precision, percentage, display)


class NumIterationsStat(MeanStat):
    """
    Class used to count the mean number of iterations to converge to equilibrium during a relaxation phase
    """

    manual = True
    option = None

    def __init__(self):
        """Creates and instance of NumIterationsStat"""

        num_iterations_fn = lambda num_iterations, **kwargs: num_iterations
        display_name = 'Num_iterations'
        name = 'Num_iterations'
        precision = 1
        percentage = False
        display = True

        MeanStat.__init__(self, num_iterations_fn, display_name, name, precision, percentage, display)


class ChangeStat(MeanStat):
    """
    Class used to measure the mean changement (over the dataset) of a given floating variable during the two phases of the algorithm.
    """

    manual = True

    def __init__(self, variable):
        """Creates and instance of ChangeStat

        Args:
            variable (FLoatingVariable): the variable whose changement between the two phases we want to track
        """

        self.option = variable.name

        change_fn = lambda change, **kwargs: change
        display_name = 'Change_{}'.format(variable.name)
        name = 'Change'
        precision = 5
        percentage = False
        display = False

        MeanStat.__init__(self, change_fn, display_name, name, precision, percentage, display)

        self._variable = variable

    def get_variable(self):
        self._variable