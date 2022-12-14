import sys



class Epoch:
    """
    Class used to process a dataset with a network once (one 'epoch') to change the internal state of the network and/or compute statistics.
    Two important subclasses of Epoch are Evaluator and Trainer.

    Attributes
    ----------
    _stats (list of Statistic): the list of all statistics to be computed over the dataset
    _stats_auto (list of Statistic): the list of statistics that do not require any argument to compute
    _stats_manual (list of Statistic): the list of statistics that require an argument to compute

    Methods
    -------
    add_statistic(stat)
        Adds a statistic to the collection
    dataset_size()
        Returns the size of the dataset
    _reset()
        Sets all the statistics to zero
    _do_measurements_auto()
        Do the measurements for each of the statistics in the 'auto' list
    _do_measurements_manual()
        Do the measurements for each of the statistics in the 'manual' list
    """

    def __init__(self):
        """Creates an instance of CollectionOfStatistics

        Args:
            stats_auto (list of Statistics, optional): list of statistics to intialize. Default: None
        """

        self._stats_auto = []
        self._stats_manual = []
        self._stats = []

    def add_statistic(self, stat):
        """Adds a statistic to the list of statistics

        Args:
            stat (Statistics): the statistics to be added
        """

        self._stats.append(stat)

        if stat.manual: self._stats_manual.append(stat)
        else: self._stats_auto.append(stat)

    def dataset_size(self):
        """Returns the size of the dataset"""
        return len(self._dataloader.dataset)

    def _reset(self):
        """Sets all the statistics to zero"""

        for stat in self._stats: stat.reset()

    def _do_measurements_auto(self, **kwargs):
        """Do measurements in all stats of the 'auto' collection"""

        for stat in self._stats_auto: stat.do_measurement(**kwargs)

    def _do_measurements_manual(self, *args):
        """Do measurements in all stats of the 'manual' collection"""

        for stat, arg in zip(self._stats_manual, args):
            stat.do_measurement(arg)

    def __str__(self):
        list_of_strings = [str(stat) for stat in self._stats if stat.display]
        string = ', '.join(list_of_strings)
                
        return string



class Evaluator(Epoch):
    """
    Class used to evaluate a network on a dataset

    Attributes
    ----------
    _network (Network): the model to evaluate
    _dataloader (Dataloader): the dataset on which to evaluate the model
    max_iterations (int): maximum number of iterations allowed to converge to equilibrium at inference

    Methods
    -------
    run(verbose)
        Evaluates the network over the dataset
    """

    def __init__(self, network, dataloader, max_iterations=100):
        """Initializes an instance of Aeqprop

        Args:
            network (Network): the model to evaluate
            dataloader (Dataloader): the dataset on which to evaluate the model
            max_iterations (int, optional): maximum number of iterations allowed to converge to equilibrium at inference. Default: 100.
        """

        Epoch.__init__(self)

        self._network = network
        self._dataloader = dataloader

        self.max_iterations = max_iterations

    def run(self, verbose=False):
        """Evaluate the model over the dataset.

        Args:
            verbose (bool, optional): if True, prints logs after every batch processed ; if False: prints logs after processing the entire dataset. Default: False.
        """

        self._reset()

        self._network.set_nudging(0.)

        for x, y, idx in self._dataloader:  # FIXME: must know that dataloader returns idx

            self._network.set_data(x, y, reset=True)

            # Inference (homeostatic phase)
            num_iterations = self._network.homeostatic_relaxation(max_iterations=self.max_iterations, adjust_control_knobs=False)
            self._do_measurements_auto()
            args = (num_iterations, idx)
            self._do_measurements_manual(*args)  # FIXME: how is statistics supposed to know about num_iterations and idx?

            if verbose:
                sys.stdout.write('\r')
                sys.stdout.write(str(self))
                sys.stdout.flush()

        if verbose:
            sys.stdout.write('\r')
            sys.stdout.write(str(self))
            sys.stdout.write('\n')

    def __str__(self):
        return 'TEST  -- ' + Epoch.__str__(self)



class Trainer(Epoch):
    """
    Class used to train a network on a dataset

    Attributes
    ----------
    _network (Network): the network to train
    _dataloader (Dataloader): the dataset on which to train the network
    _differentiator (TrainingProcedure): the method used to train the network
    use_soft_targets (bool): if True, the "one-hot code" of y is of the form [0, ..., 0, 0.9, 0, ..., 0] instead of [0, ..., 0, 1, 0, ..., 0]

    Methods
    -------
    run(verbose)
        Train the network for one epoch over the dataset
    """

    def __init__(self, network, dataloader, differentiator, use_soft_targets=False):
        """Initializes an instance of Aeqprop

        Args:
            network (Network): the network to train
            dataloader (Dataloader): the dataset on which to train the network
            differentiator (TrainingProcedure): 
            use_soft_targets (bool, optional): if True, the "one-hot code" of y is of the form [0, ..., 0, 0.9, 0, ..., 0] instead of [0, ..., 0, 1, 0, ..., 0]. Default: False
        """

        Epoch.__init__(self)

        self._network = network
        self._dataloader = dataloader
        self._differentiator = differentiator

        self.use_soft_targets = use_soft_targets


    def run(self, verbose=False):
        """Train the model for one epoch over the dataset.

        Args:
            verbose (bool, optional): if True, prints logs after every batch processed ; if False: prints logs after every epoch. Default: False.
        """

        self._reset()

        for x, y in self._dataloader:

            self._network.set_data(x, y, use_soft_targets=self.use_soft_targets)  # TODO: put an option to decide whether to reset all layers to zero or not.

            # inference
            num_iterations = self._differentiator.inference()
            self._do_measurements_auto()

            # training step
            layers_change, params_change = self._differentiator.training_step()
            args = (num_iterations,) + tuple(layers_change) + tuple(params_change)
            self._do_measurements_manual(*args)  # FIXME: how is statistics supposed to know about num_iterations and state_change?

            if verbose:
                sys.stdout.write('\r')
                sys.stdout.write(str(self))
                sys.stdout.flush()

        if verbose:
            sys.stdout.write('\r')
            sys.stdout.write(str(self))
            sys.stdout.write('\n')

    def __str__(self):
        return 'TRAIN -- ' + Epoch.__str__(self)