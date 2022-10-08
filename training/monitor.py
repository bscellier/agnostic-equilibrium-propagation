from datetime import datetime
import pickle
import sys
import time
import torch
from torch.utils.tensorboard import SummaryWriter

from training.statistics import Counter, ErrorFinder, MeanStat, NumIterationsStat, EnergyStat, CostStat, ErrorStat, ChangeStat, ViolationStat, SaturationStat
from network.variable.layer import Layer


class Monitor:
    """
    Class used to monitor the training process.

    Attributes
    ----------
    _network (Network): the network to train
    _trainer (Trainer): used to train the network on the training set
    _evaluator (Evaluator): used to evaluate the network on the test set
    _path (str): the directory where to save the model and the characteristics of the training process
    _series (list of TimeSeries): the time series of statistics that are monitored during training
    _series_layer_changes (list of TimeSeries): the time series of layer_changes (computed so as to adjust the threshold values)
    _test_error_curve (TimeSeries): the test error curve

    Methods
    -------
    run(num_epochs, verbose=False, use_tensorboard=True)
        Trains the network for num_epochs epochs
    save_network()
        Saves the model in the path
    save_series()
        Saves the state of the training process in the path
    """

    def __init__(self, network, trainer, evaluator, path=None):
        """Creates an instance of Optimizer

        Args:
            network (Network): the network to train
            trainer (Trainer): used to train the network on the training set
            evaluator (Evaluator): used to evaluate the network on the test set
            path (str, optional): the directory where to save the model and the characteristics of the training process. Default: None
        """

        self._network = network
        self._trainer = trainer
        self._evaluator = evaluator

        self._path = datetime.now().strftime("%Y%m%d-%H%M%S") if path is None else path

        self._series = []
        self._series_layer_changes = []
        self._test_error_curve = None  # initialized in _build_series()

        self._build_series()


    def run(self, num_epochs, lr_decay=1.0, verbose=False, use_tensorboard=True):
        """Launch a run for num_epochs epochs

        Logs statistics about the run either after every batch or after every epoch.

        Args:
            num_epochs (int): number of epochs of training
            lr_decay (float, optional): rate of decay of the learning rates at each epoch. Default: 1.0 (i.e. no decay)
            verbose (bool, optional): if True, prints logs after every batch processed ; if False: prints logs after every epoch. Default: False
            use_tensorboard (bool, optional): if True, uses a summary writter to monitor with tensorboard. Default: True
        """

        if use_tensorboard: writer = SummaryWriter(self._path)

        start_time = time.time()

        for epoch in range(num_epochs):

            print('Epoch {}'.format(epoch + 1))

            # Training
            self._trainer.run(verbose)

            # Evaluation
            self._evaluator.run(verbose)

            # Statistics
            for series in self._series: series.update()

            if use_tensorboard: self._update_summary_writter(writer, epoch)

            if not verbose:  # prints the statistics of training and evaluation at every epoch
                print(str(self._trainer))
                print(str(self._evaluator))

            # adjusts the threshold value for the convergence criterion
            layer_changes = [series.get_last() for series in self._series_layer_changes]
            thresholds = self._network.thresholds
            thresholds = [min(change/100., threshold) for change, threshold in zip(layer_changes, thresholds)]
            self._network.thresholds = thresholds

            # updates the learning rates (scheduler)
            learning_rates = [lr_decay * lr for lr in self._network.learning_rates]
            self._network.learning_rates = learning_rates
            
            self.save_series()  # saves the training curves
            if self._test_error_curve.is_minimum(): self.save_network()  # saves the network's parameters

            duration = (time.time() - start_time) / 60.
            print('duration = {:.1f} min\n'.format(duration))

    def save_network(self):
        """Saves the network's parameters"""

        model_path = self._path + '/model.pt'
        self._network.save(model_path)

    def save_series(self):
        """Saves the time series (`training curves')"""

        time_series_path = self._path + '/time_series.pkl'
        with open(time_series_path, 'wb') as handle:
            dictionary = {series.writer_name: series.get_series() for series in self._series if series.writer_name}
            pickle.dump(dictionary, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def _add_statistic(self, statistic, train):
        """Adds a statistic to either the trainer or the evaluator, and adds the corresponding time series to the monitor

        Args:
            statistic (Statistic): the statistic to add
            train (bool): whether we add the statistic to the trainer (True) or the evaluator (False)
        """

        if train: self._trainer.add_statistic(statistic)
        else: self._evaluator.add_statistic(statistic)

        series = TimeSeries(statistic, train)
        self._series.append(series)

        if isinstance(statistic, ChangeStat) and isinstance(statistic.get_variable(), Layer): self._series_layer_changes.append(series)
        if not train and isinstance(statistic, ErrorStat): self._test_error_curve = series

    def _build_series(self):
        """Prepares the statistics and the corresponding time series to monitor"""

        network = self._network
        train_set_size = self._trainer.dataset_size()
        test_set_size = self._evaluator.dataset_size()

        # the statistics to add to the trainer
        stats_train = [
        Counter(network, train_set_size),
        NumIterationsStat(),
        EnergyStat(network),
        CostStat(network),
        ErrorStat(network, without_clamping=False),
        ErrorStat(network, without_clamping=True),
        ]
        stats_train += [ViolationStat(layer) for layer in network.layers()]
        stats_train += [SaturationStat(layer, mode='min') for layer in network.layers()]
        stats_train += [SaturationStat(layer, mode='max') for layer in network.layers()]
        stats_train += [ChangeStat(layer) for layer in network.layers()]
        stats_train += [ChangeStat(param) for param in network.params()]

        for stat in stats_train: self._add_statistic(stat, train=True)

        # the statistics to add to the evaluator
        stats_test = [
        Counter(network, test_set_size),
        NumIterationsStat(),
        EnergyStat(network),
        CostStat(network),
        ErrorStat(network, without_clamping=False),
        ErrorStat(network, without_clamping=True)
        # ErrorFinder(network),
        ]
        stats_test += [ViolationStat(layer) for layer in network.layers()]
        stats_test += [SaturationStat(layer, mode='min') for layer in network.layers()]
        stats_test += [SaturationStat(layer, mode='max') for layer in network.layers()]

        for stat in stats_test: self._add_statistic(stat, train=False)

    def _update_summary_writter(self, writer, epoch):
        """Add the statistics to the summary writter to monitor with tensorboard

        Args:
            writer (SummaryWritter): tensorboard summary writter to update
            epoch (int): epoch of training where the statistics have been recorded
        """

        for param in self._network.params(): writer.add_histogram(param.name, param.state, epoch+1)

        for series in self._series:
            if series.writer_name: writer.add_scalar(series.writer_name, series.get_last(), epoch+1)



class TimeSeries:
    """Class for time series of statistics during training

    Attributes
    ----------
    writer_name (str): the name of the time series to be displayed in tensorboard
    _statistic (Statistic): the underlying statistic whose time series we compute during training
    _time_series (list): the time series. Each entry corresponds to the value of the statistic at a given epoch

    Methods
    -------
    update()
        Appends the current value of the statistic to the time series
    get_series()
        Returns the time series of statistics
    get_last()
        Returns the last value of the time series of statistics
    minimum()
        Returns the minimum of the time series
    is_minimum()
        Checks if the last value of the time series is strictly less than all the previous values
    """

    def __init__(self, statistic, train=True):
        """Creates an instance of Series

        Args:
            statistic (Statistic): the underlying statistic whose time series we compute during training
            train (bool, optional): whether this is a time series during training (training time) or evaluation (test time). Default: True
        """

        self._statistic = statistic
        self._time_series = []

        self._writer_name = None
        if self._statistic.display_name:
            writer_name = self._statistic.name
            if train: writer_name += '/train'
            else: writer_name += '/test'
            if self._statistic.option: writer_name += '_'+self._statistic.option
            self._writer_name = writer_name

    @property
    def writer_name(self):
        """Gets the writer_name"""
        return self._writer_name

    def update(self):
        """Appends the current value of the statistic to the time series"""
        self._time_series.append(self._statistic.get())

    def get_series(self):
        """Returns the time series of statistics"""
        return self._time_series

    def get_last(self):
        """Returns the last value of the time series of statistics"""
        return self._time_series[-1]

    def minimum(self):
        """Returns the minimum of the time series"""
        return min(self._time_series)

    def is_minimum(self):
        """Checks if the last value of the time series is strictly less than all the previous values

        Returns:
            bool: whether or not the last value of the time series is the strict minimum of the series
        """

        epoch = len(self._time_series)
        if epoch <= 1: return True

        last_value = self._time_series[-1]
        min_value = min(self._time_series[:-1])
        return last_value < min_value