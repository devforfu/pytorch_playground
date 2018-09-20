import os
import sys
from os.path import join


class Callback:
    """
    Base class for all training loop callbacks.

    The callback is a class that has a set of methods invoked withing training
    loop iterations. The class can adjust model's properties, save state, log
    output, or perform any other tuning on periodical basis.
    """
    def training_start(self):
        pass

    def training_end(self):
        pass

    def epoch_start(self, epoch):
        pass

    def epoch_end(self, epoch, metrics):
        pass

    def batch_start(self, epoch, phase):
        pass

    def batch_end(self, epoch, phase):
        pass


class CallbackGroup(Callback):
    """
    Wraps a collection of callbacks into single instance which delegates
    appropriate methods calls to the elements of collection.
    """
    def __init__(self, callbacks=None):
        callbacks = callbacks or []
        self.callbacks = callbacks
        self._callbacks = {type(cb).__name__: cb for cb in self.callbacks}

    def training_start(self):
        for cb in self.callbacks: cb.training_start()

    def training_end(self):
        for cb in self.callbacks: cb.training_end()

    def epoch_start(self, epoch):
        for cb in self.callbacks: cb.epoch_start(epoch)

    def epoch_end(self, epoch, metrics):
        for cb in self.callbacks: cb.epoch_end(epoch, metrics)

    def batch_start(self, epoch, phase):
        for cb in self.callbacks: cb.batch_start(epoch, phase)

    def batch_end(self, epoch, phase):
        for cb in self.callbacks: cb.batch_start(epoch, phase)

    def set_loop(self, loop):
        for cb in self.callbacks: cb.loop = loop

    def __getitem__(self, item):
        if item not in self._callbacks:
            raise KeyError(f'unknown callback: {item}')
        return self._callbacks[item]


class Logger(Callback):
    """
    Writes performance metrics collected during the training process into list
    of streams.

    Parameters:
        streams: A list of file-like objects with `write()` method.

    """
    def __init__(self, streams=None, log_every=1):
        self.streams = streams or [sys.stdout]
        self.log_every = log_every
        self.epoch_history = {}
        self.curr_epoch = 0

    def epoch_end(self, epoch, metrics):
        stats = [f'{name}: {value:2.4f}' for name, value in metrics.items()]
        metrics = ' - '.join(stats)
        string = f'Epoch {epoch:4d} | {metrics}\n'
        for stream in self.streams:
            stream.write(string)
            stream.flush()


class CSVLogger(Logger):
    """
    A wrapper build on top of stdout logging callback which opens a CSV file
    to write metrics.

    Parameters:
        filename: A name of CSV file to store training loss history.

    """
    def __init__(self, filename='history.csv'):
        super().__init__()
        self.filename = filename
        self.file = None

    def training_start(self):
        self.file = open(self.filename, 'w')
        self.streams = [self.file]

    def training_end(self):
        if self.file:
            self.file.close()


class History(Callback):

    def __init__(self):
        self.history = []

    def epoch_end(self, epoch, metrics):
        self.history.append(metrics)

    def training_end(self):
        history = []
        for i, record in enumerate(self.history):
            item = record.copy()
            item['epoch'] = i
            history.append(item)
        self.history = history


class ImprovementTracker(Callback):
    """
    Tracks a specific metric during training process and reports when the
    metric does not improve after the predefined number of iterations.
    """
    # def __init__(self, patience=1, phase='valid', metric='valid_avg_loss',
    #              better=min):

    def __init__(self, patience=1, metric='valid_loss', better=min):
        self.patience = patience
        # self.phase = phase
        self.metric = metric
        self.better = better
        self.no_improvement = None
        self.best_value = None
        self.stagnation = None
        self.loop = None

    def training_start(self):
        self.no_improvement = 0
        self.stagnation = False

    def epoch_end(self, epoch, metrics):
        value = metrics[self.metric]
        best_value = self.best_value or value
        improved = self.better(best_value, value) == value
        if not improved:
            self.no_improvement += 1
        else:
            self.best_value = value
            self.no_improvement = 0
        if self.no_improvement >= self.patience:
            self.stagnation = True

    @property
    def improved(self):
        return self.no_improvement == 0


class EarlyStopping(ImprovementTracker):
    """
    Stops observed training loop if the tracked performance metrics does not
    improve during predefined number of iterations.
    """

    def epoch_end(self, epoch, metrics):
        super().epoch_end(epoch, metrics)
        if self.stagnation:
            self.loop.stop = True


class Checkpoint(ImprovementTracker):
    """
    Saves model attached to the loop each time when tracked performance metric
    is improved, or on each iteration if required.
    """
    def __init__(self, folder=None, save_best_only=True,
                 filename='model_{metric}_{value:2.4f}.weights',
                 **kwargs):

        super().__init__(**kwargs)
        self.folder = folder or os.getcwd()
        self.save_best_only = save_best_only
        self.filename = filename
        self.best_model = None

    @property
    def need_to_save(self):
        if not self.save_best_only:
            return True
        return self.improved

    def get_name(self):
        return self.filename.format(metric=self.metric, value=self.best_value)

    def epoch_end(self, epoch, metrics):
        super().epoch_end(epoch, metrics)
        if self.need_to_save:
            best_model = join(self.folder, self.get_name())
            self.loop.save_model(best_model)
            self.best_model = best_model


def default_callbacks(workdir=None):
    """Returns a list with commonly used callbacks."""

    workdir = workdir or os.getcwd()
    return [
        History(),
        Logger(),
        CSVLogger(filename=join(workdir, 'history.csv')),
        Checkpoint(folder=workdir)
    ]