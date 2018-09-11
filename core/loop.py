import torch
from torch.nn import functional as F

from .callbacks import CallbackGroup


class Loop:
    """
    Simple training loop implementation.

    The loop contains two phases: training and validation. Each phase is
    computed on a separate dataset, and tracks its own parameters, like,
    average loss and batch number.

    Parameters:
        model: An optimized model.

        alpha: A value of weight used to perform linear interpolation between
            loss on the previous epoch and the new epoch, like:

                new_loss = old_loss*alpha + (1 - alpha)*new_loss

    """
    def __init__(self, model, optimizer, schedule, alpha: float=0.98):
        self.model = model
        self.optimizer = optimizer
        self.schedule = schedule
        self.alpha = alpha
        self.stop = False
        self.callbacks = None
        self.stepper = None

    def run(self, train_data, valid_data=None, loss_fn=F.nll_loss,
            epochs: int=100, callbacks=None):

        phases = [Phase(name='train', dataset=train_data)]
        if valid_data is not None:
            phases.append(Phase(name='valid', dataset=valid_data))

        cb = CallbackGroup(callbacks)
        cb.set_loop(self)
        cb.training_start()
        self.callbacks = cb
        self.stepper = self.make_stepper(loss_fn)

        a = self.alpha
        for epoch in range(epochs):
            if self.stop:
                break
            for phase in phases:
                cb.epoch_start(epoch, phase)
                is_training = phase.name == 'train'
                for x, y in phase.dataset:
                    phase.batch_num += 1
                    cb.batch_start(epoch, phase)
                    loss = self.stepper.step(x, y, is_training)
                    phase.avg_loss = phase.avg_loss*a + loss*(1 - a)
                    cb.batch_end(epoch, phase)
                cb.epoch_end(epoch, phase)
        cb.training_end()

    def make_stepper(self, loss_fn, stepper=None):
        stepper_cls = stepper or Stepper
        inst = stepper_cls(self.model, self.optimizer, self.schedule, loss_fn)
        return inst

    def save_model(self, path):
        self.stepper.save_model(path)

    @property
    def lr_schedule(self):
        return self.stepper.train_learning_rates

    def __getitem__(self, item):
        return self.callbacks[item]


class Phase:
    """
    Model training loop phase.

    Each model's training loop iteration could be separated into (at least) two
    phases: training and validation. The instances of this class track
    metrics and counters, related to the specific phase, and keep the reference
    to subset of data, used during phase.
    """
    def __init__(self, name: str, dataset):
        self.name = name
        self.dataset = dataset
        self.batch_num = 0
        self.avg_loss = 0.0

    def __repr__(self):
        return f'<Phase: {self.name}, avg_loss: {self.avg_loss:2.4f}>'


class Stepper:
    """
    A thin wrapper encapsulating the model, its optimizer, a scheduler, and a
    loss function into single object.

    The stepper instance is invoked during each training iteration and returns
    the loss on batch.
    """
    def __init__(self, model, optimizer, schedule, loss):
        if schedule.last_epoch == -1:
            schedule.step()
        self.model = model
        self.optimizer = optimizer
        self.schedule = schedule
        self.loss = loss
        self.train_learning_rates = []

    def step(self, x, y, train: bool=True):
        """
        Performs a single training step.

        Args:
            x: Features tensor.
            y: Target tensor.
            train: If False, then the gradient is not computed, and the model's
                parameters are not updated.

        Returns:
            loss: The loss value on batch.

        """
        with torch.set_grad_enabled(train):
            out = self.model(x)
            loss = self.loss(out, y)
            if train:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.schedule.step()
                lrs = self.schedule.get_lr()
                self.train_learning_rates.append(lrs)
        return loss.item()

    def save_model(self, path: str):
        """
        Saves model state into file.
        """
        torch.save(self.model.state_dict(), path)
