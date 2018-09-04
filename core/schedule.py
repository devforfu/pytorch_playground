import math
from torch.optim.lr_scheduler import _LRScheduler


class CosineAnnealingLR(_LRScheduler):
    """
    A scheduler implementing cosine annealing with restarts and an increasing
    period of the decay.
    """
    def __init__(self, optimizer, t_max=200, eta_min=0.0005,
                 cycle_mult=2, last_epoch=-1):

        self.t_max = t_max
        self.eta_min = eta_min
        self.cycle_mult = cycle_mult
        self.cycle_epoch = last_epoch
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        self.cycle_epoch += 1

        t_max = self.t_max
        eta_min = self.eta_min
        t = self.cycle_epoch % t_max

        learning_rates = []
        for lr in self.base_lrs:
            delta = lr - eta_min
            new_lr = eta_min + delta*(1 + math.cos(math.pi * t/t_max)) / 2
            learning_rates.append(new_lr)

        if t == 0:
            self.cycle_epoch = 0
            self.t_max *= self.cycle_mult

        return learning_rates
