from .callbacks import CallbackGroup


class Loop:

    def __init__(self, stepper, alpha=0.98):
        self.stepper = stepper
        self.alpha = alpha
        self.stop = False

    def run(self, train_data, valid_data, epochs=100, callbacks=None):
        phases = [
            Phase(name='train', dataset=train_data),
            Phase(name='valid', dataset=valid_data)
        ]

        cb = CallbackGroup(callbacks)
        cb.set_loop(self)
        cb.training_start()

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

    def save_model(self, path):
        self.stepper.save_model(path)


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
