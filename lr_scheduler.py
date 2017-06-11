from mxnet.lr_scheduler import LRScheduler
import logging

class StepScheduler(LRScheduler):
    """Reduce the learning rate by given a list of steps.
    Assume there exists *k* such that::
       step[k] <= num_update and num_update < step[k+1]
    Then calculate the new learning rate by::
       base_lr * pow(factor, k+1)
    Parameters
    ----------
    step: list of int
        The list of steps to schedule a change
    factor: float
        The factor to change the learning rate.
    """
    def __init__(self, base_lr, steps, learning_rates):
        super(StepScheduler, self).__init__()
        assert isinstance(steps, list) and len(steps) >= 1
        assert len(steps) == len(learning_rates)
        self.steps = steps
        self.learning_rates = learning_rates

    def __call__(self, num_update):
        
        if num_update in self.steps:
            self.base_lr = self.learning_rates[self.steps.index(num_update)]
            logging.info("Update[%d]: Change learning rate to %0.5e",
                          num_update, self.base_lr)
        
        return self.base_lr