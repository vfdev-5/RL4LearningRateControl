
from torch.nn import LSTM, Linear
from torch.optim.lr_scheduler import _LRScheduler


class AutoLRController(_LRScheduler):
    """Learning rate scheduler based on
    the article "Reinforcement Learning for Learning Rate Control"
    https://arxiv.org/abs/1705.11159

    LR Controller is composed of
    - Actor network = two-layer LSTM with 20 units in each layer
    - Critic network = DNN with a single hidden layer of 10 units

    Actor network takes as input the state of the optimizee (based model) and returns as action
    the set of learning rates.
    Critic network takes as input the action (set of learning rates), the current state and returns a reward.

    Optimizee state is an average loss on the input mini-batch
    """

    def __init__(self, optimizer, last_epoch=-1):
        super(AutoLRController, self).__init__(optimizer, last_epoch=last_epoch)

    def get_lr(self):
        raise NotImplementedError
