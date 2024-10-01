import torch
import torch.nn as nn
import torch.optim as optim
# from SF_SGD import *
from schedulefree import SGDScheduleFree , AdamWScheduleFree , ScheduleFreeWrapper

class Bilevel_Optimizer(nn.Module):
    """
    Upper-level wrapper for existing stochastic optimizers, incuding
    SGD, Adam, and AdamW. SF_SGD refers to the Schedule-Free SGD optimizer.
    """
    def __init__(self, optimizer = 'AdamW', lr = 1e-5, hypergrad = None, verbose = False \
                 , tol = 1e-3, tol_update = None, schedule = None, projection = None):
        super(Bilevel_Optimizer, self).__init__()
        self.hypergrad = hypergrad
        self.verbose = verbose
        self.projection = projection
        self.lr = lr
        if optimizer is None:
            self.optimizer = optim.Adam(self.hypergrad.lower_level_obj.regularizer.parameters(), lr=lr)
        elif optimizer == 'SGD':
            self.optimizer = optim.SGD(self.hypergrad.lower_level_obj.regularizer.parameters(), lr=lr)
        elif optimizer == 'SGD_accelerated':
            self.optimizer = optim.SGD(self.hypergrad.lower_level_obj.regularizer.parameters(), lr=lr)
        elif optimizer == 'Adam':
            self.optimizer = optim.Adam(self.hypergrad.lower_level_obj.regularizer.parameters(), lr=lr)
        elif optimizer == 'AdamW':
            self.optimizer = optim.AdamW(self.hypergrad.lower_level_obj.regularizer.parameters(), lr=lr)
        elif optimizer == 'SF_SGD':
            base_optimizer = optim.SGD(self.hypergrad.lower_level_obj.regularizer.parameters(), lr=lr)
            self.optimizer = ScheduleFreeWrapper(base_optimizer,momentum=0.9)
    def step(self):
        self.optimizer.zero_grad()
        self.hypergrad.hypergrad()
        if self.optimizer != 'SF_SGD':
            self.optimizer.param_groups[0]['lr'] = self.lr
            print(self.optimizer.param_groups[0]['lr'])
        self.optimizer.step()
        if self.projection is not None:
            self.projection()
        if self.verbose:
            print("loss: ", self.hypergrad.upper_level_obj(self.hypergrad.x_init).item()) 