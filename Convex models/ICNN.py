import torch
import torch.nn as nn
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'mps'

#the basic ICNN module

n_layers, n_filters, kernel_size = 3, 36, 5

class ICNN(nn.Module):
    def __init__(self, n_in_channels=1, n_filters=n_filters, kernel_size=kernel_size, n_layers=n_layers):
        super(ICNN, self).__init__()
        
        self.n_layers = n_layers
        #these layers should have non-negative weights
        self.wz = nn.ModuleList([nn.Conv2d(n_filters, n_filters, kernel_size=kernel_size, stride=1, padding=2, bias=False)\
                                 for i in range(self.n_layers)])
        
        #these layers can have arbitrary weights
        self.wx = nn.ModuleList([nn.Conv2d(n_in_channels, n_filters, kernel_size=kernel_size, stride=1, padding=2, bias=True)\
                                 for i in range(self.n_layers+1)])
        
        #one final conv layer with nonnegative weights
        self.final_conv2d = nn.Conv2d(n_filters, 1, kernel_size=kernel_size, stride=1, padding=2, bias=False)
        
        #slope of leaky-relu
        self.negative_slope = 0.2 
        
    def forward(self, x):
        z = torch.nn.functional.leaky_relu(self.wx[0](x), negative_slope=self.negative_slope)
        for layer in range(self.n_layers):
            z = torch.nn.functional.leaky_relu(self.wz[layer](z) + self.wx[layer+1](x), negative_slope=self.negative_slope)
        z = self.final_conv2d(z)
        z_avg = torch.nn.functional.avg_pool2d(z, z.size()[2:]).view(z.size()[0], -1)
        
        return z_avg.sum()
    
    #a weight initialization routine for the ICNN
    def initialize_weights(self, min_val=0.0, max_val=0.001, device=device):
        for layer in range(self.n_layers):
            self.wz[layer].weight.data = min_val + (max_val - min_val)\
            * torch.rand(n_filters, n_filters, kernel_size, kernel_size).to(device)
        
        self.final_conv2d.weight.data = min_val + (max_val - min_val)\
        * torch.rand(1, n_filters, kernel_size, kernel_size).to(device)
        return self
    
    #a zero clipping functionality for the ICNN (set negative weights to 0)
    def zero_clip_weights(self): 
        for layer in range(self.n_layers):
            self.wz[layer].weight.data.clamp_(0)
        
        self.final_conv2d.weight.data.clamp_(0)
        return self  