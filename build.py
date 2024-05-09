#import torch
from torch import nn, arange, pow

class TaylorSeries(nn.Module):
    def __init__(self, in_size = 1, out_size = 1, t_order = 8):
        super(TaylorSeries, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.t_order = t_order
        self.layer = nn.Linear(t_order, out_size, bias=False)
        self.orders = arange(0, t_order).float()

    def forward(self, x):
        x = x.unsqueeze(-1)
        r = pow(x, self.orders)
        r = r.view(x.shape[0], -1)
        return self.layer(r)