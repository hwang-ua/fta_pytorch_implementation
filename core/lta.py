import torch
import torch.nn.functional as functional
import numpy as np


class LTA:
    def __init__(self, tiles, bound_low, bound_high, eta, input_dim):
        # 1 tiling, binning
        self.n_tilings = 1
        self.n_tiles = tiles
        self.bound_low, self.bound_high = bound_low, bound_high
        self.delta = (self.bound_high - self.bound_low) / self.n_tiles
        self.c_mat = torch.as_tensor(np.array([self.delta * i for i in range(self.n_tiles)]) + self.bound_low, dtype=torch.float32)
        self.eta = eta
        self.d = input_dim

    def __call__(self, reps):
        temp = reps
        temp = temp.reshape([-1, self.d, 1])
        onehots = 1.0 - self.i_plus_eta(self.sum_relu(self.c_mat, temp))
        out = torch.reshape(torch.reshape(onehots, [-1]), [-1, int(self.d * self.n_tiles * self.n_tilings)])
        return out

    def sum_relu(self, c, x):
        out = functional.relu(c - x) + functional.relu(x - self.delta - c)
        return out

    def i_plus_eta(self, x):
        if self.eta == 0:
            return torch.sign(x)
        out = (x <= self.eta).type(torch.float32) * x + (x > self.eta).type(torch.float32)
        return out
