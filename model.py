import numpy as np
import math
from math import sqrt, pi
import numpy.random as npr
import torch
import itertools


class Network:
    def __init__(self, name='random', N=100, S=0, R=1, g=1.2, fb_type='random', seed=1):
        self.name = name
        npr.seed(seed)
        # network parameters
        self.N = N  # RNN units
        self.dt = .1  # time bin (in units of tau)
        self.g = g  # initial input weight scale
        self.S = S  # input
        self.R = R  # readout
        self.sig = 0.001  # initial activity noise
        self.z0 = []    # initial condition
        self.ha_before, self.ha, self.ra, self.ua = [], [], [], []      # activity, output
        self.ws = (2 * npr.random((N, S)) - 1) / sqrt(S)                # input weights
        if self.name == 'random':
            self.J = self.g * npr.standard_normal([N, N]) / np.sqrt(N)      # recurrent weights
        elif self.name == 'hub':
            self.J = self.g * npr.standard_normal([N, N]) / np.sqrt(N)      # recurrent weights
            J_diag = np.diag(np.diag(self.J))
            self.J[1:, 1:] = 0
            self.J[1:, 0] = self.J[1:, 0] * np.sqrt(N)
            # self.J += J_diag * np.sqrt(N/2)
        self.wr = (2 * npr.random((R, N)) - 1) / sqrt(N)                # readout weights
        self.fb_type = fb_type
        if fb_type == 'random':
            self.B = npr.standard_normal([N, R]) / sqrt(R)
        elif fb_type == 'aligned':
            self.B = self.wr.T * sqrt(N / R)

    # nlin
    def f(self, x):
        return np.tanh(x) if not torch.is_tensor(x) else torch.tanh(x)

    # derivative of nlin
    def df(self, x):
        return 1 / (np.cosh(10*np.tanh(x/10)) ** 2) if not torch.is_tensor(x) else 1 / (torch.cosh(10*torch.tanh(x/10)) ** 2)


class Task:
    def __init__(self, name='ComplexSine', duration=100, cycles=4, rand_init=False, dt=0.1):
        self.name = name
        self.rand_init = rand_init
        NT = int(duration / dt)
        # task parameters
        if self.name == 'ComplexSine':
            self.cycles, self.T, self.dt, self.NT = cycles, duration, dt, NT
            self.s = 0.0 * np.ones((0, NT))
            self.ustar = (np.sin(2 * pi * np.arange(NT) * cycles / (NT-1)) +
                          0.75 * np.sin(2 * 2 * pi * np.arange(NT) * cycles / (NT-1)) +
                          0.5 * np.sin(4 * 2 * pi * np.arange(NT) * cycles / (NT-1)) +
                          0.25 * np.sin(6 * 2 * pi * np.arange(NT) * cycles / (NT-1)))

    def loss(self, err):
        mse = (err ** 2).mean() / 2
        return mse


class Algorithm:
    def __init__(self, name='rflo', Nepochs=10000, lr=1e-1, online=False):
        self.name = name
        # learning parameters
        self.Nepochs = Nepochs
        self.Nstart_anneal = 30000
        self.lr = lr  # learning rate
        self.annealed_lr = 1e-6
        if self.name == 'rflo' or self.name == 'neural':
            self.online = online
        else:
            self.online = False
