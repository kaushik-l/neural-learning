from train import train_rflo, train_bptt
import sys
import numpy as np
import torch


modelname = sys.argv[1]
seed = int(sys.argv[2]) if len(sys.argv) > 2 else 1

if modelname == 'rflo':
    # initialize
    net = np.empty(1, dtype=object)
    task = np.empty(1, dtype=object)
    algo = np.empty(1, dtype=object)
    learning = np.empty(1, dtype=object)
    # train using rflo
    lr = 1e-1                       # learning rate
    net[0], task[0], algo[0], learning[0] = \
        train_rflo(arch='rnn_rflo', N=20, S=0, R=1, g=1.5,
                   task='ComplexSine', duration=40, cycles=4,
                   algo='rflo', Nepochs=10000, lr=lr, online=True, seed=seed)

elif modelname == 'bptt':
    # initialize
    net = np.empty(1, dtype=object)
    task = np.empty(1, dtype=object)
    algo = np.empty(1, dtype=object)
    learning = np.empty(1, dtype=object)
    # train using bptt
    lr = 5e-3                       # learning rate
    net[0], task[0], algo[0], learning[0] = \
        train_bptt(arch='rnn_bptt', N=20, S=0, R=1, g=1.5,
                   task='ComplexSine', duration=40, cycles=4,
                   algo='Adam', Nepochs=10000, lr=lr, seed=seed)

# save
torch.save({'net': net, 'task': task, 'algo': algo, 'learning': learning},
           '//burg//theory//users//jl5649//neural-learning//' + modelname + '//' + str(seed) + '.pt')