from train import train_rflo, train_neural, train_bptt
import sys
import numpy as np
import torch


modelname = sys.argv[1]
seed = int(sys.argv[2]) if len(sys.argv) > 2 else 1

if modelname == 'rflo_random':
    # initialize
    net = np.empty(1, dtype=object)
    task = np.empty(1, dtype=object)
    algo = np.empty(1, dtype=object)
    learning = np.empty(1, dtype=object)
    # train using rflo
    lr = 5e-3                       # learning rate
    net[0], task[0], algo[0], learning[0] = \
        train_rflo(arch='random', N=10, S=0, R=1, g=1.5,
                   task='ComplexSine', duration=10, cycles=1,
                   algo='rflo', Nepochs=10000, lr=lr, online=True, seed=seed)

elif modelname == 'neural_random':
    # initialize
    net = np.empty(1, dtype=object)
    task = np.empty(1, dtype=object)
    algo = np.empty(1, dtype=object)
    learning = np.empty(1, dtype=object)
    # train using bptt
    lr = 5e-3                       # learning rate
    net[0], task[0], algo[0], learning[0] = \
        train_neural(arch='random', N=10, S=0, R=1, g=1.5,
                     task='ComplexSine', duration=10, cycles=1,
                     algo='neural', Nepochs=10000, lr=lr, online=True, seed=seed)

elif modelname == 'bptt_random':
    # initialize
    net = np.empty(1, dtype=object)
    task = np.empty(1, dtype=object)
    algo = np.empty(1, dtype=object)
    learning = np.empty(1, dtype=object)
    # train using bptt
    lr = 5e-3                       # learning rate
    net[0], task[0], algo[0], learning[0] = \
        train_bptt(arch='random', N=10, S=0, R=1, g=1.5,
                   task='ComplexSine', duration=10, cycles=1,
                   algo='Adam', Nepochs=10000, lr=lr, seed=seed)

elif modelname == 'rflo_hub':
    # initialize
    net = np.empty(1, dtype=object)
    task = np.empty(1, dtype=object)
    algo = np.empty(1, dtype=object)
    learning = np.empty(1, dtype=object)
    # train using rflo
    lr = 5e-3                       # learning rate
    net[0], task[0], algo[0], learning[0] = \
        train_rflo(arch='hub', N=34, S=0, R=1, g=1.5,
                   task='ComplexSine', duration=10, cycles=1,
                   algo='rflo', Nepochs=10000, lr=lr, online=True, seed=seed)

elif modelname == 'neural_hub':
    # initialize
    net = np.empty(1, dtype=object)
    task = np.empty(1, dtype=object)
    algo = np.empty(1, dtype=object)
    learning = np.empty(1, dtype=object)
    # train using bptt
    lr = 5e-3                       # learning rate
    net[0], task[0], algo[0], learning[0] = \
        train_neural(arch='hub', N=34, S=0, R=1, g=1.5,
                     task='ComplexSine', duration=10, cycles=1,
                     algo='neural', Nepochs=10000, lr=lr, online=True, seed=seed)

elif modelname == 'bptt_hub':
    # initialize
    net = np.empty(1, dtype=object)
    task = np.empty(1, dtype=object)
    algo = np.empty(1, dtype=object)
    learning = np.empty(1, dtype=object)
    # train using bptt
    lr = 5e-3                       # learning rate
    net[0], task[0], algo[0], learning[0] = \
        train_bptt(arch='hub', N=34, S=0, R=1, g=1.5,
                   task='ComplexSine', duration=10, cycles=1,
                   algo='Adam', Nepochs=10000, lr=lr, seed=seed)

# save
torch.save({'net': net, 'task': task, 'algo': algo, 'learning': learning},
           '//burg//theory//users//jl5649//neural-learning//' + modelname + '//' + str(seed) + '.pt')