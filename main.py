from train import train_rflo, train_neural, train_bptt
import torch
import numpy as np

rflo_random, neural_random, bptt_random = True, True, True
rflo_hub, neural_hub, bptt_hub = False, False, False
nrepeats = 1

if rflo_random:
    net = np.empty(nrepeats, dtype=object)
    task = np.empty(nrepeats, dtype=object)
    algo = np.empty(nrepeats, dtype=object)
    learning = np.empty(nrepeats, dtype=object)
    for i in range(nrepeats):
        net[i], task[i], algo[i], learning[i] = \
            train_rflo(arch='hub', N=10, S=0, R=1, g=1.5,
                       task='ComplexSine', duration=10, cycles=1,
                       algo='rflo', Nepochs=10000, lr=5e-3, online=True, seed=6)
    torch.save({'net': net, 'task': task, 'algo': algo, 'learning': learning}, 'Data//rflo-random.pt')

if neural_random:
    net = np.empty(nrepeats, dtype=object)
    task = np.empty(nrepeats, dtype=object)
    algo = np.empty(nrepeats, dtype=object)
    learning = np.empty(nrepeats, dtype=object)
    for i in range(nrepeats):
        net[i], task[i], algo[i], learning[i] = \
            train_neural(arch='random', N=10, S=0, R=1, g=1.5,
                         task='ComplexSine', duration=10, cycles=1,
                         algo='neural', Nepochs=10000, lr=5e-3, online=True, seed=i)
    torch.save({'net': net, 'task': task, 'algo': algo, 'learning': learning}, 'Data//neural-random.pt')

if bptt_random:
    net = np.empty(nrepeats, dtype=object)
    task = np.empty(nrepeats, dtype=object)
    algo = np.empty(nrepeats, dtype=object)
    learning = np.empty(nrepeats, dtype=object)
    for i in range(nrepeats):
        net[i], task[i], algo[i], learning[i] = \
            train_bptt(arch='random', N=10, S=0, R=1, g=1.5,
                       task='ComplexSine', duration=10, cycles=1,
                       algo='Adam', Nepochs=10000, lr=5e-3, seed=i)
    torch.save({'net': net, 'task': task, 'algo': algo, 'learning': learning}, 'Data//bptt-random.pt')

if rflo_hub:
    net = np.empty(nrepeats, dtype=object)
    task = np.empty(nrepeats, dtype=object)
    algo = np.empty(nrepeats, dtype=object)
    learning = np.empty(nrepeats, dtype=object)
    for i in range(nrepeats):
        net[i], task[i], algo[i], learning[i] = \
            train_rflo(arch='hub', N=34, S=0, R=1, g=1.5,
                       task='ComplexSine', duration=10, cycles=1,
                       algo='rflo', Nepochs=10000, lr=5e-3, online=True, seed=i)
    torch.save({'net': net, 'task': task, 'algo': algo, 'learning': learning}, 'Data//rflo-hub.pt')

if neural_hub:
    net = np.empty(nrepeats, dtype=object)
    task = np.empty(nrepeats, dtype=object)
    algo = np.empty(nrepeats, dtype=object)
    learning = np.empty(nrepeats, dtype=object)
    for i in range(nrepeats):
        net[i], task[i], algo[i], learning[i] = \
            train_neural(arch='hub', N=34, S=0, R=1, g=1.5,
                         task='ComplexSine', duration=10, cycles=1,
                         algo='neural', Nepochs=10000, lr=5e-3, online=True, seed=i)
    torch.save({'net': net, 'task': task, 'algo': algo, 'learning': learning}, 'Data//neural-hub.pt')

if bptt_hub:
    net = np.empty(nrepeats, dtype=object)
    task = np.empty(nrepeats, dtype=object)
    algo = np.empty(nrepeats, dtype=object)
    learning = np.empty(nrepeats, dtype=object)
    for i in range(nrepeats):
        net[i], task[i], algo[i], learning[i] = \
            train_bptt(arch='hub', N=34, S=0, R=1, g=1.5,
                       task='ComplexSine', duration=10, cycles=1,
                       algo='Adam', Nepochs=10000, lr=5e-3, seed=i)
    torch.save({'net': net, 'task': task, 'algo': algo, 'learning': learning}, 'Data//bptt-hub.pt')