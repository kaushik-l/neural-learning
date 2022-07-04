import scipy.io

from train import train_rflo, train_bptt
import torch

rflo, bptt = True, True

if rflo:
    net, task, algo, learning = \
        train_rflo(arch='rnn_rflo', N=20, S=0, R=1, g=1.5,
                   task='ComplexSine', duration=40, cycles=1,
                   algo='rflo', Nepochs=1000, lr=1e-1, online=True, seed=1)
    torch.save({'net': net, 'task': task, 'algo': algo, 'learning': learning}, 'Data//rflo.pt')

if bptt:
    net, task, algo, learning = \
        train_bptt(arch='rnn_bptt', N=20, S=0, R=1, g=1.5,
                   task='ComplexSine', duration=40, cycles=4,
                   algo='Adam', Nepochs=1000, lr=5e-3, seed=1)
    torch.save({'net': net, 'task': task, 'algo': algo, 'learning': learning}, 'Data//bptt.pt')