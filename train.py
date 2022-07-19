import torch
from time import time
import numpy as np
import numpy.random as npr
from model import Network, Task, Algorithm
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt

# seed
npr.seed(1)


def unstack(a, axis=0):
    return [np.squeeze(e, axis) for e in np.split(a, a.shape[axis], axis=axis)]


def dimensionality(x):
    return (x.sum() ** 2) / (x ** 2).sum()


def simulate(net, task):
    # frequently used vars
    S, N, R, NT = net.S, net.N, net.R, task.NT
    # initialize output
    ha = np.zeros((NT, N))  # cortex hidden states
    ua = np.zeros((NT, R))  # angular acceleration of joints
    err = np.zeros((NT, R))  # error
    # initialize activity
    z0 = net.z0 + 0.1*npr.randn(N, 1) if task.rand_init else net.z0  # hidden state (potential)
    h0 = net.f(z0)  # hidden state (rate)
    z, h = z0, h0
    for ti in range(NT):
        # currents
        Iin = np.matmul(net.ws, task.s[:, ti])[:, None] if net.S else 0
        Irec = np.matmul(net.J, h)
        z = Iin + Irec
        # update activity
        h = (1 - net.dt) * h + net.dt * (net.f(z))  # cortex
        u = np.matmul(net.wr, h)  # output
        # error
        err[ti] = task.ustar[ti] - u
        # save values
        ha[ti], ua[ti] = h.T, u.T
    mse = task.loss(err)
    return ha, ua, err, mse


# train using rflo
def train_rflo(arch='rnn', N=100, S=0, R=1, g=1.2, task='ComplexSine', duration=100, cycles=1,
               rand_init=False, algo='rflo', fb_type='random', Nepochs=10000, lr=1e-1, online=True, seed=1):

    # instantiate model
    net = Network(arch, N, S, R, g=g, fb_type=fb_type, seed=seed)
    task = Task(task, duration, cycles, rand_init=rand_init)
    algo = Algorithm(algo, Nepochs, lr, online)

    # frequently used vars
    dt, NT, N, S, R = net.dt, task.NT, net.N, net.S, net.R
    t = dt * np.arange(NT)

    # track variables during learning
    learning = {'epoch': [], 'lr': [], 'mses': [],
                'ev_svd': [], 'ev_diag': [], 'dim_svd': [], 'J0': np.empty_like(net.J),
                'kernel': [], 'ua_ei': [], 'kernel_ei': [],
                'alignment': {'feedback': []},
                'test_ua': [], 'test_err': [], 'test_mse': []
                }

    # random initialization of hidden state
    z0 = npr.randn(N, 1)    # hidden state (potential)
    net.z0 = z0  # save

    # save initial weights
    learning['J0'][:] = net.J

    # learning rates
    lr_in = lr_cc = lr_out = algo.lr

    # statistical analysis
    reg = LinearRegression()
    svd = TruncatedSVD(n_components=int(N - 1))

    for ei in range(algo.Nepochs):

        # initialize activity
        z0 = net.z0 + 0.1*npr.randn(N, 1) if task.rand_init else net.z0  # hidden state (potential)
        h0 = net.f(z0)  # hidden state (rate)
        z, h = z0, h0

        # save tensors for plotting
        sa = np.zeros((NT, S))  # save the inputs for each time bin for plotting
        ha = np.zeros((NT, N))  # save the hidden states for each time bin for plotting
        ua = np.zeros((NT, R))  # angular acceleration of joints

        # errors
        err = np.zeros((NT, R))     # error in angular acceleration

        # eligibility traces o, p
        o = net.df(z) * sa[0]
        p = net.df(z) * ha[0]

        # store weight changes for offline learning
        dws = np.zeros_like(net.ws)
        dwr = np.zeros_like(net.wr)
        dJ = np.zeros_like(net.J)

        for ti in range(NT):

            # network update
            s = task.s[:, ti]
            Iin = np.matmul(net.ws, s)[:, None] if net.S else 0
            Irec = np.matmul(net.J, h)
            z = Iin + Irec                  # potential

            # update eligibility trace
            o = dt * net.df(z) * s.T + (1 - dt) * o
            p = dt * net.df(z) * h.T + (1 - dt) * p

            # update activity
            h = (1 - dt) * h + dt * (net.f(z))  # cortex
            u = np.matmul(net.wr, h)  # output

            # save values for plotting
            sa[ti], ha[ti], ua[ti] = s.T, h.T, u.T

            # error
            err[ti] = task.ustar[ti] - u

            # online weight update
            if algo.online:
                if lr_in:
                    net.ws += ((lr_in / NT) * np.matmul(net.B, err[ti]).reshape(N, 1) * o)
                if lr_out:
                    net.wr += (((lr_out / NT) * h) * err[ti]).T
                if lr_cc:
                    net.J += ((lr_cc / NT) * np.matmul(net.B, err[ti]).reshape(N, 1) * p)
                net.B = net.wr.T * np.sqrt(N / R) if fb_type == 'aligned' else net.B    # realign feedback if needed
            else:
                if lr_in:
                    dws += ((lr_in / NT) * np.matmul(net.B, err[ti]).reshape(N, 1) * o)
                if lr_out:
                    dwr += (((lr_out / NT) * h) * err[ti]).T
                if lr_cc:
                    dJ += ((lr_cc / NT) * np.matmul(net.B, err[ti]).reshape(N, 1) * p)

        # offline update
        if not algo.online:
            net.ws += dws
            net.wr += dwr
            net.J += dJ
            net.B = net.wr.T * np.sqrt(N / R) if fb_type == 'aligned' else net.B    # realign feedback if needed

        # compute overlap
        learning['alignment']['feedback'].append((net.wr.flatten() @ net.B.flatten('F')) /
                                                 (np.linalg.norm(net.wr.flatten()) * np.linalg.norm(net.B.flatten('F'))))

        # kernel
        epochs = np.concatenate((np.arange(1, 256, 2), 2 ** np.arange(8, int(np.ceil(np.log2(Nepochs))))))
        if ei+1 in epochs and lr_cc > 0:
            ha_ = ha - ha.mean(axis=1)[:, None].repeat(N,1)
            learning['kernel'].append(np.matmul(ha_, ha_.T))
            learning['ua_ei'].append(ua)
            learning['kernel_ei'].append(ei+1)

        # print loss
        mse = task.loss(err)
        if (ei+1) % 10 == 0: print('\r' + str(ei + 1) + '/' + str(algo.Nepochs) + '\t Err:' + str(mse), end='')

        # save mse list and cond list
        learning['mses'].append(mse)

        # adaptive learning rate
        if lr_cc:
            lr_cc *= np.exp(np.log(np.minimum(1e-2, lr_cc) / lr_cc) / Nepochs)
            learning['lr'].append(lr_cc)
            learning['epoch'].append(ei)

    # save input, activity, output
    net.ha, net.ua = ha, ua
    _, learning['test_ua'], learning['test_err'], learning['test_mse'] = simulate(net, task)

    return net, task, algo, learning


# train using neural
def train_neural(arch='rnn', N=100, S=0, R=1, g=1.2, task='ComplexSine', duration=100, cycles=1,
               rand_init=False, algo='neural', fb_type='random', Nepochs=10000, lr=1e-1, online=True, seed=1):

    # instantiate model
    net = Network(arch, N, S, R, g=g, fb_type=fb_type, seed=seed)
    task = Task(task, duration, cycles, rand_init=rand_init)
    algo = Algorithm(algo, Nepochs, lr, online)

    # frequently used vars
    dt, NT, N, S, R = net.dt, task.NT, net.N, net.S, net.R
    t = dt * np.arange(NT)

    # track variables during learning
    learning = {'epoch': [], 'lr': [], 'mses': [],
                'ev_svd': [], 'ev_diag': [], 'dim_svd': [], 'J0': np.empty_like(net.J),
                'kernel': [], 'ua_ei': [], 'kernel_ei': [],
                'alignment': {'feedback': []},
                'test_ua': [], 'test_err': [], 'test_mse': []
                }

    # random initialization of hidden state
    z0 = npr.randn(N, 1)    # hidden state (potential)
    net.z0 = z0  # save

    # save initial weights
    learning['J0'][:] = net.J

    # learning rates
    lr_in = lr_cc = lr_out = algo.lr

    # statistical analysis
    reg = LinearRegression()
    svd = TruncatedSVD(n_components=int(N - 1))

    for ei in range(algo.Nepochs):

        # initialize activity
        z0 = net.z0 + 0.1*npr.randn(N, 1) if task.rand_init else net.z0  # hidden state (potential)
        h0 = net.f(z0)  # hidden state (rate)
        z, h = z0, h0

        # save tensors for plotting
        sa = np.zeros((NT, S))  # save the inputs for each time bin for plotting
        ha = np.zeros((NT, N))  # save the hidden states for each time bin for plotting
        ua = np.zeros((NT, R))  # angular acceleration of joints

        # errors
        err = np.zeros((NT, R))     # error in angular acceleration

        # eligibility traces o, p
        o = np.zeros((N, N, S))
        p_old = np.zeros((N, N, N))

        # store weight changes for offline learning
        dws = np.zeros_like(net.ws)
        dwr = np.zeros_like(net.wr)
        dJ = np.zeros_like(net.J)

        for ti in range(NT):

            # network update
            s = task.s[:, ti]
            Iin = np.matmul(net.ws, s)[:, None] if net.S else 0
            Irec = np.matmul(net.J, h)
            z = Iin + Irec                  # potential

            # update eligibility trace
            p = np.zeros((N, N, N))
            cov = net.df(z) * h.T
            for i in range(N):
                cov_i = np.zeros((N, N))
                cov_i[i] = cov[i]
                p[i] = (1 - dt) * p_old[i] + dt * cov_i + dt * net.df(z[i]) * np.einsum('j,jkl', net.J[i], p_old)

            # update activity
            h = (1 - dt) * h + dt * (net.f(z))  # cortex
            u = np.matmul(net.wr, h)  # output

            # save values for plotting
            sa[ti], ha[ti], ua[ti] = s.T, h.T, u.T

            # error
            err[ti] = task.ustar[ti] - u

            # online weight update
            if algo.online:
                if lr_out:
                    net.wr += (((lr_out / NT) * h) * err[ti]).T
                if lr_cc:
                    for i in range(N):
                        net.J[i] += ((lr_cc / NT) * net.B[i] * err[ti] * p[i, i])
                net.B = net.wr.T * np.sqrt(N / R) if fb_type == 'aligned' else net.B    # realign feedback if needed
            else:
                if lr_out:
                    dwr += (((lr_out / NT) * h) * err[ti]).T
                if lr_cc:
                    for i in range(N):
                        dJ[i] += ((lr_cc / NT) * net.B[i] * err[ti] * p[i, i])
            p_old = p

        # offline update
        if not algo.online:
            net.ws += dws
            net.wr += dwr
            net.J += dJ
            net.B = net.wr.T * np.sqrt(N / R) if fb_type == 'aligned' else net.B    # realign feedback if needed

        # compute overlap
        learning['alignment']['feedback'].append((net.wr.flatten() @ net.B.flatten('F')) /
                                                 (np.linalg.norm(net.wr.flatten()) * np.linalg.norm(net.B.flatten('F'))))

        # kernel
        epochs = np.concatenate((np.arange(1, 256, 2), 2 ** np.arange(8, int(np.ceil(np.log2(Nepochs))))))
        if ei+1 in epochs and lr_cc > 0:
            ha_ = ha - ha.mean(axis=1)[:, None].repeat(N,1)
            learning['kernel'].append(np.matmul(ha_, ha_.T))
            learning['ua_ei'].append(ua)
            learning['kernel_ei'].append(ei+1)

        # print loss
        mse = task.loss(err)
        if (ei+1) % 10 == 0: print('\r' + str(ei + 1) + '/' + str(algo.Nepochs) + '\t Err:' + str(mse), end='')

        # save mse list and cond list
        learning['mses'].append(mse)

        # adaptive learning rate
        if lr_cc:
            lr_cc *= np.exp(np.log(np.minimum(1e-2, lr_cc) / lr_cc) / Nepochs)
            learning['lr'].append(lr_cc)
            learning['epoch'].append(ei)

    # save input, activity, output
    net.ha, net.ua = ha, ua
    _, learning['test_ua'], learning['test_err'], learning['test_mse'] = simulate(net, task)

    return net, task, algo, learning


# train using backprop
def train_bptt(arch='ThCtx', N=256, S=0, R=1, g=1.2, task='ComplexSine', duration=100, cycles=1,
               rand_init=False, algo='Adam', Nepochs=10000, lr=1e-3, seed=1):

    # sites
    sites = ('ws', 'J', 'wr')

    # instantiate model
    net = Network(arch, N, S, R, g=g, seed=seed)
    task = Task(task, duration, cycles, rand_init=rand_init)
    algo = Algorithm(algo, Nepochs, lr)

    # convert to tensor
    for site in sites:
        setattr(net, site, torch.tensor(getattr(net, site), requires_grad=True))

    # frequently used vars
    dt, NT, N, S, R = net.dt, task.NT, net.N, net.S, net.R
    t = dt * np.arange(NT)

    # track variables during learning
    learning = {'epoch': [], 'ev_svd': [], 'ev_diag': [], 'dim_svd': [], 'kernel': [], 'kernel_ei': [], 'ua_ei': [],
                'J0': [], 'mses': [], 'snr_mse': [], 'lr': []}

    # optimizer
    opt = None
    if algo.name == 'Adam':
        opt = torch.optim.Adam([getattr(net, site) for site in sites], lr=algo.lr)

    # random initialization of hidden state
    z0 = npr.randn(N, 1)    # hidden state (potential)
    net.z0 = z0             # save

    # save initial weights
    learning['J0'] = net.J.detach().clone()
    svd = TruncatedSVD(n_components=int(N - 1))
    reg = LinearRegression()

    # train
    for ei in range(algo.Nepochs):

        # initialize activity
        z0 = net.z0 + 0.1*npr.randn(N, 1) if task.rand_init else net.z0  # hidden state (potential)
        h0 = net.f(z0)  # hidden state (rate)
        z, h = torch.as_tensor(z0), torch.as_tensor(h0)

        # save tensors for plotting
        sa = torch.zeros(NT, S)  # save the inputs for each time bin for plotting
        ha = torch.zeros(NT, N)  # save the hidden states for each time bin for plotting
        ua = torch.zeros(NT, R)  # angular acceleration of joints

        # errors
        err = torch.zeros(NT, R)     # error in angular acceleration

        for ti in range(NT):
            # network update
            s = torch.tensor(task.s)  # input
            Iin = net.ws.mm(s[:, ti]) if net.S else 0
            Irec = net.J.mm(h) #np.matmul(net.J, h)
            z = Iin + Irec    # potential

            # update activity
            h = (1 - dt) * h + dt * (net.f(z))  # cortex
            u = net.wr.mm(h)  # output

            # save values for plotting
            ha[ti], ua[ti] = h.T, u.T

            # error
            err[ti] = task.ustar[ti] - u

        # print loss
        loss = task.loss(err)
        if (ei+1) % 10 == 0: print('\r' + str(ei + 1) + '/' + str(algo.Nepochs) + '\t Err:' + str(loss.item()), end='')

        # save mse list and cond list
        learning['mses'].append(loss.item())

        # do BPTT
        loss.backward()
        opt.step()
        opt.zero_grad()

        # kernel
        epochs = np.concatenate((np.arange(1, 256, 2), 2 ** np.arange(8, int(np.ceil(np.log2(Nepochs))))))
        if ei+1 in epochs:
            ha_ = (ha - ha.mean(dim=1).repeat(N, 1).T).detach().numpy()
            learning['kernel'].append(np.matmul(ha_, ha_.T))
            learning['ua_ei'].append(ua.detach().numpy())
            learning['kernel_ei'].append(ei+1)

    # save input, activity, output
    net.ha, net.ua = ha.detach().numpy(), ua.detach().numpy()
    for site in sites:
        setattr(net, site, getattr(net, site).detach().numpy())
    _, learning['test_ua'], learning['test_err'], learning['test_mse'] = simulate(net, task)

    return net, task, algo, learning
