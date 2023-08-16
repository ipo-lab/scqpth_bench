import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


def plot_profile_bars(backward_times, forward_times, total_times, n_sims,  **kwargs):
    dat = pd.concat({
        'Backward': backward_times.median(axis=0),
        'Forward': forward_times.median(axis=0),
        "Total": total_times.median(axis=0)}, axis=1)

    # --- error bars:
    error = pd.concat({
        'Backward': backward_times.std(axis=0),
        'Forward': forward_times.std(axis=0),
        "Total": total_times.std(axis=0)}, axis=1)
    error = 1.96 * error / n_sims ** 0.5

    # --- set y lims:
    ymin = backward_times.min().min()
    ymin = np.log10(ymin)
    ymin = 10**np.floor(ymin)
    ymax = total_times.max().max()
    ymax = np.log10(ymax)
    ymax = 10**np.ceil(ymax)

    color = ["#E69F00", "#56B4E9", "#999999"]
    dat.plot.bar(ylabel='time (s)', rot=0, color=color, yerr=error, **kwargs)
    plt.ylabel('time (s)', fontsize=16)
    plt.ylim(ymin=ymin, ymax=ymax)
    return None


def torch_uniform(*size, lower=0, upper=1, dtype=torch.float64):
    r = torch.rand(*size, dtype=dtype)
    r = r * (upper - lower) + lower
    return r


def create_box_qp_data(n_x, n_batch, prob, seed=0, requires_grad=True, dtype=torch.float64, eps=0.01):
    torch.manual_seed(seed)
    np.random.seed(seed)

    # --- Q:
    Q = generate_random_Q_torch(n_x=n_x, n_batch=n_batch, prob=prob, eps=eps, dtype=dtype, requires_grad=requires_grad)

    # --- p:
    p = torch.randn(n_batch, n_x, 1, requires_grad=requires_grad, dtype=dtype)

    # --- A:
    A = torch.eye(n_x, dtype=dtype)
    A = A.unsqueeze(0) * torch.ones(n_batch, 1, 1, dtype=dtype)

    # --- lb and ub
    lb = -torch_uniform(n_batch, n_x, 1, lower=1, upper=2, dtype=dtype) / n_x
    ub = torch_uniform(n_batch, n_x, 1, lower=1, upper=2, dtype=dtype) / n_x

    # --- G and hfor optnet:
    G = torch.cat((-torch.eye(n_x, dtype=dtype), torch.eye(n_x, dtype=dtype)))
    G = G.unsqueeze(0) * torch.ones(n_batch, 1, 1, dtype=dtype)
    h = torch.cat((-lb, ub), dim=1)

    return Q, p, A, lb, ub, G, h


def create_qp_data(n_x, m, n_batch, prob=0.15, seed=0, requires_grad=True, dtype=torch.float64, eps=0.01):
    torch.manual_seed(seed)
    np.random.seed(seed)

    # --- Q:
    Q = generate_random_Q_torch(n_x=n_x, n_batch=n_batch, prob=0.50, eps=eps, dtype=dtype, requires_grad=requires_grad)

    # --- p:
    p = torch.randn(n_batch, n_x, 1, requires_grad=requires_grad, dtype=dtype)

    # --- A:
    A = generate_random_A_torch(n_x=n_x, m=m, n_batch=n_batch, prob=prob, dtype=dtype, requires_grad=requires_grad)

    # --- feasible and add slack:
    lb = -torch_uniform(n_batch, m, 1, lower=0, upper=1, dtype=dtype)
    ub = torch_uniform(n_batch, m, 1, lower=0, upper=1, dtype=dtype)

    # --- G and h for optnet:
    G = torch.cat((-A, A), dim=1)
    h = torch.cat((-lb, ub), dim=1)

    return Q, p, A, lb, ub, G, h


def generate_hard_qp(n_x, m, prob, seed=None):
    if seed is not None:
        np.random.seed(seed)
    M = np.random.normal(size=(n_x, n_x))
    M = M * np.random.binomial(1, prob, size=(n_x, n_x))
    # --- Q:
    Q = np.dot(M.T, M)/n_x + 1e-2 * np.eye(n_x)
    # --- p:
    p = np.random.normal(size=(n_x, 1))
    # --- A:
    A = generate_random_A(n_x=n_x, m=m, prob=prob)
    # --- lb and ub:
    lb = -np.random.uniform(size=(m, 1))
    ub = np.random.uniform(size=(m, 1))
    return Q, p, A, lb, ub


def generate_hard_qp_torch(n_x, m, n_batch, prob=0.15, seed=0, dtype=torch.float64, requires_grad=True):
    torch.manual_seed(seed)
    np.random.seed(seed)

    Q = np.zeros((n_batch, n_x, n_x))
    p = np.zeros((n_batch, n_x, 1))
    A = np.zeros((n_batch, m, n_x))
    lb = np.zeros((n_batch, m, 1))
    ub = np.zeros((n_batch, m, 1))
    for i in range(n_batch):
        Q_i, p_i, A_i, lb_i, ub_i = generate_hard_qp(n_x=n_x, m=m, prob=prob)
        Q[i, :, :] = Q_i
        p[i, :, :] = p_i
        A[i, :, :] = A_i
        lb[i, :, :] = lb_i
        ub[i, :, :] = ub_i

    Q = torch.tensor(Q, dtype=dtype, requires_grad=requires_grad)
    p = torch.tensor(p, dtype=dtype,  requires_grad=requires_grad)
    A = torch.tensor(A, dtype=dtype,  requires_grad=requires_grad)
    lb = torch.tensor(lb, dtype=dtype,  requires_grad=requires_grad)
    ub = torch.tensor(ub, dtype=dtype,  requires_grad=requires_grad)
    # --- add G, h
    G = torch.cat((-A, A), dim=1)
    h = torch.cat((-lb, ub), dim=1)

    return Q, p, A, lb, ub, G, h


def generate_random_A(n_x, m, prob):
    # --- A:
    A = np.zeros((m, n_x))
    for i in range(m):
        a = np.random.normal(size=(1, n_x))
        b = np.zeros(1)
        while b.sum() == 0:
            b = np.random.binomial(1, prob, size=(1, n_x))
        a = a*b
        A[i, :] = a

    return A


def generate_random_A_torch(n_x, m, n_batch, prob, dtype=torch.float64, requires_grad=True):

    A = np.zeros((n_batch, m, n_x))
    for i in range(n_batch):
        A[i, :, :] = generate_random_A(n_x=n_x, m=m, prob=prob)

    A = torch.tensor(A, dtype=dtype, requires_grad=requires_grad)

    return A


def generate_random_Q_torch(n_x, n_batch, prob, eps=0.01, dtype=torch.float64, requires_grad=True):
    Q = np.zeros((n_batch, n_x, n_x))
    # --- Q:
    for i in range(n_batch):
        M = np.random.normal(size=(n_x, n_x))
        M = M * np.random.binomial(1, prob, size=(n_x, n_x))
        # --- Q:
        Q[i, :, :] = np.dot(M.T, M) + eps * np.eye(n_x)

    Q = torch.tensor(Q, dtype=dtype, requires_grad=requires_grad)
    return Q

