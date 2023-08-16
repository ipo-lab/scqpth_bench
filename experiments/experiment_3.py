import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scqpth.control import scqpth_control
from scqpth.scqpth import SCQPTHNet
from experiments.utils import create_qp_data, plot_profile_bars
from qpth.qp import QPFunction
import time as time
import matplotlib.pyplot as plt

# --- create problem data
n_x = 100
m = round(1 * n_x)#1 and 5
prob = 0.15
n_features = 5
learning_rate = 5e-3
n_batch = 128
n_sims = 10
tol = 10 ** -3
n_epochs = 100
n_mini_batch = 32

if tol < 10**-3:
    folder = 'images/experiment_3/high_tol/'
else:
    folder = 'images/experiment_3/low_tol/'
file_name_profile = folder + 'n_x_' + str(n_x) + '_m_' + str(m) + '.pdf'
file_name_loss = folder + 'n_x_' + str(n_x) + '_m_' + str(m) + '_loss.pdf'

# --- set models:
control_scqpth = scqpth_control(eps_abs=tol, eps_rel=tol)
QP_scqpth = SCQPTHNet(control_scqpth)

control_optnet = {"eps": tol, "verbose": 0, "notImprovedLim": 3, "maxIter": 20, "check_Q_spd": False}
QP_optnet = QPFunction(**control_optnet)

models = {"SCQPTH": QP_scqpth, "QPTH": QP_optnet}
model_names = models.keys()

# --- storage:
loss_hist = np.zeros((n_sims, n_epochs, len(model_names)))
forward_times = np.zeros((n_sims, len(model_names)))
forward_times = pd.DataFrame(forward_times, columns=model_names)
backward_times = np.zeros((n_sims, len(model_names)))
backward_times = pd.DataFrame(backward_times, columns=model_names)
total_times = np.zeros((n_sims, len(model_names)))
total_times = pd.DataFrame(total_times, columns=model_names)

# --- main loop:
for i in range(n_sims):
    print('simulation = {:d}'.format(i))
    k = -1
    for model_name in model_names:
        # --- define data:
        torch.manual_seed(i)
        k = k+1
        Q, p, A, lb, ub, G, h = create_qp_data(n_x=n_x, m=m, n_batch=n_batch, prob=prob, seed=i)
        QP = models.get(model_name)
        x = torch.normal(mean=0, std=1, size=(n_batch, n_features), dtype=Q.dtype)
        beta = torch.normal(mean=0, std=1, size=(n_features, n_x), dtype=Q.dtype)
        p = torch.matmul(x, beta).unsqueeze(2)

        # --- define the model and optimizer:
        model = torch.nn.Linear(n_features, n_x, dtype=Q.dtype)
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

        # ---- main training loop
        for epoch in range(n_epochs):

            # --- mini batch index:
            idx = np.random.randint(low=0, high=n_batch, size=n_mini_batch)

            # get output from the model, given the inputs
            p_hat = model(x[idx, :])
            p_hat = p_hat.unsqueeze(2)

            # --- invoke optimization layer:
            if model_name == "QPTH":
                # --- Forward
                start = time.time()
                z = QP(Q[idx, :, :], p_hat[:, :, 0], G[idx, :, :], h[idx, :, 0],
                       torch.empty(0, dtype=Q.dtype), torch.empty(0, dtype=Q.dtype))
                z = z.unsqueeze(2)
                forward_time = time.time() - start
            else:
                # --- Forward
                start = time.time()
                z = QP.forward(Q=Q[idx, :, :], p=p_hat, A=A[idx, :, :], lb=lb[idx, :, :],
                               ub=ub[idx, :, :])
                forward_time = time.time() - start
            # --- evaluate losss
            loss = 0.5 * torch.matmul(torch.matmul(torch.transpose(z, 1, 2), Q[idx, :, :]), z).sum() + (
                        p[idx, :, :] * z).sum()

            optimizer.zero_grad()
            # --- compute gradients
            start = time.time()
            loss.backward()
            backward_time = time.time() - start
            # --- update parameters
            optimizer.step()
            print('epoch {}, loss {}'.format(epoch, loss.item()))

            # --- storage:
            loss_hist[i, epoch, k] = loss.item()
            total_time = forward_time + backward_time
            forward_times[model_name][i] += forward_time
            backward_times[model_name][i] += backward_time
            total_times[model_name][i] += total_time

# --- median runtime
plot_profile_bars(backward_times, forward_times, total_times, n_sims, logy=True)
plt.savefig(file_name_profile)

# --- convergence plots:
loss_min = loss_hist.min()
loss_dat = loss_hist / abs(loss_min)
loss_mean = pd.concat({
    "SCQPTH": pd.DataFrame(loss_dat[:, :, 0].mean(axis=0))[0],
    "QPTH": pd.DataFrame(loss_dat[:, :, 1].mean(axis=0))[0]
}, axis=1)
loss_error = pd.concat({
    "SCQPTH": pd.DataFrame(loss_dat[:, :, 0].std(axis=0))[0],
    "QPTH": pd.DataFrame(loss_dat[:, :, 1].std(axis=0))[0]
}, axis=1)
loss_error = loss_error / n_sims ** 0.5
color2 = ["#0000EE", "#CD3333"]
loss_mean.plot.line(ylabel='Normalized QP Loss', xlabel='Epoch', color=color2, linewidth=4)  # title=title
plt.fill_between(np.arange(n_epochs), loss_mean['SCQPTH'] - 2 * loss_error['SCQPTH'],
                 loss_mean['SCQPTH'] + 2 * loss_error['SCQPTH'], alpha=0.25, color=color2[0])
plt.fill_between(np.arange(n_epochs), loss_mean['QPTH'] - 2 * loss_error['QPTH'],
                 loss_mean['QPTH'] + 2 * loss_error['QPTH'], alpha=0.25, color=color2[1])
plt.savefig(file_name_loss)
