import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scqpth.control import scqpth_control
from scqpth.scqpth import SCQPTHNet

from lqp_py.solve_box_qp_admm_torch import SolveBoxQP
from lqp_py.control import box_qp_control

from qpth.qp import QPFunction

import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer

from experiments.utils import create_box_qp_data, plot_profile_bars
import time as time

# --- create problem data
n_list = [10, 25, 50, 100, 250, 500, 750, 1000]
n_x = n_list[0]
prob = 0.50
n_batch = 32
n_sims = 10
tol = 10 ** -3
if tol < 10**-3:
    folder = 'images/experiment_1/high_tol/'
else:
    folder = 'images/experiment_1/low_tol/'
file_name = folder + '/n_x_' + str(n_x) + '.pdf'


# --- set models:
control_scqpth = scqpth_control(eps_abs=tol, eps_rel=tol)
QP_scqpth = SCQPTHNet(control_scqpth)

control_fp = box_qp_control(max_iters=10_000, eps_rel=tol, eps_abs=tol)
QP_lqp = SolveBoxQP(control=control_fp)

control_optnet = {"eps": tol, "verbose": 0, "notImprovedLim": 3, "maxIter": 20, "check_Q_spd": False}
QP_optnet = QPFunction(**control_optnet)

# --- cvx:
if n_x <= 500:
    Q_sqrt = cp.Parameter((n_x, n_x))
    p_cvx = cp.Parameter(n_x)
    G_cvx = cp.Parameter((2 * n_x, n_x))
    h_cvx = cp.Parameter(2 * n_x)
    x_cvx = cp.Variable(n_x)
    z_cvx = cp.Variable(1)
    obj = cp.Minimize(0.5 * z_cvx + p_cvx.T @ x_cvx)
    cons = [G_cvx @ x_cvx <= h_cvx, cp.sum_squares(Q_sqrt @ x_cvx) <= z_cvx]
    problem = cp.Problem(obj, cons)

    QP_scs = CvxpyLayer(problem, parameters=[Q_sqrt, p_cvx, G_cvx, h_cvx], variables=[x_cvx, z_cvx])  # gp=False
    control_scs = {'max_iters': 10_000, "eps": tol}
else:
    QP_scs = None

models = {"SCQPTH": QP_scqpth,
          "LQP": QP_lqp,
          "QPTH": QP_optnet}
if QP_scs is not None:
    models["Cvxpylayers"] = QP_scs
model_names = list(models.keys())

# --- storage:
forward_times = np.zeros((n_sims, len(model_names)))
forward_times = pd.DataFrame(forward_times, columns=model_names)
backward_times = np.zeros((n_sims, len(model_names)))
backward_times = pd.DataFrame(backward_times, columns=model_names)
total_times = np.zeros((n_sims, len(model_names)))
total_times = pd.DataFrame(total_times, columns=model_names)

# --- main loop:
for i in range(n_sims):
    print('simulation = {:d}'.format(i))

    for model_name in model_names:
        print("model: {:s}".format(model_name))
        Q, p, A, lb, ub, G, h = create_box_qp_data(n_x=n_x, n_batch=n_batch, prob=prob, seed=i)
        QP = models.get(model_name)
        # --- forward:
        if model_name == "QPTH":
            p = p[:, :, 0]
            h = h[:, :, 0]
            lb = lb[:, :, 0]
            ub = ub[:, :, 0]
            dl_dz = torch.ones((n_batch, n_x))
        else:
            dl_dz = torch.ones((n_batch, n_x, 1))

        if model_name == "QPTH":
            # --- Forward
            start = time.time()
            x = QP(Q, p, G, h, torch.empty(0, dtype=Q.dtype), torch.empty(0, dtype=Q.dtype))
            forward_time = time.time() - start
            print('forward time: {:f}'.format(forward_time))
        elif model_name == "Cvxpylayers":
            # --- Forward
            start = time.time()
            x = QP_scs(torch.linalg.cholesky(Q, upper=True), p[:, :, 0], G, h[:, :, 0],
                       solver_args=control_scs)
            x = x[0]
            forward_time = time.time() - start
            print('forward time: {:f}'.format(forward_time))
        elif model_name == "LQP":
            # --- Forward
            start = time.time()
            x = QP.forward(Q=Q, p=p, A=None, b=None, lb=lb, ub=ub)
            forward_time = time.time() - start
            print('forward time: {:f}'.format(forward_time))
        else:
            # --- Forward
            start = time.time()
            x = QP.forward(Q=Q, p=p, A=A, lb=lb, ub=ub)
            forward_time = time.time() - start
            print('forward time: {:f}'.format(forward_time))

        # --- backward
        if model_name == "Cvxpylayers":
            start = time.time()
            x.sum().backward()
            backward_time = time.time() - start
            print('backward time: {:f}'.format(backward_time))
        else:
            start = time.time()
            test = x.backward(dl_dz)
            backward_time = time.time() - start
            print('backward time: {:f}'.format(backward_time))

        # --- total time:
        total_time = forward_time + backward_time
        print('total time: {:f}'.format(total_time))

        # --- storage:
        forward_times[model_name][i] = forward_time
        backward_times[model_name][i] = backward_time
        total_times[model_name][i] = total_time

# --- bar plot:
plot_profile_bars(backward_times, forward_times, total_times, n_sims, fontsize=10, logy=True)
plt.savefig(file_name)
