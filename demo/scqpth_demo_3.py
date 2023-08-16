import torch
import time as time
from scqpth.control import scqpth_control
from scqpth.scqpth import SCQPTHNet
from experiments.utils import create_qp_data

# --- params
n_x = 100
n_batch = 32
m = 2 * n_x
prob = 0.15
seed = 0
dtytpe = torch.float64

# --- create problem data
Q, p, A, lb, ub, G, h = create_qp_data(n_x=n_x, m=m, n_batch=n_batch, prob=prob, seed=seed)

# --- params
eps_abs = 10 ** -5
eps_rel = 10 ** -5
check_solved = 10

control = scqpth_control(eps_abs=eps_abs, eps_rel=eps_rel, verbose=True, scale=True, check_solved=check_solved,
                         alpha=1.2, adaptive_rho_tol=5, adaptive_rho=True, unroll=False)

# --- Solve time
model = SCQPTHNet(control)

start = time.time()
x = model.forward(Q, p, A, lb, ub)
end = time.time() - start
print('computation time: {:f}'.format(end))

dl_dz = torch.rand((n_batch, n_x, 1), dtype=Q.dtype)
start = time.time()
test = x.backward(dl_dz)
end = time.time() - start
print('computation time: {:f}'.format(end))


