import torch
import time as time
from scqpth.control import scqpth_control
from scqpth.scqpth import SCQPTHNet


# --- create problem data
torch.manual_seed(0)
dtype = torch.float64
n_x = 100
n_batch = 32
n_samples = 2 * n_x
L = torch.randn(n_batch, n_samples, n_x, dtype=dtype)

Q = torch.matmul(torch.transpose(L, 1, 2), L)
Q = Q / n_samples
p = torch.randn(n_batch, n_x, 1)

A = torch.eye(n_x, dtype=dtype)
A = A.unsqueeze(0) * torch.ones(n_batch, 1, 1, dtype=dtype)
n_A = A.shape[1]
lb = -torch.ones(n_batch, A.shape[1], 1, dtype=dtype)
ub = torch.ones(n_batch, A.shape[1], 1, dtype=dtype)

Q.requires_grad = True
p.requires_grad = True
A.requires_grad = True
lb.requires_grad = True
ub.requires_grad = True

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

dl_dz = torch.rand((n_batch, n_x, 1), dtype=dtype)
start = time.time()
test = x.backward(dl_dz)
end = time.time() - start
print('computation time: {:f}'.format(end))


