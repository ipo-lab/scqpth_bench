import torch
import time as time
from scqpth.control import scqpth_control
from scqpth.scqpth import SCQPTH, scqpth_solve, scqpth_grad, scqpth_grad_kkt
import matplotlib.pyplot as plt

# --- create problem data
torch.manual_seed(0)
dtype = torch.float64
n_x = 500
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
model = SCQPTH(Q, p, A, lb, ub, control)

start = time.time()
sol = model.solve()
end = time.time() - start
print('computation time: {:f}'.format(end))

control = scqpth_control(eps_abs=eps_abs, eps_rel=eps_rel, verbose=True, scale=True, check_solved=check_solved,
                         alpha=1.2, adaptive_rho_tol=5, adaptive_rho=True, unroll=True)
start = time.time()
x = scqpth_solve(Q, p, A, lb, ub, control)
end = time.time() - start
print('computation time: {:f}'.format(end))

# --- unroll grads:
dl_dz = torch.rand((n_batch, n_x, 1), dtype=dtype)
test = x.backward(dl_dz)

# --- robust:
x = model.sol.get('x')
z = model.sol.get('z')
y = model.sol.get('y')
rho = model.sol.get('rho')

start = time.time()
grads = scqpth_grad(dl_dz, x, z, y, Q, A, lb, ub, rho=rho, ATA=None, LU=None, P=None, eps=1e-6)
end = time.time() - start
print('computation time: {:f}'.format(end))

start = time.time()
scqpth_grad_kkt(dl_dz, x, y, Q, A, lb, ub)
end = time.time() - start
print('computation time: {:f}'.format(end))


plt.plot(p.grad.sum(dim=0))
plt.plot(grads[1].detach().sum(dim=0))

plt.plot(A.grad.sum(dim=0).reshape(n_x * n_A))
plt.plot(grads[2].detach().sum(dim=0).reshape(n_x * n_A))

plt.plot(lb.grad.sum(dim=0))
plt.plot(grads[3].detach().sum(dim=0))

plt.plot(ub.grad.sum(dim=0))
plt.plot(grads[4].detach().sum(dim=0))
