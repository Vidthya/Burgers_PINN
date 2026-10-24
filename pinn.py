"""
Training a neural network to satisfy the PDE directly. Using automatic differentiation
to compute the physics loss (residual of Burgers' equation) rather than needing
labelled data at all.

Unlike a standard surrogate model, this network is trained to satisfy three things
simultaneously:

1. The initial condition, u(x,0) = -sin(pi x)

2. The boundary conditions, u(-1,t) = u(1,t) = 0

3. The PDE residual itself, u_t + uu_x - nuu_xx = 0, evaluated at randomly sampled
   "collocation" points throughout the domain interior.
"""

import numpy as np
import torch
import torch.nn as nn

nu = 0.01/np.pi

class PINN(nn.Module):
  def __init__(self, n_hidden=6, width=32):
    super().__init__()
    layers = [nn.Linear(2,width), nn.Tanh()]
    for _ in range(n_hidden - 1):
      layers += [nn.Linear(width, width), nn.Tanh()]
    layers += [nn.Linear(width, 1)]
    self.net = nn.Sequential(*layers)

  def forward(self,x,t):
    xt = torch.cat([x,t], dim=1)
    return self.net(xt)


def pde_residual(model, x, t):
  """
  Compute the Burgers' equation residual u_t + u*u_x - nu*u_xx at (x,t)
  using automatic differentiation through the network.
  """
  x.requires_grad_(True)
  t.requires_grad_(True)
  u = model(x,t)

  u_t = torch.autograd.grad(u,t,grad_outputs=torch.ones_like(u), create_graph=True)[0]
  u_x = torch.autograd.grad(u,x,grad_outputs=torch.ones_like(u), create_graph=True)[0]
  u_xx = torch.autograd.grad(u_x,x,grad_outputs=torch.ones_like(u_x), create_graph=True)[0]

  return u_t + u*u_x - nu*u_xx




def train_pinn(
     n_collocation=10000,
     n_ic=200,
     n_bc=200,
     epochs=8000,
     lr=1e-3,
     device=None,
     log_every=500,
     seed=0,
):
  device = device or ("cuda" if torch.cuda.is_available() else "cpu")
  print(f"Training on device: {device}")

  ##Fix the random seed so results are reproducible
  torch.manual_seed(seed)
  np.random.seed(seed)

  model = PINN().to(device)
  optimizer = torch.optim.Adam(model.parameters(),lr=lr)

  ##Collocation points: sampled randomly across the interior of the domain
  ##Enforce PDE residual=0
  x_f = (torch.rand(n_collocation, 1)*2-1).to(device)
  t_f = torch.rand(n_collocation, 1).to(device)

  ##Initial condition points: t=0, u=-sin(pi x)
  x_ic = (torch.rand(n_ic,1)*2-1).to(device)
  t_ic = torch.zeros(n_ic,1).to(device)
  u_ic = -torch.sin(np.pi*x_ic)

  ##Boundary condition points: x=-1 and x=1, u=0
  t_bc = torch.rand(n_bc,1).to(device)
  x_bc_left = -torch.ones(n_bc,1).to(device)
  x_bc_right = torch.ones(n_bc,1).to(device)

  mse = nn.MSELoss()
  history = {"total": [], "pde": [], "ic": [], "bc": []}

  for epoch in range(epochs):
    optimizer.zero_grad()
    residual = pde_residual(model,x_f.clone(), t_f.clone())
    loss_pde = mse(residual, torch.zeros_like(residual))

    u_pred_ic = model(x_ic, t_ic)
    loss_ic = mse(u_pred_ic, u_ic)

    u_pred_bc_left = model(x_bc_left, t_bc)
    u_pred_bc_right = model(x_bc_right, t_bc)
    loss_bc = mse(u_pred_bc_left, torch.zeros_like(u_pred_bc_left)) + \
              mse(u_pred_bc_right, torch.zeros_like(u_pred_bc_right))

    loss = loss_pde + loss_ic + loss_bc
    loss.backward()
    optimizer.step()

    history["total"].append(loss.item())
    history["pde"].append(loss_pde.item())
    history["ic"].append(loss_ic.item())
    history["bc"].append(loss_bc.item())

    if epoch % log_every == 0 or epoch == epochs -1:
      print(f"epoch {epoch:5d}; total {loss.item():.3e}"
      f"pde {loss_pde.item():.3e}; ic {loss_ic.item():.3e}; bc {loss_bc.item():.3e}")

  return model, history, device
