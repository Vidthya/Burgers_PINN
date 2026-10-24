"""
Finite-difference reference solver for the 1D viscous Burgers' equation:

u_t + uu_x = nu u_xx, x in [-1,1], t in [0,1]

u(x,0) = -sin(pi*x)

u(-1,t) = u(1,t) = 0

The standard benchmark problem from Raissi, Perdikaris & Karniadakis (2019),
"Physics-Informed Neural Networks". nu = 0.01/pi gives a solution that develops a
sharp near-discontinuity around x=0 as t increases, which makes it a nontrivial test
for a PINN.
"""

import numpy as np

nu = 0.01/np.pi

def solve_burgers_fd(nx: int=401, nt:int=4000, x_range=(-1.0, 1.0), t_max:float=1.0):
    """
    Solve viscous Burgers' equation with explicit finite differences

    Returns
    x: (nx,) array of spatial grid points
    t: (nt+1,) array of time grid points
    u: (nt+1, nx) array, u[i,j]=u(x[j], t[i])
    """

    x = np.linspace(x_range[0],x_range[1], nx)
    dx = x[1] - x[0]

    ##Explicit scheme stability: dt must be small relative to dx^2 (diffusion)
    ##Pick dt conservatively and re-derive nt from it
    dt_diffusion_limit = 0.4 * dx**2 / nu
    dt = min(dt_diffusion_limit, t_max / nt)
    n_steps = int(np.ceil(t_max / dt))
    dt = t_max / n_steps

    u = np.zeros((n_steps + 1, nx))
    u[0] = -np.sin(np.pi * x)
    u[0, 0] = 0.0
    u[0, -1] = 0.0

    for n in range(n_steps):
        un = u[n]
        u_next = un.copy()

        ##Central-difference diffusion term for all interior points
        diffusion = nu * (un[2:] - 2 * un[1:-1] + un[:-2]) / dx**2

        ##Upwind advection term (Burgers' flux is nonlinear, so upwind
        ##direction depends on the local sign of u)
        du_dx_backward = (un[1:-1] - un[:-2]) / dx
        du_dx_forward = (un[2:] - un[1:-1]) / dx
        advection = np.where(
            un[1:-1] >= 0,
            un[1:-1] * du_dx_backward,
            un[1:-1] * du_dx_forward,
        )

        u_next[1:-1] = un[1:-1] + dt * (-advection + diffusion)
        u_next[0] = 0.0
        u_next[-1] = 0.0
        u[n + 1] = u_next

    ##Subsample time axis down to a manageable number of stored snapshots
    stride = max(1, n_steps // nt)
    t_full = np.linspace(0, t_max, n_steps + 1)
    idx = np.arange(0, n_steps + 1, stride)
    if idx[-1] != n_steps:
        idx = np.append(idx, n_steps)

    return x, t_full[idx], u[idx]


if __name__ == "__main__":
    x, t, u = solve_burgers_fd()
    print(f"Solved on grid: {len(x)} x-points x {len(t)} t-points")
    print(f"u range: [{u.min():.4f}, {u.max():.4f}]")
    np.savez("fd_reference.npz", x=x, t=t, u=u)
    print("Saved reference solution to fd_reference.npz")
