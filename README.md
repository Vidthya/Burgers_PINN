### Physics-Informed Neural Network for the Viscous Burgers' Equation



A PyTorch implementation of a Physics-Informed Neural Network (PINN) that solves the 1D viscous Burgers' equation using only the governing PDE and the initial and the boundary conditions. No simulator-generated training data is used inside the domain. The result is checked against a finite-difference reference solver.



Everything is in one notebook: Burgers\_PINN\_notebook.ipynb (solver, PINN, training, evaluation). Click the "Open in Colab" badge in the notebook to run it yourself.



#### The problem

This is the standard benchmark from Raissi, Perdikaris \& Karniadakis (2019):



u\_t + u \* u\_x = nu \* u\_xx,	x in \[-1, 1], t in \[0, 1]

u(x, 0) = -sin(pi \* x)

u(-1, t) = u(1, t) = 0

nu = 0.01 / pi



This is a good test because it is nonlinear and develops a sharp, near-discontinuous front around x = 0 as t increases. That is much harder for a smooth network to represent than plain linear diffusion.



#### How it works

The notebook has three parts.

1. Finite-difference reference solver. 
An explicit scheme (upwind advection, central-difference diffusion) that produces a ground-truth solution on a 401 x 4001 grid.
2. The PINN.
A fully connected network (x, t) -> u (6 hidden layers, 32 units, tanh) trained on three loss terms:
- PDE residual: at 10,000 random interior points, PyTorch autograd computes u\_t, u\_x and u\_xx of the network output, and the loss penalises any violation of u\_t + u\*u\_x - nu\*u\_xx =0.
- Initial condition: u(x, 0) matches -sin(pi x).
- Boundary condition: u(-1, t) = u(1, t) = 0.
3. Evaluation.

&#x20;  The trained network is evaluated on the same grid as the reference. The notebook reports L2 error and the maximum absolute error, and plots the solution, the error map, loss curves and snapshots at fixed times.

#### 

#### Results

Trained for 8,000 Adam epochs (learning rate 1e-3), CPU only. Because training starts
from random weights, the result depends on the random seed, so the same network was
trained with six seeds in the notebook.

| Seed | Relative L2 error | Max abs error | Final training loss |
|---|---|---|---|
| 0 | 4.47% | 0.490 | 6.5e-04 |
| 1 | 6.80% | 0.719 | 7.0e-04 |
| 2 | 2.70% | 0.222 | 1.15e-03 |
| 3 | 4.45% | 0.502 | 1.06e-03 |
| 4 | 10.19% | 1.008 | 1.38e-03 |
| 5 | 7.28% | 0.724 | 8.3e-04 |

**Summary across six seeds:** relative L2 error median 5.6%, mean 6.0%, SD 2.7,
range 2.70% to 10.19%. Relative L2 error is defined as
‖u_PINN − u_ref‖₂ / ‖u_ref‖₂, evaluated on the 401 × 4001 reference grid.

![Seed 0: reference, PINN, error and training loss](results/seed0.png)

*Seed 0 (relative L2 error 4.47%). Top: finite-difference reference and PINN
prediction. Bottom left: absolute error. Bottom right: training loss.*

Details for seed 0:

| Metric | Value |
|---|---|
| Relative L2 error vs. FD reference | 4.47% |
| Max absolute error | 0.490 |
| Final training loss (epoch 8,000) | 6.5e-04 |
| PINN training time (8,000 epochs) | 610 s |
| PINN inference time (full 401 x 4001 grid) | 2.2 s |
| Finite-difference solve time (same grid) | 0.30 s |

#### Limitations

- **Small sample.** Six seeds, one architecture (6 hidden layers of 32 units, tanh),
  and no tuning of the network, collocation points or loss weights. The seed-to-seed
  spread is large (2.70% to 10.19%), so these numbers describe the spread seen here,
  not a guaranteed accuracy or bound.
- **Final loss does not rank the runs.** Seed 0 has the lowest final loss (6.5e-04)
  but not the lowest error; seed 2 has a higher final loss (1.15e-03) and the lowest
  error (2.70%). The training loss also shows periodic spikes under Adam, so the
  loss at the final epoch is a noisy number. With six seeds this is an observation,
  not a conclusion, but training loss alone should not be used to choose a run.
- **Errors are concentrated at the shock.** In seed 0 the error is a thin band along
  x = 0 for t > 0.4, reaching the maximum of 0.49; elsewhere it is mostly below about
  0.1. Across seeds the maximum absolute error ranges from 0.22 (seed 2) to 1.01
  (seed 4). 
- **The reference solver resolves the shock only coarsely.** The reference is a
  first-order upwind scheme on a grid with dx = 0.005, while the shock width is of
  order 2ν ≈ 0.006. Part of the error near the shock may therefore come from the
  reference, not only from the PINN. 
- **Not faster than a conventional solver.** The finite-difference solve takes
  0.30 s; training the PINN takes 610 s and evaluating it 2.2 s. For a simple,
  low-dimensional problem like this, classical numerical methods win on speed, and
  this project does not claim otherwise. It demonstrates the PINN method on a problem
  with a known reference.
- **One problem only.** A single viscosity (ν = 0.01/π) and a single set of initial
  and boundary conditions were tested.



#### Running it

Open the notebook in Colab, then choose Runtime, Run all. It needs torch, numpy and matplotlib, which Colab provides. 

To run locally: pip install torch numpy matplotlib jupyter, then jupyter notebook Burgers\_PINN\_notebook.ipynb .



#### Possible extensions

* Switch to L-BFGS after Adam to reduce the final error.
* Treat nu as a learnable parameter and recover it from sparse observations (an inverse problem, which a standard finite-difference solver cannot do without an external optimiser).
* Extend to 2D.



