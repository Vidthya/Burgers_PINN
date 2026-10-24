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

Trained for 8,000 Adam epochs (learning rate = 1e-3), CPU only.

In the Colab notebook (seed = 0): relative L2 error 4.5%.



|Metric|Value|
|-|-|
|Relative L2 error vs. FD reference|4.47%|
|Max absolute error|0.490|
|Final training loss (epoch 8,000)|6.5e-4|
|PINN training time (8,000 epochs)|610 s|
|PINN inference time (full 401 x 4001 grid)|2.2 s|
|Finite-difference solve time (same grid)|0.30 s|



Because training starts from random weights, the result depends on the random seed. To measure how much, I trained the same network with six seeds in the same notebook. Seed 0 reproduced the figure above exactly.



|Seed|Relative L2 error|Max abs error|Final loss|
|-|-|-|-|
|0|4.47%|0.490|6.5e-04|
|1|6.80%|0.719|7.0e-04|
|2|2.70%|0.222|1.2e-03|
|3|4.45%|0.502|1.1e-03|
|4|10.19%|1.008|1.4e-03|
|5|7.28%|0.724|8.3e-04|



Relative L2 error: 2.70% to 10.19% across six seeds, mean 6.9%. Six seeds is a small sample, so this describes the spread seen here, not a guaranteed bound. A lower final training loss did not always mean a lower error against the reference (seed 2 has a higher final loss than seed 0 but a lower error).



## Limitations

- **Small sample.** Results come from six random seeds on a single architecture
  (6 hidden layers of 32 units, tanh, 8,000 Adam epochs, learning rate 1e-3).
  The seed-to-seed spread is large, with relative L2 error from 2.70% to 10.19%
  (median 5.6%, mean 6.0%, SD 2.7), so the numbers above should not be read as
  a precise accuracy figure for the method.
- **Final loss does not rank the runs.** Seed 0 has the lowest final training
  loss (6.5e-4) but not the lowest error, and seed 2 has a higher loss (1.15e-3)
  with the lowest error (2.70%). With six seeds this is an observation, not a
  conclusion, but training loss alone should not be used to choose a run.
- **Errors are concentrated in ** The maximum absolute error ranges from 0.22
  (seed 2) to 1.01 (seed 4). I have not tuned the network, the collocation points or the loss
  weights to improve this.
- **Not faster than a conventional solver.** On CPU, training takes about 610 s
  against 0.30 s for the finite-difference reference. This project is a
  demonstration of the PINN method on a problem with a known reference solution,
  not a replacement for a numerical solver.
- **Reference solver not independently verified.** The finite-difference solver
  is my own implementation. 
- **Single problem.** Only one viscosity and one set of initial and boundary
  conditions were tested.


#### Speed

The 1D finite-difference solve takes 0.30s on the full grid, whereas training the PINN took 610s and evaluating it over the same grid took 2.2s. For a simple, low-dimensional problem like this one, classical numerical methods win on speed, and this project does not claim otherwise. 



#### Running it

Open the notebook in Colab, then choose Runtime, Run all. It needs torch, numpy and matplotlib, which Colab provides. 

To run locally: pip install torch numpy matplotlib jupyter, then jupyter notebook Burgers\_PINN\_notebook.ipynb .



#### Possible extensions

* Switch to L-BFGS after Adam to reduce the final error.
* Treat nu as a learnable parameter and recover it from sparse observations (an inverse problem, which a standard finite-difference solver cannot do without an external optimiser).
* Extend to 2D.



