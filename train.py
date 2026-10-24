"""
Training the PINN, evaluating it against the finite-difference reference solution, times
both and producing comparison plots.

Usage: python solver.py - generates fd_reference.npz (run once)

python train.py - trains PINN, evaluates, plots
"""

import time
import numpy as np
import torch
import matplotlib.pyplot as plt

from solver import solve_burgers_fd
from pinn import train_pinn

ref_file = 'fd_reference.npz'



def evaluate_on_grid(model, x, t, device):
    X, T = np.meshgrid(x, t) # shape (nt, nx)
    x_flat = torch.tensor(X.flatten(), dtype=torch.float32).unsqueeze(1).to(device)
    t_flat = torch.tensor(T.flatten(), dtype=torch.float32).unsqueeze(1).to(device)

    model.eval()
    with torch.no_grad():
        u_flat = model(x_flat, t_flat).cpu().numpy()
    return u_flat.reshape(X.shape)



def main():
    data = np.load(ref_file)
    x, t, u_fd = data["x"], data["t"], data["u"]
    print(f"Loaded FD reference: {u_fd.shape} grid")

    t0 = time.perf_counter()
    solve_burgers_fd(nx=len(x), nt=len(t))
    fd_time = time.perf_counter() - t0
    print(f"FD solver time (full grid): {fd_time:.4f}s")

    t0 = time.perf_counter()
    model, history, device = train_pinn(epochs=8000)
    train_time = time.perf_counter() - t0
    print(f"PINN training time: {train_time:.2f}s on {device}")

    t0 = time.perf_counter()
    u_pinn = evaluate_on_grid(model, x, t, device)
    pinn_inference_time = time.perf_counter() - t0
    print(f"PINN inference time (full grid): {pinn_inference_time:.4f}s")

    error = np.abs(u_pinn - u_fd)
    l2_rel_error = np.linalg.norm(u_pinn - u_fd) / np.linalg.norm(u_fd)
    print(f"Relative L2 error vs FD reference: {l2_rel_error:.4%}")
    print(f"Max absolute error: {error.max():.4f}")
    print(f"Inference vs FD solve speedup (post-training): {fd_time / pinn_inference_time:.1f}x")

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    im0 = axes[0, 0].pcolormesh(x, t, u_fd, shading="auto", cmap="RdBu_r", vmin=-1, vmax=1)
    axes[0, 0].set_title("Finite-difference reference u(x,t)")
    axes[0, 0].set_xlabel("x"); axes[0, 0].set_ylabel("t")
    fig.colorbar(im0, ax=axes[0, 0])

    im1 = axes[0, 1].pcolormesh(x, t, u_pinn, shading="auto", cmap="RdBu_r", vmin=-1, vmax=1)
    axes[0, 1].set_title("PINN prediction u(x,t)")
    axes[0, 1].set_xlabel("x"); axes[0, 1].set_ylabel("t")
    fig.colorbar(im1, ax=axes[0, 1])

    im2 = axes[1, 0].pcolormesh(x, t, error, shading="auto", cmap="viridis")
    axes[1, 0].set_title(f"Absolute error (rel. L2 = {l2_rel_error:.2%})")
    axes[1, 0].set_xlabel("x"); axes[1, 0].set_ylabel("t")
    fig.colorbar(im2, ax=axes[1, 0])

    axes[1, 1].semilogy(history["total"], label="total")
    axes[1, 1].semilogy(history["pde"], label="pde residual", alpha=0.7)
    axes[1, 1].semilogy(history["ic"], label="initial condition", alpha=0.7)
    axes[1, 1].semilogy(history["bc"], label="boundary condition", alpha=0.7)
    axes[1, 1].set_title("Training loss")
    axes[1, 1].set_xlabel("epoch"); axes[1, 1].set_ylabel("loss (log scale)")
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig("results.png", dpi=150)
    print("Saved plots to results.png")

    fig2, axes2 = plt.subplots(1, 3, figsize=(15, 4))
    for ax, t_snap in zip(axes2, [0.25, 0.5, 0.75]):
        idx = np.argmin(np.abs(t - t_snap))
        ax.plot(x, u_fd[idx], label="FD reference", linewidth=2)
        ax.plot(x, u_pinn[idx], "--", label="PINN", linewidth=2)
        ax.set_title(f"t = {t[idx]:.2f}")
        ax.set_xlabel("x"); ax.set_ylabel("u")
        ax.legend()
    plt.tight_layout()
    plt.savefig("snapshots.png", dpi=150)
    print("Saved snapshot comparison to snapshots.png")

    with open("results_summary.txt", "w") as f:
        f.write(f"Device: {device}\n")
        f.write(f"FD solver time (grid {u_fd.shape}): {fd_time:.4f}s\n")
        f.write(f"PINN training time (8000 epochs): {train_time:.2f}s\n")
        f.write(f"PINN inference time (full grid): {pinn_inference_time:.4f}s\n")
        f.write(f"Relative L2 error vs FD: {l2_rel_error:.4%}\n")
        f.write(f"Max absolute error: {error.max():.4f}\n")
        f.write(f"Inference speedup vs re-solving: {fd_time / pinn_inference_time:.1f}x\n")


if __name__ == "__main__":
    main()

import os
os.makedirs("results", exist_ok=True)
...
plt.savefig("results/seed0.png", dpi=150)
...
plt.savefig("results/snapshots_seed0.png", dpi=150)
plt.close("all")   

