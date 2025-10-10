import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import imageio.v2 as imageio
import tensorflow as tf
import nibabel as nib
import scipy.io

from matplotlib.animation import FuncAnimation

def fdm_fisher_kpp_2d(diff_slice, phi_slice, D_phys=0.013, rho_phys=0.012, 
                      Nx=180, Ny=150, Nt=1000, t_max=200):
    """
    FDM solver for 2D Fisher-KPP equation on a normalized domain [-1,1]x[-1,1].
    
    Parameters
    ----------
    diff_slice : 2D array
        Spatial mask for diffusion
    phi_slice : 2D array
        Spatial mask for growth
    config : dict
        Configuration dictionary containing initial condition center
    D_phys : float
        Diffusion coefficient in mm^2/day
    rho_phys : float
        Proliferation rate in 1/day
    Nx, Ny : int
        Number of grid points in x and y
    Nt : int
        Number of time steps
    t_max : float
        Maximum time in domain time units (days or years)
    """
    
    # Physical domain size
    Lx = 180.0  # mm
    Ly = 150.0  # mm
    
    # Convert physical D to normalized domain
    D_norm = D_phys * 0.5 * ((2./Lx)**2 + (2./Ly)**2)
    D_norm_bis = D_phys / (( (Lx/2)**2 + (Ly/2)**2 ) / 2)
    print("D_norm", D_norm)
    print("D_norm_bis", D_norm_bis)
    
    # Keep rho in days
    rho_norm = rho_phys
    
    # Create normalized spatial grid
    x = np.linspace(-1, 1, Nx)
    y = np.linspace(-1, 1, Ny)
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    
    # Time grid
    t = np.linspace(0, t_max, Nt)
    dt = t[1] - t[0]
    
    # Initialize solution
    u = np.zeros((Nx, Ny, Nt))
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    # Initial condition: Gaussian centered at config["sampling"]["ic_center"]
    ic_x, ic_y = 0.4, 0.4
    u[:, :, 0] = 0.5*np.exp(-2*((X - ic_x)**2 + (Y - ic_y)**2) / 0.1**2)
    
    # FDM time stepping
    for k in range(Nt - 1):
        u[1:-1, 1:-1, k+1] = (
            u[1:-1, 1:-1, k]
            + D_norm * dt / dx**2 * (u[2:, 1:-1, k] - 2 * u[1:-1, 1:-1, k] + u[:-2, 1:-1, k]) * diff_slice[1:-1, 1:-1]
            + D_norm * dt / dy**2 * (u[1:-1, 2:, k] - 2 * u[1:-1, 1:-1, k] + u[1:-1, :-2, k]) * diff_slice[1:-1, 1:-1]
            + rho_norm * dt * u[1:-1, 1:-1, k] * (1 - u[1:-1, 1:-1, k])
        ) * phi_slice[1:-1, 1:-1]
    
    return x, y, t, u

def visualize_solution_evolution(model, diff_slice, phi_slice, save_dir, config, gif_name="comparison.gif",
                                 t_snap=None, value_threshold=0.005, L=1.0):

    if t_snap is None:
        t_snap = [0, 25, 50, 75, 100, 125, 150, 175, 200]

    D = config["physics"]["D"]
    r = config["physics"]["r"]  

    Ny, Nx = 150, 180
    x = np.linspace(-L, L, Nx)
    y = np.linspace(-L, L, Ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    X_flat, Y_flat = X.flatten()[:, None], Y.flatten()[:, None]

    os.makedirs(save_dir, exist_ok=True)
    filenames = []

    # FDM reference once for all time steps
    x_fdm, y_fdm, t_fdm, u_fdm = fdm_fisher_kpp_2d(diff_slice, phi_slice, D_phys=D, rho_phys=r)

    for i, t_plot in enumerate(t_snap):
        # PINN prediction
        T_flat = t_plot * np.ones_like(X_flat)
        T_flat = T_flat/200.0
        XYT = np.hstack([X_flat, Y_flat, T_flat])
        u_pred = model.predict(XYT, verbose=0).reshape((Nx,Ny))
        
        # FDM slice
        idx_t = np.argmin(np.abs(t_fdm - t_plot))
        u_fdm_t = u_fdm[:, :, idx_t]

        # Error
        error = np.abs(u_pred - u_fdm_t)
        error = np.where(error < 0.01, 0, error)
        epsilon = 1e-4
        relative_error = np.where((u_fdm_t < 0.01) & (u_pred < 0.01), 0, error / ((u_fdm_t) + epsilon))
        relative_error_in_percent = relative_error * 100
        relative_error_in_percent = np.clip(relative_error_in_percent, 0, 99)
        error_max = np.max(relative_error_in_percent)
        error_min = np.min(relative_error_in_percent)
        mean_relative_error = np.mean(relative_error_in_percent)
        mean_absolute_error = np.mean(error[error > 0])
        max_absolute_error = np.max(error)

        # Save error stats
        with open(f"{save_dir}/new_error_stats.txt", "a") as f:
            f.write(f"t={t_plot:.2f}, max relative error: {error_max:.4f}, min relative error: {error_min:.4f}, mean relative error: {mean_relative_error:.4f}, mean absolute error: {mean_absolute_error:.4f}, max absolute error: {max_absolute_error:.4f}\n")

        fig, ax = plt.subplots(1, 4, figsize=(20, 5))
        for a in ax: a.set_axis_off()
        # PINN plot
        pcm0 = ax[0].imshow(u_pred, cmap="plasma", origin="lower", vmin=0, vmax=1, label = "Tumor cells density [-]")
        ax[0].imshow(diff_slice, cmap="gray", origin="lower", alpha=0.3)
        ax[0].set_title(f"PINN solution u [-] | t={(t_plot)} days", fontsize=13)

        # FDM plot
        pcm1 = ax[1].imshow(u_fdm_t, cmap="plasma", origin="lower", vmin=0, vmax=1, label = "Tumor cells density [-]")
        ax[1].imshow(diff_slice, cmap="gray", origin="lower", alpha=0.3)
        ax[1].set_title(f"FDM solution u [-] | t={(t_plot)} days", fontsize=13)

        # Error plot
        pcm2 = ax[2].imshow(relative_error_in_percent, cmap="viridis", origin="lower", vmin=0, vmax=100, label = "Relative error [%]")
        ax[2].imshow(diff_slice, cmap="gray", origin="lower", alpha=0.3)
        ax[2].set_title(f"Relative error [%] | t={(t_plot)} days", fontsize=13)

        pcm3 = ax[3].imshow(error, cmap="viridis", origin="lower", vmin=0, vmax=0.03, label = "Absolute error [-]")
        ax[3].imshow(diff_slice, cmap="gray", origin="lower", alpha=0.3)
        ax[3].set_title(f"Absolute error [-] | t={(t_plot)} days", fontsize=13)


        fig.colorbar(pcm0, ax=ax[0])
        fig.colorbar(pcm1, ax=ax[1])
        fig.colorbar(pcm2, ax=ax[2])
        fig.colorbar(pcm3, ax=ax[3])

        plt.suptitle(r"Comparison | $D_w=0.013 \ [mm²/day]$ | $\rho=0.012 \ [1/day]$", fontsize=13)
        filename = f"{save_dir}/comparison_{i:03d}_inverse.png"
        filenames.append(filename)
        plt.savefig(filename, dpi=500)
        plt.savefig(f"{save_dir}/comparison_{i:03d}_inverse.svg", dpi=500)
        plt.close()

    # GIF
    with imageio.get_writer(f"{save_dir}/{gif_name}", mode='I', duration=0.5) as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)

    print(f"Comparison GIF saved: {save_dir}/{gif_name}")
