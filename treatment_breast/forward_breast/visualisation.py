import os
import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from treatment_breast.forward_breast.data_utils import load_nifti, load_mat
import tensorflow as tf


import numpy as np

import numpy as np
import matplotlib.pyplot as plt

def fdm_fisher_kpp_2d_treatment(
    diff_slice,
    phi_slice,
    treatment,
    D_phys=0.013,
    rho_phys=0.012,
    Nx=180,
    Ny=150,
    Nt=1000,
    t_max=130,
    ic_center=(-0.4, 0.0),
    ic_sigma=0.1,
    ic_amplitude=0.5,
    plot_lambda=True,
):
    """
    FDM solver for 2D Fisher-KPP equation on a normalized domain [-1,1]x[-1,1].
    """

    # --- Physical domain size ---
    Lx = diff_slice.shape[0]  # mm
    Ly = diff_slice.shape[1]  # mm

    # --- Convert physical D to normalized domain ---
    D_norm = D_phys * 0.5 * ((2. / Lx) ** 2 + (2. / Ly) ** 2)
    
    # --- Keep rho in days ---
    rho_norm = rho_phys

    # --- Create normalized spatial grid ---
    x = np.linspace(-1, 1, Lx)
    y = np.linspace(-1, 1, Ly)
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    
    # --- Time grid ---
    t = np.linspace(0, t_max, Nt)
    dt = t[1] - t[0]

    # --- Initialize solution ---
    u = np.zeros((Lx, Ly, Nt))
    X, Y = np.meshgrid(x, y, indexing="ij")

    # --- Initial condition: Gaussian centered at config["sampling"]["ic_center"] ---
    ic_x, ic_y = ic_center
    u[:, :, 0] = ic_amplitude * np.exp(-2 * ((X - ic_x) ** 2 + (Y - ic_y) ** 2) / ic_sigma ** 2)

    # --- Compute total lambda(t) at each time step ---
    def compute_lambda(t_curr):
        lam_total = np.zeros_like(phi_slice, dtype=np.float64)
        for drug in treatment.get("drugs", []):
            alpha_i = float(drug.get("alpha", 0.3))
            beta_i = float(drug.get("beta", 1.0))
            times = drug.get("times", [])
            k_smooth = 5.0
            temporal = sum(
                alpha_i * np.exp(-beta_i * (t_curr - tau)) * (0.5 * (1.0 + np.tanh(k_smooth * (t_curr - tau))))
                for tau in times
            )
            # spatial weighting
            if treatment.get("spatial_C", "pff") == "pff":
                lam_total += temporal * phi_slice  
            else:
                lam_total += temporal * (phi_slice > 0).astype(np.float64)
        return lam_total

    if plot_lambda:
        plt.figure(figsize=(8, 4))
        t_dense = np.linspace(0, t_max, 2000)
        total_lambda = np.zeros_like(t_dense)

        for drug in treatment.get("drugs", []):
            alpha_i = float(drug.get("alpha", 0.3))
            beta_i = float(drug.get("beta", 1.0))
            times = drug.get("times", [])
            lam_i = np.zeros_like(t_dense)
            k_smooth = 5.0
            for tau in times:
                # Contribution of the drug only after administration time
                heaviside = 0.5 * (1.0 + np.tanh(k_smooth * (t_dense - tau)))
                lam_i += alpha_i * np.exp(-beta_i *(t_dense - tau))* heaviside

            total_lambda += lam_i

        plt.plot(t_dense, total_lambda, "--", color="black", label="Total λ(t)")
        plt.title("Treatment intensity λ(t) over time")
        plt.xlabel("Time (days)")
        plt.ylabel("λ(t) [-]")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend()
        plt.tight_layout()
        plt.savefig("lambda_treatment_plot_final.png", dpi=300)
        plt.show()


    # --- FDM time stepping ---
    for k in range(Nt - 1):
        lap_x = (u[2:, 1:-1, k] - 2 * u[1:-1, 1:-1, k] + u[:-2, 1:-1, k]) / dx**2
        lap_y = (u[1:-1, 2:, k] - 2 * u[1:-1, 1:-1, k] + u[1:-1, :-2, k]) / dy**2
        lap = (lap_x + lap_y) * diff_slice[1:-1, 1:-1]

        lam_map = compute_lambda(t[k])[1:-1, 1:-1]
        diffusion_term = D_norm * lap
        reaction_term = rho_norm * u[1:-1, 1:-1, k] * (1 - u[1:-1, 1:-1, k])
        treatment_term = lam_map * u[1:-1, 1:-1, k]


        u_next = (
            u[1:-1, 1:-1, k]
            + dt * (
                diffusion_term
                + reaction_term
                - treatment_term
            )
        )

        u[1:-1, 1:-1, k + 1] = u_next * phi_slice[1:-1, 1:-1]

    return x, y, t, u


def visualize_solution_evolution(model, diff_slice, phi_slice, save_dir, config, gif_name="comparison.gif",
                                 t_snap=None, L=1.0):

    if t_snap is None:
        t_snap = [0, 22, 43, 64, 85, 106, 127]

    D = config["physics"]["D"]
    r = config["physics"]["r"]

    Ny, Nx = diff_slice.shape[1], diff_slice.shape[0]
    x = np.linspace(-L, L, Nx)
    y = np.linspace(-L, L, Ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    X_flat, Y_flat = X.flatten()[:, None], Y.flatten()[:, None]

    os.makedirs(save_dir, exist_ok=True)
    filenames = []

    treatment_cfg = {
    "spatial_C": "pff",
    "drugs": [
        {"name": "doxorubicin", "alpha": 0.0815, "beta": 0.07, "times": [21, 42, 63, 84]},
        {"name": "cyclophosphamide", "alpha": 0.04, "beta": 0.21, "times": [21, 42, 63, 84]},
        ],
    }

    x_fdm, y_fdm, t_fdm, u_fdm = fdm_fisher_kpp_2d_treatment(
        diff_slice,
        phi_slice,
        treatment_cfg,
        D_phys=0.025,
        rho_phys=0.012,
        Nt=2000,
        t_max=130,
        plot_lambda=True
    )

    for i, t_plot in enumerate(t_snap):
        print("i",i)
        # --- PINN prediction ---
        T_flat = t_plot * np.ones_like(X_flat)
        T_flat = T_flat/130.0
        XYT = np.hstack([X_flat, Y_flat, T_flat])
        u_pred = model.predict(XYT, verbose=0).reshape(Nx,Ny)
        
        # --- FDM slice ---
        idx_t = np.argmin(np.abs(t_fdm - t_plot))
        u_fdm_t = u_fdm[:, :, idx_t]

        # --- Error ---
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

        # --- Save error stats ---
        with open(f"{save_dir}/new_error_stats.txt", "a") as f:
            f.write(f"t={t_plot:.2f}, max relative error: {error_max:.4f}, min relative error: {error_min:.4f}, mean relative error: {mean_relative_error:.4f}, mean absolute error: {mean_absolute_error:.4f}, max absolute error: {max_absolute_error:.4f}\n")

        fig, ax = plt.subplots(1, 4, figsize=(20, 5))
        for a in ax: a.set_axis_off()
        # --- PINN plot ---
        pcm0 = ax[0].imshow(u_pred, cmap="plasma", origin="lower", vmin=0, vmax=1, label = "Tumor cells density [-]")
        ax[0].imshow(diff_slice, cmap="gray", origin="lower", alpha=0.3)
        ax[0].set_title(f"PINN solution u [-] | t={(t_plot)} days", fontsize=13)

        # --- FDM plot ---
        pcm1 = ax[1].imshow(u_fdm_t, cmap="plasma", origin="lower", vmin=0, vmax=1, label = "Tumor cells density [-]")
        ax[1].imshow(diff_slice, cmap="gray", origin="lower", alpha=0.3)
        ax[1].set_title(f"FDM solution u [-] | t={(t_plot)} days", fontsize=13)

        # --- Error plot ---
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

        plt.suptitle(fr"Comparison | $D_w={D} \ [mm²/day]$ | $\rho={r} \ [1/day]$", fontsize=13)
        filename = f"{save_dir}/comparison_{i:03d}_one.png"
        filenames.append(filename)
        plt.savefig(filename, dpi=500)
        plt.savefig(f"{save_dir}/comparison_{i:03d}_one.svg", dpi=500)
        plt.close()

    # --- GIF ---
    with imageio.get_writer(f"{save_dir}/{gif_name}", mode='I', duration=200) as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)

    print(f"Comparison GIF saved: {gif_name}")


def plot_center_time_evolution(model, config, ic_x=0.0, ic_y=0.0, n_times=130, save_path=None):
    """
    Plot u(x_ic, y_ic, t) over time using the trained model.
    """

    # Create time grid in [0,1] (normalized)
    t_norm = np.linspace(0, 1, n_times)[:, None]
    x = np.full_like(t_norm, ic_x)
    y = np.full_like(t_norm, ic_y)

    # Predict u(x_ic, y_ic, t)
    inputs = np.concatenate([x, y, t_norm], axis=1)
    u_pred = model(tf.constant(inputs, dtype=tf.float64)).numpy().flatten()

    # Convert to physical time (days)
    t_phys = t_norm.flatten() * float(config.get("t_max", 130.0))

    # Plot
    plt.figure(figsize=(8,5))
    plt.plot(t_phys, u_pred, lw=2, color='navy')
    plt.xlabel("Time (days)")
    plt.ylabel("u(x_ic, y_ic, t)")
    plt.title(f"Evolution of u at center ({ic_x:.2f}, {ic_y:.2f})")
    plt.grid(True)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

    return t_phys, u_pred

def plot_center_time_evolution_with_fdm(model, config, diff_slice, phi_slice, treatment_cfg,
                                        ic_x=-0.4, ic_y=0.0, n_times=130, save_path=None):
    """
    Compare temporal evolution at tumor center between PINN and FDM.
    Produces a figure with both curves.
    """

    t_max = float(config.get("t_max", 130.0))

    # --- PINN prediction ---
    t_norm = np.linspace(0, 1, n_times)[:, None]
    x = np.full_like(t_norm, ic_x)
    y = np.full_like(t_norm, ic_y)
    inputs = np.concatenate([x, y, t_norm], axis=1)
    u_pred = model(tf.constant(inputs, dtype=tf.float64)).numpy().flatten()
    t_phys = t_norm.flatten() * t_max

    # --- FDM computation ---
    print("Running FDM for comparison...")
    x_fdm, y_fdm, t_fdm, u_fdm = fdm_fisher_kpp_2d_treatment(
        diff_slice,
        phi_slice,
        treatment_cfg,
        D_phys=0.025,
        rho_phys=0.012,
        Nt=2000,
        t_max=t_max,
        plot_lambda=False
    )

    ix = np.argmin(np.abs(x_fdm - ic_x))
    iy = np.argmin(np.abs(y_fdm - ic_y))
    u_fdm_center = u_fdm[ix, iy, :]
    u_fdm_interp = np.interp(t_phys, t_fdm, u_fdm_center)

    # --- Plot ---
    plt.figure(figsize=(8, 5))
    plt.plot(t_phys, u_pred, lw=2.5, color='navy', label='PINN')
    plt.plot(t_phys, u_fdm_interp, '--', lw=2.5, color='darkorange', label='FDM')

    plt.xlabel("Time (days)")
    plt.ylabel(r"$u(x_{ic}, y_{ic}, t)$")
    plt.title(f"Evolution of u at center ({ic_x:.2f}, {ic_y:.2f})")
    plt.grid(True, alpha=0.5)
    plt.legend(frameon=False, fontsize=11)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

    return t_phys, u_pred, u_fdm_interp


# #---Example ---

# treatment_cfg = {
#     "spatial_C": "pff",
#     "drugs": [
#         {"name": "doxorubicin", "alpha": 0.08, "beta": 0.07, "times": [21, 42, 63, 84]},
#         {"name": "cyclophosphamide", "alpha": 0.04, "beta": 0.21, "times": [21, 42, 63, 84]},
#     ],
# }

# config = {"t_max": 130}

# pff_volume = load_mat("../phi.mat", 'phi') 
# diff_volume = load_mat("../diff_map.mat", "diff_map")
# diff_slice = diff_volume[:, :, 7]
# phi_slice = pff_volume[:, :, 7]
# model = tf.keras.models.load_model("../model.h5")

# t_phys, u_pinn, u_fdm = plot_center_time_evolution_with_fdm(
#     model,
#     config,
#     diff_slice,
#     phi_slice,
#     treatment_cfg,
#     ic_x=-0.4,
#     ic_y=0.0,
#     n_times=2000,
#     save_path="center_comparison_25_06.png"
# )