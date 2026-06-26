import numpy as np
import scipy.io
import nibabel as nib
from scipy.stats import qmc
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator

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
    FDM solver for 2D Fisher-KPP equation with treatment-induced death term.
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

def load_mat(file_path, key):

    data = scipy.io.loadmat(file_path)[key]

    return data[:, :, :]


def load_nifti(file_path):

    data = nib.load(file_path).get_fdata()
    data = np.flip(np.transpose(data, axes=[1, 0, 2]), 1)

    return data[30:210, 10:200, 6:250]

def sample_data_points(n_points, phi_slice, diff_slice, x0, y0, radius, config, t_min=0, t_max=130):
    
    treatment_cfg = {
    "spatial_C": "pff",
    "drugs": [
        {"name": "doxorubicin", "alpha": 0.12, "beta": 0.07, "times": [21, 42, 63, 84]},
        ],
    }
    fraction = 1.0

    x, y, t, u = fdm_fisher_kpp_2d_treatment(
        diff_slice,
        phi_slice,
        treatment_cfg,
        D_phys=0.025,
        rho_phys=0.012,
        Nt=2000,
        t_max=130,
        plot_lambda=True
    ) 
    theta = np.random.uniform(0, 2 * np.pi, n_points)
    r = radius * np.sqrt(np.random.uniform(0, 1, n_points))
    x_circ = x0 + r * np.cos(theta)
    y_circ = y0 + r * np.sin(theta)
    times = [0, 21, 42, 63, 84]
    t_circ = np.random.choice(times, n_points*int(fraction))
    t_circ_rest = np.random.uniform(t_min, t_max, n_points - int(n_points * fraction))
    t_circ = np.concatenate((t_circ, t_circ_rest))

    data = np.vstack((x_circ, y_circ, t_circ)).T 

    interpolator = RegularGridInterpolator((x, y, t), u, bounds_error=False, fill_value=None)
    u_sampled = interpolator(data)
    u_sampled = np.clip(u_sampled, 0, None)  

    return np.hstack((data, u_sampled[:, None]))  

def sample_data_points_in_domain(n_points, phi_slice, diff_slice, config,
                                   x_min=-1, x_max=1, y_min=-1, y_max=1, t_min=0, t_max=130):

    treatment_cfg = {
    "spatial_C": "pff",
    "drugs": [
        {"name": "doxorubicin", "alpha": 0.12, "beta": 0.07, "times": [21, 42, 63, 84]},
         ],
    }
    fraction = 1
    times = [0, 21, 42, 63, 84]

    x, y, t, u = fdm_fisher_kpp_2d_treatment(
        diff_slice,
        phi_slice,
        treatment_cfg,
        D_phys=0.025,
        rho_phys=0.012,
        Nt=2000,
        t_max=130,
        plot_lambda=True
    ) 
    x_random = np.random.uniform(x_min, x_max, n_points)
    y_random = np.random.uniform(y_min, y_max, n_points)
    t_times = np.random.choice(times, int(n_points * fraction))
    t_random_rest = np.random.uniform(t_min, t_max, n_points - int(n_points * fraction))
    t_random = np.concatenate((t_times, t_random_rest))

    data = np.vstack((x_random, y_random, t_random)).T  

    interpolator = RegularGridInterpolator((x, y, t), u, bounds_error=False, fill_value=None)
    u_sampled = interpolator(data)
    u_sampled = np.clip(u_sampled, 0, None)  

    return np.hstack((data, u_sampled[:, None]))

def sample_collocation_points_in_domain(n_points, x_min=-1, x_max=1, y_min=-1, y_max=1, t_min=0, t_max=130):
    
    sampler = qmc.LatinHypercube(d=3, scramble=True)
    points = sampler.random(n_points)
    x = x_min + (x_max - x_min) * points[:, 0]
    y = y_min + (y_max - y_min) * points[:, 1]
    t = t_min + (t_max - t_min) * points[:, 2]
    return np.column_stack((x, y, t))

def sample_collocation_points(n_points, x0, y0, rad, t_min=0, t_max=130):

    data = np.zeros((n_points, 4))
    theta = np.random.uniform(0, 2 * np.pi, n_points)
    r = (rad) * np.sqrt(np.random.uniform(0, 1, n_points))
    x_circ = x0 + r * np.cos(theta)
    y_circ = y0 + r * np.sin(theta)
    t_circ = t_min + (t_max - t_min) * np.random.rand(n_points)
    data[:, 0] = x_circ
    data[:, 1] = y_circ
    data[:, 2] = t_circ

    return data

def prepare_data(config, save_dir):

    # --- Load data ---
    pff_slice = load_mat(config["phi_slice_path"], 'phi') 
    pff_slice = pff_slice[:, :, config["phi_slice_z"]]
    image = load_mat(config["phi_slice_path"], 'phi')
    image = image[:, :, config["phi_slice_z"]]
    diff_volume = load_mat(config["diff_map_path"], "diff_map")
    diff_slice = diff_volume[:, :, config["phi_slice_z"]]

    plt.figure(figsize=(6,6))
    plt.imshow(image, extent=(-1, 1, -1, 1), cmap="gray", origin="lower")
    plt.title("Perfusion-like map (phi slice)", fontsize=12)
    plt.xlabel("x", fontsize=10)
    plt.ylabel("y", fontsize=10)
    plt.colorbar(label="phi [-]")
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/phi_slice.png", dpi=300)
    plt.close()

    # --- Sample points ---
    if config["sampling"]["strategy"] == "centered":

        data_points = sample_data_points(
            config["sampling"]["n_inside"], 
            pff_slice, 
            diff_slice,
            config["sampling"]["ic_center"][0], 
            config["sampling"]["ic_center"][1],
            config["sampling"]["ic_radius"], config)
        
        collocation_data = sample_collocation_points(
            config["sampling"]["n_inside"], 
            config["sampling"]["ic_center"][0], 
            config["sampling"]["ic_center"][1], 
            config["sampling"]["ic_radius"])
        
    if config["sampling"]["strategy"] == "distributed":
        data_points = sample_data_points_in_domain(
            config["sampling"]["n_inside"], 
            pff_slice, 
            diff_slice, 
            config)
        
        collocation_data = sample_collocation_points_in_domain(
            config["sampling"]["n_inside"])

    # --- Time normalization ---
    t_max = 130
    x_data, y_data, t_data, u_data = [np.expand_dims(data_points[:, i], axis=1) for i in range(4)]
    t_data = t_data/t_max
    x_c, y_c, t_c = [np.expand_dims(collocation_data[:, i], axis=1) for i in range(3)]
    t_c = t_c/t_max

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.set_title("Data points distribution", fontsize=9)
    sc1 = ax.scatter(collocation_data[:, 1], collocation_data[:, 0], s=10, marker="o", c="#90BFE8", label="Collocation points", alpha=0.7, edgecolor='k', linewidth=0.2)
    sc2 = ax.scatter(data_points[:, 1], data_points[:, 0], s=10, marker="s", c = data_points[:, 3], label="Data points", alpha=0.7, cmap='viridis', linewidth=0.2)
    ax.imshow(image, extent=(-1, 1, -1, 1), cmap="gray", origin="lower", alpha=0.3)
    cbar = plt.colorbar(sc2, ax=ax, label="u [-]")
    ax.set_xlabel("x", fontsize=9)
    ax.set_ylabel("y", fontsize=9)
    ax.grid(True, which='both', linestyle=':', linewidth=0.5, alpha=0.7)
    ax.legend(loc='upper right', fontsize=9, borderaxespad=0., frameon=True)
    fig.savefig(f"{save_dir}/points_data_inverse.png", dpi=300, bbox_inches='tight')
    fig.savefig(f"{save_dir}/points_data_inverse.svg", dpi=300, bbox_inches='tight')
    plt.close(fig)

    return {
        "x_data": tf.convert_to_tensor(x_data, dtype=tf.float64),
        "y_data": tf.convert_to_tensor(y_data, dtype=tf.float64),
        "t_data": tf.convert_to_tensor(t_data, dtype=tf.float64),
        "u_data": tf.convert_to_tensor(u_data, dtype=tf.float64),
        "x_c": tf.convert_to_tensor(x_c, dtype=tf.float64),
        "y_c": tf.convert_to_tensor(y_c, dtype=tf.float64),
        "t_c": tf.convert_to_tensor(t_c, dtype=tf.float64),
    }


#--- Test ---
# config = {
#     "phi_slice_path": "../phi.mat",
#     "diff_map_path": "../diff_map.mat",
#     "phi_slice_z": 7,
#     "sampling": {
#         "strategy": "centered",  # "centered" or "distributed"
#         "n_inside": 12000,
#         "ic_center": (-0.4, 0.0),
#         "ic_radius": 0.4,
#     }
# }

# prepare_data(config, save_dir="...")