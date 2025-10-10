import numpy as np
import scipy.io
import nibabel as nib
from scipy.stats import qmc
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator

def fdm_fisher_kpp_2d(diff_slice, phi_slice, config, D_phys=0.013, rho_phys=0.012, 
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
    print("D_norm", D_norm)
    
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
    ic_x, ic_y = config["sampling"]["ic_center"][0], config["sampling"]["ic_center"][1]
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

def load_mat(file_path):
    data = scipy.io.loadmat(file_path)['phi']
    return data[30:210, 10:200, 6:250]

def load_nifti(file_path):
    data = nib.load(file_path).get_fdata()
    data = np.flip(np.transpose(data, axes=[1, 0, 2]), 1)
    return data[30:210, 10:200, 6:250]

def sample_data_points(n_points, phi_slice, diff_slice, x0, y0, radius, config,
                       x_min=-1, x_max=1, y_min=-1, y_max=1, t_min=0, t_max=200):
    x, y, t, u = fdm_fisher_kpp_2d(diff_slice, phi_slice, config) 
    theta = np.random.uniform(0, 2 * np.pi, n_points)
    r = radius * np.sqrt(np.random.uniform(0, 1, n_points))
    x_circ = x0 + r * np.cos(theta)
    y_circ = y0 + r * np.sin(theta)
    t_circ = np.linspace(t_min, t_max, n_points)

    data = np.vstack((x_circ, y_circ, t_circ)).T 

    interpolator = RegularGridInterpolator((x, y, t), u, bounds_error=False, fill_value=None)
    u_sampled = interpolator(data)
    u_sampled = np.clip(u_sampled, 0, None)  

    return np.hstack((data, u_sampled[:, None]))  

def sample_data_points_in_domain(n_points, phi_slice, diff_slice, config,
                                   x_min=-1, x_max=1, y_min=-1, y_max=1, t_min=0, t_max=200):
    tf.random.set_seed(42)
    x, y, t, u = fdm_fisher_kpp_2d(diff_slice, phi_slice, config)  
    x_random = np.random.uniform(x_min, x_max, n_points)
    y_random = np.random.uniform(y_min, y_max, n_points)
    t_random = np.random.uniform(t_min, t_max, n_points)
    data = np.vstack((x_random, y_random, t_random)).T  

    interpolator = RegularGridInterpolator((x, y, t), u, bounds_error=False, fill_value=None)
    u_sampled = interpolator(data)
    u_sampled = np.clip(u_sampled, 0, None)  

    return np.hstack((data, u_sampled[:, None]))

def sample_collocation_points_in_domain(n_points, x_min=-1, x_max=1, y_min=-1, y_max=1, t_min=0, t_max=200):
    
    sampler = qmc.LatinHypercube(d=3, scramble=True)
    points = sampler.random(n_points)
    x = x_min + (x_max - x_min) * points[:, 0]
    y = y_min + (y_max - y_min) * points[:, 1]
    t = t_min + (t_max - t_min) * points[:, 2]
    return np.column_stack((x, y, t))

def sample_collocation_points(n_points, x0, y0, rad, x_min=-1, x_max=1, y_min=-1, y_max=1, t_min=0, t_max=200):
    tf.random.set_seed(42)
    data = np.zeros((n_points, 4))
    radius = rad  
    theta = np.random.uniform(0, 2 * np.pi, n_points)
    r = (radius) * np.sqrt(np.random.uniform(0, 1, n_points))
    x_circ = x0 + r * np.cos(theta)
    y_circ = y0 + r * np.sin(theta)
    t_circ = t_min + (t_max - t_min) * np.random.rand(n_points)
    data[:, 0] = x_circ
    data[:, 1] = y_circ
    data[:, 2] = t_circ

    return data

def prepare_data(config, save_dir):

    pff = load_mat(config["phi_slice_path"]) 
    pff_slice = pff[:, :, config["phi_slice_z"]]
    image = load_nifti("../data/P1/t1_masked_syn.nii")
    image = image[:, :, config["phi_slice_z"]]
    gm = load_nifti(config["gm_path"])
    wm = load_nifti(config["wm_path"])
    csf = load_nifti(config["csf_path"])

    # Diffusion map
    tissue = wm + gm
    pWM = (tissue > csf) * wm
    pGM = (tissue > csf) * gm
    total = pWM + pGM
    pWM = np.where(total > 0, pWM / total, 0.)
    pGM = np.where(total > 0, pGM / total, 0.)
    diff_volume = (pWM + 0.1 * pGM)
    diff_slice = diff_volume[:, :, config["phi_slice_z"]]

    
    if config["sampling"]["strategy"] == "centered":
        data_points = sample_data_points(config["sampling"]["n_inside"], pff_slice, diff_slice,
                                        config["sampling"]["ic_center"][0], config["sampling"]["ic_center"][1],
                                        config["sampling"]["ic_radius"], config)
        collocation_data = sample_collocation_points(config["sampling"]["n_inside"], config["sampling"]["ic_center"][0], config["sampling"]["ic_center"][1], config["sampling"]["ic_radius"])
    if config["sampling"]["strategy"] == "distributed":
        data_points = sample_data_points_in_domain(config["sampling"]["n_inside"], pff_slice, diff_slice, config)
        collocation_data = sample_collocation_points_in_domain(config["sampling"]["n_inside"])

    t_max = 200.0
    x_data, y_data, t_data, u_data = [np.expand_dims(data_points[:, i], axis=1) for i in range(4)]
    t_data = t_data/t_max
    x_c, y_c, t_c = [np.expand_dims(collocation_data[:, i], axis=1) for i in range(3)]
    t_c = t_c/t_max


    fig, ax = plt.subplots(figsize=(9, 7))
    ax.set_title("Data points distribution", fontsize=9)
    # Collocation points: blue, 'o'
    sc1 = ax.scatter(collocation_data[:, 0], collocation_data[:, 1], s=10, marker="o", c="#90BFE8", label="Collocation points", alpha=0.7, edgecolor='k', linewidth=0.2)
    sc2 = ax.scatter(data_points[:, 0], data_points[:, 1], s=10, marker="s", c = data_points[:, 3], label="Data points", alpha=0.7, cmap='viridis', linewidth=0.2)
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