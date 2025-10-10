# data_utils2.py

import numpy as np
import scipy.io
import nibabel as nib
from scipy.stats import qmc
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow_graphics.math.interpolation import trilinear as interp3d

tf.random.set_seed(42)


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

import numpy as np
from scipy.interpolate import RegularGridInterpolator

def sample_data_points(n_points, phi_slice, diff_slice, x0, y0, radius, config,
                       x_min=-1, x_max=1, y_min=-1, y_max=1, t_min=0, t_max=200):
    # Génère la solution u(x, y, t)
    tf.random.set_seed(42)
    x, y, t, u = fdm_fisher_kpp_2d(diff_slice, phi_slice, config)  # Assumer u.shape = (len(x), len(y), len(t))


    # Génère les coordonnées sur un cercle de rayon donné
    theta = np.random.uniform(0, 2 * np.pi, n_points)
    r = radius * np.sqrt(np.random.uniform(0, 1, n_points))
    x_circ = x0 + r * np.cos(theta)
    y_circ = y0 + r * np.sin(theta)
    #t_circ with values through tmin to tmax
    t_circ = np.linspace(t_min, t_max, n_points)

    data = np.vstack((x_circ, y_circ, t_circ)).T  # shape (n_points, 3)

    # Interpolateur 3D
    interpolator = RegularGridInterpolator((x, y, t), u, bounds_error=False, fill_value=None)
    u_sampled = interpolator(data)
    u_sampled = np.clip(u_sampled, 0, None)  # assure u ≥ 0

    return np.hstack((data, u_sampled[:, None]))  # shape (n_points, 4)

def sample_data_points_in_domain(n_points, phi_slice, diff_slice, config,
                                   x_min=-1, x_max=1, y_min=-1, y_max=1, t_min=0, t_max=200):
    # Génère la solution u(x, y, t)
    tf.random.set_seed(42)
    x, y, t, u = fdm_fisher_kpp_2d(diff_slice, phi_slice, config)  # Assumer u.shape = (len(x), len(y), len(t))

    # Échantillonne des points aléatoires dans le domaine
    x_random = np.random.uniform(x_min, x_max, n_points)
    y_random = np.random.uniform(y_min, y_max, n_points)
    t_random = np.random.uniform(t_min, t_max, n_points)
    data = np.vstack((x_random, y_random, t_random)).T  # shape (n_points, 3)

    # Interpolateur 3D
    interpolator = RegularGridInterpolator((x, y, t), u, bounds_error=False, fill_value=None)
    u_sampled = interpolator(data)
    u_sampled = np.clip(u_sampled, 0, None)  # assure u ≥ 0

    return np.hstack((data, u_sampled[:, None]))  # shape (n_points, 4)

def sample_collocation_points_in_domain(n_points, x_min=-1, x_max=1, y_min=-1, y_max=1, t_min=0, t_max=200):
    
    sampler = qmc.LatinHypercube(d=3, scramble=True)
    points = sampler.random(n_points)
    x = x_min + (x_max - x_min) * points[:, 0]
    y = y_min + (y_max - y_min) * points[:, 1]
    t = t_min + (t_max - t_min) * points[:, 2]
    return np.column_stack((x, y, t))

def sample_collocation_points(n_points, x0, y0, rad, x_min=-1, x_max=1, y_min=-1, y_max=1, t_min=0, t_max=200):
    tf.random.set_seed(42)
    data = np.zeros((n_points, 3))
    radius = rad  # radius of the circle around (0.3, 0.3)
    theta = np.random.uniform(0, 2 * np.pi, n_points)
    r = (radius) * np.sqrt(np.random.uniform(0, 1, n_points))
    x_circ = x0 + r * np.cos(theta)
    y_circ = y0 + r * np.sin(theta)
    # t is randomly sampled between t_min and t_max
    t_circ = t_min + (t_max - t_min) * np.random.rand(n_points)
    print("t_circ", t_circ)
    data[:, 0] = x_circ
    data[:, 1] = y_circ
    data[:, 2] = t_circ

    return data

from scipy.stats import qmc
import numpy as np

def sampled_boundary_points(n_points, x_min=-1, x_max=1, y_min=-1, y_max=1, t_min=0, t_max=200):
    tf.random.set_seed(42)

    n_bc = 4
    n_data_per_bc = 30
    #
    engine = qmc.LatinHypercube(d=1)
    data = np.zeros([n_bc, n_data_per_bc, 4])

    for i, j in zip(range(n_bc), [-1, +1, -1, +1]):
        points = (engine.random(n=n_data_per_bc)[:, 0] - 0.5) * 2
        if i < 2:
            data[i, :, 0] = j
            data[i, :, 1] = points
        else:
            data[i, :, 0] = points
            data[i, :, 1] = j

        
    # BC training data Values ....for u .....
    # for x BC and y BC ......................
    for j in range(0,n_data_per_bc):    
        data[0, j, 3] = 0
        data[1, j, 3] = 0  
    for i in range(0,n_data_per_bc):
        data[2, i, 3] = 0
        data[3, i, 3] = 0

    # add time values
    data[:, :, 2] = t_min + (t_max - t_min) * data[:, :, 2]


    return data.reshape(n_data_per_bc * n_bc, 4)

def sample_anchors_outside_brain(n_points, phi_slice):
    anchors = []
    Ny, Nx = phi_slice.shape
    for _ in range(n_points):
        while True:
            x = np.random.uniform(-1, 1)
            y = np.random.uniform(-1, 1)
            t = np.random.uniform(0, 200)
            ix = int((x + 1) / 2 * (Nx - 1))
            iy = int((y + 1) / 2 * (Ny - 1))
            if phi_slice[iy, ix] < 0.01 or  0.01 < phi_slice[iy, ix] < 0.05:
                anchors.append([x, y, t])
                break
    return np.array(anchors)

def sample_initial_condition_points(pff_slice,n_points, x0, y0, rad,  x_min=-1, x_max=1, y_min=-1, y_max=1, t_min=0, t_max=200):
    #sample points around the initial condition in circle
    n_ic = 1
    n_data_per_ic = n_points
    engine = qmc.LatinHypercube(d=1)
    data = np.zeros([n_ic, n_data_per_ic, 4])
    radius = rad/2 # radius of the circle around (0.3, 0.3)
    theta = np.random.uniform(0, 2 * np.pi, n_data_per_ic)
    r = radius * np.sqrt(np.random.uniform(0, 1, n_data_per_ic))
    x_circ = x0 + r * np.cos(theta)
    y_circ = y0 + r * np.sin(theta)
    t_circ = np.zeros(n_data_per_ic)
    data[0, :, 0] = x_circ
    data[0, :, 1] = y_circ
    data[0, :, 2] = t_circ
    #multiply the initial condition by the pff_slice to have a mask
    data[0, :, 3] = 0.5*np.exp(-2*((x_circ - x0)**2 + (y_circ - y0)**2) / 0.1**2)
    data[0, :, 3] *= pff_slice[int((x0 + 1) / 2 * (pff_slice.shape[0] - 1)), int((y0 + 1) / 2 * (pff_slice.shape[1] - 1))]


    return data.reshape(n_data_per_ic * n_ic, 4)

def sample_initial_condition_points_in_domain(pff_slice,n_points, x0, y0, rad,  x_min=-1, x_max=1, y_min=-1, y_max=1, t_min=0, t_max=200):
    #sample points everywhere in the domain at t=0
    n_ic = 1
    n_data_per_ic = n_points
    engine = qmc.LatinHypercube(d=1)
    data = np.zeros([n_ic, n_data_per_ic, 4])
    x_random = np.random.uniform(x_min, x_max, n_data_per_ic)
    y_random = np.random.uniform(y_min, y_max, n_data_per_ic)
    t_circ = np.zeros(n_data_per_ic)
    data[0, :, 0] = x_random
    data[0, :, 1] = y_random
    data[0, :, 2] = t_circ
    #multiply the initial condition by the pff_slice to have a mask
    data[0, :, 3] = 0.5*np.exp(-2*((x_random - x0)**2 + (y_random - y0)**2) / 0.1**2)
    data[0, :, 3] *= pff_slice[int((x0 + 1) / 2 * (pff_slice.shape[0] - 1)), int((y0 + 1) / 2 * (pff_slice.shape[1] - 1))]

    return data.reshape(n_data_per_ic * n_ic, 4)

"""def prepare_data(config, save_dir):

    pff = load_mat(config["phi_slice_path"]) 
    pff_slice = pff[:, :, config["phi_slice_z"]]
    image = load_nifti("/home/vanderhaeghen/research/pinns_for_glioma_reproduction/archive/P1 (1)/t1_masked_syn.nii")
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

    if config["sampling"]["n_outside"] is not None:
        anchors_out = sample_anchors_outside_brain(config["sampling"]["n_outside"], pff_slice)
        anchors_out = np.hstack([anchors_out, np.zeros((anchors_out.shape[0], 1))])

    if config["sampling"]["strategy"] == "centered":
        print("centered")
        ic_data = sample_initial_condition_points(pff_slice, config["sampling"]["n_ic"], config["sampling"]["ic_center"][0], config["sampling"]["ic_center"][1],
                                               config["sampling"]["ic_radius"])
        print("ic_data", ic_data.shape)
        data_points = sample_data_points(config["sampling"]["n_inside"], pff_slice, diff_slice, config["sampling"]["ic_center"][0], config["sampling"]["ic_center"][1], config["sampling"]["ic_radius"], config)
        collocation_data = sample_collocation_points(config["sampling"]["n_inside"], config["sampling"]["ic_center"][0], config["sampling"]["ic_center"][1], config["sampling"]["ic_radius"])
    if config["sampling"]["strategy"] == "distributed":
        print("distributed")
        data_points = sample_data_points_in_domain(config["sampling"]["n_inside"], pff_slice, diff_slice, config)
        data_points = np.hstack([data_points, np.ones((data_points.shape[0], 1))])  # add a column of ones for u=1
        collocation_data = sample_collocation_points_in_domain(config["sampling"]["n_inside"])
        ic_data = sample_initial_condition_points_in_domain(pff_slice, config["sampling"]["n_ic"], config["sampling"]["ic_center"][0], config["sampling"]["ic_center"][1],
                                               config["sampling"]["ic_radius"])
        ic_data = np.hstack([ic_data, np.ones((ic_data.shape[0], 1))])  # add a column of ones for u=1
    

    
    if config["sampling"]["n_outside"] is not None:
        x_d, y_d, t_d, u_d = [np.expand_dims(anchors_out[:, i], axis=1) for i in range(4)]
    x_ic, y_ic, t_ic, u_ic = [np.expand_dims(ic_data[:, i], axis=1) for i in range(4)]
    x_data, y_data, t_data, u_data = [np.expand_dims(data_points[:, i], axis=1) for i in range(4)]
    x_c, y_c, t_c = [np.expand_dims(collocation_data[:, i], axis=1) for i in range(3)]


    fig, ax = plt.subplots(figsize=(9, 7))
    ax.set_title("Collocation points distribution", fontsize=9)
    # Collocation points: blue, 'o'
    #sc1 = ax.scatter(collocation_data[:, 0], collocation_data[:, 1], s=10, marker="o", c="#90BFE8", label="Collocation points", alpha=0.7, edgecolor='k', linewidth=0.2)
    #sc2 = ax.scatter(data_points[:, 0], data_points[:, 1], s=10, marker="s", c = data_points[:, 3], label="Data points", alpha=0.7, cmap='viridis', linewidth=0.2)
    # Initial condition: red, 's'
    sc3 = ax.scatter(ic_data[:, 0], ic_data[:, 1], s=9, marker="o", c = ic_data[:, 3], label="Data points (t=0)", alpha=0.7, linewidth=0.2)
    # Inside data: viridis colormap, '.'
    if config["sampling"]["n_outside"] is not None:
         sc4 = ax.scatter(anchors_out[:, 0], anchors_out[:, 1], s=12, marker="x", c= anchors_out[:, 3], label="Data points outside the brain", alpha=0.8)
    ax.imshow(image, extent=(-1, 1, -1, 1), cmap="gray", origin="lower", alpha=0.3)
    #cbar = plt.colorbar(sc3, ax=ax, label="u [-]")
    ax.set_xlabel("x", fontsize=9)
    ax.set_ylabel("y", fontsize=9)
    ax.grid(True, which='both', linestyle=':', linewidth=0.5, alpha=0.7)
    # Légende hors du cadre à droite
    ax.legend(loc='upper right', fontsize=9, borderaxespad=0., frameon=True)
    fig.savefig(f"{save_dir}/points_data_bigger_radius.png", dpi=300, bbox_inches='tight')
    fig.savefig(f"{save_dir}/points_data_radius.svg", dpi=300, bbox_inches='tight')
    plt.close(fig)

    return {
        "x_d": tf.convert_to_tensor(x_d, dtype=tf.float64),
        "y_d": tf.convert_to_tensor(y_d, dtype=tf.float64),
        "t_d": tf.convert_to_tensor(t_d, dtype=tf.float64),
        "u_d": tf.convert_to_tensor(u_d, dtype=tf.float64),
        "x_data": tf.convert_to_tensor(x_data, dtype=tf.float64),
        "y_data": tf.convert_to_tensor(y_data, dtype=tf.float64),
        "t_data": tf.convert_to_tensor(t_data, dtype=tf.float64),
        "u_data": tf.convert_to_tensor(u_data, dtype=tf.float64),
        "x_ic": tf.convert_to_tensor(x_ic, dtype=tf.float64),
        "y_ic": tf.convert_to_tensor(y_ic, dtype=tf.float64),
        "t_ic": tf.convert_to_tensor(t_ic, dtype=tf.float64),
        "u_ic": tf.convert_to_tensor(u_ic, dtype=tf.float64),
        "x_c": tf.convert_to_tensor(x_c, dtype=tf.float64),
        "y_c": tf.convert_to_tensor(y_c, dtype=tf.float64),
        "t_c": tf.convert_to_tensor(t_c, dtype=tf.float64),
    }"""

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

    if config["sampling"]["n_outside"] is not None:
        anchors_out = sample_anchors_outside_brain(config["sampling"]["n_outside"], pff_slice)
        anchors_out = np.hstack([anchors_out, np.zeros((anchors_out.shape[0], 1))])

    if config["sampling"]["strategy"] == "centered":
        print("centered")
        ic_data = sample_initial_condition_points(
            pff_slice,
            config["sampling"]["n_ic"],
            config["sampling"]["ic_center"][0],
            config["sampling"]["ic_center"][1],
            config["sampling"]["ic_radius"]
        )
        data_points = sample_data_points(
            config["sampling"]["n_inside"],
            pff_slice, diff_slice,
            config["sampling"]["ic_center"][0],
            config["sampling"]["ic_center"][1],
            config["sampling"]["ic_radius"],
            config
        )
        collocation_data = sample_collocation_points(
            config["sampling"]["n_inside"],
            config["sampling"]["ic_center"][0],
            config["sampling"]["ic_center"][1],
            config["sampling"]["ic_radius"]
        )

    if config["sampling"]["strategy"] == "distributed":
        print("distributed")
        data_points = sample_data_points_in_domain(config["sampling"]["n_inside"], pff_slice, diff_slice, config)
        collocation_data = sample_collocation_points_in_domain(config["sampling"]["n_inside"])
        ic_data = sample_initial_condition_points_in_domain(
            pff_slice,
            config["sampling"]["n_ic"],
            config["sampling"]["ic_center"][0],
            config["sampling"]["ic_center"][1],
            config["sampling"]["ic_radius"]
        )

    # --- Normalisation du temps ---
    t_max = 200.0  # temps physique max
    if config["sampling"]["n_outside"] is not None:
        x_d, y_d, t_d, u_d = [np.expand_dims(anchors_out[:, i], axis=1) for i in range(4)]
        t_d = t_d / t_max
    else:
        x_d = y_d = t_d = u_d = None

    x_ic, y_ic, t_ic, u_ic = [np.expand_dims(ic_data[:, i], axis=1) for i in range(4)]

    x_data, y_data, t_data, u_data = [np.expand_dims(data_points[:, i], axis=1) for i in range(4)]
    t_data = t_data / t_max

    x_c, y_c, t_c = [np.expand_dims(collocation_data[:, i], axis=1) for i in range(3)]
    t_c = t_c / t_max

    # --- Plot points ---
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.set_title("Collocation points distribution", fontsize=9)
    sc3 = ax.scatter(ic_data[:, 0], ic_data[:, 1], s=9, marker="o",
                     c=ic_data[:, 3], label="Data points (t=0)", alpha=0.7, linewidth=0.2)
    sc1 = ax.scatter(collocation_data[:, 0], collocation_data[:, 1], s=10, marker="o",
                     c="#90BFE8", label="Collocation points", alpha=0.7, edgecolor='k', linewidth=0.2)
    if config["sampling"]["n_outside"] is not None:
        sc4 = ax.scatter(anchors_out[:, 0], anchors_out[:, 1], s=12, marker="x",
                         c=anchors_out[:, 3], label="Data points outside the brain", alpha=0.8)
    ax.imshow(image, extent=(-1, 1, -1, 1), cmap="gray", origin="lower", alpha=0.3)
    ax.set_xlabel("x", fontsize=9)
    ax.set_ylabel("y", fontsize=9)
    cbar = plt.colorbar(sc3, ax=ax, label="u [-]")
    ax.grid(True, linestyle=':', linewidth=0.5, alpha=0.7)
    ax.legend(loc='upper right', fontsize=9, borderaxespad=0., frameon=True)
    fig.savefig(f"{save_dir}/points_data_bigger_radius.png", dpi=300, bbox_inches='tight')
    fig.savefig(f"{save_dir}/points_data_radius.svg", dpi=300, bbox_inches='tight')
    plt.close(fig)

    return {
        "x_d": tf.convert_to_tensor(x_d, dtype=tf.float64),
        "y_d": tf.convert_to_tensor(y_d, dtype=tf.float64), 
        "t_d": tf.convert_to_tensor(t_d, dtype=tf.float64),
        "u_d": tf.convert_to_tensor(u_d, dtype=tf.float64),
        "x_data": tf.convert_to_tensor(x_data, dtype=tf.float64),
        "y_data": tf.convert_to_tensor(y_data, dtype=tf.float64),
        "t_data": tf.convert_to_tensor(t_data, dtype=tf.float64),
        "u_data": tf.convert_to_tensor(u_data, dtype=tf.float64),
        "x_ic": tf.convert_to_tensor(x_ic, dtype=tf.float64),
        "y_ic": tf.convert_to_tensor(y_ic, dtype=tf.float64),
        "t_ic": tf.convert_to_tensor(t_ic, dtype=tf.float64),
        "u_ic": tf.convert_to_tensor(u_ic, dtype=tf.float64),
        "x_c": tf.convert_to_tensor(x_c, dtype=tf.float64),
        "y_c": tf.convert_to_tensor(y_c, dtype=tf.float64),
        "t_c": tf.convert_to_tensor(t_c, dtype=tf.float64),
        "t_max": tf.constant(t_max, dtype=tf.float64),  # <-- pour la loss PINN
    }


config = {
    "phi_slice_path": "/home/vanderhaeghen/research/pinns_for_glioma_reproduction/dataset/ants_reg_syn/P1/phi.mat",
    "phi_slice_z": 166,
    "gm_path": "/home/vanderhaeghen/research/pinns_for_glioma_reproduction/dataset/ants_reg_syn/P1/GM_syn.nii",
    "wm_path": "/home/vanderhaeghen/research/pinns_for_glioma_reproduction/dataset/ants_reg_syn/P1/WM_syn.nii",
    "csf_path": "/home/vanderhaeghen/research/pinns_for_glioma_reproduction/dataset/ants_reg_syn/P1/CSF_syn.nii",
    "sampling": {
        "strategy": "centered",  # "centered" or "distributed"
        "n_inside": 12000,  # number of data points inside the brain
        "n_outside": 4000,  # number of data points outside the brain
        "n_ic": 4000,  # number of initial condition points
        "ic_center": (0.4, 0.4),  # center of the initial condition
        "ic_radius": 0.35,  # radius around the initial condition center for sampling data points
    }
}

save_dir = "/home/vanderhaeghen/research/biostec_paper"

def prepare_data_bis(config, save_dir):

    pff = load_mat(config["phi_slice_path"]) 
    pff_slice = pff[:, :, config["phi_slice_z"]]

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

    if config["sampling"]["n_outside"] is not None:
        anchors_out = sample_anchors_outside_brain(config["sampling"]["n_outside"], pff_slice)
        anchors_out = np.hstack([anchors_out, np.zeros((anchors_out.shape[0], 1))])

    if config["sampling"]["strategy"] == "centered":
        print("centered")
        ic_data = sample_initial_condition_points(pff_slice, config["sampling"]["n_ic"], config["sampling"]["ic_center"][0], config["sampling"]["ic_center"][1],
                                               config["sampling"]["ic_radius"])
        print("ic_data", ic_data.shape)
        data_points = sample_data_points(config["sampling"]["n_inside"], pff_slice, diff_slice, config["sampling"]["ic_center"][0], config["sampling"]["ic_center"][1], config["sampling"]["ic_radius"], config)
        collocation_data = sample_collocation_points(config["sampling"]["n_inside"], config["sampling"]["ic_center"][0], config["sampling"]["ic_center"][1], config["sampling"]["ic_radius"])
    if config["sampling"]["strategy"] == "distributed":
        print("distributed")
        data_points = sample_data_points_in_domain(config["sampling"]["n_inside"], pff_slice, diff_slice, config)
        data_points = np.hstack([data_points, np.ones((data_points.shape[0], 1))])  # add a column of ones for u=1
        collocation_data = sample_collocation_points_in_domain(config["sampling"]["n_inside"])
        ic_data = sample_initial_condition_points_in_domain(pff_slice, config["sampling"]["n_ic"], config["sampling"]["ic_center"][0], config["sampling"]["ic_center"][1],
                                               config["sampling"]["ic_radius"])

    
     # --- Normalisation du temps ---
    t_max = 200.0  # temps physique max
    if config["sampling"]["n_outside"] is not None:
        x_d, y_d, t_d, u_d = [np.expand_dims(anchors_out[:, i], axis=1) for i in range(4)]
        t_d = t_d / t_max
    else:
        x_d = y_d = t_d = u_d = None

    x_ic, y_ic, t_ic, u_ic = [np.expand_dims(ic_data[:, i], axis=1) for i in range(4)]
    t_ic = t_ic / t_max

    x_data, y_data, t_data, u_data = [np.expand_dims(data_points[:, i], axis=1) for i in range(4)]
    t_data = t_data / t_max

    x_c, y_c, t_c = [np.expand_dims(collocation_data[:, i], axis=1) for i in range(3)]
    t_c = t_c / t_max

    print('max t_data', np.max(t_data))
    print('min t_data', np.min(t_data))
    print('max t_ic', np.max(t_ic))
    print('min t_ic', np.min(t_ic))
    print('min x_ic', np.min(x_ic))
    print('min y_ic', np.min(y_ic))
    print('max x_ic', np.max(x_ic))
    print('max y_ic', np.max(y_ic))
    print('max u_ic', np.max(u_ic))
    print('min u_ic', np.min(u_ic))

    print('max t_c', np.max(t_c))
    print('min t_c', np.min(t_c))
    print('max y_c', np.max(y_c))
    print('min y_c', np.min(y_c))
    print('max x_c', np.max(x_c))


    # --- Plot points  at t= 150 ---
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.set_title("Collocation points distribution at t=150", fontsize=9)
    sc3 = ax.scatter(ic_data[:, 0], ic_data[:, 1], s=9, marker="o",
                     c=ic_data[:, 3], label="Data points (t=0)", alpha=0.7, linewidth=0.2)
    if config["sampling"]["n_outside"] is not None:
        sc4 = ax.scatter(anchors_out[:, 0], anchors_out[:, 1], s=12, marker="x",
                         c=anchors_out[:, 3], label="Data points outside the brain", alpha=0.8)
    #sc2 = ax.scatter(data_points[:, 0], data_points[:, 1], s=10, marker="s", c = data_points[:, 3], label="Data points", alpha=0.7, cmap='viridis', linewidth=0.2)
    ax.imshow(pff_slice, extent=(-1, 1, -1, 1), cmap="gray", origin="lower", alpha=0.3)
    ax.set_xlabel("x", fontsize=9)
    ax.set_ylabel("y", fontsize=9)
    cbar = plt.colorbar(sc3, ax=ax, label="u [-]")
    ax.grid(True, linestyle=':', linewidth=0.5, alpha=0.7)
    ax.legend(loc='upper right', fontsize=9, borderaxespad=0., frameon=True)
    fig.savefig(f"{save_dir}/points_data_bigger_radius_t150.png", dpi=300, bbox_inches='tight')
    fig.savefig(f"{save_dir}/points_data_radius_t150.svg", dpi=300, bbox_inches='tight')
    plt.close(fig)



prepare_data_bis(config, save_dir)