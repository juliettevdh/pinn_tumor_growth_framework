import numpy as np
import scipy.io
import nibabel as nib
from scipy.stats import qmc
import tensorflow as tf
import matplotlib.pyplot as plt

def load_mat(file_path):

    data = scipy.io.loadmat(file_path)['phi']

    return data[30:210, 10:200, 6:250]

def load_nifti(file_path):

    data = nib.load(file_path).get_fdata()
    data = np.flip(np.transpose(data, axes=[1, 0, 2]), 1)

    return data[30:210, 10:200, 6:250]

def sample_collocation_points_in_domain(n_points, x_min=-1, x_max=1, y_min=-1, y_max=1, t_min=0, t_max=200):
    
    sampler = qmc.LatinHypercube(d=3, scramble=True)
    points = sampler.random(n_points)
    x = x_min + (x_max - x_min) * points[:, 0]
    y = y_min + (y_max - y_min) * points[:, 1]
    t = t_min + (t_max - t_min) * points[:, 2]

    return np.column_stack((x, y, t))

def sample_collocation_points(n_points, x0, y0, rad, x_min=-1, x_max=1, y_min=-1, y_max=1, t_min=0, t_max=200):

    data = np.zeros((n_points, 3))
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

def sample_anchors_outside_brain(n_points, phi_slice, t_max = 200):

    anchors = []
    Ny, Nx = phi_slice.shape
    for _ in range(n_points):
        while True:
            x = np.random.uniform(-1, 1)
            y = np.random.uniform(-1, 1)
            t = np.random.uniform(0, t_max)
            ix = int((x + 1) / 2 * (Nx - 1))
            iy = int((y + 1) / 2 * (Ny - 1))
            if phi_slice[iy, ix] < 0.01 or  0.01 < phi_slice[iy, ix] < 0.05:
                anchors.append([x, y, t])
                break

    return np.array(anchors)

def sample_initial_condition_points(pff_slice,n_points, x0, y0, rad):
    
    n_ic = 1
    n_data_per_ic = n_points
    data = np.zeros([n_ic, n_data_per_ic, 4])
    radius = rad/2
    theta = np.random.uniform(0, 2 * np.pi, n_data_per_ic)
    r = radius * np.sqrt(np.random.uniform(0, 1, n_data_per_ic))
    x_circ = x0 + r * np.cos(theta)
    y_circ = y0 + r * np.sin(theta)
    t_circ = np.zeros(n_data_per_ic)
    data[0, :, 0] = x_circ
    data[0, :, 1] = y_circ
    data[0, :, 2] = t_circ
    data[0, :, 3] = 0.5*np.exp(-2*((x_circ - x0)**2 + (y_circ - y0)**2) / 0.1**2)
    data[0, :, 3] *= pff_slice[int((x0 + 1) / 2 * (pff_slice.shape[0] - 1)), int((y0 + 1) / 2 * (pff_slice.shape[1] - 1))]

    return data.reshape(n_data_per_ic * n_ic, 4)

def sample_initial_condition_points_in_domain(pff_slice,n_points, x0, y0, x_min=-1, x_max=1, y_min=-1, y_max=1, t_min=0, t_max=200):
    
    n_ic = 1
    n_data_per_ic = n_points
    data = np.zeros([n_ic, n_data_per_ic, 4])
    x_random = np.random.uniform(x_min, x_max, n_data_per_ic)
    y_random = np.random.uniform(y_min, y_max, n_data_per_ic)
    t_circ = np.zeros(n_data_per_ic)
    data[0, :, 0] = x_random
    data[0, :, 1] = y_random
    data[0, :, 2] = t_circ
    data[0, :, 3] = 0.5*np.exp(-2*((x_random - x0)**2 + (y_random - y0)**2) / 0.1**2)
    data[0, :, 3] *= pff_slice[int((x0 + 1) / 2 * (pff_slice.shape[0] - 1)), int((y0 + 1) / 2 * (pff_slice.shape[1] - 1))]

    return data.reshape(n_data_per_ic * n_ic, 4)


def prepare_data(config, save_dir):

    # --- Load data ---
    pff = load_mat(config["phi_slice_path"]) 
    pff_slice = pff[:, :, config["phi_slice_z"]]
    image = load_nifti("../data/P1/t1_masked_syn.nii")
    image = image[:, :, config["phi_slice_z"]]

    # --- Sample points ---
    if config["sampling"]["n_outside"] is not None:
        anchors_out = sample_anchors_outside_brain(config["sampling"]["n_outside"], pff_slice)
        anchors_out = np.hstack([anchors_out, np.zeros((anchors_out.shape[0], 1))])

    if config["sampling"]["strategy"] == "centered":

        ic_data = sample_initial_condition_points(
            pff_slice,
            config["sampling"]["n_ic"],
            config["sampling"]["ic_center"][0],
            config["sampling"]["ic_center"][1],
            config["sampling"]["ic_radius"]
        )
    
        collocation_data = sample_collocation_points(
            config["sampling"]["n_inside"],
            config["sampling"]["ic_center"][0],
            config["sampling"]["ic_center"][1],
            config["sampling"]["ic_radius"]
        )

    if config["sampling"]["strategy"] == "distributed":
        print("distributed")
        collocation_data = sample_collocation_points_in_domain(config["sampling"]["n_inside"])
        ic_data = sample_initial_condition_points_in_domain(
            pff_slice,
            config["sampling"]["n_ic"],
            config["sampling"]["ic_center"][0],
            config["sampling"]["ic_center"][1]        
            )

    # --- Time normalization ---
    t_max = 200.0  
    if config["sampling"]["n_outside"] is not None:
        x_d, y_d, t_d, u_d = [np.expand_dims(anchors_out[:, i], axis=1) for i in range(4)]
        t_d = t_d / t_max
    else:
        x_d = y_d = t_d = u_d = None

    x_ic, y_ic, t_ic, u_ic = [np.expand_dims(ic_data[:, i], axis=1) for i in range(4)]

    x_c, y_c, t_c = [np.expand_dims(collocation_data[:, i], axis=1) for i in range(3)]
    t_c = t_c / t_max

    # --- Plot ---
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
        "x_ic": tf.convert_to_tensor(x_ic, dtype=tf.float64),
        "y_ic": tf.convert_to_tensor(y_ic, dtype=tf.float64),
        "t_ic": tf.convert_to_tensor(t_ic, dtype=tf.float64),
        "u_ic": tf.convert_to_tensor(u_ic, dtype=tf.float64),
        "x_c": tf.convert_to_tensor(x_c, dtype=tf.float64),
        "y_c": tf.convert_to_tensor(y_c, dtype=tf.float64),
        "t_c": tf.convert_to_tensor(t_c, dtype=tf.float64),
    }