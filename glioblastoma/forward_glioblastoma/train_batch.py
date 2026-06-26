### This script implements the training of the PINN model using 
### some batching to improve computation time. 

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow_graphics.math.interpolation import trilinear as interp3d
from glioblastoma.forward_glioblastoma.data_utils import load_mat, load_nifti

tf.keras.backend.set_floatx("float64")

@tf.function
def u(model, x, y, t):
    return model(tf.concat([x, y, t], axis=1))

def mse(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))

def train_pinn(model, data, config, save_dir):

    pff_volume = load_mat(config["phi_slice_path"]) 
    gm = load_nifti(config["gm_path"])
    wm = load_nifti(config["wm_path"])
    csf = load_nifti(config["csf_path"])

    # --- Diffusion map ---
    tissue = wm + gm
    pWM = (tissue > csf) * wm
    pGM = (tissue > csf) * gm
    total = pWM + pGM
    pWM = np.where(total > 0, pWM / total, 0.)
    pGM = np.where(total > 0, pGM / total, 0.)
    diff_volume = (pWM + 0.1 * pGM)
    diff_slice = diff_volume[:, :, config["phi_slice_z"]]
    pff_slice = pff_volume[:, :, config["phi_slice_z"]]
    pff_mapped = pff_volume.astype("float32")[..., None]
    diff_mapped = diff_volume.astype("float32")[..., None]

    # ---Interpolators ---
    def pff_interp(v):
        v32 = tf.cast(v, tf.float32)
        i_x = (v32[:,0:1]+1.)/2.*(pff_mapped.shape[0]-1)
        i_y = (v32[:,1:2]+1.)/2.*(pff_mapped.shape[1]-1)
        i_z = tf.fill([tf.shape(v32)[0], 1], float(config["phi_slice_z"]))
        coords = tf.concat([i_x, i_y, i_z], axis=1)
        return interp3d.interpolate(tf.convert_to_tensor(pff_mapped, dtype=tf.float32), coords)

    def diff_interp(v):
        v32 = tf.cast(v, tf.float32)
        i_x = (v32[:,0:1]+1.)/2.*(diff_mapped.shape[0]-1)
        i_y = (v32[:,1:2]+1.)/2.*(diff_mapped.shape[1]-1)
        i_z = tf.fill([tf.shape(v32)[0], 1], float(config["phi_slice_z"]))
        coords = tf.concat([i_x, i_y, i_z], axis=1)
        return interp3d.interpolate(tf.convert_to_tensor(diff_mapped, dtype=tf.float32), coords)
    
    @tf.function
    def f(x, y, t):
        v = tf.concat([x, y, t], axis=1)
        diff_map = tf.cast(diff_interp(v), tf.float64)
        pff_map = tf.cast(pff_interp(v), tf.float64)
        with tf.GradientTape(persistent=True) as tape2:
            tape2.watch([x, y, t])
            with tf.GradientTape(persistent=True) as tape1:
                tape1.watch([x, y, t])
                u = model(tf.concat([x, y, t], axis=1))
            u_x = tape1.gradient(u, x)
            u_y = tape1.gradient(u, y)
            u_tau = tape1.gradient(u, t)
            del tape1
        u_xx = tape2.gradient(u_x, x)
        u_yy = tape2.gradient(u_y, y)
        del tape2

        D = config["physics"]["D"]
        r = config["physics"]["r"]
        D_norm = D * 0.5 * ((2./180)**2 + (2./150)**2)
        t_max = 200.0

        F = (u_tau - t_max * (D_norm * diff_map * (u_xx + u_yy) + r * u * (1 - u))) * pff_map
        return F
    
    # ---Training step ---
    @tf.function
    def train_step(x_ic_b, y_ic_b, t_ic_b, u_ic_b, x_bc_b, y_bc_b, t_bc_b, u_bc_b, x_c_b, y_c_b, t_c_b):
        with tf.GradientTape() as tape:
            u_pred = model(tf.concat([x_ic_b, y_ic_b, t_ic_b], axis=1))
            l_data = mse(u_ic_b, u_pred)
            u_bc_pred = model(tf.concat([x_bc_b, y_bc_b, t_bc_b], axis=1))
            l_bc = mse(u_bc_b, u_bc_pred)
            L_phys = tf.reduce_mean(tf.square(f(x_c_b, y_c_b, t_c_b)))
            loss = alpha * l_data + beta * l_bc + gamma * L_phys

        grads = tape.gradient(loss, model.trainable_weights)  # r excluded
        opt.apply_gradients(zip(grads, model.trainable_weights))
        return loss, l_data, l_bc, L_phys, grads
    
    alpha = config["train"]["alpha"]
    beta = config["train"]["beta"]
    gamma = config["train"]["gamma"]

    # Optimization
    if config["train"]["optimizer"] == "Adam":
        opt = tf.keras.optimizers.Adam(learning_rate=config["train"]["lr"])
    elif config["train"]["optimizer"] == "SGD":
        opt = tf.keras.optimizers.SGD(learning_rate=config["train"]["lr"])
    elif config["train"]["optimizer"] == "RMSprop":
        opt = tf.keras.optimizers.RMSprop(learning_rate=config["train"]["lr"])
    elif config["train"]["optimizer"] == "AdamW":
        opt = tf.keras.optimizers.AdamW(learning_rate=config["train"]["lr"])
    else:
        raise ValueError("Unsupported optimizer")

    loss_history = []
    loss_data = []
    loss_pde = []
    grad_list_norm = []

    dataset_ic = tf.data.Dataset.from_tensor_slices((data['x_ic'], data['y_ic'], data['t_ic'], data['u_ic'])).shuffle(buffer_size=data['x_ic'].shape[0], reshuffle_each_iteration=True).batch(config["train"]["batch_size"]).repeat()
    dataset_bc = tf.data.Dataset.from_tensor_slices((data['x_d'], data['y_d'], data['t_d'], data['u_d'])).shuffle(buffer_size=data['x_d'].shape[0], reshuffle_each_iteration=True).batch(config["train"]["batch_size"]).repeat()
    dataset_c = tf.data.Dataset.from_tensor_slices((data['x_c'], data['y_c'], data['t_c'])).shuffle(buffer_size=data['x_c'].shape[0], reshuffle_each_iteration=True).batch(config["train"]["batch_size"]).repeat()
    iter_ic = iter(dataset_ic)
    iter_bc = iter(dataset_bc)
    iter_c = iter(dataset_c)

    # ---Training loop ---
    for epoch in range(config["train"]["epochs"]):
        x_ic_b, y_ic_b, t_ic_b, u_ic_b = next(iter_ic)
        x_bc_b, y_bc_b, t_bc_b, u_bc_b = next(iter_bc)
        x_c_b, y_c_b, t_c_b = next(iter_c)

        #visualisation of points every 15000 epochs
        if epoch % 15000 == 0:
            plt.figure(figsize=(6,6))
            plt.scatter(x_ic_b, y_ic_b, c=u_ic_b, s=1, label='IC')
            plt.scatter(x_bc_b, y_bc_b, c=u_bc_b, s=1, label='BC')
            plt.scatter(x_c_b, y_c_b, c='green', s=1, label='Collocation')
            plt.colorbar(label='u value')
            plt.xlim([-1, 1])
            plt.ylim([-1, 1])
            plt.xlabel('x')
            plt.ylabel('y')
            plt.title(f'Training Points at Epoch {epoch}')
            plt.legend(markerscale=5)
            plt.grid(True)
            plt.savefig(os.path.join(save_dir, f'training_points_epoch_{epoch}.png'), dpi=300)
            plt.close()

        loss, l_data, l_bc, L_phys, grads = train_step(x_ic_b, y_ic_b, t_ic_b, u_ic_b, x_bc_b, y_bc_b, t_bc_b, u_bc_b, x_c_b, y_c_b, t_c_b)

        if epoch % 1000 == 0:
            print(f"[Phase 1] Epoch {epoch}, Loss: {loss.numpy():.6e}")
            loss_history.append(loss.numpy())
            loss_data.append(l_data.numpy()+l_bc.numpy())
            loss_pde.append(L_phys.numpy())
            grad_norm = tf.linalg.global_norm(grads).numpy()
            grad_list_norm.append(grad_norm)
            print("loss data:", l_data.numpy()+l_bc.numpy(), "L_phys:", L_phys.numpy())

    # --- Save model and loss ---
    model.save(os.path.join(save_dir, "model.h5"))
    np.savetxt(os.path.join(save_dir, "loss_history.txt"), np.array(loss_history))
    np.savetxt(os.path.join(save_dir, "loss_data.txt"), np.array(loss_data))
    np.savetxt(os.path.join(save_dir, "loss_pde.txt"), np.array(loss_pde))
    np.savetxt(os.path.join(save_dir, "grad_norm_history.txt"), np.array(grad_list_norm))

    # --- Loss history plot with separated losses ---
    plt.rcParams.update({'font.size': 9})
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.semilogy(np.arange(len(loss_data)), loss_data, label='Loss data', color="#237764", linewidth=1.4)
    ax.semilogy(np.arange(len(loss_pde)), loss_pde, label='Loss physics', color="#FB7E12", linewidth=1.4)
    ax.set_xlabel('Epochs x1000', fontsize=9)
    ax.set_ylabel('Loss (log scale)', fontsize=9)
    ax.set_title('Training Loss History', fontsize=9)
    ax.tick_params(axis='both', which='major', labelsize=9)
    ax.tick_params(axis='both', which='minor', labelsize=9)
    ax.legend(fontsize=9)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, 'loss_history_separated.png'), dpi=300)
    fig.savefig(os.path.join(save_dir, 'loss_history_separated.svg'), dpi=300)
    plt.close(fig)
    plt.rcParams.update(plt.rcParamsDefault)


    return loss_history, pff_slice, diff_slice