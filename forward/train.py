import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow_graphics.math.interpolation import trilinear as interp3d
from data_utils import load_mat, load_nifti

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
        u_xx = tape2.gradient(u_x, x)
        u_yy = tape2.gradient(u_y, y)

        D = config["physics"]["D"]
        r = config["physics"]["r"]
        D_norm = D * 0.5 * ((2./180)**2 + (2./150)**2)
        t_max = 200.0
        F = (u_tau - t_max * (D_norm * diff_map * (u_xx + u_yy) + r * u * (1 - u))) * pff_map
        return F
    
    alpha = config["train"]["alpha"]
    beta = config["train"]["beta"]
    gamma = config["train"]["gamma"]

    # --- Optimization ---
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

    for epoch in range(config["train"]["epochs"]):
        with tf.GradientTape() as tape:
            u_pred = model(tf.concat([data['x_ic'], data['y_ic'], data['t_ic']], axis=1))
            l_data = mse(data['u_ic'], u_pred)
            u_bc_pred = model(tf.concat([data['x_d'], data['y_d'], data['t_d']], axis=1))
            l_bc = mse(data['u_d'], u_bc_pred)
            L_phys = tf.reduce_mean(tf.square(f(data['x_c'], data['y_c'], data['t_c'])))
            loss = alpha * l_data + beta * l_bc + gamma * L_phys

        grads = tape.gradient(loss, model.trainable_weights)
        opt.apply_gradients(zip(grads, model.trainable_weights))

        if epoch % 1000 == 0:
            print(f"[Phase 1] Epoch {epoch}, Loss: {loss.numpy():.6e}")
            loss_history.append(loss.numpy())
            loss_data.append(l_data.numpy()+l_bc.numpy())
            loss_pde.append(L_phys.numpy())
            print("loss data:", l_data.numpy()+l_bc.numpy(), "L_phys:", L_phys.numpy())

    # --- Save model and loss ---
    model.save(os.path.join(save_dir, "model.h5"))
    np.savetxt(os.path.join(save_dir, "loss_history.txt"), np.array(loss_history))
    np.savetxt(os.path.join(save_dir, "loss_data.txt"), np.array(loss_data))
    np.savetxt(os.path.join(save_dir, "loss_pde.txt"), np.array(loss_pde))

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