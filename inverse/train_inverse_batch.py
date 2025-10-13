import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow_graphics.math.interpolation import trilinear as interp3d
from data_utils_inverse import load_mat, load_nifti
from matplotlib.ticker import LogLocator, LogFormatterSciNotation

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
    
    r = tf.Variable(tf.math.log(tf.constant(1e-2, dtype=tf.float64)), dtype=tf.float64, trainable=True)
    D = tf.Variable(tf.math.log(tf.constant(1e-6, dtype=tf.float64)), dtype=tf.float64, trainable=True)
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

        t_max = 200.0

        F = (u_tau - t_max*(tf.exp(D) * diff_map * (u_xx + u_yy) + tf.exp(r) * u * (1 - u))) * pff_map
        return F
    
    alpha = config["train"]["alpha"]
    gamma = config["train"]["gamma"]
    
    @tf.function
    def train_step(x_data_b, y_data_b, t_data_b, u_data_b, x_c_b, y_c_b, t_c_b):
        with tf.GradientTape() as tape:
            u_pred = model(tf.concat([x_data_b, y_data_b, t_data_b], axis=1))
            l_data = mse(u_data_b, u_pred)
            L_phys = tf.reduce_mean(tf.square(f(x_c_b, y_c_b, t_c_b)))
            loss = alpha * l_data + gamma * L_phys

        grads = tape.gradient(loss, model.trainable_weights + [D,r])
        opt.apply_gradients(zip(grads, model.trainable_weights + [D,r]))
        return loss, l_data, L_phys, grads, D, r

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
    r_list = []
    D_list = []
    D_val_phys = []
    loss_data = []
    loss_pde = []
    grad_list_norm = []

    dataset_c = tf.data.Dataset.from_tensor_slices((data['x_c'], data['y_c'], data['t_c'])).shuffle(buffer_size=data['x_c'].shape[0], reshuffle_each_iteration=True).batch(config["train"]["batch_size"]).repeat()
    dataset_data = tf.data.Dataset.from_tensor_slices((data['x_data'], data['y_data'], data['t_data'], data['u_data'])).shuffle(buffer_size=data['x_data'].shape[0], reshuffle_each_iteration=True).batch(config["train"]["batch_size"]).repeat()
    iter_c = iter(dataset_c)
    iter_data = iter(dataset_data)

    for epoch in range(config["train"]["epochs"]):
        x_c_b, y_c_b, t_c_b = next(iter_c)
        x_data_b, y_data_b, t_data_b, u_data_b = next(iter_data)

        #visualisation of points every 15000 epochs
        if epoch % 15000 == 0:
            plt.figure(figsize=(6,6))
            plt.scatter(x_data_b, y_data_b, c='blue', s=1, label='Data')
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

        loss, l_data, L_phys, grads, D_train, r_train = train_step(x_data_b, y_data_b, t_data_b, u_data_b, x_c_b, y_c_b, t_c_b)
        r_list.append(tf.exp(r_train).numpy())
        D_list.append(tf.exp(D_train).numpy())
        D_val_phys.append(tf.exp(D_train).numpy()/(0.5 * ((2./180)**2 + (2./150)**2)))

        if epoch % 1000 == 0:
            print(f"[Phase 1] Epoch {epoch}, Loss: {loss.numpy():.6e}")
            print(f"D: {tf.exp(D_train).numpy():.6e}", f"r: {tf.exp(r_train).numpy():.6e}")
            print("Estimated D (physical):", tf.exp(D_train).numpy()/(0.5 * ((2./180)**2 + (2./150)**2)))
            loss_history.append(loss.numpy())
            loss_data.append(l_data.numpy())
            loss_pde.append(L_phys.numpy())
            grad_norm = tf.linalg.global_norm(grads).numpy()
            grad_list_norm.append(grad_norm)
            print("loss data:", l_data.numpy(), "L_phys:", L_phys.numpy())

    # --- Save model and training history ---
    model.save(os.path.join(save_dir, "model.h5"))
    np.savetxt(os.path.join(save_dir, "loss_history.txt"), np.array(loss_history))
    np.savetxt(os.path.join(save_dir, "D_history.txt"), np.array(D_list))
    np.savetxt(os.path.join(save_dir, "D_val_phys_history.txt"), np.array(D_val_phys))
    np.savetxt(os.path.join(save_dir, "r_history.txt"), np.array(r_list))
    np.savetxt(os.path.join(save_dir, "loss_data_history.txt"), np.array(loss_data))
    np.savetxt(os.path.join(save_dir, "loss_pde_history.txt"), np.array(loss_pde))

    # --- Plots ---
    epochs_D = np.arange(len(D_list))
    epochs_r = np.arange(len(r_list))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # --- Coefficient of diffusion D plot ---
    ax1.plot(epochs_D, D_list, label=r'$D_w$', color='#1f77b4', linewidth=1.8)
    ax1.axhline(y=0.013, color='#1f77b4', linestyle='--', label=r'True $D_w$', linewidth=2.2)
    ax1.set_yscale('log')
    ax1.set_ylabel(r'$D_w$ value [mm²/day]', fontsize=13)
    ax1.set_title(r'Diffusion Coefficient $D_w$ over Training', fontsize=13)
    ax1.grid(which='major', linestyle='-', alpha=0.6)
    ax1.grid(which='minor', linestyle=':', alpha=0.4)
    ax1.yaxis.set_major_locator(LogLocator(base=10.0, numticks=12))
    ax1.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10)*0.1, numticks=100))
    ax1.yaxis.set_major_formatter(LogFormatterSciNotation())
    ax1.legend(fontsize=13, frameon=True)

    # --- Proliferation rate r plot ---
    ax2.plot(epochs_r, r_list, label=r'$\rho$', color="#bc7bc4", linewidth=1.8)
    ax2.axhline(y=0.012, color="#bc7bc4", linestyle='--', label=r'True $\rho$', linewidth=2.2)
    ax2.set_yscale('log')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel(r'$\rho$ value [day⁻¹]', fontsize=13)
    ax2.set_title(r'Proliferation Rate $\rho$ over Training', fontsize=13)
    ax2.grid(which='major', linestyle='-', alpha=0.6)
    ax2.grid(which='minor', linestyle=':', alpha=0.4)
    ax2.yaxis.set_major_locator(LogLocator(base=10.0, numticks=13))
    ax2.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10)*0.1, numticks=100))
    ax2.yaxis.set_major_formatter(LogFormatterSciNotation())
    ax2.legend(fontsize=13, frameon=True, loc='lower right')

    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "D_r_history_bis.png"), dpi=300)
    fig.savefig(os.path.join(save_dir, "D_r_history_bis.svg"), dpi=300)
    plt.close(fig)

    # --- Plot loss history but separated, loss for data and loss for physics ---
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