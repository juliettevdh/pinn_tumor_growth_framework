### This script implements the training of the PINN model using 
### some batching to improve computation time. 

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow_graphics.math.interpolation import trilinear as interp3d
from data_utils import load_mat
import cv2
from visualisation import plot_center_time_evolution

tf.keras.backend.set_floatx("float64") 

@tf.function
def u(model, x, y, t):
    return model(tf.concat([x, y, t], axis=1))

def mse(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))

def train_pinn(model, data, config, save_dir, debug=False):

    os.makedirs(save_dir, exist_ok=True)

    # ---Load maps ---
    pff_volume = load_mat(config["phi_slice_path"], 'phi')
    diff_volume = load_mat(config["diff_map_path"], 'diff_map')
    zslice = int(config.get("phi_slice_z", 0))
    pff_slice = pff_volume[:, :, zslice].astype(np.float32)
    diff_slice = diff_volume[:, :, zslice].astype(np.float32)

    pff_slice_vis = cv2.GaussianBlur(pff_slice, (15, 15), 0)

    # Save visual slices
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.imshow(pff_slice_vis, cmap='jet')
    plt.title(f"PFF slice z={zslice}")
    plt.colorbar()
    plt.subplot(1,2,2)
    plt.imshow(diff_slice, cmap='jet')
    plt.title(f"Diffusion slice z={zslice}")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'maps_slices.png'), dpi=300)
    plt.close()

    # Put volumes into arrays for interpolation
    pff_mapped = pff_volume.astype(np.float32)[..., None]  # (X,Y,Z,1)
    diff_mapped = diff_volume.astype(np.float32)[..., None]

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

    # ---Treatment parameters ---
    treatment_cfg = config.get("treatment", {})
    drugs = treatment_cfg.get("drugs", [])
    train_alpha = bool(treatment_cfg.get("train_alpha", False))



    # ---lambda / treatment computation ---
    t_max = float(config.get("t_max", 130.0))

    @tf.function
    def compute_lambda_tf(v, t_norm):
        """
        Compute treatment across time
        """
        phys_t = tf.cast(t_norm, tf.float64) * tf.cast(t_max, tf.float64)  # days
        pff_vals = tf.cast(pff_interp(v), tf.float64)

        lam_total = tf.zeros_like(pff_vals, dtype=tf.float64)
        spatial_C = treatment_cfg.get("spatial_C", "pff")
        k_smooth = tf.cast(5.0, tf.float64)  
        for i, drug in enumerate(drugs):
            beta_i = tf.cast(float(drug.get("beta", 1.0)), tf.float64)
            alpha_i = tf.cast(float(drug.get("alpha", 0.3)), tf.float64)
            times = drug.get("times", [])
            # if no administrations -> skip
            if not times:
                continue
            # for each tau, compute alpha * exp(-beta*(phys_t - tau)) * H(phys_t - tau)
            contribs = []
            for tau in times:
                tauf = tf.cast(float(tau), tf.float64)
                decay = alpha_i * tf.exp(-beta_i * (phys_t - tauf))
                heaviside = 0.5 * (1.0 + tf.tanh(k_smooth * (phys_t - tauf)))
                contribs.append(decay * heaviside)
            temporal = tf.add_n(contribs)
            if spatial_C == "pff":
                lam_total += temporal * pff_vals
            else:
                # uniform inside tissue: use indicator pff_vals>0 as mask
                lam_total += temporal * tf.cast(pff_vals > 0.0, tf.float64)

        return lam_total  # [N,1]
    
    def plot_lambda_map(model, config, t_phys, save_path=None):
        """
        Plot de lambda(x,y) pour un temps physique donné.
        """
        t_max = 130
        t_norm = t_phys / t_max  # normalised in [0,1]

        # grid on [-1,1]x[-1,1]
        n = 256
        x = np.linspace(-1, 1, n)
        y = np.linspace(-1, 1, n)
        X, Y = np.meshgrid(x, y)
        t = np.full_like(X, t_norm)

        v = tf.convert_to_tensor(np.stack([X.flatten(), Y.flatten(), t.flatten()], axis=1), dtype=tf.float64)
        t_tensor = tf.convert_to_tensor(t.flatten()[:, None], dtype=tf.float64)

        lam_map = compute_lambda_tf(v, t_tensor).numpy().reshape(n, n)

        plt.figure(figsize=(6,5))
        plt.imshow(lam_map, extent=[-1,1,-1,1], origin='lower', cmap='magma')
        plt.colorbar(label='λ(x,y)')
        plt.title(f"λ map at t = {t_phys:.2f} days (t_norm={t_norm:.3f})")
        if save_path:
            plt.savefig(os.path.join(save_path, f"lambda_map_t{int(t_phys)}.png"), dpi=300)
            plt.close()
        else:
            plt.show()

    for t in [0, 20, 21, 25, 50, 100, 122, 126, 128, 150]:
        plot_lambda_map(model, config, t, save_path=save_dir)


    # ---PDE residual ---
    @tf.function
    def pde_residual(x, y, t_norm):
        v = tf.concat([x, y, t_norm], axis=1)  # [N,3]
        diff_map = tf.cast(diff_interp(v), tf.float64)  # [N,1] float64
        pff_map = tf.cast(pff_interp(v), tf.float64)    # [N,1] float64
        with tf.GradientTape(persistent=True) as tape2:
            tape2.watch([x, y, t_norm])
            with tf.GradientTape(persistent=True) as tape1:
                tape1.watch([x, y, t_norm])
                u = model(tf.concat([x, y, t_norm], axis=1)) 
            u_x = tape1.gradient(u, x)
            u_y = tape1.gradient(u, y)
            u_t = tape1.gradient(u, t_norm)
        u_xx = tape2.gradient(u_x, x)
        u_yy = tape2.gradient(u_y, y)
        del tape1
        del tape2

        D = tf.cast(config["physics"]["D"], tf.float64)
        r = tf.cast(config["physics"]["r"], tf.float64)
        D_norm = D * tf.cast(0.5 * ((2.0/350.0)**2 + (2.0/256.0)**2), tf.float64)

        lam_map = compute_lambda_tf(v, t_norm)  # [N,1]


        diffusion_term = D_norm * diff_map * (u_xx + u_yy)
        reaction_term = r * u * (1.0 - u)
        treatment_term = lam_map * u

        F_inner = diffusion_term + reaction_term - treatment_term 
        # convert normalized-time derivative back to physical scale t_max
        F = (u_t - (tf.cast(t_max, tf.float64) * F_inner)) * pff_map 
        return F

    # ---Training step ---
    @tf.function
    def train_step(x_ic_b, y_ic_b, t_ic_b, u_ic_b, x_bc_b, y_bc_b, t_bc_b, u_bc_b, x_c_b, y_c_b, t_c_b):
        with tf.GradientTape() as tape:
            # data losses
            u_pred_ic = model(tf.concat([x_ic_b, y_ic_b, t_ic_b], axis=1))
            l_data = mse(u_ic_b, u_pred_ic)

            u_pred_bc = model(tf.concat([x_bc_b, y_bc_b, t_bc_b], axis=1))
            l_bc = mse(u_bc_b, u_pred_bc)

            # physics loss
            F = pde_residual(x_c_b, y_c_b, t_c_b)
            L_phys = tf.reduce_mean(tf.square(F))

            loss = alpha * l_data + beta * l_bc + gamma * L_phys

        train_vars = model.trainable_weights
        grads = tape.gradient(loss, train_vars)
        opt.apply_gradients(zip(grads, train_vars))

        return loss, l_data, l_bc, L_phys, grads
    
    alpha = float(config["train"]["alpha"])
    beta = float(config["train"]["beta"])
    gamma = float(config["train"]["gamma"])

    opt_name = config["train"].get("optimizer", "Adam")
    lr = float(config["train"].get("lr", 1e-3))
    if opt_name == "Adam":
        opt = tf.keras.optimizers.Adam(learning_rate=lr)
    elif opt_name == "SGD":
        opt = tf.keras.optimizers.SGD(learning_rate=lr)
    elif opt_name == "RMSprop":
        opt = tf.keras.optimizers.RMSprop(learning_rate=lr)
    elif opt_name == "AdamW":
        opt = tf.keras.optimizers.AdamW(learning_rate=lr)
    else:
        raise ValueError("Unsupported optimizer")
    
    loss_history = []
    loss_data = []
    loss_pde = []
    grad_norms = []

    # ---------- Prepare datasets ----------
    batch_size = int(config["train"]["batch_size"])
    epochs = int(config["train"]["epochs"])

    dataset_ic = tf.data.Dataset.from_tensor_slices((data['x_ic'], data['y_ic'], data['t_ic'], data['u_ic'])) \
                .shuffle(buffer_size=int(data['x_ic'].shape[0])) \
                .batch(batch_size).repeat()
    dataset_bc = tf.data.Dataset.from_tensor_slices((data['x_d'], data['y_d'], data['t_d'], data['u_d'])) \
                .shuffle(buffer_size=int(data['x_d'].shape[0])) \
                .batch(batch_size).repeat()
    dataset_c = tf.data.Dataset.from_tensor_slices((data['x_c'], data['y_c'], data['t_c'])) \
                .shuffle(buffer_size=int(data['x_c'].shape[0])) \
                .batch(batch_size*2).repeat()

    iter_ic = iter(dataset_ic)
    iter_bc = iter(dataset_bc)
    iter_c = iter(dataset_c)

    # ---Training loop ---
    for epoch in range(epochs):
        x_ic_b, y_ic_b, t_ic_b, u_ic_b = next(iter_ic)
        x_bc_b, y_bc_b, t_bc_b, u_bc_b = next(iter_bc)
        x_c_b, y_c_b, t_c_b = next(iter_c)

        loss, l_data, l_bc, L_phys, grads = train_step(
            tf.cast(x_ic_b, tf.float64), tf.cast(y_ic_b, tf.float64), tf.cast(t_ic_b, tf.float64), tf.cast(u_ic_b, tf.float64),
            tf.cast(x_bc_b, tf.float64), tf.cast(y_bc_b, tf.float64), tf.cast(t_bc_b, tf.float64), tf.cast(u_bc_b, tf.float64),
            tf.cast(x_c_b, tf.float64), tf.cast(y_c_b, tf.float64), tf.cast(t_c_b, tf.float64)
        )

        if epoch % 1000 == 0:
            print(f"[Training] Epoch {epoch}/{epochs} Loss: {loss.numpy():.4e} (data: {l_data.numpy()+l_bc.numpy():.4e}, pde: {L_phys.numpy():.4e})")
            loss_history.append(loss.numpy())
            loss_data.append((l_data.numpy()+l_bc.numpy()))
            loss_pde.append(L_phys.numpy())
            grad_norms.append(tf.linalg.global_norm(grads).numpy())

        # visualisation of points every 15000 epochs
        if epoch % 15000 == 0:
            plt.figure(figsize=(6,6))
            plt.scatter(data['x_ic'].numpy().flatten(), data['y_ic'].numpy().flatten(), c=data['u_ic'].numpy().flatten(), s=1, label='IC')
            plt.scatter(data['x_d'].numpy().flatten(), data['y_d'].numpy().flatten(), c=data['u_d'].numpy().flatten(), s=1, label='BC')
            # collocation subset (for plotting)
            xc = x_c_b.numpy().flatten(); yc = y_c_b.numpy().flatten()
            plt.scatter(xc, yc, c='green', s=1, label='Collocation')
            plt.xlim([-1,1]); plt.ylim([-1,1]); plt.legend()
            plt.title(f"Points epoch {epoch}")
            plt.savefig(os.path.join(save_dir, f"training_points_epoch_{epoch}.png"), dpi=300)
            plt.close()

            t_phys, u_center = plot_center_time_evolution(
        model, config, config["sampling"]["ic_center"][0], config["sampling"]["ic_center"][1], n_times=130,
        save_path=os.path.join(save_dir, f"u_center_vs_time_{epoch}.png")
    )

    # ---Save results ---
    model.save(os.path.join(save_dir, "model.h5"))
    np.savetxt(os.path.join(save_dir, "loss_history.txt"), np.array(loss_history))
    np.savetxt(os.path.join(save_dir, "loss_data.txt"), np.array(loss_data))
    np.savetxt(os.path.join(save_dir, "loss_pde.txt"), np.array(loss_pde))
    np.savetxt(os.path.join(save_dir, "grad_norm_history.txt"), np.array(grad_norms))

    # ---Loss plot ---
    plt.figure(figsize=(10,5))
    if loss_data:
        plt.semilogy(np.arange(len(loss_data)), loss_data, label='Loss data')
    if loss_pde:
        plt.semilogy(np.arange(len(loss_pde)), loss_pde, label='Loss physics')
    plt.xlabel('Checkpoint (every 1000 epochs)')
    plt.ylabel('Loss (log)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'loss_history_separated.png'), dpi=300)
    plt.close()

    return loss_history, pff_slice, diff_slice
