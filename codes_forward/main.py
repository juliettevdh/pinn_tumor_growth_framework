import os
from train_batch import train_pinn
from data_utils import prepare_data
from visualisation import visualize_solution_evolution
from model import DNN_builder
import tensorflow as tf
import matplotlib.pyplot as plt
import time
import numpy as np  # added for optional smoothing

def run_experiment(config):
    # Create unique folder for experiment
    exp_id = f"{config['experiment_name']}"
    save_dir = os.path.join("results/experiments_biostec_batch_200", exp_id)
    os.makedirs(save_dir, exist_ok=True)

    # Save config
    with open(os.path.join(save_dir, "config.txt"), "w") as f:
        for key, value in config.items():
            f.write(f"{key}: {value}\n")

    # Prepare data
    data = prepare_data(config, save_dir)

    # Build model
    tf.keras.backend.clear_session()
    model = DNN_builder(**config['model'])

    # Train
    start_time = time.time()
    loss_history, phi_slice, diff_slice = train_pinn(model, data, config, save_dir)
    end_time = time.time()
    
    #write to config file
    with open(os.path.join(save_dir, "config.txt"), "a") as f:
        f.write(f"Training time: {end_time - start_time:.2f} seconds\n")
        f.write(f"Final loss: {loss_history[-1] if len(loss_history) > 0 else -1}\n")
        
    #plot loss (enhanced, no smoothing)
    # Force fontsize 9 globally for this plot
    plt.rcParams.update({'font.size': 9})
    plt.style.use('seaborn-v0_8-darkgrid')  # modern style with grid
    fig, ax = plt.subplots(figsize=(10, 5))
    epochs = np.arange(1, len(loss_history) + 1)
    ax.semilogy(epochs, loss_history, label='Loss', color='#1f77b4', linewidth=1.4, alpha=0.9)

    ax.set_title('Training Loss History', fontsize=9, pad=12)
    ax.set_xlabel('Epochs x1000', fontsize=9)
    ax.set_ylabel('Loss (log scale)', fontsize=9)

    # Minor ticks + grid refinement
    ax.minorticks_on()
    ax.grid(which='major', linestyle='-', linewidth=0.6, alpha=0.7)
    ax.grid(which='minor', linestyle=':', linewidth=0.4, alpha=0.5)
    
    # Set tick label fontsize explicitly
    ax.tick_params(axis='both', which='major', labelsize=9)
    ax.tick_params(axis='both', which='minor', labelsize=9)

    # Annotate final loss
    final_loss = loss_history[-1]
    ax.text(0.99, 0.95, f"Final: {final_loss:.2e}", transform=ax.transAxes, ha='right', va='top', fontsize=9,
        bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='#888', alpha=0.8))

    ax.legend(frameon=True, fontsize=9)
    fig.savefig(os.path.join(save_dir, 'loss_history.svg'), dpi=300)
    fig.savefig(os.path.join(save_dir, 'loss_history.png'), dpi=300)
    plt.close(fig)
    
    # Reset rcParams to default after plot
    plt.rcParams.update(plt.rcParamsDefault)

    # Visualization
    visualize_solution_evolution(
        model, diff_slice, phi_slice, save_dir, config,
        t_snap=config.get('visualization', {}).get('t_snap', None),
        value_threshold=config.get('visualization', {}).get('value_threshold', 0.05),
        L=config.get('visualization', {}).get('L', 1.0)
    )

    print(f"✅ Experiment {exp_id} done. Outputs saved in {save_dir}")

    return loss_history
