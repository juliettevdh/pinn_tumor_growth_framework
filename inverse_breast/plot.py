import numpy as np
import matplotlib.pyplot as plt
import os

def plot_alpha_history(save_dir):
    """
    Plot evolution of exp(alpha) during PINN training.
    
    """

    file_path = os.path.join(save_dir, "alpha_history_dox.txt")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    alpha_vals = np.loadtxt(file_path)
    if alpha_vals.ndim == 0:
        alpha_vals = np.array([alpha_vals])

    epochs = np.arange(len(alpha_vals))

    plt.figure(figsize=(8,5))
    plt.plot(epochs, alpha_vals, lw=2, color="tab:blue")
    plt.axhline(y=0.12, color='r', linestyle='--', label='Ground Truth')
    plt.legend()
    plt.xlabel("Epochs", fontsize=12)
    plt.ylabel("Treatment efficacy α", fontsize=12)
    plt.title("Evolution of treatment efficacy α during training", fontsize=14)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.yscale("log")
    plt.tight_layout()

    out_path = os.path.join(save_dir, "alpha_history_plot_dox2.png")
    plt.savefig(out_path, dpi=300)
    plt.show()

    print(f"✅ Saved plot to {out_path}")

if __name__ == "__main__":
    save_directory = "/home/vanderhaeghen/research/pinn_tumor_growth_framework/inverse_breast/results/experiments_biostec_25_06/arch_02_25_06/" 
    plot_alpha_history(save_directory)
