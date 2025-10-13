# This script runs a series of experiments for the inverse problem
# using different hyperparameters and configurations. Each experiment's
# settings are defined in the EXPERIMENTS list, and the run_experiment
# function is called with the corresponding configuration.

from main_inverse import run_experiment

EXPERIMENTS = [
    # === ARCHITECTURE ===
     {"name": "inverse_unknown_D_r", "ic_x": 0.4, "ic_y": 0.4,
    "D": 0.013, "r": 0.012, "radius": 0.35,
     "n_inside": 12000, "n_ic": 4000, "n_outside": 4000,
     "neurons": 32, "n_hidden_layers": 5, "actfn": "swish",
     "epochs": 500000, "lr": 1e-3, "optimizer": "Adam", "batch_size": 512,
     "alpha": 1, "beta": 1, "gamma": 1, "phi_slice_z": 166, "strategy": "distributed"},
]

for exp in EXPERIMENTS:
    print(f"\n🚀 Running: {exp}")
    config = {
        "experiment_name": exp["name"],
        "model": {
            "in_shape": 3,
            "out_shape": 1,
            "n_hidden_layers": exp["n_hidden_layers"],
            "neuron_per_layer": exp["neurons"],
            "actfn": exp["actfn"]
        },
        "train": {
            "epochs": exp["epochs"],
            "lr": exp["lr"],
            "optimizer": exp["optimizer"],
            "alpha": exp["alpha"],
            "beta": exp["beta"],
            "gamma": exp["gamma"],
            "batch_size": exp["batch_size"]
        },
        "sampling": {
            "n_ic": exp["n_ic"],
            "n_outside": exp["n_outside"],
            "n_inside": exp["n_inside"],
            "ic_center": [exp["ic_x"], exp["ic_y"]],
            "ic_radius": exp["radius"],
            "diff_radius_min": 0.07,
            "diff_radius_max": 0.25,
            "diff_steps": 50,
            "strategy": exp["strategy"],
        },
        "phi_slice_path": "../data/P1/phi.mat",
        "phi_slice_z": exp["phi_slice_z"],
        "gm_path": "../data/P1/GM_syn.nii",
        "wm_path": "../data/P1/WM_syn.nii",
        "csf_path": "../data/P1/CSF_syn.nii",
        "physics": {
            "D": exp["D"],
            "r": exp["r"],
        },
    }

    loss_history = run_experiment(config)
    final_loss = loss_history[-1] if len(loss_history) > 0 else -1
