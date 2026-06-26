from main import run_experiment

EXPERIMENTS = [

    #=== ARCHITECTURE ===

    {"name": "arch_02", "ic_x": -0.4, "ic_y": 0.0,
    "D": 0.025, "r": 0.012, "radius": 0.4,
    "n_inside": 12000, "n_ic": 10000, "n_outside": 10000,
    "neurons": 32, "n_hidden_layers": 5, "actfn": "swish",
    "epochs":150000, "lr": 1e-3, "optimizer": "Adam", "batch_size": 512,
    "alpha": 1, "beta": 1, "gamma": 1, "phi_slice_z": 7, "strategy": "centered"},

]

for exp in EXPERIMENTS:
    print(f"\n Running: {exp}")
    config = {
        "experiment_name": exp["name"],
        "model": {
            "in_shape": 3,
            "out_shape": 1,
            "n_hidden_layers": exp["n_hidden_layers"],
            "neuron_per_layer": exp["neurons"],
            "actfn": exp["actfn"]
        },
        "model_u": {
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
            "delta": 1.0,
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
        "phi_slice_path": "/home/vanderhaeghen/research/pinn_tumor_growth_framework/data_breast/phi.mat",
        "phi_slice_z": exp["phi_slice_z"],
        "diff_map_path": "/home/vanderhaeghen/research/pinn_tumor_growth_framework/data_breast/diff_map.mat",
        "physics": {
            "D": exp["D"],
            "r": exp["r"],
            "nu": 0.45,
            "lambda": 1.0,
            "G": 1.0,
            "coupling_gamma": 1.0,
            "t_max": 130.0,
        },
        "treatment": {
            "train_alpha": False,
            "alpha_clip": [0.0, 5.0],

            
            # "pff": drug exposure ∝ pff map intensity (zero outside breast)
            # "uniform": constant inside breast region (still multiplied by pff mask)
            "spatial_C": "pff",

            # List of drugs and their pharmacokinetic parameters
            "drugs": [
            {"name": "doxorubicin", "alpha": 0.08, "beta": 0.07, "times": [21, 42, 63, 84]},
            {"name": "cyclophosphamide", "alpha": 0.04, "beta": 0.21, "times": [21, 42, 63, 84]},
            ],
        }
    }

    loss_history = run_experiment(config)
    final_loss = loss_history[-1] if len(loss_history) > 0 else -1

    {'name': 'paclitaxel', 'alpha': 0.08, 'beta': 0.07, 'times': [21, 42, 63, 84]}, {'name': 'epirubicine', 'alpha': 0.04, 'beta': 0.21, 'times': [21, 42, 63, 84]}