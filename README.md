# Tumor Growth Simulation with PINN Project

Python project for **forward** and **inverse** modeling of tumor growth using Physics-Informed Neural Networks (PINNs). 
The repository presents two applications. The first framework addresses forward and inverse problems in the context of glioblastoma growth modeling. 
The second framework extends the first by incorporating a treatment term into the governing PDE and applies the methodology to breast MRI data.
All experiments are performed on synthetic tumors.
This repository contains the processing, training, and visualization scripts along with the requirements needed to run the project.

---

## Table of Contents
- [Project Structure](#-project-structure)
- [Installation](#️-installation)
- [Usage](#-usage)

---

## Project Structure

The repository is organized according to the considered application (glioblastoma or breast cancer under treatment) and the type of problem (forward simulation or inverse parameter estimation).

```
.
├── glioblastoma/
│   │
│   ├── forward_glioblastoma/     # Forward PINN for glioblastoma growth prediction
│   │   ├── automatic.py          # Automates forward simulations and creates several experiments
│   │   ├── data_utils.py         # Data loading and preprocessing
│   │   ├── main.py               # Main workflow: configuration, training, visualization
│   │   ├── model.py              # PINN architecture definition
│   │   ├── train.py              # Standard training procedure
│   │   ├── train_batch.py        # Batch training procedure
│   │   ├── visualisation.py      # Plotting and visualization utilities
│   │
│   └── inverse_glioblastoma/     # Inverse PINN for parameter estimation
│       ├── automatic_inverse.py  # Automates inverse experiments
│       ├── data_utils_inverse.py # Data loading and preprocessing
│       ├── main_inverse.py       # Main workflow for inverse experiments
│       ├── model.py              # PINN architecture definition
│       ├── train_inverse_batch.py# Batch training for inverse problems
│       ├── visualisation.py      # Plotting and visualization utilities
│
├── treatment_breast/
│   │
│   ├── forward_breast/           # Forward PINN for treatment simulations
│   │   ├── automatic.py          # Automates forward experiments
│   │   ├── data_utils.py         # Data loading and preprocessing
│   │   ├── main.py               # Main workflow
│   │   ├── model.py              # PINN architecture definition
│   │   ├── train_batch.py        # Mini-batch training procedure
│   │   ├── visualisation.py      # Plotting and visualization utilities
│   │   └── results/              # Generated figures and experiment outputs
│   │
│   └── inverse_breast/           # Inverse PINN for treatment parameter estimation
│       ├── automatic.py          # Automates inverse experiments
│       ├── data_utils.py         # Data loading and preprocessing
│       ├── main.py               # Main workflow
│       ├── model.py              # PINN architecture definition
│       ├── train_batch_inverse.py# Inverse training procedure
│       ├── plot.py               # Additional plotting utilities
│       ├── visualisation.py      # Visualization utilities
│       └── results/              # Generated figures and experiment outputs
│
├── requirements.txt              # Python dependencies
├── README.md                     # Project documentation
├── LICENSE                       # Project license
```

## Installation
A virtual environment with the following requirements should be created: 
```
absl-py==2.3.1
array_record==0.8.1
astunparse==1.6.3
attrs==25.3.0
certifi==2025.8.3
charset-normalizer==3.4.3
contourpy==1.3.2
cycler==0.12.1
dm-tree==0.1.9
docstring_parser==0.17.0
einops==0.8.1
etils==1.13.0
flatbuffers==25.2.10
fonttools==4.59.2
fsspec==2025.9.0
gast==0.6.0
google-pasta==0.2.0
grpcio==1.74.0
h5py==3.14.0
idna==3.10
imageio==2.37.0
immutabledict==4.2.1
importlib_resources==6.5.2
keras==3.11.3
kiwisolver==1.4.9
libclang==18.1.1
Markdown==3.9
markdown-it-py==4.0.0
MarkupSafe==3.0.2
matplotlib==3.10.6
mdurl==0.1.2
ml_dtypes==0.5.3
namex==0.1.0
networkx==3.4.2
nibabel==5.3.2
numpy==2.1.3
nvidia-cublas-cu12==12.9.1.4
nvidia-cuda-cupti-cu12==12.9.79
nvidia-cuda-nvcc-cu12==12.9.86
nvidia-cuda-nvrtc-cu12==12.9.86
nvidia-cuda-runtime-cu12==12.9.79
nvidia-cudnn-cu12==9.13.0.50
nvidia-cufft-cu12==11.4.1.4
nvidia-curand-cu12==10.3.10.19
nvidia-cusolver-cu12==11.7.5.82
nvidia-cusparse-cu12==12.5.10.65
nvidia-nccl-cu12==2.28.3
nvidia-nvjitlink-cu12==12.9.86
OpenEXR==3.4.0
opt_einsum==3.4.0
optree==0.17.0
packaging==25.0
pillow==11.3.0
promise==2.3
protobuf==4.21.12
psutil==7.0.0
pyarrow==21.0.0
Pygments==2.19.2
pyparsing==3.2.3
python-dateutil==2.9.0.post0
requests==2.32.5
rich==14.1.0
scipy==1.15.3
simple-parsing==0.1.7
six==1.17.0
tensorboard==2.19.0
tensorboard-data-server==0.7.2
tensorflow==2.19.1
tensorflow-addons==0.23.0
tensorflow-datasets==4.9.9
tensorflow-graphics==2021.12.3
tensorflow-io-gcs-filesystem==0.37.1
tensorflow-metadata==1.17.2
termcolor==3.1.0
toml==0.10.2
tqdm==4.67.1
trimesh==4.8.1
typeguard==2.13.3
typing_extensions==4.15.0
urllib3==2.5.0
Werkzeug==3.1.3
wrapt==1.17.3
zipp==3.23.0
```

## Usage

To launch experiments, please run the automatic.py and automatic_inverse.py files. Details about experiments can be changed. 
