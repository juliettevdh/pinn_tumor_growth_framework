# 🧠 Tumor Growth Simulation with PINN Project

Python project for **forward** and **inverse** modeling of tumor growth using Physics-Informed Neural Networks (PINNs).  
This repository contains the processing, training, and visualization scripts, along with the corresponding experimental data files.

---

## 📚 Table of Contents
- [📁 Project Structure](#-project-structure)
- [⚙️ Installation](#️-installation)
- [🚀 Usage](#-usage)
  - [Forward Model](#1-forward-model)
  - [Inverse Model](#2-inverse-model)
- [📊 Script Details](#-script-details)
- [📂 Data](#-data)
- [📈 Results](#-results)
- [✍️ Author](#️-author)

---

## 📁 Project Structure

The repository is organized into several main directories, separating forward modeling, inverse modeling, data, and results.
```
.
├── .venv_biostec/ # Python virtual environment
│
├── codes_forward/ # Forward modeling and simulation scripts
│ ├── automatic.py # Automates forward simulations
│ ├── data_utils.py # Data loading and preprocessing functions
│ ├── main.py # Main entry point for the forward model
│ ├── model.py # Defines the forward PINN architecture
│ ├── train.py # Model training script
│ ├── train_batch.py # Batch training version 
│ ├── visualisation.py # Plotting and visualization utilities
│ └── pycache/ # Compiled Python cache files
│
├── codes_inverse/ # Inverse problem scripts (parameter estimation)
│ ├── automatic_inverse.py # Automates inverse simulations
│ ├── data_utils_inverse.py# Data utilities for inverse modeling
│ ├── main_inverse.py # Main entry point for the inverse model
│ ├── model.py # Defines the inverse PINN architecture
│ ├── train_inverse_batch.py # Training for inverse estimation
│ ├── visualisation.py # Visualization tools for inverse results
│ └── pycache/ # Compiled Python cache files
│
├── data/ # Data for 8 patients
│ ├── P1/ 
│ ├── P2/
│ ├── P3/
│ ├── P4/
│ ├── P5/
│ ├── P6/
│ ├── P7/
│ └── P8/
│
├── results/ # Output directory for generated results and figures
```
