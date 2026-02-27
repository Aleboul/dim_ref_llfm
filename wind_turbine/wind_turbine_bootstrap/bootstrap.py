import os
import numpy as np
import pandas as pd
from config import FILE_PATH, Q, KAPPA, LAMBDA, GPD_FIT_Q
from data.loader import load_data, preprocess_data
from utils.transforms import pareto_rank_transform_matrix, compute_tpdm
from utils.helpers import complement_partitions
from utils.io import save_results
from estimation.purevar import PureVar, estimate_A_hat_I
from estimation.factors import compute_C_hat
from estimation.lasso import update_A_hat_with_refit
from evaluation.exceedances_bootstrap import plot_bootstrap_failure, compute_failure_sets_bootstrap_weighted, plot_bootstrap_failure_alpha


# ============================================================
# PARAMETERS
# ============================================================
BOOTSTRAP_DIR = "bootstrap"
pixels_with_wind_farm_counts = pd.read_csv('data/prod_data/pixels_with_wind_farm_counts.csv')
X_raw = load_data(FILE_PATH)
X = preprocess_data(X_raw)
Y = pareto_rank_transform_matrix(X)
n, p = X.shape
print(f"Data shape: n={n}, p={p}")

ell_values = np.array([0.1,0.2,0.3,0.4,0.5,0.6, 0.7,0.8,0.9])
weights = pixels_with_wind_farm_counts['total_power'].values / pixels_with_wind_farm_counts['total_power'].sum()

print(pixels_with_wind_farm_counts['total_power'].sum())

import matplotlib.pyplot as plt

# Création de l'histogramme
plt.figure(figsize=(8, 6))
plt.hist(weights, bins=10, color='skyblue', edgecolor='black')

# Ajout des titres et labels
plt.title("Histogramme des poids")
plt.xlabel("Poids")
plt.ylabel("Fréquence")

# Affichage de l'histogramme
plt.grid(axis='y', alpha=0.75)
plt.show()

results_dir = "results"
os.makedirs(results_dir, exist_ok=True)

results_path = os.path.join(results_dir, "bootstrap_failure_weighted.npz")

# Check if results already exist
if os.path.exists(results_path):
    print(f"Loading precomputed results from {results_path}")
    data = np.load(results_path, allow_pickle=True)
    results = {key: data[key].item() if data[key].dtype == object else data[key] for key in data.files}
else:
    print("Results not found. Computing bootstrap failure sets...")
    results = compute_failure_sets_bootstrap_weighted(X, Y, weights, ell_values, fit_q = Q)
    np.savez(results_path, **results)
    print(f"Results saved to {results_path}")
results['u_values'] = 1/results['u_values']

# Now plot as usual

plot_bootstrap_failure(results, save=True)

plot_bootstrap_failure_alpha(results, index = 30, save =True)

plot_bootstrap_failure_alpha(results, index = 71, save =True, ylim=[0,0.006])