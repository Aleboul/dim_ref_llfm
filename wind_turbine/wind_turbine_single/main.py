import numpy as np
from config import FILE_PATH, Q, KAPPA, LAMBDA, USE_CACHE, GPD_FIT_Q
from data.loader import load_data, preprocess_data
from utils.transforms import pareto_rank_transform_matrix, compute_tpdm
from utils.plots import plot_matrix, plot_hist
from utils.helpers import complement_partitions
from utils.io import load_results, save_results, load_gpd_params

from estimation.purevar import PureVar, estimate_A_hat_I
from estimation.factors import compute_C_hat
from estimation.lasso import update_A_hat_with_refit
from evaluation.fit_quality import compute_fitted_tpdm, compute_bivariate_failure_sets, plot_comparison_hexbin
from evaluation.exceedances import fit_gpd_exceedances, compute_failure_sets_weighted

import matplotlib.pyplot as plt

import pandas as pd

pixels_with_wind_farm_counts = pd.read_csv('../data/prod_data/pixels_with_wind_farm_counts.csv')


# -----------------------------
# Load and preprocess data
# -----------------------------
X_raw = load_data(FILE_PATH)
X = preprocess_data(X_raw)  # original inverted data
X_unique = np.unique(X, axis=0)

Y = pareto_rank_transform_matrix(X)
# compute TPDM and get full radii r
TPDM, r0, mask, w, r_full = compute_tpdm(Y, q=Q)
plot_matrix(TPDM, "TPDM Matrix")
plot_hist(TPDM[np.triu_indices_from(TPDM, k=1)],
          "Histogram of Upper-Diagonal TPDM")

# -----------------------------
# Load or compute A_hat, C_hat, K_hat, I_hat
# -----------------------------
cached = load_results(Q, KAPPA, LAMBDA) if USE_CACHE else None

if cached is not None:
    A_hat = cached["A_hat"]
    C_hat = cached["C_hat"]
    K_hat = cached["K_hat"]
    I_hat = cached["I_hat"]
    pure_lists = [sorted(group) for group in I_hat]
    estimated_impure = complement_partitions(I_hat, X.shape[1])
else:
    # Pure variable estimation
    I_hat, K_hat = PureVar(np.abs(TPDM), KAPPA)
    A_hat_I = estimate_A_hat_I(I_hat, K_hat, TPDM)
    estimated_impure = complement_partitions(I_hat, X.shape[1])
    pure_lists = [sorted(group) for group in I_hat]

    # Factor covariance estimation
    C_hat = compute_C_hat(TPDM, A_hat_I, pure_lists)
    plot_matrix(C_hat, "C_hat")

    # Lasso refinement
    A_hat = update_A_hat_with_refit(
        Sigma_hat=TPDM,
        A_hat=A_hat_I.copy(),
        pure_lists=pure_lists,
        estimated_impure=estimated_impure,
        lambda_=LAMBDA
    )
    # Save results
    save_results(A_hat, C_hat, K_hat, I_hat, Q, KAPPA, LAMBDA)

# Ensure non-negativity if you want
A_hat = np.maximum(A_hat, 0)

def project_onto_simplex(v):
    """
    Project a vector v onto the probability simplex:
        S = {x | x >= 0, sum(x) = 1}
    """
    n = len(v)
    if np.all(v == 0):
        # all zeros → return zeros
        return v
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, n+1) > (cssv - 1))[0][-1]
    theta = (cssv[rho] - 1) / (rho + 1)
    w = np.maximum(v - theta, 0)
    return w

def row_simplex_projection(M):
    """
    Project each row of M onto the simplex using only the non-zero support.
    Zeros stay zeros.
    """
    M_proj = np.zeros_like(M, dtype=float)

    for i in range(M.shape[0]):
        row = M[i, :]
        support = row != 0
        # Only project the supported entries
        if np.any(support):
            proj = project_onto_simplex(row[support])
            M_proj[i, support] = proj

    return M_proj

A_hat = row_simplex_projection(A_hat)

plot_matrix(A_hat, "Refined A_hat")
total_entries = A_hat.size  # same as shape[0] * shape[1]
strictly_positive_count = np.sum(A_hat > 0)
print("Total entries:", total_entries)
print("Strictly positive entries:", strictly_positive_count)
print("Zero or negative entries:", total_entries - strictly_positive_count)
print(np.sum(A_hat > 0, axis = 1))

# -----------------------------
# Model evaluation
# -----------------------------
TPDM_fitted = compute_fitted_tpdm(A_hat, C_hat)
plot_comparison_hexbin(TPDM_fitted, TPDM)
for q in [0.05, 0.03]:
    Chi_fitted, Chi_hat = compute_bivariate_failure_sets(Y=Y,
                                                        A_hat=A_hat, I_hat=I_hat, q=q)
    plot_comparison_hexbin(Chi_fitted, Chi_hat)

# -----------------------------
# Fit (or load) GPD params and compute failure sets
# -----------------------------
# Try to load cached GPD params first (fit threshold GPD_FIT_Q)
params_cached = load_gpd_params(GPD_FIT_Q) if USE_CACHE else None
if params_cached is not None:
    params = params_cached
else:
    params, r2_qq_list = fit_gpd_exceedances(
        X, fit_q=Q)
    

# -------------------------
# Boxplot of all r2_qq
# -------------------------
xis = params[:, 0]

plt.figure(figsize=(8, 4))
plt.hist(xis, bins=30, color='skyblue', edgecolor='black')
plt.xlabel("Shape parameter ξ")
plt.ylabel("Frequency")
plt.title("Histogram of GPD Shape Parameters")
plt.grid(axis='y', alpha=0.75)
plt.show()

plt.figure(figsize=(6, 6))
plt.boxplot(
    r2_qq_list,
    patch_artist=True,
    showfliers=False,       # hide outliers
    whis=95,                # whiskers extend to 95% confidence interval
    boxprops=dict(facecolor='lightblue', color='blue'),
    medianprops=dict(color='red'),
    whiskerprops=dict(color='blue'),
    capprops=dict(color='blue')
)
plt.ylabel("R² (Q-Q Plot)")
plt.title(
    r"Q–Q $R^2$ for GPD fit to conditional excesses "
    r"$X_j - u \mid X_j > u$ (95% CI)"
)
plt.grid(axis='y', alpha=0.75)
plt.show()

weigths = np.ones(Y.shape[1])
ell_values = np.array([0.5,0.7,0.9])
print(ell_values)
print(pixels_with_wind_farm_counts['total_power'])
weigths = pixels_with_wind_farm_counts['total_power'].values / pixels_with_wind_farm_counts['total_power'].sum()

# compute failure sets and compare fitted vs empirical
results = compute_failure_sets_weighted(X=X, Y=Y, q=Q, weights = weigths, ell_values = ell_values,
                                     A_hat=A_hat, I_hat=I_hat, params=params,
                                     u_values=None)
