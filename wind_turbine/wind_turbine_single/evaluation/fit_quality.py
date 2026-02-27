import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from matplotlib.colors import ListedColormap
import seaborn as sns
import swiftascmaps

arcane_colors = ["#C71585", "#9D7FBA", "#324AB2"]
arcane_cmap = ListedColormap(arcane_colors, name='arcane')


# ----------------------------------------------------------
# 1. Compute fitted TPDM
# ----------------------------------------------------------
def compute_fitted_tpdm(A_hat, C_hat):
    """
    Computes the fitted extremal covariance matrix using the model: A_hat @ C_hat @ A_hat.T
    """
    # Compute the fitted TPDM using the model
    return A_hat @ C_hat @ A_hat.T


# ----------------------------------------------------------
# 2. Compute fitted and empirical extremal correlation
# ----------------------------------------------------------
def compute_bivariate_failure_sets(Y, A_hat, I_hat, q=0.1):
    """
    Estimates and compares fitted vs. empirical pairwise extremal correlations \chi(j,\ell) for j,ℓ∈[d] and j\ne \ell.

    \[
    \chi(j, \ell) = K \int_{\mathbb{S}^{K-1}_+} \left( \sum_{a \in [K]} A_{ja} z_a \right) \wedge \left( \sum_{a \in [K]} A_{\ell a} z_a \right) \psi(d\bm{z})
    \]

    By regular variation, the above integral is also equal to:

    \[
    \chi(j, \ell) = K \lim_{x \to \infty} \mathbb{E}\left[\left(\sum_{a=1}^K A_{ja} \frac{Z_a}{\|Z\|_1}\right) \wedge \left(\sum_{a=1}^K A_{\ell a} \frac{Z_a}{\|Z\|_1}\right) \mid \|Z\|_1 > x\right].
    \]

    Given the estimated pure variable groups \(\hat{\mathcal{I}}\), we define the proxy vector \(Z'\) for each observation i as:

    \[
    Z'_i = \left( \frac{1}{|\hat{I}_1|} \sum_{j \in \hat{I}_1} X_{i,j}, \dots, \frac{1}{|\hat{I}_K|} \sum_{j \in \hat{I}_K} X_{i,j} \right),
    \]

    where \(\hat{I}_a\) is the estimated set of pure variables for factor a. Assuming \(\hat{I}_a = I_{\pi(a)}\) for some permutation π on [K], \(Z'\) serves as a good proxy for the extremal dependence structure \(\Phi_Z\).

    Given estimates of the loading matrix \(\hat{A}\), we obtain a natural estimator of \chi(j,\ell) as:

    \[
    \hat{\chi}_n(j, \ell) = \frac{\hat{K}}{k} \sum_{i=1}^n \left( \frac{\sum_{a=1}^{\hat{K}} \hat{A}_{ja} Z'_{i,a}}{\|\bm{Z}'_i\|_1} \wedge \frac{\sum_{a=1}^{\hat{K}} \hat{A}_{\ell a} Z'_{i,a}}{\|\bm{Z}'_i\|_1} \right) \mathds{1}_{\{ \|\bm{Z}'_i\|_1 > \hat{z}_{n,k} \}},
    \]

    where \(\hat{z}_{n,k}\) is the \((1 - k/n)\)-quantile of the norms \(\|\bm{Z}'_1\|_1, \dots, \|\bm{Z}'_n\|_1\). This approach focuses on pairwise extremal correlations, providing a computationally efficient and interpretable estimator.

    Steps:
    1. Construct \(Z'\) by averaging over pure-variable groups.
    2. Compute the L1 norms of \(Z'\) and determine the threshold for extreme observations.
    3. Normalize the contributions of each variable to the factors for observations exceeding the threshold.
    4. Compute fitted extremal correlations as the pairwise minima of normalized contributions.
    5. Compute empirical extremal correlations based on exceedances in the data.
    6. Compare fitted and empirical correlations using R² score and a hexbin plot.

    Args:
        Y (np.ndarray): Pareto-margins Data matrix of shape (n, d), where n is the number of observations and d is the number of variables.
        A_hat (np.ndarray): Estimated loading matrix of shape (d, K), where K is the number of factors.
        I_hat (list of sets): List of pure variable groups, where each set contains indices of pure variables.
        q (float): Quantile threshold for defining extreme observations (default: 0.1).

    Returns:
        fitted_chi (np.ndarray): Fitted extremal correlations (d x d matrix)
        empirical_chi (np.ndarray): Empirical extremal correlations (d x d matrix)
    """
    n, d = Y.shape
    u = (1 - q)  # Threshold for extreme observations

    # Step 1: Construct Z' by averaging over pure-variable groups
    pure_lists = [sorted(group) for group in I_hat]
    Z_prime = np.vstack([np.mean(Y[:, group], axis=1) for group in pure_lists])

    # Step 2: Compute the L1 norms of Z' (sum of absolute values across factors for each observation)
    Z_norms = np.linalg.norm(Z_prime, axis=0, ord=1)

    # Step 3: Determine the threshold for extreme observations
    t_hat = np.quantile(Z_norms, u)
    K_hat = len(I_hat)

    # Step 4: Identify observations exceeding the threshold
    mask_valid = Z_norms > t_hat
    if not np.any(mask_valid):
        raise ValueError("No observations exceed the threshold.")

    # Step 5: Normalize the inner terms (d × n_valid)
    normalized_inner = (A_hat @ Z_prime[:, mask_valid]) / Z_norms[mask_valid]

    # --- Fitted probabilities ---
    mins = np.minimum(
        normalized_inner[:, None, :], normalized_inner[None, :, :])
    fitted_chi = np.minimum(K_hat * mins.mean(axis=2), 1)

    # --- Empirical probabilities ---
    exceed = (1 - 1 / Y > u).astype(float)
    both = exceed.T @ exceed
    empirical_chi = both / int(n*q)

    return fitted_chi, empirical_chi


# ----------------------------------------------------------
# 3. Generic hexbin plot function
# ----------------------------------------------------------
def plot_comparison_hexbin(fitted_matrix, empirical_matrix, xlabel='Fitted', ylabel='Empirical', title=None, cmap=None):
    """
    Plots a hexbin comparison between two matrices by extracting the upper-triangle elements.

    Args:
        fitted_matrix (np.ndarray): Fitted values (d × d matrix)
        empirical_matrix (np.ndarray): Empirical values (d × d matrix)
        xlabel: x-axis label
        ylabel: y-axis label
        title: plot title
        cmap: colormap (can be string or matplotlib colormap)
    """
    # Use sns.crest if no cmap provided
    if cmap is None:
        cmap = 'swift.lover_r'

    # Extract upper-triangle elements (excluding diagonal)
    rows, cols = np.triu_indices(fitted_matrix.shape[0], k=1)
    x = fitted_matrix[rows, cols]
    y = empirical_matrix[rows, cols]

    # Compute R²
    r2 = r2_score(y, x)
    print(f"R² score: {r2:.4f}")

    # Create hexbin plot
    fig, ax = plt.subplots(figsize=(7, 6))
    hb = ax.hexbin(x, y, gridsize=50, cmap=cmap, mincnt=1)

    lims = [min(x.min(), y.min()), max(x.max(), y.max())]
    ax.plot(lims, lims, '--', linewidth=2, color='#C71585')

    ax.set_xlabel(xlabel, fontsize=18)
    ax.set_ylabel(ylabel, fontsize=18)
    ax.tick_params(labelsize=16)             # Change both axes at once
    ax.set_title(title or f'R² = {r2:.4f}', fontsize=20)
    ax.grid(True, alpha=0.3)

    cb = fig.colorbar(hb, ax=ax)
    cb.set_label('Counts', fontsize=18)
    cb.ax.tick_params(labelsize=16)
    plt.show()

    

