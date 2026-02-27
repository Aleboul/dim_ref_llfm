import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import genpareto
from sklearn.metrics import r2_score

from utils.io import save_gpd_params, load_gpd_params


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import genpareto
from sklearn.metrics import r2_score

def fit_gpd_exceedances(X, fit_q=0.1):
    """
    Fit GPD to exceedances per column of X and compute r2_qq for each.
    Returns:
        - params: array of shape (d,4) [xi, loc, sigma, threshold_u]
        - r2_qq_list: list of R^2 for Q-Q plots per column
    """
    d = X.shape[1]
    params = []
    r2_qq_list = []

    for j in range(d):
        sample = X[:, j]
        u = np.quantile(sample, 1-fit_q)
        exceedances = sample[sample > u] - u

        if exceedances.size == 0:
            xi, loc, sigma = np.nan, 0.0, np.nan
            r2_qq = np.nan
        else:
            try:
                xi, loc, sigma = genpareto.fit(exceedances, floc=0)
            except Exception:
                xi, loc, sigma = np.nan, 0.0, np.nan

            # Q-Q R^2
            try:
                exceedances_sorted = np.sort(exceedances)
                n_exc = exceedances.size
                empirical_cdf = np.arange(1, n_exc + 1) / (n_exc + 1)
                theoretical_quantiles = genpareto.ppf(empirical_cdf, c=xi, loc=0, scale=sigma)
                r2_qq = r2_score(exceedances_sorted, theoretical_quantiles)
            except Exception:
                r2_qq = np.nan

        params.append([xi, loc, sigma, u])
        r2_qq_list.append(r2_qq)

    params = np.array(params).reshape(-1, 4)
    return params, r2_qq_list


def empirical_prob_with_bootstrap_basic(X, u, ell, B=1000):
    """
    Compute empirical probability P(at least ell exceedances of u)
    with basic bootstrap confidence interval.
    """
    np.random.seed(29041996)
    n = X.shape[0]
    counts = np.sum(X > u, axis=1)
    est = np.mean(counts >= ell)

    boots = []
    for _ in range(B):
        idx = np.random.randint(0, n, size=n)  # resample rows
        boots.append(np.mean(counts[idx] >= ell))
    boots = np.array(boots)

    # basic bootstrap interval
    lower = np.percentile(boots, 2.5)
    upper = np.percentile(boots, 97.5)

    return est, (lower, upper)


def empirical_prob_with_bootstrap_weighted(X, u, ell, weights, B=1000):
    """
    Compute empirical probability P(weighted sum of exceedances >= ell)
    with basic bootstrap confidence interval.

    Args:
        X (np.ndarray): Data matrix of shape (n, d)
        u (float): Threshold
        ell (float): Threshold for weighted exceedances
        weights (np.ndarray): 1D array of length d with weights for each component
        B (int): Number of bootstrap samples
    Returns:
        est (float): Empirical estimate
        (lower, upper) (tuple): 95% basic bootstrap confidence interval
    """
    np.random.seed(29041996)
    n, d = X.shape
    weights = np.asarray(weights)
    if weights.shape[0] != d:
        raise ValueError("weights must have length d")

    # Weighted exceedances
    weighted_counts = np.sum((X > u) * weights, axis=1)
    est = np.mean(weighted_counts >= ell)

    # Bootstrap
    boots = []
    for _ in range(B):
        idx = np.random.randint(0, n, size=n)  # resample rows
        boots.append(np.mean(weighted_counts[idx] >= ell))
    boots = np.array(boots)

    # Basic bootstrap interval
    lower = np.percentile(boots, 2.5)
    upper = np.percentile(boots, 97.5)

    return est, (lower, upper)

def compute_pareto_failure_sets(Y, A_hat, I_hat, q=0.1,
                                u_values=None, percentages=None,
                                save_results=False, B=1000):
    """
    Compute fitted vs empirical failure probabilities using Z' (averaged pure-variable vectors)
    and their norms instead of Y and its norms.

    Parameters
    ----------
    Y : ndarray (n, d)
        Sample in Pareto space.
    A_hat : ndarray (d, K)
        Factor loading estimates.
    I_hat : list of sets
        Pure groups.
    u_values : ndarray or None
        Quantile thresholds for exceedances (default: np.linspace(0.95, 0.9999, 200)).
    percentages : list of float
        Fractions of d for ell-values. Default: [0.1, ..., 1.0].
    save_results : bool
        If True, saves results into results/failure_set_ell*.npz
    B : int
        Bootstrap replicates for confidence intervals.

    Returns
    -------
    results : dict
        Dictionary with results per ell value.
    """

    n, d = Y.shape

    # Step 1: Construct Z' by averaging over pure-variable groups
    pure_lists = [sorted(group) for group in I_hat]
    group_averages = []
    for group in pure_lists:
        # average over group variables
        group_avg = np.mean(Y[:, group], axis=1)
        group_averages.append(group_avg)
    Z_prime = np.vstack(group_averages)  # shape (K × n)

    # Step 2: Norms of Z'
    Z_norms = np.linalg.norm(Z_prime, axis=0, ord=1)
    zeta_hat = len(I_hat)

    if u_values is None:
        u_values = np.linspace(0.95, 0.9999, 200)

    if percentages is None:
        percentages = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    ell_values = [max(1, int(p * d)) for p in percentages]

    results = {}
    # Step 3: threshold on norms

    t_hat = np.quantile(Z_norms, 1-q)
    mask_valid = Z_norms > t_hat

    normalized_inner = (A_hat @ Z_prime[:, mask_valid]) / Z_norms[mask_valid]

    for ell in ell_values:
        empirical_results = np.zeros_like(u_values, dtype=float)
        fitted_results = np.zeros_like(u_values, dtype=float)
        lower_bounds = np.zeros_like(u_values, dtype=float)
        upper_bounds = np.zeros_like(u_values, dtype=float)

        for idx, u in enumerate(u_values):
            if not np.any(mask_valid):
                continue

            # Step 4: normalize inner terms

            # Fitted probability: ell-th largest component exceedance
            target_index = normalized_inner.shape[0] - ell
            kth_largest = np.partition(
                normalized_inner * (1 - u), target_index, axis=0
            )[target_index, :]
            fitted_val = np.mean(kth_largest)
            fitted_results[idx] = zeta_hat * fitted_val

            # empirical probability with basic bootstrap CI
            est, (lower, upper) = empirical_prob_with_bootstrap_basic(
                Y, 1/(1-u), ell, B=B)
            empirical_results[idx] = est
            lower_bounds[idx] = lower
            upper_bounds[idx] = upper

        # Plotting
        plt.figure(figsize=(10, 6))
        plt.plot(u_values, fitted_results, linewidth=2,
                 label='Fitted', color='#324AB2')
        plt.plot(u_values, empirical_results, linewidth=2, linestyle='--',
                 label='Empirical', color='#C71585')
        plt.fill_between(u_values, lower_bounds, upper_bounds, color='#C71585', alpha=0.2,
                         label='95% Basic-Bootstrap CI')
        plt.xlabel('Threshold (u)')
        plt.ylabel('Probability / fitted quantity')
        plt.title(
            f'Empirical vs Fitted (At least {ell} components > u, {ell/d*100:.1f}% of d)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim([0, 0.05])
        plt.tight_layout()
        plt.show()

        avg_error = np.mean(np.abs(empirical_results - fitted_results))
        max_error = np.max(np.abs(empirical_results - fitted_results))

        results[ell] = {
            "u_values": u_values,
            "fitted": fitted_results,
            "empirical": empirical_results,
            "lower_CI": lower_bounds,
            "upper_CI": upper_bounds,
            "avg_error": avg_error,
            "max_error": max_error
        }

        if save_results:
            fname = f"results/failure_set_ell{ell}.npz"
            np.savez(fname, u_values=u_values, fitted=fitted_results, empirical=empirical_results,
                     lower=lower_bounds, upper=upper_bounds)
            print(f"Saved failure-set results: {fname}")

    return results
def compute_failure_sets(
    X, Y, A_hat, I_hat, params,
    q=0.05,
    percentages=None
):
    import numpy as np

    n, d = X.shape
    K = len(I_hat)

    # -------------------------
    # Step 1: Construct Z'
    # -------------------------
    pure_lists = [sorted(group) for group in I_hat]

    Z_prime = np.zeros((K, n))
    for a, group in enumerate(pure_lists):
        Z_prime[a, :] = np.mean(Y[:, group], axis=1)

    # -------------------------
    # Step 2: Extreme norms
    # -------------------------
    Z_norms = np.linalg.norm(Z_prime, ord=1, axis=0)
    t_hat = np.quantile(Z_norms, 1 - q)

    mask_extreme = Z_norms > t_hat
    k_exceed = np.sum(mask_extreme)

    if k_exceed == 0:
        raise ValueError("No extreme observations above threshold")

    Z_prime_ext = Z_prime[:, mask_extreme]
    Z_norms_ext = Z_norms[mask_extreme]

    # -------------------------
    # Step 3: Normalize Z'
    # -------------------------
    Z_bar = Z_prime_ext / Z_norms_ext

    # -------------------------
    # Step 4: Linear form
    # -------------------------
    S = A_hat @ Z_bar   # shape (d, k_exceed)

    # -------------------------
    # Thresholds and ell values
    # -------------------------
    u_values = np.linspace(0.1, 0.75, 50)

    if percentages is None:
        percentages = [0.10, 0.2, 0.3, 0.4, 0.5,
                        0.6, 0.7, 0.8, 0.9, 1.0]

    ell_values = [max(1, int(p * d)) for p in percentages]

    # -------------------------
    # Tail parameters
    # -------------------------
    xis = params[:, 0]
    sigmas = params[:, 2]
    u_refs = params[:, 3]

    results = {}

    # -------------------------
    # Loop over ell
    # -------------------------
    for ell in ell_values:

        fitted = np.zeros(len(u_values))

        order_index = d - ell
        order_index = max(0, min(order_index, d - 1))

        # -------------------------
        # Loop over thresholds
        # -------------------------
        for iu, u in enumerate(u_values):

            # ---- Step 5: q_hat ----
            q_hat = np.zeros(d)

            for j in range(d):

                if u <= u_refs[j]:
                    q_hat[j] = np.mean(X[:, j] > u)

                else:
                    y = u - u_refs[j]
                    xi = xis[j]
                    sigma = sigmas[j]
                    q_ref = np.mean(X[:, j] > u_refs[j])

                    if abs(xi) < 1e-8:
                        q_hat[j] = q_ref * np.exp(-y / sigma)
                    else:
                        inner = 1 + xi * y / sigma
                        q_hat[j] = q_ref * inner ** (-1 / xi)

            # ---- Step 6: c_{i,j} ----
            C = S * q_hat[:, None]

            # ---- Step 7: order statistic ----
            c_order = np.partition(C, order_index, axis=0)[order_index, :]

            # ---- Step 8: fitted probability ----
            fitted[iu] = K * np.mean(c_order)

        results[ell] = {
            "u_values": u_values,
            "p_fitted": fitted
        }
        plt.figure(figsize=(10, 6))
        plt.plot(u_values, fitted, linewidth=2,
                 label='Fitted', color='#324AB2')
        plt.show()

    return results

def compute_failure_sets(X, Y, A_hat, I_hat, params, q=0.05,
                         u_values=None, percentages=None, save_results=False, B=1000):
    r"""
    Computes and compares fitted vs. empirical failure probabilities with bootstrap confidence intervals,
    using extremes of \(Z'\) and its norms. The threshold is chosen by quantile \(Q\).

    The target parameter is the probability of joint failure for a threshold vector \(\bm{x} = (x_j)_{j \in [d]}\):

    \[
    p := p(\bm{x}, \mathcal{J}) := \mathbb{P}\left\{ \exists J \in \bm{\mathcal{J}}, \forall j \in J : X_j > x_j \right\}.
    \]

    Let \(q_j = 1 - F_j(x_j)\) and \(\bm{q} = (q_1, \dots, q_d)\). Using Lemma \ref{lemma:survival_stdf_lfm_tail_dep}, we can write:

    \[
    p = \mathbb{P}\left\{ \exists J \in \bm{\mathcal{J}}, \forall j \in J : X_j > x_j \right\} \approx t^{-1} R^\cup_{\mathcal{\bm{J}}} (t \bm{q})
    = \zeta_{\bm{Z}} \int_{\mathbb{S}^{K-1}_+} \bigvee_{J \in \mathcal{\bm{J}}} \bigwedge_{j \in J} \left( \sum_{a=1}^K A_{ja} z_a q_j \right) \Phi_{\bm{Z}}(d\bm{z})
    \]

    The estimation of \(\hat{q}_j\) is done in two steps: \(u_{ref,j}\) is the 1-q quantile of the jth margin.
    1. For thresholds \(u\) below the reference threshold \(u_{ref,j}\) (stored in `params[:, 3]`),
       \(\hat{q}_j(u)\) is computed empirically as \(\hat{q}_j(u) = \frac{1}{n} \sum_{i=1}^n \mathds{1}_{\{X_{i,j} > u\}}\).

    2. For thresholds \(u\) above the reference threshold \(u_{ref,j}\), \(\hat{q}_j(u)\) is extrapolated using the Generalized Pareto Distribution (GPD) tail model:
       \[
       \hat{q}_j(u) = \hat{q}_j(u_{ref,j}) \cdot \left(1 + \xi_j \cdot \frac{u - u_{ref,j}}{\sigma_j}\right)^{-1/\xi_j}
       \]
       where \(\xi_j\) is the shape parameter, \(\sigma_j\) is the scale parameter, and \(\hat{q}_j(u_{ref,j})\) is the empirical tail probability at the reference threshold.
       For the special case where \(\xi_j \approx 0\), the extrapolation uses the exponential distribution:
       \[
       \hat{q}_j(u) = \hat{q}_j(u_{ref,j}) \cdot \exp\left(-\frac{u - u_{ref,j}}{\sigma_j}\right).
       \]

    Given estimates for \(\hat{\mathcal{I}}\), we define:

    \[
    \bm{Z}'_i = \left( \frac{1}{|\hat{I}_1|} \sum_{j \in \hat{I}_1} X_{i,j}, \dots, \frac{1}{|\hat{I}_K|} \sum_{j \in \hat{I}_K} X_{i,j} \right),
    \]

    which is a good proxy for \(\psi_{\bm{Z}}\) up to a permutation of labels. Given estimates of \(A\) and \(q_j\), we obtain the natural estimator of \(p\):

    \[
    \hat{p}_n = \frac{\hat{K}}{k} \sum_{i=1}^n \bigvee_{J \in \mathcal{\bm{J}}} \bigwedge_{j \in J} \left( \frac{\sum_{a=1}^{\hat{K}} \hat{A}_{ja} Z'_{i,a} \hat{q}_j}{\|\bm{Z}'_i\|_1} \right) \mathds{1}_{\{ \|\bm{Z}'_i\|_1 > \hat{z}_{n,k} \}},
    \]

    where \(\hat{z}_{n,k}\) is the \((1-k/n)\)-quantile of the norms \(\|\bm{Z}'_1\|_1, \dots, \|\bm{Z}'_n\|_1\). The formula for \(\hat{p}_n\) requires evaluating expressions of the form \(\bigvee_{J \in \mathcal{\bm{J}}} \bigwedge_{j \in J} c_j\), which simplifies to \(c_{(d-\ell+1)}\), the \((d-\ell+1)\)-th order statistic of \(\bm{c}\).

    The final estimator of \(p\) is given by:

    \[
    \hat{p}_n = \frac{\hat{K}}{k} \sum_{i=1}^n \hat{c}_{i,(d-\ell+1)} \mathds{1}_{\{ \|\bm{Z}'_i\| > \hat{z}_{n,k} \}}, \quad \text{where} \quad \hat{c}_{i,j} = \frac{\sum_{a=1}^K \hat{A}_{ja} Z'_{i,a} \hat{q}_j}{\|\bm{Z}'_i\|_1}.
    \]

    Steps:
    1. Construct \(Z'\) by averaging over pure-variable groups.
    2. Compute the L1 norms of \(Z'\) and determine the threshold for extreme observations.
    3. Estimate \(\hat{q}_j\) using empirical and extrapolated tail probabilities.
    4. Compute the normalized contributions of each variable to the factors for extreme observations.
    5. Estimate the fitted failure probabilities using the \((d-\ell+1)\)-th order statistic.
    6. Estimate the empirical failure probabilities with bootstrap confidence intervals.
    7. Compare fitted and empirical probabilities using plots and error metrics.

    Args:
        X (np.ndarray): Data matrix of shape (n, d) for tail estimation.
        Y (np.ndarray): Pareto margins Data matrix of shape (n, d) derived from X.
        A_hat (np.ndarray): Estimated loading matrix of shape (d, K), where K is the number of factors.
        I_hat (list of sets): List of pure variable groups, where each set contains indices of pure variables.
        params (np.ndarray): Parameters for tail estimation, shape (d, 4), where each row contains (xi, mu, sigma, u_ref).
        q (float): Quantile threshold for defining extreme observations (default: 0.05).
        u_values (np.ndarray): Threshold values for tail estimation (default: None).
        percentages (list): Percentages of d for determining \(\ell\) values (default: None).
        save_results (bool): Whether to save results to file (default: False).
        B (int): Number of bootstrap samples (default: 1000).

    Returns:
        dict: Dictionary containing results for each \(\ell\) value, including fitted and empirical probabilities,
              confidence intervals, and error metrics.
    """
    n, d = X.shape

    # Step 1: Build Z' (K × n)
    pure_lists = [sorted(group) for group in I_hat]
    group_averages = [np.mean(Y[:, group], axis=1) for group in pure_lists]
    Z_prime = np.vstack(group_averages)

    # Step 2: Compute norms of Z'
    Z_norms = np.linalg.norm(Z_prime, axis=0, ord=1)

    # Step 3: Thresholding
    t_hat = np.quantile(Z_norms, 1-q)
    mask_valid = Z_norms > t_hat
    if not np.any(mask_valid):
        raise ValueError("No samples exceed the chosen threshold; adjust Q.")

    Z_prime_masked = Z_prime[:, mask_valid]
    Z_norms_masked = Z_norms[mask_valid]
    k_exceed = Z_norms_masked.shape[0]
    zeta_hat = len(I_hat)

    # Step 4: Compute normalized inner term
    group_avg_matrix = Z_prime_masked / Z_norms_masked
    S = A_hat @ group_avg_matrix  # shape (d, n_masked)

    if u_values is None:
        u_values = np.linspace(0.1, 0.75, 200)
    if percentages is None:
        percentages = [0.10, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    ell_values = [max(1, int(p * d)) for p in percentages]

    results = {}
    for ell in ell_values:
        empirical_results = np.zeros_like(u_values, dtype=float)
        fitted_results = np.zeros_like(u_values, dtype=float)
        lower_bounds = np.zeros_like(u_values, dtype=float)
        upper_bounds = np.zeros_like(u_values, dtype=float)

        # Vectorized loop over thresholds u
        for idx, u in enumerate(u_values):
            # Compute q_hat (vectorized over d)
            q_hat = np.zeros(d)
            u_refs = params[:, 3]
            xis = params[:, 0]
            sigmas = params[:, 2]

            tails = np.array([np.mean(X[:, j] > u_refs[j]) if np.isfinite(
                u_refs[j]) else 0.0 for j in range(d)])

            below_mask = u <= u_refs
            above_mask = ~below_mask

            # Direct empirical exceedances
            q_hat[below_mask] = np.array(
                [np.mean(X[:, j] > u) for j in np.where(below_mask)[0]])

            # Extrapolated tail
            for j in range(d):

                if u <= u_refs[j]:
                    # Empirical regime
                    q_hat[j] = np.mean(X[:, j] > u)

                else:
                    y = u - u_refs[j]
                    xi = xis[j]
                    sigma = sigmas[j]
                    #q_ref = np.mean(X[:, j] > u_refs[j])
                    #print(q_ref)
                    q_ref = q
                    # GPD extrapolation
                    if abs(xi) < 1e-8:
                        # Exponential tail (xi = 0)
                        q_hat[j] = q_ref * np.exp(-y / sigma)

                    elif xi > 0:
                        # Heavy tail, unbounded support
                        inner = 1 + xi * y / sigma
                        q_hat[j] = q_ref * inner ** (-1 / xi)

                    else:
                        # xi < 0 : bounded support
                        y_max = -sigma / xi

                        if y >= y_max:
                            # Beyond the upper endpoint of the distribution
                            q_hat[j] = 0.0
                        else:
                            inner = 1 + xi * y / sigma
                            q_hat[j] = q_ref * inner ** (-1 / xi)

            # Combine with S
            combined = S * q_hat[:, None]
            target_index = max(
                0, min(combined.shape[0] - ell, combined.shape[0]-1))
            kth_largest = np.partition(combined, target_index, axis=0)[
                target_index, :]
            if k_exceed > 0:
                fitted_results[idx] = zeta_hat * \
                    (1.0 / k_exceed) * np.sum(kth_largest)
            else:
                fitted_results[idx] = 0.0

            # Empirical probability with bootstrap
            est, (lower, upper) = empirical_prob_with_bootstrap_basic(
                X, u, ell, B=B)
            empirical_results[idx] = est
            lower_bounds[idx] = lower
            upper_bounds[idx] = upper

        # Plotting
        plt.figure(figsize=(10, 6))
        plt.plot(1/u_values, fitted_results, linewidth=2,
                 label='Fitted', color='#324AB2')
        plt.plot(1/u_values, empirical_results, '--', linewidth=2,
                 label='Empirical', color='#C71585')
        plt.fill_between(1/u_values, lower_bounds, upper_bounds, color='#C71585', alpha=0.2,
                         label='95% Basic-Bootstrap CI')
        plt.xlabel('Threshold (x)')
        plt.ylabel('Probability / fitted quantity')
        plt.title(
            f'Empirical vs Fitted (At least {ell} components > x, {ell/d*100:.1f}% of d)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim([0, 0.05])
        plt.tight_layout()
        plt.show()

        results[ell] = {
            "u_values": u_values,
            "fitted": fitted_results,
            "empirical": empirical_results,
            "lower_CI": lower_bounds,
            "upper_CI": upper_bounds
        }

        if save_results:
            fname = f"results/failure_set_ell{ell}.npz"
            np.savez(fname, u_values=u_values, fitted=fitted_results, empirical=empirical_results,
                     lower=lower_bounds, upper=upper_bounds)
            print(f"Saved failure-set results: {fname}")

    return results
# ============================================================
# Weighted Semi-Parametric Estimator of Failure Probabilities
#
# Implements the semi-parametric estimator:
#   \tilde{p}_{n,k'}(\bm{x}) = (K_hat / k') * sum_i \hat{c}_{i,\pi_i(k_i)} 1_{||Z_i||_1 > z_{n,k'}}
# where
#   - Z_i are latent factor contributions (normalized),
#   - \hat{c}_{i,j} = sum_a (A_hat[j,a] * Z_i[a] * q_hat[j]) / ||Z_i||_1,
#   - \pi_i(k_i) is the weighted order statistic selection,
#   - k_i is the smallest index such that cumulative weight >= alpha (ell),
# and also computes the empirical weighted probability estimator:
#   \hat{p}_n(\bm{x}) = (1/n) sum_i 1_{sum_j omega_j 1_{X_{i,j} > x_j} >= alpha}
# ============================================================

import numpy as np
import matplotlib.pyplot as plt

def compute_failure_sets_weighted(
    X, Y, A_hat, I_hat, params, weights, ell_values,
    q=0.05, u_values=None, save_results=False, B=1000
):
    """
    Weighted semi-parametric and empirical failure probability estimator.

    Parameters:
    - X : data matrix (n x d) for empirical exceedances
    - Y : Pareto data matrix used to compute latent factors Z_i
    - A_hat : estimated factor matrix (d x K_hat)
    - I_hat : list of index groups corresponding to factors
    - params : marginal semi-parametric parameters (xi, etc.)
    - weights : weight vector (length d) for weighted subsets
    - ell_values : list of threshold weights (alpha) for top subsets
    - q : quantile for thresholding norms of Z_i (1-q corresponds to k')
    - u_values : optional thresholds for marginal exceedances
    - save_results : whether to save plots/data
    - B : number of bootstrap replicates for empirical estimator
    """

    n, d = X.shape
    weights = np.asarray(weights)
    if weights.shape[0] != d:
        raise ValueError("weights must have length d")

    # ------------------------------
    # Step 1: Construct latent factors Z_i
    #   Z_prime[:, i] corresponds to Z_i in the math formula
    #   Each row is one factor across all samples
    # ------------------------------
    pure_lists = [sorted(group) for group in I_hat]
    group_averages = [np.mean(Y[:, group], axis=1) for group in pure_lists]
    Z_prime = np.vstack(group_averages)  # shape (K_hat, n)

    # ------------------------------
    # Step 2: Compute L1 norms ||Z_i||_1
    # ------------------------------
    Z_norms = np.linalg.norm(Z_prime, axis=0, ord=1)

    # ------------------------------
    # Step 3: Threshold to select extreme samples
    #   Corresponds to 1_{||Z_i||_1 > z_{n,k'}}
    # ------------------------------
    t_hat = np.quantile(Z_norms, 1-q)
    mask_valid = Z_norms > t_hat
    if not np.any(mask_valid):
        raise ValueError("No samples exceed the chosen threshold; adjust Q.")

    Z_prime_masked = Z_prime[:, mask_valid]
    Z_norms_masked = Z_norms[mask_valid]
    k_exceed = Z_prime_masked.shape[1]  # number of samples exceeding threshold
    zeta_hat = len(I_hat)  # estimated number of factors K_hat

    # ------------------------------
    # Step 4: Compute normalized inner term
    #   Corresponds to \hat{c}_{i,j} = sum_a A_hat[j,a] * Z_i[a] / ||Z_i||_1
    # ------------------------------
    group_avg_matrix = Z_prime_masked / Z_norms_masked
    S = A_hat @ group_avg_matrix  # shape (d, k_exceed)

    if u_values is None:
        u_values = np.linspace(0.1, 0.75, 200)

    results = {}
    for ell in ell_values:
        # containers for semi-parametric fitted probabilities and empirical estimates
        empirical_results = np.zeros_like(u_values, dtype=float)
        fitted_results = np.zeros_like(u_values, dtype=float)
        lower_bounds = np.zeros_like(u_values, dtype=float)
        upper_bounds = np.zeros_like(u_values, dtype=float)

        # Loop over thresholds u (marginal exceedance levels)
        for idx, u in enumerate(u_values):

            # ------------------------------
            # Step 4a: Compute marginal semi-parametric estimates q_hat[j] = \hat{q}_j(x_j)
            # ------------------------------
            q_hat = np.zeros(d)
            u_refs = params[:, 3]
            xis = params[:, 0]
            sigmas = params[:, 2]

            for j in range(d):
                if u <= u_refs[j]:
                    q_hat[j] = np.mean(X[:, j] > u)
                else:
                    y = u - u_refs[j]
                    xi = xis[j]
                    sigma = sigmas[j]
                    q_ref = np.mean(X[:, j] > u_refs[j]) if u_refs[j] > 0 else q
                    if abs(xi) < 1e-8:
                        q_hat[j] = q_ref * np.exp(-y / sigma)
                    elif xi > 0:
                        q_hat[j] = q_ref * (1 + xi * y / sigma) ** (-1 / xi)
                    else:
                        y_max = -sigma / xi
                        q_hat[j] = 0.0 if y >= y_max else q_ref * (1 + xi * y / sigma) ** (-1 / xi)

            # ------------------------------
            # Step 4b: Combine latent factors with marginal estimates
            #   combined[j,i] corresponds to \hat{c}_{i,j} in formula
            # ------------------------------
            combined = S * q_hat[:, None]  # shape (d, k_exceed)

            # ------------------------------
            # Step 4c: Weighted order statistic to select top subset
            #   - Sort each column descending -> pi_i permutation
            #   - Cumulative weights S_{i,k} to find minimal k_i >= ell
            #   - kth_largest corresponds to \hat{c}_{i, pi_i(k_i)}
            # ------------------------------
            sorted_idx = np.argsort(-combined, axis=0)
            sorted_combined = np.take_along_axis(combined, sorted_idx, axis=0)
            sorted_weights = np.take_along_axis(weights[:, None], sorted_idx, axis=0)
            cum_weights = np.cumsum(sorted_weights, axis=0)
            cum_weights[-1,:] = np.ones(cum_weights.shape[1])
            mask = cum_weights >= ell
            kth_indices = np.argmax(mask, axis=0)  # index of first True
            kth_largest = sorted_combined[kth_indices, np.arange(k_exceed)]

            # ------------------------------
            # Step 4d: Compute fitted semi-parametric estimator \tilde{p}_{n,k'}(\bm{x})
            # ------------------------------
            fitted_results[idx] = zeta_hat * (1.0 / k_exceed) * np.sum(kth_largest)

            # ------------------------------
            # Step 4e: Compute empirical weighted estimator \hat{p}_n(\bm{x}) via bootstrap
            # ------------------------------
            est, (lower, upper) = empirical_prob_with_bootstrap_weighted(X, u, ell, weights, B=B)
            empirical_results[idx] = est
            lower_bounds[idx] = lower
            upper_bounds[idx] = upper

        # ------------------------------
        # Step 5: Plot results
        # ------------------------------
        plt.figure(figsize=(10, 6))
        plt.plot(1/u_values, fitted_results, linewidth=2,
                 label='Fitted', color='#324AB2')
        plt.plot(1/u_values, empirical_results, '--', linewidth=2,
                 label='Empirical', color='#C71585')
        plt.fill_between(1/u_values, lower_bounds, upper_bounds,
                         color='#C71585', alpha=0.2, label='95% CI')
        plt.xlabel('Threshold (x)')
        plt.ylabel('Probability / fitted quantity')
        plt.title(f'Empirical vs Fitted (Weighted, threshold weight >= {ell})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim([0, 0.05])
        plt.tight_layout()
        plt.show()

        results[ell] = {
            "u_values": u_values,
            "fitted": fitted_results,
            "empirical": empirical_results,
            "lower_CI": lower_bounds,
            "upper_CI": upper_bounds
        }

        # Optionally save results
        if save_results:
            fname = f"results/failure_set_weighted_ell{ell}.npz"
            np.savez(fname, u_values=u_values, fitted=fitted_results, empirical=empirical_results,
                     lower=lower_bounds, upper=upper_bounds)
            print(f"Saved weighted failure-set results: {fname}")

    return results

