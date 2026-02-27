from joblib import Parallel, delayed
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.stats import genpareto
import swiftascmaps
import matplotlib.colors as mcolors

def extract_pure_rows(A, tol=1e-10):
    A = np.asarray(A)
    d, K = A.shape
    
    pure_groups = [[] for _ in range(K)]
    
    for i, row in enumerate(A):
        nz = np.flatnonzero(np.abs(row) > tol)
        
        if len(nz) == 1 and np.isclose(abs(row[nz[0]]), 1, atol=tol):
            pure_groups[nz[0]].append(i)
    
    return pure_groups


def compute_failure_sets_bootstrap_weighted(
    X,
    Y_full,
    weights,
    ell_values,
    bootstrap_dir="bootstrap",
    B=999,
    fit_q=0.1,
    conf_level=0.95,
    u_values=None,
    n_jobs=-1,
):
    
    n, d = X.shape
    if u_values is None:
        u_values = np.linspace(0.1, 0.75, 200)

    n_u = len(u_values)

    files = sorted([f for f in os.listdir(bootstrap_dir)
                    if f.startswith("bootstrap_") and f.endswith(".npz")])
    if len(files) < B + 1:
        raise RuntimeError(f"Expected {B+1} bootstrap files, found {len(files)}")

    ell_indices = {ell: max(0, d - ell) for ell in ell_values}

    # --- Bootstrap worker ---
    def process_one_bootstrap(fname):
        data = np.load(os.path.join(bootstrap_dir, fname), allow_pickle=True)
        A_hat_boot = data["A_hat"]
        I_hat_boot = extract_pure_rows(A_hat_boot)
        idx_boot = data["indices"]

        X_boot = X[idx_boot, :]
        #X_boot += 1e-12*np.random.randn(*X_boot.shape)
        Y_boot = Y_full[idx_boot, :]

        # GPD fitting
        params = np.empty((d, 4))
        u_refs = np.quantile(X_boot, 1 - fit_q, axis=0)

        for j in range(d):
            sample = X_boot[:, j]
            u_ref = u_refs[j]
            exceedances = sample[sample > u_ref] - u_ref
            if exceedances.size == 0:
                params[j] = [np.nan, 0.0, np.nan, u_ref]
            else:
                try:
                    xi, loc, sigma = genpareto.fit(exceedances, floc=0)
                    params[j] = [xi, loc, sigma, u_ref]
                except Exception:
                    params[j] = [np.nan, 0.0, np.nan, u_ref]

        pure_lists = [sorted(g) for g in I_hat_boot]
        group_averages = []
        for g in pure_lists:
            if len(g) > 0:
                group_averages.append(np.mean(Y_boot[:, g], axis=1))
            else:
                group_averages.append(np.zeros(Y_boot.shape[0]))

        Z_prime = np.vstack(group_averages)
        Z_norms = np.linalg.norm(Z_prime, axis=0, ord=1)
        t_hat = np.quantile(Z_norms, 1 - fit_q)
        mask_valid = Z_norms > t_hat

        if not np.any(mask_valid):
            zero = np.zeros(n_u)
            return {ell: (zero.copy(), zero.copy()) for ell in ell_values}

        Z_prime_masked = Z_prime[:, mask_valid]
        Z_norms_masked = Z_norms[mask_valid]
        S = A_hat_boot @ (Z_prime_masked / Z_norms_masked)
        zeta_hat = len(I_hat_boot)
        k_exceed = np.sum(mask_valid)

        u_refs = params[:, 3]
        xis = params[:, 0]
        sigmas = params[:, 2]
        tails = np.array([np.mean(X_boot[:, j] > u_refs[j]) for j in range(d)])

        result = {}
        for ell in ell_values:
            fitted_results = np.zeros(n_u)
            empirical_results = np.zeros(n_u)
            target_index = ell_indices[ell]

            for iu, u in enumerate(u_values):
                weighted_counts = np.sum((X_boot > u) * weights[None, :], axis=1)
                empirical_results[iu] = np.mean(weighted_counts >= ell)

            for iu, u in enumerate(u_values):
                q_hat = np.zeros(d)
                below_mask = u <= u_refs
                above_mask = ~below_mask

                if np.any(below_mask):
                    q_hat[below_mask] = np.mean(X_boot[:, below_mask] > u, axis=0)

                if np.any(above_mask):
                    idxA = np.where(above_mask)[0]
                    y = u - u_refs[idxA]
                    xi_vals = xis[idxA]
                    sigma_vals = sigmas[idxA]
                    tail_vals = tails[idxA]
                    valid = (np.isfinite(xi_vals)) & (np.isfinite(sigma_vals)) & (sigma_vals > 0) & (tail_vals > 0)
                    q_hat[idxA[~valid]] = 0.0
                    idxV = idxA[valid]
                    if len(idxV) > 0:
                        xiV = xi_vals[valid]
                        sigmaV = sigma_vals[valid]
                        tailV = tail_vals[valid]
                        yV = y[valid]
                        inner = 1 + xiV * yV / sigmaV
                        good = inner > 0
                        idxG = idxV[good]
                        if len(idxG) > 0:
                            xiG = xiV[good]
                            sigmaG = sigmaV[good]
                            tailG = tailV[good]
                            yG = yV[good]
                            innerG = inner[good]
                            small = np.abs(xiG) < 1e-8
                            if np.any(small):
                                idxS = idxG[small]
                                q_hat[idxS] = tailG[small] * np.exp(-yG[small]/sigmaG[small])
                            reg = ~small
                            if np.any(reg):
                                idxR = idxG[reg]
                                q_hat[idxR] = tailG[reg] * innerG[reg]**(-1/xiG[reg])

                combined = S * q_hat[:, None]
                sorted_idx = np.argsort(-combined, axis=0)
                sorted_combined = np.take_along_axis(combined, sorted_idx, axis=0)
                sorted_weights = np.take_along_axis(weights[:, None], sorted_idx, axis=0)
                cum_weights = np.cumsum(sorted_weights, axis=0)
                mask = cum_weights >= ell
                kth_indices = np.argmax(mask, axis=0)
                kth_largest = sorted_combined[kth_indices, np.arange(k_exceed)]
                fitted_results[iu] = zeta_hat * (1.0/k_exceed) * np.sum(kth_largest)

            result[ell] = (fitted_results, empirical_results)
        return result

    results_list = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(process_one_bootstrap)(fname) for fname in files
    )

    all_fitted = {ell: np.zeros((len(results_list), n_u)) for ell in ell_values}
    all_empirical = {ell: np.zeros((len(results_list), n_u)) for ell in ell_values}

    for i, res in enumerate(results_list):
        for ell in ell_values:
            f, e = res[ell]
            all_fitted[ell][i] = f
            all_empirical[ell][i] = e

    alpha = 1 - conf_level
    q_low, q_high = 100*(alpha/2), 100*(1-alpha/2)


    results = {"u_values": u_values, "ell_values": ell_values}
    results["fitted_hat"] = np.vstack([all_fitted[ell][0] for ell in ell_values]).T
    results["empirical_hat"] = np.vstack([all_empirical[ell][0] for ell in ell_values]).T
    results["fitted_CI"] = (
        np.vstack([np.clip(2*results["fitted_hat"][:, j] - np.percentile(all_fitted[ell][1:], q_high, axis=0), 0, None)
                   for j, ell in enumerate(ell_values)]).T,
        np.vstack([np.clip(2*results["fitted_hat"][:, j] - np.percentile(all_fitted[ell][1:], q_low, axis=0), 0, None)
                   for j, ell in enumerate(ell_values)]).T
    )
    results["empirical_CI"] = (
        np.vstack([np.clip(2*results["empirical_hat"][:, j] - np.percentile(all_empirical[ell][1:], q_high, axis=0), 0, 1)
                   for j, ell in enumerate(ell_values)]).T,
        np.vstack([np.clip(2*results["empirical_hat"][:, j] - np.percentile(all_empirical[ell][1:], q_low, axis=0), 0, 1)
                   for j, ell in enumerate(ell_values)]).T
    )

    return results


def plot_bootstrap_failure(results, save=False, zoom_x=3, zoom_width=0.4, zoom_height=0.01):

    u_values = results["u_values"]
    ell_values = results["ell_values"]
    fitted_hat = results["fitted_hat"]
    empirical_hat = results["empirical_hat"]
    fitted_CI = results["fitted_CI"]
    empirical_CI = results["empirical_CI"]

    #emp_color = '#C71585'
    #fit_color = '#324AB2'
    cmap = plt.get_cmap("swift.lover")

    emp_color = mcolors.to_hex(cmap(0.0))   # first color
    fit_color = mcolors.to_hex(cmap(0.5))   # last color

    if save:
        os.makedirs('results/plot', exist_ok=True)

    for j, ell in enumerate(ell_values):
        fig, ax = plt.subplots(figsize=(10, 6))

        # ========================
        # MAIN PLOT
        # ========================

        ax.fill_between(
            u_values,
            empirical_CI[0][:, j], empirical_CI[1][:, j],
            color=emp_color, alpha=0.3,
            label='95% Basic-Bootstrap CI (Empirical)'
        )

        ax.fill_between(
            u_values,
            fitted_CI[0][:, j], fitted_CI[1][:, j],
            color=fit_color, alpha=0.3,
            label='95% Basic-Bootstrap CI (Fitted)'
        )

        ax.plot(u_values, empirical_hat[:, j],
                color=emp_color, linewidth=2, linestyle='--', label='Empirical')

        ax.plot(u_values, fitted_hat[:, j],
                color=fit_color, linewidth=2, linestyle='--', label='Fitted')

        ax.set_xlabel('Threshold w', fontsize=24)
        ax.set_ylabel('Empirical / Fitted probability', fontsize=24)
        ax.grid(alpha=0.3)
        ax.set_ylim([0, 0.025])
        ax.set_xlim([2,7])
        ax.tick_params(labelsize=20)             # Change both axes at once

        # ========================
        # ZOOM INSET
        # ========================

        #axins = inset_axes(ax, width="30%", height="30%", loc='upper right')
#
        #axins.fill_between(
        #    u_values,
        #    empirical_CI[0][:, j], empirical_CI[1][:, j],
        #    color=emp_color, alpha=0.3
        #)
#
        #axins.fill_between(
        #    u_values,
        #    fitted_CI[0][:, j], fitted_CI[1][:, j],
        #    color=fit_color, alpha=0.3
        #)
#
        #axins.plot(u_values, empirical_hat[:, j], color=emp_color, linestyle='--')
        #axins.plot(u_values, fitted_hat[:, j], color=fit_color, linestyle='--')
#
        ## zoom limits around x = 3
        #axins.set_xlim(zoom_x - zoom_width, zoom_x + zoom_width)
        ##axins.set_ylim(0, zoom_height)
        #axins.set_ylim(0, 0.004)
        #axins.grid(alpha=0.2)
        #
#
        #mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")
#
        if save:
            filename = f'results/plot/bootstrap_failure_alpha_{ell}.pdf'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Saved: {filename}")

        plt.show()

import matplotlib.colors as mcolors


def plot_bootstrap_failure_alpha(results, index=30, save=False, ylim=(0, 0.1)):
    """
    Plot empirical vs fitted curves with confidence intervals.

    Parameters
    ----------
    results : dict
        Dictionary containing:
        'u_values', 'ell_values', 'fitted_hat', 'empirical_hat',
        'fitted_CI', 'empirical_CI'
    index : int
        Row to plot
    save : bool
        Whether to save figure to results/plot/
    ylim : tuple
        y-axis limits
    """

    # ========================
    # Extract
    # ========================
    u_values = results["u_values"][-1]
    ell_values = results["ell_values"]
    fitted_hat = results["fitted_hat"]
    empirical_hat = results["empirical_hat"]
    fitted_CI = results["fitted_CI"]
    empirical_CI = results["empirical_CI"]
    u_values = results["u_values"] 
    ell_values = results["ell_values"][:-1] 
    fitted_hat = results["fitted_hat"][:,:-1]
    empirical_hat = results["empirical_hat"][:,:-1] 
    fitted_CI = results["fitted_CI"][:, :,:-1] 
    empirical_CI = results["empirical_CI"][:,: ,:-1]
    print(u_values[index])
    print(fitted_hat[index])
    print(empirical_hat[index])
    print(fitted_hat.shape)
    print(ell_values)

    # ========================
    # Colors
    # ========================
    cmap = plt.get_cmap("swift.lover")
    emp_color = mcolors.to_hex(cmap(0.0))
    fit_color = mcolors.to_hex(cmap(0.5))

    if save:
        os.makedirs("results/plot", exist_ok=True)

    # ========================
    # Plot
    # ========================
    fig, ax = plt.subplots(figsize=(10, 6))

    # CI bands
    ax.fill_between(
        ell_values,
        empirical_CI[0][index], empirical_CI[1][index],
        color=emp_color, alpha=0.3,
        label="CI (Empirical)"
    )

    ax.fill_between(
        ell_values,
        fitted_CI[0][index], fitted_CI[1][index],
        color=fit_color, alpha=0.3,
        label="CI (Fitted)"
    )

    # Lines
    ax.plot(
        ell_values, empirical_hat[index],
        color=emp_color, linewidth=4, linestyle="--", label="Empirical"
    )

    ax.plot(
        ell_values, fitted_hat[index],
        color=fit_color, linewidth=4, linestyle="--", label="Fitted"
    )

    # Formatting
    ax.set_xlabel(r'Capacity proportion $\alpha$', fontsize=24)
    ax.set_ylabel('Empirical / Fitted probability', fontsize=24)
    ax.set_ylim(ylim)
    ax.grid(alpha=0.3)
    ax.tick_params(labelsize=20)
    ax.legend(fontsize=24)

    # Save
    if save:
        filename = f"results/plot/bootstrap_failure_u_{u_values[index]}.pdf"
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        print(f"Saved: {filename}")

    plt.show()