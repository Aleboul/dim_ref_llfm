import numpy as np
from estimation.factors import compute_C_hat, compute_theta_j


def fista_using_C(C_hat, theta_j, lambda_, max_iter=2000, tol=1e-6):
    """
    Solves the Lasso problem using the Fast Iterative Shrinkage-Thresholding Algorithm (FISTA):

    \[
    \hat{\beta}_j^{\mathrm{LASSO}} = \arg\min_{\beta \in \mathbb{R}^K}
    \Bigl\{ \frac{1}{2}\|\hat{C}\beta - \hat{\theta}_j\|_2^2 + \lambda \|\beta\|_1 \Bigr\},
    \]

    where \(\lambda > 0\) controls sparsity. The FISTA algorithm iteratively updates \(\beta\) via:

    \[
    \begin{aligned}
    \beta^{(k+1)} &= \mathcal{S}_{\lambda/L}\left( y^{(k)} - \frac{1}{L} \hat{C}^\top(\hat{C}y^{(k)} - \hat{\theta}_j) \right), \\
    t^{(k+1)} &= \frac{1+\sqrt{1+4 (t^{(k)})^2}}{2}, \\
    y^{(k+1)} &= \beta^{(k+1)} + \frac{t^{(k)}-1}{t^{(k+1)}}(\beta^{(k+1)} - \beta^{(k)}),
    \end{aligned}
    \]

    where \(\mathcal{S}_{\alpha}\) is the soft-thresholding operator defined componentwise as:

    \[
    (S_\alpha(\bm{z}))_i = \textrm{sign}(z_i)\max\{|z_i|-\alpha,0\}, \quad \alpha > 0.
    \]

    Args:
        C_hat (np.ndarray): Extremal covariance matrix \(\hat{C}\) of shape (K, K).
        theta_j (np.ndarray): Vector \(\hat{\theta}_j\) of shape (K,).
        lambda_ (float): Regularization parameter \(\lambda\) controlling sparsity.
        max_iter (int): Maximum number of iterations (default: 2000).
        tol (float): Tolerance for convergence (default: 1e-6).

    Returns:
        np.ndarray: \(\hat{\beta}_j^{\mathrm{LASSO}}\), the estimated sparse coefficient vector of shape (K,).
    """
    K = C_hat.shape[1]
    # Initialize beta and beta_prev as zero vectors of length K
    beta = np.zeros(K)
    beta_prev = np.zeros_like(beta)
    # Initialize acceleration term
    t_prev = 1.0

    # Compute the Lipschitz constant L = ||C_hat||_2^2
    L = np.linalg.norm(C_hat, 2)**2
    if L <= 0:
        L = 1.0  # Ensure L is positive

    # Main FISTA loop
    for _ in range(max_iter):
        # Update acceleration term t
        t = (1.0 + np.sqrt(1.0 + 4.0 * t_prev**2)) / 2.0
        # Update y using the acceleration term
        y = beta + ((t_prev - 1.0) / t) * (beta - beta_prev)
        # Store current beta for convergence check
        beta_prev = beta.copy()
        t_prev = t

        # Compute the gradient of the smooth part of the objective function
        grad = C_hat.T @ (C_hat @ y - theta_j)
        # Update z using gradient descent step
        z = y - (1.0 / L) * grad
        # Apply soft-thresholding to obtain the new beta
        beta = np.sign(z) * np.maximum(np.abs(z) - lambda_ / L, 0.0)

        # Check for convergence
        if np.linalg.norm(beta - beta_prev) < tol:
            break

    return beta


def refit_on_support(beta_lasso, C_hat, theta_j, tol=1e-12):
    """
    Refits the non-zero coefficients of the LASSO solution using ordinary least squares (OLS).

    Given the LASSO solution \(\hat{\beta}_j^{\textrm{LASSO}}\), this function identifies the support set
    \(\hat{S}_\gamma = \{a \in [\hat{K}] : |\hat{\beta}_{ja}^{\textrm{LASSO}}| > \gamma\}\),
    where \(\gamma\) is a threshold (here, `tol`). It then refits the coefficients on this support using OLS:

    \[
    \hat{\beta}_j = \arg \min_{\beta_{\hat{S}_\gamma}} \|\hat{C} \beta_{\hat{S}_\gamma} - \hat{\theta}_j\|_2^2.
    \]

    Args:
        beta_lasso (np.ndarray): LASSO solution \(\hat{\beta}_j^{\textrm{LASSO}}\) of shape (K,).
        C_hat (np.ndarray): Extremal covariance matrix \(\hat{C}\) of shape (K, K).
        theta_j (np.ndarray): Vector \(\hat{\theta}_j\) of shape (K,).
        tol (float): Threshold \(\gamma\) for identifying the support set (default: 1e-12).

    Returns:
        tuple: (beta_refit, support)
               - beta_refit (np.ndarray): Refitted coefficient vector \(\hat{\beta}_j\) of shape (K,).
               - support (np.ndarray): Indices of the support set \(\hat{S}_\gamma\).
    """
    # Identify the support set: indices where |beta_lasso| > tol
    support = np.where(np.abs(beta_lasso) > tol)[0]
    # Initialize the refitted beta as zeros
    beta_refit = np.zeros_like(beta_lasso)

    if support.size > 0:
        # Extract the submatrix of C_hat corresponding to the support set
        C_support = C_hat[np.ix_(support, support)]
        # Extract the subvector of theta_j corresponding to the support set
        theta_support = theta_j[support]
        # Solve the least squares problem on the support set
        beta_support = np.linalg.solve(C_support, theta_support)
        # Update the refitted beta with the solution on the support set
        beta_refit[support] = beta_support

    return beta_refit, support


def update_A_hat_with_refit(Sigma_hat, A_hat, pure_lists, estimated_impure,
                            lambda_=1e-3, tol=1e-12, max_iter=2000):
    """
    Updates the loading matrix A_hat by refitting the coefficients for impure variables.

    For each impure variable \(j\), this function:
    1. Computes the extremal covariance matrix \(\hat{C}\) and the vector \(\hat{\theta}_j\).
    2. Estimates the sparse coefficient vector \(\hat{\beta}_j^{\textrm{LASSO}}\) using FISTA.
    3. Refits the non-zero coefficients of \(\hat{\beta}_j^{\textrm{LASSO}}\) using OLS.

    Args:
        Sigma_hat (np.ndarray): Covariance or similarity matrix.
        A_hat (np.ndarray): Loading matrix to be updated.
        pure_lists (list of sets): List of pure variable groups.
        estimated_impure (list): Indices of impure variables.
        lambda_ (float): Regularization parameter for LASSO (default: 1e-3).
        tol (float): Threshold for identifying the support set (default: 1e-12).
        max_iter (int): Maximum number of iterations for FISTA (default: 2000).

    Returns:
        np.ndarray: Updated loading matrix A_hat.
    """
    # Compute the extremal covariance matrix C_hat
    C_hat = compute_C_hat(Sigma_hat, A_hat, pure_lists)

    # Update the loading matrix for each impure variable
    for j in estimated_impure:
        # Compute the vector theta_j
        theta_j = compute_theta_j(Sigma_hat, A_hat, pure_lists, j)
        # Estimate the sparse coefficient vector using FISTA
        beta_lasso = fista_using_C(C_hat, theta_j, lambda_, max_iter=max_iter)
        # Refit the non-zero coefficients using OLS
        beta_refit, _ = refit_on_support(beta_lasso, C_hat, theta_j, tol=tol)
        # Update the loading matrix with the refitted coefficients
        A_hat[j, :] = beta_refit

    return A_hat
