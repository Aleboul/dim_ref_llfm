import numpy as np


def compute_C_hat(Sigma_hat, A_hat, pure_lists):
    """
    Computes the extremal covariance matrix \(\hat{C}\) based on the provided covariance matrix (\(\Sigma_{n,k}\)),
    the estimated loading matrix (\(\hat{A}\)), and the list of pure variable groups (\(\hat{I}_a\)).

    The diagonal elements of \(\hat{C}\) are computed as:
    \[
    \hat{C}(a,a) = \frac{2}{|\hat{I}_a|(|\hat{I}_a| + 1)} \sum_{j, \ell \in \hat{I}_a, j \le \ell} |\hat{\Sigma}_{n,k}(j,\ell)|,
    \]
    and the off-diagonal elements as:
    \[
    \hat{C}(a,b) = \frac{1}{|\hat{I}_a||\hat{I}_b|} \sum_{j \in \hat{I}_a, \ell \in \hat{I}_b} \hat{A}_{ja} \hat{A}_{\ell b} \hat{\Sigma}_{n,k}(j,\ell),
    \]
    for each \(a \in [\hat{K}]\) and \(b \in [\hat{K}] \setminus \{a\}\).

    Note: The A_hat argument is expected to be the output of the function estimate_A_hat_I, i.e.,
          a matrix A_hat_I of shape (p, K_hat), where p is the number of variables.
          For each pure group a, the entries in the corresponding columns are set to ±1,
          representing the canonical basis vector (up to a sign). Variables not in any pure group remain 0.

    Args:
        Sigma_hat (np.ndarray): Covariance or similarity matrix (\(\hat{\Sigma}_{n,k}\)).
        A_hat (np.ndarray): Estimated loading matrix (\(\hat{A}\)), as provided by the function estimate_A_hat_I.
        pure_lists (list of sets): List of pure variable groups (\(\hat{I}_a\)), where each set contains indices of pure variables.

    Returns:
        np.ndarray: \(\hat{C}\), a K x K matrix representing the extremal covariances between factors.
    """
    # K: Number of factors
    K = len(pure_lists)
    # Initialize C_hat as a K x K zero matrix
    C_hat = np.zeros((K, K))

    # Loop over each factor a
    for a in range(K):
        # Get indices of elements in group a
        I_a = pure_lists[a]
        # Number of elements in group a
        size_a = len(I_a)
        # Extract the submatrix of Sigma_hat corresponding to group a
        block = Sigma_hat[np.ix_(I_a, I_a)]
        # Sum of absolute values of the upper triangular part of the submatrix
        block_sum = np.sum(np.triu(np.abs(block)))
        # Compute the diagonal element of C_hat for group a, according to the equation
        C_hat[a, a] = (2.0 / (size_a * (size_a + 1))) * block_sum

        # Loop over groups b > a to compute off-diagonal elements
        for b in range(a + 1, K):
            # Get indices of elements in group b
            I_b = pure_lists[b]
            # Number of elements in group b
            size_b = len(I_b)
            # Initialize the sum for the off-diagonal element
            val = 0.0
            # Sum over all pairs (i, j) where i is in group a and j is in group b
            for i in I_a:
                for j in I_b:
                    val += A_hat[i, a] * A_hat[j, b] * Sigma_hat[i, j]
            # Compute the off-diagonal element of C_hat for groups a and b, according to the equation
            C_hat[a, b] = val / (size_a * size_b)
            # Ensure symmetry in C_hat
            C_hat[b, a] = C_hat[a, b]

    # Return the computed C_hat matrix
    return C_hat


def compute_theta_j(Sigma_hat, A_hat, pure_lists, j):
    """
    Estimates the vector \(\hat{\theta}_j\) for a given variable \(j\) based on the equation:

    \[
    \hat{\theta}_{ja} = \frac{1}{|\hat{I}_a|} \sum_{\ell \in \hat{I}_a} \hat{A}_{\ell a} \Sigma_{n,k}(\ell,j), \quad a \in [\hat{K}],
    \]

    where:
    - \(\hat{\theta}_{ja}\) is the \(a\)-th entry of \(\hat{\theta}_j\),
    - \(\hat{I}_a\) is the set of pure variables in group \(a\),
    - \(\hat{A}_{\ell a}\) is the loading of variable \(\ell\) on factor \(a\),
    - \(\Sigma_{n,k}(\ell,j)\) is the covariance between variables \(\ell\) and \(j\).

    Note: The A_hat argument is expected to be the output of the function estimate_A_hat_I, i.e.,
          a matrix A_hat_I of shape (p, K_hat), where p is the number of variables.
          For each pure group a, the entries in the corresponding columns are set to ±1,
          representing the canonical basis vector (up to a sign). Variables not in any pure group remain 0.

    Args:
        Sigma_hat (np.ndarray): Covariance or similarity matrix.
        A_hat (np.ndarray): Estimated loading matrix.
        pure_lists (list of sets): List of pure variable groups, where each set contains indices of pure variables.
        j (int): Index of the variable for which \(\hat{\theta}_j\) is estimated.

    Returns:
        np.ndarray: \(\hat{\theta}_j\), a vector of length \(\hat{K}\) (number of groups/clusters),
                    where each entry \(\hat{\theta}_{ja}\) is the estimated value for group \(a\).
    """
    # K_hat: Number of groups/clusters (columns in A_hat)
    K_hat = A_hat.shape[1]
    # Initialize theta_j as a zero vector of length K_hat
    theta_j = np.zeros(K_hat)
    # d: Total number of elements (rows in Sigma_hat)
    d = Sigma_hat.shape[0]

    # Loop over each group a
    for a in range(K_hat):
        # Get indices of elements in group a
        I_a = pure_lists[a]
        # Number of elements in group a
        size_a = len(I_a)
        # Compute the product of A_hat[i, a] and Sigma_hat[i, j] for all i in group a
        vals = [A_hat[i, a] * Sigma_hat[i, j] for i in I_a]
        # Compute theta_j[a] as the average of vals, according to the equation
        theta_j[a] = np.sum(vals) / size_a

    # Return the computed theta_j vector
    return theta_j
