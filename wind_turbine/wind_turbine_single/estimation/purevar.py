import numpy as np
import random
from utils.helpers import merge_groups


def PureVar(Sigma_hat, delta):
    """
    Identifies pure variable in the extremal covariance Sigma_hat.

    Args:
        Sigma_hat (np.ndarray): Extremal Covariance or Tail Pairwise Dependence Measure.
        delta (float): Threshold parameter for determining pure groups.

    Returns:
        tuple: (I_hat, number of factor)
               I_hat is a list of sets, where each set represents a pure set.
    """
    p = Sigma_hat.shape[0]  # Number of variables (dimension of Sigma_hat)
    I_hat = []  # List to store pure groups

    # Loop over each variable i
    for i in range(p):
        # Find the maximum absolute extremal covariance of variable i with any other variable
        max_ij = max(abs(Sigma_hat[i, j]) for j in range(p))

        # Identify candidate variables for the pure group of i:
        # Variables l where the correlation with i is within 2*delta of the maximum
        I_i = {l for l in range(p) if l != i and max_ij <=
               abs(Sigma_hat[i, l]) + 2 * delta}

        # Check if the group I_i is pure:
        # For all j in I_i, the difference between |Sigma_hat[i,j]| and the maximum correlation of j
        # with any other variable should be <= 2*delta, see Lemma 9.1 in the paper for this identification step
        is_pure = all(abs(abs(Sigma_hat[i, j]) - max(abs(Sigma_hat[j, k]) for k in range(p))) <= 2*delta
                      for j in I_i)

        # If the group is pure, add i to I_i and merge with existing groups
        if is_pure:
            I_i.add(i)
            I_hat = merge_groups(I_i, I_hat)

    # Return the list of pure groups and the number of groups
    return I_hat, len(I_hat)


def estimate_A_hat_I(I_hat, K_hat, Sigma_hat):
    """
    Estimates the matrix A_hat_I based on pure groups and Sigma_hat.

    Args:
        I_hat (list of sets): Pure groups identified by PureVar.
        K_hat (int): Number of pure groups.
        Sigma_hat (np.ndarray): Covariance or similarity matrix.

    Returns:
        np.ndarray: A matrix A_hat_I of shape (p, K_hat), where p is the number of variables.
                    For each pure group a, the entries in the corresponding columns are set to ±1,
                    representing the canonical basis vector (up to a sign). Variables not in any pure group remain 0.
    """
    p = Sigma_hat.shape[0]  # Number of variables
    # Initialize A_hat_I as a p x K_hat zero matrix
    A_hat_I = np.zeros((p, K_hat))

    # Loop over each pure group a
    for a, cluster in enumerate(I_hat):
        # Convert the set to a list for easier indexing
        cluster = list(cluster)
        if not cluster:  # Skip if the cluster is empty
            continue

        # Randomly select a reference variable from the cluster
        i_ref = random.choice(cluster)
        # Set the reference variable's coefficient to 1
        A_hat_I[i_ref, a] = 1

        # For all other variables in the cluster, set their coefficient to the sign of their correlation with the reference
        for j in cluster:
            if j != i_ref:
                A_hat_I[j, a] = np.sign(Sigma_hat[i_ref, j])

    # Return the estimated A_hat_I matrix
    return A_hat_I
