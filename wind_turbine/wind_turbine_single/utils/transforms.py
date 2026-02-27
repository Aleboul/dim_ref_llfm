import numpy as np
from scipy.stats import rankdata

def pareto_rank_transform_matrix(X, alpha=1.0, axis=0):
    X = np.asarray(X)
    if axis not in [0, 1]:
        raise ValueError("axis must be 0 or 1")
    
    n = X.shape[axis]
    if np.isscalar(alpha):
        alpha = np.full(X.shape[1-axis], alpha)
    else:
        alpha = np.asarray(alpha)
        if len(alpha) != X.shape[1-axis]:
            raise ValueError("alpha length mismatch")
    
    transformed = np.empty_like(X, dtype=float)
    for idx in range(X.shape[1-axis]):
        if axis == 0:
            data = X[:, idx]
        else:
            data = X[idx, :]
        ranks = rankdata(data, method='average')
        p = ranks / (n + 1)
        transformed_val = np.power(1 - p, -1/alpha[idx])
        if axis == 0:
            transformed[:, idx] = transformed_val
        else:
            transformed[idx, :] = transformed_val
    return transformed

def compute_tpdm(X, indices=None, q=0.1):
    if indices is None:
        sample = X
    else:
        sample = X[:, indices]
    r = np.linalg.norm(sample, axis=1, ord=np.inf)   # 1, 2, np.inf
    r0 = np.quantile(r, 1-q)
    mask = r >= r0
    if np.sum(mask) == 0:
        # avoid division by zero later on
        w = np.empty((0, sample.shape[1]))
        TPDM = np.zeros((sample.shape[1], sample.shape[1]))
        return TPDM, r0, mask, w, r
    w = sample[mask] / r[mask, None]
    TPDM = (w.T @ w) / w.shape[0]
    print(np.where(r == r0))
    stop
    return TPDM, r0, mask, w, r

