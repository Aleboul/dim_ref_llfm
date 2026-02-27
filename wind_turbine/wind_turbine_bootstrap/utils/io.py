import os
import numpy as np
import pickle

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

def get_result_filenames(q, kappa, lam):
    """Generate parameter-specific filenames for factorization results."""
    prefix = f"Q{q}_KAPPA{kappa}_LAMBDA{lam}"
    return {
        "A_hat": os.path.join(RESULTS_DIR, f"A_hat_{prefix}.npy"),
        "C_hat": os.path.join(RESULTS_DIR, f"C_hat_{prefix}.npy"),
        "K_hat": os.path.join(RESULTS_DIR, f"K_hat_{prefix}.txt"),
        "I_hat": os.path.join(RESULTS_DIR, f"I_hat_{prefix}.pkl"),
    }

def save_results(A_hat, C_hat, K_hat, I_hat, q, kappa, lam):
    """Save factorization results to results directory with param-specific names."""
    files = get_result_filenames(q, kappa, lam)
    np.save(files["A_hat"], A_hat)
    np.save(files["C_hat"], C_hat)
    with open(files["K_hat"], "w") as f:
        f.write(str(K_hat))
    with open(files["I_hat"], "wb") as f:
        pickle.dump(I_hat, f)
    print(f"✅ Results saved: {files}")
    return files

def load_results(q, kappa, lam):
    """Try to load core factorization results (A_hat, C_hat, K_hat, I_hat)."""
    files = get_result_filenames(q, kappa, lam)
    if all(os.path.exists(path) for path in files.values()):
        A_hat = np.load(files["A_hat"])
        C_hat = np.load(files["C_hat"])
        with open(files["K_hat"], "r") as f:
            K_hat = int(f.read().strip())
        with open(files["I_hat"], "rb") as f:
            I_hat = pickle.load(f)
        print(f"📂 Loaded cached results: {files}")
        return {"A_hat": A_hat, "C_hat": C_hat, "K_hat": K_hat, "I_hat": I_hat}
    return None

# GPD params (saved independent of factorization parameters)
def get_gpd_filename(fit_q):
    """Filename for stored GPD parameters (fit threshold q)."""
    # include fit_q value (rounded nicely)
    q_str = str(fit_q).replace('.', 'p')
    return os.path.join(RESULTS_DIR, f"gpd_params_q{q_str}.npy")

def save_gpd_params(params, fit_q):
    """Save GPD fit params array (shape d x 4 or similar)."""
    path = get_gpd_filename(fit_q)
    np.save(path, params)
    print(f"✅ GPD params saved: {path}")
    return path

def load_gpd_params(fit_q):
    """Load GPD params if present, otherwise return None."""
    path = get_gpd_filename(fit_q)
    if os.path.exists(path):
        params = np.load(path)
        print(f"📂 Loaded cached GPD params: {path}")
        return params
    return None

