# General configuration and hyperparameters

FILE_PATH = "../data/prod_data/daily_100m_corrected.csv"

# TPDM threshold quantile
Q = 0.05 #0.025

# Pure variable estimation
KAPPA = 0.006 #0.000075

# Lasso regularization
LAMBDA = 0.0008400000000000001 #12.5e-07

# Whether to use cached results (A_hat, C_hat, I_hat, K_hat)
USE_CACHE = True

# GPD fitting threshold for exceedances (used in fit_gpd_exceedances)
GPD_FIT_Q = 0.05 #0.025


# General configuration and hyperparameters

#FILE_PATH = "data/prod_data/daily_100m_corrected.csv"
#
## TPDM threshold quantile
#Q = 0.05 #0.025
#
## Pure variable estimation
#KAPPA = 0.004 #0.000075
#
## Lasso regularization
#LAMBDA = 12.5e-04 #12.5e-07
#
## Whether to use cached results (A_hat, C_hat, I_hat, K_hat)
#USE_CACHE = True
#
## GPD fitting threshold for exceedances (used in fit_gpd_exceedances)
#GPD_FIT_Q = 0.05 #0.025
#
