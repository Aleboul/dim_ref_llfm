# Linear Latent Factor Model and Flaute event in Schleswig-Holstein

The script main.py performs an end-to-end analysis of high-dimensional extreme value data using a factor model approach. It includes:

- **Data loading and preprocessing** – loading raw data, applying Pareto rank transformations, and computing the Tail Pairwise Dependence Matrix (TPDM).  
- **Factor and pure variable estimation** – estimating pure variables, factor loadings (`A_hat`), and factor covariances (`C_hat`), with Lasso refinement.  
- **Simplex projection** – ensuring non-negativity and probability simplex constraints on the factor loadings.  
- **Model evaluation** – computing fitted TPDM, bivariate failure sets, and visual comparison of model vs empirical data.  
- **Extreme value fitting** – fitting Generalized Pareto Distributions (GPD) to exceedances, computing shape parameters, and generating histograms and Q-Q plots.  
- **Weighted failure set computation** – optionally weighting observations (e.g., by wind farm power) and computing multivariate failure sets.  
- **Visualization** – plotting matrices, histograms, hexbin comparisons, and boxplots for model diagnostics.

Dependencies include `numpy`, `pandas`, `matplotlib`, and several project-specific modules (`estimation`, `evaluation`, `utils`, `data.loader`, `config`). 

All parameters are defined in config.py and correspond to the optimal kappa and lambda values, chosen to best match the empirical extremal correlations with those implied by the model.

For efficiency, the weighted failure sets are computed only on the full dataset, while bootstrapped estimates for the empirical data are supported. For reproducing the bootstrap results, please refer to the wind_turbine_single folder.