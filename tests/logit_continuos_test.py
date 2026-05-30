import pymc as pm
import numpy as np
import arviz as az
import time
from blackjax.mcmc.pymc_bridge import sample_slingshot

def run_linear_imputation():
    print("\n" + "="*50)
    print("Benchmarking: Hierarchical Linear with Log_Income Imputation (Non-Centered)")
    print("="*50)
    
    np.random.seed(802)
    N = 400
    J = 5  # Number of groups
    
    group_idx = np.random.randint(0, J, N)
    true_log_income = np.random.normal(10.5, 0.8, N)
    
    # Introduce ~15% Missing Completely at Random (MCAR) in Log_Income
    log_income_obs = true_log_income.copy()
    missing_mask = np.random.rand(N) < 0.15
    log_income_obs[missing_mask] = np.nan 
    
    # True Parameters
    true_alpha = np.random.normal(-1.0, 0.5, J)
    true_beta = 1.25
    true_sigma_y = 0.5
    
    # Standardize ground truth to generate Y data correctly
    z_income_true = (true_log_income - np.mean(true_log_income)) / np.std(true_log_income)
    
    # Continuous Target Variable (e.g., Household Wealth Metric or Savings Rate)
    y_data = true_alpha[group_idx] + true_beta * z_income_true + np.random.normal(0, true_sigma_y, N)
    
    # Standardize observed covariate (ignoring NaNs for baseline scaling)
    obs_mean = np.nanmean(log_income_obs)
    obs_std = np.nanstd(log_income_obs)
    z_income_obs = (log_income_obs - obs_mean) / obs_std

    with pm.Model() as linear_imputation_model:
        # 1. Covariate Imputation Subgraph
        mu_income = pm.Normal("mu_income", mu=0.0, sigma=1.0)
        sigma_income = pm.HalfNormal("sigma_income", sigma=1.0)
        z_income_imputed = pm.Normal("z_income", mu=mu_income, sigma=sigma_income, observed=z_income_obs)
        
        # 2. Hierarchical Intercepts (Non-Centered Formulation)
        mu_alpha = pm.Normal("mu_alpha", mu=0, sigma=2.0)
        sigma_alpha = pm.HalfNormal("sigma_alpha", sigma=1.0)
        alpha_offset = pm.Normal("alpha_offset", mu=0, sigma=1.0, shape=J)
        alpha = pm.Deterministic("alpha", mu_alpha + alpha_offset * sigma_alpha)
        
        # 3. Structural Slope
        beta = pm.Normal("beta", mu=0, sigma=2.0)
        
        # 4. Observation Noise
        sigma_y = pm.HalfNormal("sigma_y", sigma=1.0)
        
        # 5. Continuous Gaussian Likelihood
        mu_y = alpha[group_idx] + beta * z_income_imputed
        y = pm.Normal("y", mu=mu_y, sigma=sigma_y, observed=y_data)

    start_time = time.time()
    # Keeping a solid 16-temperature ladder to evaluate multi-GPU scaling on HPCC
    idata = sample_slingshot(
        linear_imputation_model, draws=1000, tune=2000, chains=8, proposals=800, 
        num_temperatures=16, target_swap_accept=0.65, random_seed=803
    )
    
    # --- THE FIX: Invert the unconstrained log transform ---
    post = idata.posterior
    
    # Find the log-transformed key for sigma_y and exponentiate it
    sigma_y_key = [k for k in post.data_vars if "sigma_y" in k][0]
    post["sigma_y"] = np.exp(post[sigma_y_key])
    
    # Optional: You can do the same for mu_income if its hyperparameters were bounded, 
    # but mu_income is a flat Normal, so it should be fine.
    
    print(f"Execution Time: {time.time() - start_time:.2f} seconds")
    print(az.summary(idata, var_names=["mu_alpha", "beta", "sigma_y", "mu_income"]))

if __name__ == "__main__":
    run_linear_imputation()