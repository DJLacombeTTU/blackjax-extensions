import pymc as pm
import numpy as np
import arviz as az
import time
from blackjax.mcmc.pymc_bridge import sample_slingshot

def run_imputation_logit():
    print("\n" + "="*50)
    print("Benchmarking: Hierarchical Logit with Log_Income Imputation (Non-Centered)")
    print("="*50)
    
    np.random.seed(801)
    N = 300
    J = 5  
    
    group_idx = np.random.randint(0, J, N)
    true_log_income = np.random.normal(10.5, 0.8, N)
    
    log_income_obs = true_log_income.copy()
    missing_mask = np.random.rand(N) < 0.15
    log_income_obs[missing_mask] = np.nan 
    
    true_alpha = np.random.normal(-1.0, 0.5, J)
    true_beta = 0.75
    
    z_income_true = (true_log_income - np.mean(true_log_income)) / np.std(true_log_income)
    logits = true_alpha[group_idx] + true_beta * z_income_true
    probs = 1 / (1 + np.exp(-logits))
    y_data = np.random.binomial(1, probs)
    
    obs_mean = np.nanmean(log_income_obs)
    obs_std = np.nanstd(log_income_obs)
    z_income_obs = (log_income_obs - obs_mean) / obs_std

    with pm.Model() as imputation_model:
        # 1. Imputation Model 
        mu_income = pm.Normal("mu_income", mu=0.0, sigma=1.0)
        sigma_income = pm.HalfNormal("sigma_income", sigma=1.0)
        z_income_imputed = pm.Normal("z_income", mu=mu_income, sigma=sigma_income, observed=z_income_obs)
        
        # 2. Hierarchical Intercepts (THE FIX: NON-CENTERED)
        mu_alpha = pm.Normal("mu_alpha", mu=0, sigma=1.5)
        sigma_alpha = pm.HalfNormal("sigma_alpha", sigma=1.0)
        
        # Sample raw offsets from a perfectly round standard normal
        alpha_offset = pm.Normal("alpha_offset", mu=0, sigma=1.0, shape=J)
        
        # Deterministically build the true alpha
        alpha = pm.Deterministic("alpha", mu_alpha + alpha_offset * sigma_alpha)
        
        # 3. Structural Parameter
        beta = pm.Normal("beta", mu=0, sigma=1.5)
        
        # 4. Logit Likelihood
        logits_pred = alpha[group_idx] + beta * z_income_imputed
        y = pm.Bernoulli("y", logit_p=logits_pred, observed=y_data)

    start_time = time.time()
    idata = sample_slingshot(
        imputation_model, draws=1000, tune=5000, chains=8, proposals=800, 
        num_temperatures=16, target_swap_accept=0.65, random_seed=802
    )
    
    print(f"Execution Time: {time.time() - start_time:.2f} seconds")
    # Note: We now ask for alpha_offset to ensure the base space is mixing
    print(az.summary(idata, var_names=["mu_alpha", "beta", "mu_income"]))

if __name__ == "__main__":
    run_imputation_logit()