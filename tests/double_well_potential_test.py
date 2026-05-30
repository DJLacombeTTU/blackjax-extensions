import pymc as pm
import pymc.sampling.jax as pm_jax
import numpy as np
import arviz as az
import time
from blackjax.mcmc.pymc_bridge import sample_slingshot

def run_comparison_suite():
    print("Initializing Physics Double-Well Potential Model...")
    
    # Define the bistable landscape: V(x) = x^4 - 10*x^2
    # Swapping pm.Flat for a highly uninformative Normal to stabilize JAX trace initialization
    with pm.Model() as double_well_model:
        x = pm.Normal("x", mu=0.0, sigma=15.0)
        energy = 1.0 * x**4 - 10.0 * x**2
        pm.Potential("double_well_potential", -energy)

    # -----------------------------------------------------------------
    # EXPERIMENT 1: Standard PyMC BlackJax NUTS (Forced Single Device Vectorization)
    # -----------------------------------------------------------------
    print("\n" + "="*60)
    print("Running Experiment 1: Standard PyMC Single-Temperature BlackJax NUTS")
    print("="*60)
    
    start_nuts = time.time()
    try:
        # chain_method="vectorized" prevents PyMC from choking on multi-GPU dimension formatting
        idata_nuts = pm_jax.sample_blackjax_nuts(
            draws=1000, 
            tune=1000, 
            chains=8, 
            random_seed=999,
            model=double_well_model,
            chain_method="vectorized",
            progressbar=False
        )
        time_nuts = time.time() - start_nuts
        summary_nuts = az.summary(idata_nuts, var_names=["x"])
        mean_nuts = summary_nuts.loc["x", "mean"]
        sd_nuts = summary_nuts.loc["x", "sd"]
        ess_nuts = summary_nuts.loc["x", "ess_bulk"]
        rhat_nuts = summary_nuts.loc["x", "r_hat"]
    except Exception as e:
        print(f"Standard NUTS failed or crashed: {e}")
        time_nuts, mean_nuts, sd_nuts, ess_nuts, rhat_nuts = np.nan, np.nan, np.nan, np.nan, np.nan

    # -----------------------------------------------------------------
    # EXPERIMENT 2: Parallel Tempered Slingshot Engine
    # -----------------------------------------------------------------
    print("\n" + "="*60)
    print("Running Experiment 2: Multi-Temperature Slingshot Engine")
    print("NOTE: Initial compilation on multi-GPU cluster nodes may take up to 2-3 minutes.")
    print("Please do not interrupt; sampling will begin automatically once compiled.")
    print("="*60)
    
    start_slingshot = time.time()
    try:
        idata_sling = sample_slingshot(
            double_well_model, 
            draws=3000, 
            tune=3000, 
            chains=8, 
            proposals=3000, 
            num_temperatures=16, 
            target_swap_accept=0.60, 
            random_seed=999
        )
        time_sling = time.time() - start_slingshot
        summary_sling = az.summary(idata_sling, var_names=["x"])
        mean_sling = summary_sling.loc["x", "mean"]
        sd_sling = summary_sling.loc["x", "sd"]
        ess_sling = summary_sling.loc["x", "ess_bulk"]
        rhat_sling = summary_sling.loc["x", "r_hat"]
    except Exception as e:
        print(f"Slingshot engine encountered an error: {e}")
        return

    # -----------------------------------------------------------------
    # FINAL METRIC COMPILATION & PRINTING
    # -----------------------------------------------------------------
    print("\n" + "="*70)
    print("                     FINAL PERFORMANCE COMPARISON                  ")
    print("="*70)
    print(f"{'Metric':<20} | {'True Value':<12} | {'Standard NUTS':<15} | {'Slingshot (PT)':<15}")
    print("-" * 70)
    print(f"{'Posterior Mean':<20} | {'0.000':<12} | {mean_nuts:<15.3f} | {mean_sling:<15.3f}")
    print(f"{'Posterior SD':<20} | {'~2.236':<12} | {sd_nuts:<15.3f} | {sd_sling:<15.3f}")
    print(f"{'Bulk ESS':<20} | {'> 1000':<12} | {ess_nuts:<15.1f} | {ess_sling:<15.1f}")
    print(f"{'Gelman-Rubin (R_hat)':<20} | {'1.000':<12} | {rhat_nuts:<15.3f} | {rhat_sling:<15.3f}")
    print(f"{'Execution Time (s)':<20} | {'N/A':<12} | {time_nuts:<15.2f} | {time_sling:<15.2f}")
    print("="*70)

if __name__ == "__main__":
    run_comparison_suite()