import os

# =====================================================================
# 1. MOCK MULTI-DEVICE ENVIRONMENT (Must run BEFORE importing JAX/PyMC)
# =====================================================================
# Force JAX to use the host CPU instead of your single laptop GPU
os.environ["JAX_PLATFORMS"] = "cpu"
# Split the host CPU into 2 distinct logical devices to mimic 2 cluster GPUs
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"

import jax
import pymc as pm
import pymc.sampling.jax as pm_jax
import numpy as np
import arviz as az
from blackjax.mcmc.pymc_bridge import sample_slingshot

def run_multi_device_regression():
    print("="*70)
    print(" JAX MULTI-DEVICE HARDWARE VERIFICATION")
    print("="*70)
    
    # Verify the host platform spoofing worked
    devices = jax.devices()
    num_devices = len(devices)
    print(f"[*] Total Logical Devices Detected by JAX: {num_devices}")
    for idx, dev in enumerate(devices):
        print(f"    -> Device {idx}: {dev}")
    
    if num_devices < 2:
        print("[!] Setup failed: JAX does not see 2 distinct devices.")
        return
        
    print("\n" + "="*70)
    print(" GENERATING SIMULATED REGRESSION DATA")
    print("="*70)
    
    # True underlying parameters
    true_intercept = 2.5
    true_slope = 4.2
    true_sigma = 1.0
    num_observations = 200
    
    # Generate synthetic X and Y
    np.random.seed(42)
    X_data = np.random.uniform(-3, 3, size=num_observations)
    noise = np.random.normal(0, true_sigma, size=num_observations)
    Y_data = true_intercept + true_slope * X_data + noise
    
    print(f"[*] Generated {num_observations} observations.")
    print(f"    True Line: Y = {true_intercept} + {true_slope}*X + Normal(0, {true_sigma})")

    print("\n" + "="*70)
    print(" BUILDING PYMC TOY REGRESSION MODEL")
    print("="*70)
    
    with pm.Model() as regression_model:
        # Priors for unknown model parameters
        intercept = pm.Normal("Intercept", mu=0.0, sigma=10.0)
        slope = pm.Normal("Slope", mu=0.0, sigma=10.0)
        sigma = pm.HalfNormal("Sigma", sigma=5.0)
        
        # Expected value of outcome
        likelihood_mu = intercept + slope * X_data
        
        # Likelihood (sampling distribution) of observations
        Y_obs = pm.Normal("Y_obs", mu=likelihood_mu, sigma=sigma, observed=Y_data)

    print("\n" + "="*70)
    print(" SAMPLING WITH SLINGSHOT VIA JAX.PMAP")
    print("="*70)
    
    num_chains = 4
    num_temperatures = 8
    
    print(f"[*] Launching {num_chains} MCMC chains.")
    print(f"[*] VERIFICATION NOTICE:")
    print(f"    Your sample_slingshot engine will invoke jax.pmap internally.")
    print(f"    - Device 0 ({devices[0]}) will handle 2 chains (with 8 temps each).")
    print(f"    - Device 1 ({devices[1]}) will handle 2 chains (with 8 temps each).")
    print("-" * 70)
    print("Compiling and sampling across mock hardware topology...\n")
    
    # Execute the parallelized sampling using YOUR engine
    idata = sample_slingshot(
        regression_model, 
        draws=1000, 
        tune=1000, 
        chains=num_chains, 
        proposals=800, 
        num_temperatures=num_temperatures, 
        target_swap_accept=0.60, 
        random_seed=42
    )
    
    # --- THE FIX: Invert the unconstrained log transform for Sigma ---
    post = idata.posterior
    sigma_key = [k for k in post.data_vars if "Sigma" in k][0]
    post["Sigma"] = np.exp(post[sigma_key])
    
    print("\n" + "="*70)
    print(" POSTERIOR SUMMARY STATISTICS")
    print("="*70)
    summary = az.summary(idata, var_names=["Intercept", "Slope", "Sigma"])
    print(summary)
    
    print("\n[*] Success! Code seamlessly mapped and ran on multi-device grid.")

if __name__ == "__main__":
    run_multi_device_regression()