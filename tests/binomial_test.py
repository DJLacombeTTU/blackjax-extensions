import pymc as pm
import numpy as np
import arviz as az
import blackjax
# Add this explicit import pointing to your custom bridge:
from blackjax.mcmc.pymc_bridge import sample_slingshot

# 1. Generate strictly bimodal data
np.random.seed(42)
true_means = np.array([-6.0, 6.0])
data = np.concatenate([np.random.normal(m, 0.5, 100) for m in true_means])

# 2. Build the Marginalized Bimodal Model
with pm.Model() as bimodal_model:
    # Priors for the two mixture components
    weights = pm.Dirichlet("weights", a=np.array([2.0, 2.0]))
    means = pm.Normal("means", mu=0, sigma=10, shape=2)
    
    # Marginalized Likelihood
    y = pm.NormalMixture("y", w=weights, mu=means, sigma=0.5, observed=data)

# 3. Execute the Engine (HPCC Optimized)
print("Starting bimodal production sampling...")
slingshot_idata = blackjax.sample_slingshot(
    bimodal_model, 
    draws=1000, 
    tune=1000, 
    chains=8, 
    proposals=800,
    num_temperatures=8,
    target_swap_accept=0.60,
    random_seed=42
)

# 4. Display Results
print("\n--- Model Convergence Metrics ---")
summary = az.summary(slingshot_idata)
print(summary)