import pymc as pm
import numpy as np
import blackjax
import arviz as az

# 1. Generate "hard" bimodal/trimodal data
# We create three distinct, well-separated clusters.
np.random.seed(42)
true_means = np.array([-10.0, 0.0, 10.0])
data = np.concatenate([np.random.normal(m, 0.5, 100) for m in true_means])

# 2. Build the Marginalized Model
# By using NormalMixture, we analytically marginalize out the discrete 'assignment'
# variable, allowing the sampler to focus on the continuous parameter space.
with pm.Model() as hard_model:
    # Priors for mixture components (slightly informative prior to prevent edge-trapping)
    weights = pm.Dirichlet("weights", a=np.array([2.0, 2.0, 2.0]))
    means = pm.Normal("means", mu=0, sigma=10, shape=3)
    
    # Marginalized Likelihood (The core engine of this model)
    y = pm.NormalMixture("y", w=weights, mu=means, sigma=0.5, observed=data)

# 3. Execute the Engine (HPCC Optimized Parameters)
print("Starting production sampling...")
slingshot_idata = blackjax.sample_slingshot(
    hard_model, 
    draws=2000, 
    tune=2000, 
    chains=8, 
    proposals=800,
    num_temperatures=16,
    target_swap_accept=0.60,
    random_seed=42
)

# 4. Display Results
print("\n--- Model Convergence Metrics ---")
summary = az.summary(slingshot_idata)
print(summary)