import pymc as pm
import numpy as np
import arviz as az
import time
from blackjax.mcmc.pymc_bridge import sample_slingshot

# --- MODEL 1: LINEAR REGRESSION ---
def run_linear_regression():
    print("\n" + "="*50)
    print("Benchmarking: 1. Linear Regression")
    print("="*50)
    
    np.random.seed(42)
    N, D = 500, 3
    X_data = np.random.normal(0, 1, (N, D))
    true_beta = np.array([1.5, -2.0, 0.8])
    true_sigma = 0.5
    y_data = X_data @ true_beta + np.random.normal(0, true_sigma, N)

    with pm.Model() as model:
        beta = pm.Normal("beta", mu=0, sigma=1.0, shape=D)
        log_sigma = pm.Normal("log_sigma", mu=0, sigma=1.0)
        sigma = pm.Deterministic("sigma", pm.math.exp(log_sigma))
        
        mu = pm.math.dot(X_data, beta)
        y = pm.Normal("y", mu=mu, sigma=sigma, observed=y_data)

    start_time = time.time()
    idata = sample_slingshot(
        model, draws=1000, tune=1000, chains=8, proposals=800, 
        num_temperatures=8, target_swap_accept=0.65, random_seed=42
    )
    print(f"Execution Time: {time.time() - start_time:.2f} seconds")
    print(az.summary(idata, var_names=["beta", "log_sigma"]))


# --- MODEL 2: LOGIT REGRESSION ---
def run_logit_regression():
    print("\n" + "="*50)
    print("Benchmarking: 2. Logit Regression")
    print("="*50)
    
    np.random.seed(101)
    N, D = 500, 3
    X_data = np.random.normal(0, 1, (N, D))
    true_beta = np.array([-1.0, 2.5, 0.5])
    logits = X_data @ true_beta
    probs = 1 / (1 + np.exp(-logits))
    y_data = np.random.binomial(1, probs)

    with pm.Model() as model:
        beta = pm.Normal("beta", mu=0, sigma=1.0, shape=D)
        logits_pred = pm.math.dot(X_data, beta)
        # PyMC's Bernoulli accepts logits directly for numerical stability
        y = pm.Bernoulli("y", logit_p=logits_pred, observed=y_data)

    start_time = time.time()
    idata = sample_slingshot(
        model, draws=1000, tune=1000, chains=8, proposals=800, 
        num_temperatures=8, target_swap_accept=0.65, random_seed=101
    )
    print(f"Execution Time: {time.time() - start_time:.2f} seconds")
    print(az.summary(idata, var_names=["beta"]))

# --- MODEL 3: HIERARCHICAL MODEL ---
def run_hierarchical_model():
    print("\n" + "="*50)
    print("Benchmarking: 3. Hierarchical Model")
    print("="*50)
    
    np.random.seed(201)
    J, M = 8, 20
    true_mu, true_tau, true_sigma = 2.0, 0.8, 1.0
    
    alphas = np.random.normal(true_mu, true_tau, J)
    group_indices = np.repeat(np.arange(J), M)
    y_data = alphas[group_indices] + np.random.normal(0, true_sigma, J * M)

    with pm.Model() as model:
        mu_global = pm.Normal("mu_global", mu=0, sigma=1.0)
        log_tau = pm.Normal("log_tau", mu=0, sigma=1.0)
        tau = pm.Deterministic("tau", pm.math.exp(log_tau))
        
        alphas_eval = pm.Normal("alphas", mu=mu_global, sigma=tau, shape=J)
        
        log_sigma = pm.Normal("log_sigma", mu=0, sigma=1.0)
        sigma = pm.Deterministic("sigma", pm.math.exp(log_sigma))
        
        mu_obs = alphas_eval[group_indices]
        y = pm.Normal("y", mu=mu_obs, sigma=sigma, observed=y_data)

    start_time = time.time()
    idata = sample_slingshot(
        model, draws=1000, tune=1000, chains=8, proposals=800, 
        num_temperatures=8, target_swap_accept=0.65, random_seed=201
    )
    print(f"Execution Time: {time.time() - start_time:.2f} seconds")
    print(az.summary(idata, var_names=["mu_global", "log_tau", "log_sigma"]))

# --- MODEL 4: NEAL'S FUNNEL ---
def run_neals_funnel():
    print("\n" + "="*50)
    print("Benchmarking: 4. Neal's Funnel (Non-Centered)")
    print("="*50)
    
    with pm.Model() as model:
        v = pm.Normal("v", mu=0, sigma=3.0)
        
        # 1. Sample from a perfectly easy standard normal (mean 0, sd 1)
        x_raw = pm.Normal("x_raw", mu=0, sigma=1.0, shape=9)
        
        # 2. Scale it deterministically by v
        x = pm.Deterministic("x", x_raw * pm.math.exp(v / 2.0))

    start_time = time.time()
    idata = sample_slingshot(
        model, draws=1000, tune=1000, chains=8, proposals=800, 
        num_temperatures=12, target_swap_accept=0.75, random_seed=301
    )
    print(f"Execution Time: {time.time() - start_time:.2f} seconds")
    print(az.summary(idata, var_names=["v"]))

    # --- MODEL 5: CORRELATED GAUSSIAN ---
def run_correlated_gaussian():
    print("\n" + "="*50)
    print("Benchmarking: 5. Correlated Gaussian")
    print("="*50)
    
    dim = 5
    # Create covariance matrix with 1s on diagonal and 0.5s elsewhere
    cov = np.eye(dim) + 0.5 * (np.ones((dim, dim)) - np.eye(dim))
    
    with pm.Model() as model:
        # PyMC handles the multivariate normal density natively
        theta = pm.MvNormal("theta", mu=np.zeros(dim), cov=cov, shape=dim)

    start_time = time.time()
    idata = sample_slingshot(
        model, draws=1000, tune=1000, chains=8, proposals=800, 
        num_temperatures=8, target_swap_accept=0.65, random_seed=401
    )
    print(f"Execution Time: {time.time() - start_time:.2f} seconds")
    print(az.summary(idata))

# --- MODEL 6: ROSENBROCK TWISTED BANANA ---
def run_rosenbrock():
    print("\n" + "="*50)
    print("Benchmarking: 6. Rosenbrock Twisted Banana")
    print("="*50)
    
    D = 10
    
    with pm.Model() as model:
        # We use a Flat prior so the Potential completely dictates the geometry
        theta = pm.Flat("theta", shape=D)
        
        # We add the Rosenbrock density directly to the model log-probability
        term1 = 100.0 * (theta[1:] - theta[:-1]**2)**2
        term2 = (1.0 - theta[:-1])**2
        pm.Potential("rosenbrock_potential", -pm.math.sum(term1 + term2))

    start_time = time.time()
    idata = sample_slingshot(
        model, draws=1000, tune=1000, chains=8, proposals=800, 
        num_temperatures=12, target_swap_accept=0.75, random_seed=501
    )
    print(f"Execution Time: {time.time() - start_time:.2f} seconds")
    print(az.summary(idata))

# --- MODEL 7: HIGH-DIMENSIONAL HORSESHOE ---
def run_horseshoe():
    print("\n" + "="*50)
    print("Benchmarking: 7. High-Dimensional Horseshoe")
    print("="*50)
    
    np.random.seed(601)
    D, N = 20, 100
    X_data = np.random.normal(0, 1, (N, D))
    true_beta = np.zeros(D)
    true_beta[[2, 7, 15]] = [3.5, -2.1, 4.0]
    y_data = X_data @ true_beta + np.random.normal(0, 1.0, N)

    with pm.Model() as model:
        # Standard Horseshoe Priors
        tau = pm.HalfCauchy("tau", beta=1.0)
        lambdas = pm.HalfCauchy("lambdas", beta=1.0, shape=D)
        
        # Non-centered formulation for beta to prevent the sampler from getting trapped
        z = pm.Normal("z", mu=0.0, sigma=1.0, shape=D)
        beta = pm.Deterministic("beta", z * tau * lambdas)
        
        y = pm.Normal("y", mu=pm.math.dot(X_data, beta), sigma=1.0, observed=y_data)

    start_time = time.time()
    idata = sample_slingshot(
        model, draws=1000, tune=1000, chains=8, proposals=800, 
        num_temperatures=16, target_swap_accept=0.60, random_seed=602
    )
    
    # --- THE FIX ---
    post = idata.posterior
    
    # Dynamically find the log-transformed keys returned by the bridge
    tau_key = [k for k in post.data_vars if 'tau' in k][0]
    lambdas_key = [k for k in post.data_vars if 'lambda' in k][0]
    
    # Invert the log-transform with np.exp() and compute beta
    tau_constrained = np.exp(post[tau_key])
    lambdas_constrained = np.exp(post[lambdas_key])
    post["beta"] = post["z"] * tau_constrained * lambdas_constrained
    
    print(f"Execution Time: {time.time() - start_time:.2f} seconds")
    print(az.summary(idata, var_names=["beta"]))

# --- MODEL 8: LOG-GAUSSIAN COX PROCESS ---
def run_lgcp():
    print("\n" + "="*50)
    print("Benchmarking: 8. Log-Gaussian Cox Process")
    print("="*50)
    
    np.random.seed(701)
    M = 4
    D = M * M
    mu_val = 1.0
    
    # Create the distance matrix for the spatial covariance
    coords = np.stack(np.meshgrid(np.arange(M), np.arange(M)), axis=-1).reshape(-1, 2)
    dists_sq = np.sum((coords[:, None, :] - coords[None, :, :])**2, axis=-1)
    Sigma = 0.5**2 * np.exp(-dists_sq / 2.0) + 1e-6 * np.eye(D)
    
    # Generate true counts
    true_Y = np.random.multivariate_normal(np.ones(D) * mu_val, Sigma)
    y_counts = np.random.poisson(np.exp(true_Y))

    with pm.Model() as model:
        # Spatial Latent Field
        Y = pm.MvNormal("Y", mu=np.ones(D) * mu_val, cov=Sigma, shape=D)
        
        # Poisson Likelihood based on exponentiated latent field
        rates = pm.math.exp(Y)
        pm.Poisson("y_obs", mu=rates, observed=y_counts)

    start_time = time.time()
    idata = sample_slingshot(
        model, draws=1000, tune=1000, chains=8, proposals=800, 
        num_temperatures=12, target_swap_accept=0.65, random_seed=702
    )
    print(f"Execution Time: {time.time() - start_time:.2f} seconds")
    print(az.summary(idata, var_names=["Y"]))

if __name__ == "__main__":
    run_linear_regression()
    run_logit_regression()
    run_hierarchical_model()
    run_neals_funnel()
    # Adding the new ones:
    run_correlated_gaussian()
    run_rosenbrock()
    run_horseshoe()
    run_lgcp()