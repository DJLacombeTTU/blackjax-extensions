import pymc as pm
import pymc.sampling.jax as pm_jax
import numpy as np
import arviz as az
import time
import matplotlib.pyplot as plt
from blackjax.mcmc.pymc_bridge import sample_slingshot

def run_rastrigin_comparison():
    print("\n" + "="*60)
    print("Benchmarking: The 2D Rastrigin Multimodal Torture Test")
    print("="*60)
    
    with pm.Model() as rastrigin_model:
        # Wide priors to allow broad exploration space
        x = pm.Normal("x", mu=0.0, sigma=5.0)
        y = pm.Normal("y", mu=0.0, sigma=5.0)
        
        # Mathematical definition of the Rastrigin energy surface
        # Global max density is at (0,0), surrounded by an infinite grid of traps
        energy = 20.0 + (x**2 - 10.0 * pm.math.cos(2.0 * np.pi * x)) \
                      + (y**2 - 10.0 * pm.math.cos(2.0 * np.pi * y))
        
        pm.Potential("rastrigin_potential", -energy)

    # -----------------------------------------------------------------
    # EXPERIMENT 1: Standard PyMC BlackJax NUTS
    # -----------------------------------------------------------------
    print("\nRunning Experiment 1: Standard Single-Temperature NUTS...")
    start_nuts = time.time()
    try:
        idata_nuts = pm_jax.sample_blackjax_nuts(
            draws=1000, tune=1000, chains=8, random_seed=42,
            model=rastrigin_model, chain_method="vectorized", progressbar=False
        )
        time_nuts = time.time() - start_nuts
        summary_nuts = az.summary(idata_nuts, var_names=["x", "y"])
        rhat_x_nuts = summary_nuts.loc["x", "r_hat"]
        ess_x_nuts = summary_nuts.loc["x", "ess_bulk"]
    except Exception as e:
        print(f"Standard NUTS crashed: {e}")
        time_nuts, rhat_x_nuts, ess_x_nuts = np.nan, np.nan, np.nan

    # -----------------------------------------------------------------
    # EXPERIMENT 2: Parallel Tempered Slingshot Engine
    # -----------------------------------------------------------------
    print("\nRunning Experiment 2: Multi-Temperature Slingshot Engine...")
    print("Melting energy barriers... Please allow 1-2 minutes for JAX compilation.")
    start_slingshot = time.time()
    try:
        idata_sling = sample_slingshot(
            rastrigin_model, draws=1000, tune=1000, chains=8, proposals=800, 
            num_temperatures=24, target_swap_accept=0.60, random_seed=42
        )
        time_sling = time.time() - start_slingshot
        summary_sling = az.summary(idata_sling, var_names=["x", "y"])
        rhat_x_sling = summary_sling.loc["x", "r_hat"]
        ess_x_sling = summary_sling.loc["x", "ess_bulk"]
    except Exception as e:
        print(f"Slingshot engine crashed: {e}")
        return

    # -----------------------------------------------------------------
    # PRINT RESULTS
    # -----------------------------------------------------------------
    print("\n" + "="*75)
    print("                     RASTRIGIN BENCHMARK RESULTS                    ")
    print("="*75)
    print(f"Engine          | Execution Time | Mean (X, Y)       | R_hat (X) | Bulk ESS")
    print("-" * 75)
    
    # Extract means for clean printing
    mean_x_n = idata_nuts.posterior["x"].mean().values
    mean_y_n = idata_nuts.posterior["y"].mean().values
    mean_x_s = idata_sling.posterior["x"].mean().values
    mean_y_s = idata_sling.posterior["y"].mean().values
    
    print(f"Standard NUTS   | {time_nuts:<14.2f}s | ({mean_x_n:>5.3f}, {mean_y_n:>5.3f}) | {rhat_x_nuts:<9.3f} | {ess_x_nuts:<8.1f}")
    print(f"Slingshot (PT)  | {time_sling:<14.2f}s | ({mean_x_s:>5.3f}, {mean_y_s:>5.3f}) | {rhat_x_sling:<9.3f} | {ess_x_sling:<8.1f}")
    print("="*75)
    print("True Theoretical Global Mean: (0.000, 0.000)")

    # -----------------------------------------------------------------
    # VISUALIZATION & TRACE PLOTS
    # -----------------------------------------------------------------
    import matplotlib.pyplot as plt
    print("\nGenerating and saving trace plots...")

    # 1. Plot Standard NUTS (The "Harp" Trap)
    try:
        axes_nuts = az.plot_trace(idata_nuts, var_names=["x", "y"])
        fig_nuts = axes_nuts.ravel()[0].figure
        fig_nuts.suptitle("Standard NUTS: Trapped Local Chains", fontsize=16)
        fig_nuts.tight_layout()
        fig_nuts.savefig("rastrigin_trace_nuts.png", dpi=300)
        plt.close(fig_nuts)
        print(" -> Saved 'rastrigin_trace_nuts.png'")
    except Exception as e:
        print(f" -> Could not generate NUTS plot: {e}")

    # 2. Plot Slingshot PT (The "Fuzzy Caterpillar")
    try:
        axes_sling = az.plot_trace(idata_sling, var_names=["x", "y"])
        fig_sling = axes_sling.ravel()[0].figure
        fig_sling.suptitle("Slingshot (PT): Global Convergence", fontsize=16)
        fig_sling.tight_layout()
        fig_sling.savefig("rastrigin_trace_slingshot.png", dpi=300)
        plt.close(fig_sling)
        print(" -> Saved 'rastrigin_trace_slingshot.png'")
    except Exception as e:
        print(f" -> Could not generate Slingshot plot: {e}")

if __name__ == "__main__":
    run_rastrigin_comparison()