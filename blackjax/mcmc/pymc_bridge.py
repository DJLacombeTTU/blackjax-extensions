import numpy as np
import jax
import jax.numpy as jnp
import scipy.optimize
import arviz as az
from pymc.sampling.jax import get_jaxified_logp
from blackjax.mcmc.tempered_slingshot import tempered_slingshot

def sample_slingshot(
    pymc_model, 
    draws=1000, 
    tune=1000, 
    chains=16, 
    proposals=1000, 
    target_accept=0.65, 
    target_swap_accept=0.30,
    num_temperatures=1,
    random_seed=42
):
    """High-level user API bridging PyMC model extraction to the standalone core engine."""
    dim = len(pymc_model.value_vars)
    var_names = [v.name for v in pymc_model.value_vars]
    
    # 1. PPL-Specific Extraction Work
    raw_logp = get_jaxified_logp(pymc_model, negative_logp=True)
    logdensity_fn = lambda theta: raw_logp([theta[i] for i in range(dim)])
    
    # 2. Local MAP Initialization Optimization
    def neg_log_density(theta): return -logdensity_fn(theta)
    val_and_grad_fn = jax.jit(jax.value_and_grad(neg_log_density))
    def scipy_objective(theta_np):
        val, grad = val_and_grad_fn(jnp.array(theta_np))
        return np.array(val).astype(np.float64), np.array(grad).astype(np.float64)
        
    opt_result = scipy.optimize.minimize(scipy_objective, jnp.zeros(dim), method="BFGS", jac=True)
    map_estimate = jnp.array(opt_result.x)
    
    rng_key = jax.random.PRNGKey(random_seed)
    init_key, warmup_key, sample_key = jax.random.split(rng_key, 3)
    warm_start_positions = map_estimate + jax.random.normal(init_key, (chains, dim)) * 0.01

    if num_temperatures == 1:
        # Standard Single-Chain Fallback Path
        import blackjax
        from blackjax.mcmc.slingshot import init_adaptation, dual_averaging_step
        states = jax.vmap(lambda p: blackjax.slingshot(logdensity_fn, step_size=1.0, num_proposals=proposals).init(p))(warm_start_positions)
        da_states = jax.vmap(lambda ss: init_adaptation(ss, dim))(jnp.ones(chains) * 0.1)
        
        @jax.jit
        def warmup_step(carry, step_key):
            states, da_states = carry
            keys = jax.random.split(step_key, chains)
            def single_chain_warmup(key, state, da_state):
                step_size = jnp.exp(da_state.log_step_size)
                algo = blackjax.slingshot(logdensity_fn, step_size=step_size, num_proposals=proposals, cholesky=da_state.cholesky)
                next_state, info = algo.step(key, state)
                acc_rate = getattr(info, "acceptance_rate", target_accept)
                next_da_state = dual_averaging_step(da_state, acc_rate, next_state.position, target_rate=target_accept)
                return next_state, next_da_state
            return jax.vmap(single_chain_warmup)(keys, states, da_states), None

        @jax.jit
        def sample_step(carry_states, step_key):
            keys = jax.random.split(step_key, chains)
            next_states, _ = jax.vmap(lambda k, s: blackjax.slingshot(logdensity_fn, step_size=jnp.exp(da_states.log_step_size_bar), num_proposals=proposals, cholesky=da_states.cholesky).step(k, s))(keys, carry_states)
            return next_states, next_states.position

        (states, da_states), _ = jax.lax.scan(warmup_step, (states, da_states), jax.random.split(warmup_key, tune))
        _, positions = jax.lax.scan(sample_step, states, jax.random.split(sample_key, draws))
        swap_rates_out = None
    else:
        # 3. Utilizing the Decoupled Standalone Core Sampler
        warmup_algo = tempered_slingshot(logdensity_fn, num_temperatures, proposals, target_accept, target_swap_accept, is_warmup=True)
        production_algo = tempered_slingshot(logdensity_fn, num_temperatures, proposals, target_accept, target_swap_accept, is_warmup=False)
        
        # Initialize Core State
        state = warmup_algo.init(warm_start_positions)
        
        # Execute Warmup Scan
        warmup_keys = jax.random.split(warmup_key, tune)
        state, _ = jax.lax.scan(lambda s, k: warmup_algo.step(k, s), state, warmup_keys)
        
        # Execute Production Scan
        sample_keys = jax.random.split(sample_key, draws)
        state, (positions_history, swap_history) = jax.lax.scan(lambda s, k: production_algo.step(k, s), state, sample_keys)
        
        # Isolate target cold-chain positions (Beta = 1.0)
        positions = positions_history[:, 0, :, :]
        swap_rates_out = np.repeat(np.array(swap_history)[np.newaxis, :, :], chains, axis=0)

    # 4. ArviZ Structural Output Map Assembly
    posterior_dict = {name: np.swapaxes(positions[:, :, idx], 0, 1) for idx, name in enumerate(var_names)}
    sample_stats = {"swap_acceptance_rate": swap_rates_out} if swap_rates_out is not None else None
    return az.from_dict(posterior=posterior_dict, sample_stats=sample_stats)
