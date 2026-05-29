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
    """High-level bridge deploying Dual-Tree Segregation for perfect XLA compilation."""
    all_vars = pymc_model.value_vars
    cont_vars = pymc_model.continuous_value_vars
    disc_vars = pymc_model.discrete_value_vars
    var_names = [v.name for v in all_vars]
    
    raw_logp = get_jaxified_logp(pymc_model)
    logdensity_fn = lambda tree_dict: raw_logp([tree_dict[name] for name in var_names])
    
    init_point = pymc_model.initial_point()
    base_pytree = {}
    for v in cont_vars:
        base_pytree[v.name] = jnp.array(init_point[v.name], dtype=jnp.float64)
    for v in disc_vars:
        base_pytree[v.name] = jnp.array(init_point[v.name], dtype=jnp.int32)
    
    cont_sizes = [base_pytree[v.name].size for v in cont_vars]
    total_cont_dim = sum(cont_sizes)
    
    def pack_cont(tree):
        return jnp.concatenate([tree[v.name].flatten() for v in cont_vars]) if cont_vars else jnp.array([0.0])
        
    def unpack_cont_to_tree(c_flat, target_tree):
        new_tree = dict(target_tree)
        curr = 0
        for v, size in zip(cont_vars, cont_sizes):
            flat_slice = c_flat[curr:curr+size]
            new_tree[v.name] = flat_slice.reshape(target_tree[v.name].shape)
            curr += size
        return new_tree

    def opt_objective(c_np):
        val, grad = jax.jit(jax.value_and_grad(lambda c: -logdensity_fn(unpack_cont_to_tree(c, base_pytree))))(jnp.array(c_np))
        return np.array(val).astype(np.float64), np.array(grad).astype(np.float64)
        
    init_c_flat = pack_cont(base_pytree)
    if total_cont_dim > 0:
        opt_result = scipy.optimize.minimize(opt_objective, init_c_flat, method="BFGS", jac=True)
        optimized_tree = unpack_cont_to_tree(jnp.array(opt_result.x), base_pytree)
    else:
        optimized_tree = base_pytree
    
    rng_key = jax.random.PRNGKey(random_seed)
    init_key, warmup_key, sample_key = jax.random.split(rng_key, 3)
    jitter_keys = jax.random.split(init_key, chains)
    
    # Generate explicit, separated initial states for continuous and discrete components
    def generate_chain_start(key):
        c_tree, d_tree = {}, {}
        for v in all_vars:
            if v in cont_vars:
                c_tree[v.name] = optimized_tree[v.name] + jax.random.normal(key, shape=optimized_tree[v.name].shape) * 0.01
            else:
                d_tree[v.name] = optimized_tree[v.name]
        return c_tree, d_tree
        
    initial_cont, initial_disc = jax.vmap(generate_chain_start)(jitter_keys)

    warmup_algo = tempered_slingshot(logdensity_fn, num_temperatures, proposals, target_accept, target_swap_accept, is_warmup=True)
    production_algo = tempered_slingshot(logdensity_fn, num_temperatures, proposals, target_accept, target_swap_accept, is_warmup=False)
    
    # Hand off the explicit Dual-Tree tuple to the native core
    state = warmup_algo.init((initial_cont, initial_disc))
    state, _ = jax.lax.scan(lambda s, k: warmup_algo.step(k, s), state, jax.random.split(warmup_key, tune))
    state, info_history = jax.lax.scan(lambda s, k: production_algo.step(k, s), state, jax.random.split(sample_key, draws))
    
    cold_chain_cont = jax.tree_util.tree_map(lambda x: x[:, 0, ...], info_history.position_cont)
    cold_chain_disc = jax.tree_util.tree_map(lambda x: x[:, 0, ...], info_history.position_disc)
    swap_history = info_history.swap_acceptance_rate
    
    posterior_dict = {}
    for v in all_vars:
        if v in cont_vars:
            raw_data = np.array(cold_chain_cont[v.name])
        else:
            raw_data = np.array(cold_chain_disc[v.name])
        posterior_dict[v.name] = np.swapaxes(raw_data, 0, 1)
                    
    swap_rates_out = np.repeat(np.array(swap_history)[np.newaxis, :, :], chains, axis=0)
    return az.from_dict(posterior=posterior_dict, sample_stats={"swap_acceptance_rate": swap_rates_out})
