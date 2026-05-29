from typing import Callable, NamedTuple, Any
import jax
import jax.numpy as jnp
import blackjax
from blackjax.mcmc.slingshot import init_adaptation, dual_averaging_step
from blackjax.base import SamplingAlgorithm

class TemperedSlingshotState(NamedTuple):
    """Functional atomic state for the Parallel Tempered Slingshot engine."""
    positions: Any
    slingshot_states: Any
    da_states: Any
    logit_r: jnp.ndarray
    betas: jnp.ndarray

class TemperedSlingshotInfo(NamedTuple):
    """Metadata and diagnostics returned at each iteration step."""
    swap_acceptance_rate: jnp.ndarray

def init(initial_positions: Any, logdensity_fn: Callable, num_temperatures: int) -> TemperedSlingshotState:
    """Initialize the global thermal multi-chain grid state from an initial position PyTree."""
    leaves = jax.tree_util.tree_leaves(initial_positions)
    chains = leaves[0].shape[0] if len(leaves) > 0 else 1
    
    init_betas = jnp.array([1.0 / (2.0**i) for i in range(num_temperatures)])
    init_logit_r = jnp.zeros(num_temperatures - 1)
    
    def init_temp_level(beta):
        tempered_fn = lambda theta: beta * logdensity_fn(theta)
        states_level = jax.vmap(lambda p: blackjax.slingshot(tempered_fn, step_size=1.0, num_proposals=1000).init(p))(initial_positions)
        dim = sum(jnp.prod(jnp.array(l.shape[1:])) for l in leaves)
        da_states_level = jax.vmap(lambda ss: init_adaptation(ss, dim))(jnp.ones(chains) * 0.1)
        return states_level, da_states_level
        
    slingshot_states, da_states = jax.vmap(init_temp_level)(init_betas)
    return TemperedSlingshotState(
        positions=slingshot_states.position,
        slingshot_states=slingshot_states,
        da_states=da_states,
        logit_r=init_logit_r,
        betas=init_betas
    )

def build_kernel(
    logdensity_fn: Callable,
    num_temperatures: int,
    proposals: int = 1000,
    target_accept: float = 0.65,
    target_swap_accept: float = 0.30,
    is_warmup: bool = False
) -> Callable:
    """Build a pure, compiled JAX Markov transition kernel featuring Mass Matrix Pooling."""
    
    def one_step(rng_key: jax.Array, state: TemperedSlingshotState) -> tuple[TemperedSlingshotState, TemperedSlingshotInfo]:
        slingshot_states = state.slingshot_states
        da_states = state.da_states
        logit_r = state.logit_r
        betas = state.betas
        
        leaves = jax.tree_util.tree_leaves(slingshot_states.position)
        chains = leaves[0].shape[1]
        
        sample_key, swap_key = jax.random.split(rng_key)
        
        if is_warmup:
            r = jax.nn.sigmoid(logit_r)
            betas_list = [1.0]
            for idx in range(num_temperatures - 1):
                betas_list.append(betas_list[-1] * r[idx])
            betas = jnp.array(betas_list)

        def single_temp_step(beta, states_level, da_states_level, keys_level):
            tempered_fn = lambda theta: beta * logdensity_fn(theta)
            def single_chain_step(key, s, da):
                step_size = jnp.exp(da.log_step_size) if is_warmup else jnp.exp(da.log_step_size_bar)
                algo = blackjax.slingshot(tempered_fn, step_size=step_size, num_proposals=proposals, cholesky=da.cholesky)
                next_s, info = algo.step(key, s)
                
                if is_warmup:
                    acc_rate = getattr(info, "acceptance_rate", target_accept)
                    next_da = dual_averaging_step(da, acc_rate, next_s.position, target_rate=target_accept)
                    min_log_step = jnp.log(0.05)
                    next_da = next_da._replace(
                        log_step_size=jnp.maximum(next_da.log_step_size, min_log_step),
                        log_step_size_bar=jnp.maximum(next_da.log_step_size_bar, min_log_step)
                    )
                else:
                    next_da = da
                return next_s, next_da
            return jax.vmap(single_chain_step)(keys_level, states_level, da_states_level)

        keys = jax.random.split(sample_key, num_temperatures * chains).reshape(num_temperatures, chains, 2)
        next_states, next_da_states = jax.vmap(single_temp_step)(betas, slingshot_states, da_states, keys)
        
        # --- MASS MATRIX POOLING SUBROUTINE ---
        if is_warmup:
            # Extract the stabilized, structured Cholesky factor from the cold chain (Index 0)
            cold_cholesky = next_da_states.cholesky[0]  # Shape: (chains, dim, dim)
            cold_cholesky_expanded = jnp.expand_dims(cold_cholesky, 0)  # Shape: (1, chains, dim, dim)
            
            # Broadcast inverse temperatures across tensor dimensions
            betas_expanded = betas[:, None, None, None]  # Shape: (num_temperatures, 1, 1, 1)
            
            # Execute beta-weighted shrinkage regularization across the full thermal grid
            pooled_cholesky = betas_expanded * next_da_states.cholesky + (1.0 - betas_expanded) * cold_cholesky_expanded
            next_da_states = next_da_states._replace(cholesky=pooled_cholesky)
        
        # 3. Replica Exchange Swap Executions
        step_swaps = jnp.zeros(num_temperatures - 1)
        for i in range(num_temperatures - 1):
            j = i + 1
            state_i = jax.tree_util.tree_map(lambda x: x[i], next_states)
            state_j = jax.tree_util.tree_map(lambda x: x[j], next_states)
            logp_i = jax.vmap(logdensity_fn)(state_i.position)
            logp_j = jax.vmap(logdensity_fn)(state_j.position)
            
            log_alpha = (betas[i] - betas[j]) * (logp_j - logp_i)
            mean_p_accept = jnp.mean(jnp.minimum(1.0, jnp.exp(log_alpha)))
            
            swap_key, subkey = jax.random.split(swap_key)
            do_swap = jnp.log(jax.random.uniform(subkey, shape=(chains,))) < log_alpha
            step_swaps = step_swaps.at[i].set(jnp.mean(do_swap.astype(jnp.float32)))
            
            def update_full_tree(full_leaf, leaf_i, leaf_j):
                mask = jnp.reshape(do_swap, (chains,) + (1,) * (leaf_i.ndim - 1))
                return full_leaf.at[i].set(jnp.where(mask, leaf_j, leaf_i)).at[j].set(jnp.where(mask, leaf_i, leaf_j))
            next_states = jax.tree_util.tree_map(update_full_tree, next_states, state_i, state_j)
            
            if is_warmup:
                logit_r = logit_r.at[i].add(- (1.0 / jnp.power(100, 0.6)) * (mean_p_accept - target_swap_accept))
                
        return TemperedSlingshotState(
            positions=next_states.position,
            slingshot_states=next_states,
            da_states=next_da_states,
            logit_r=logit_r,
            betas=betas
        ), TemperedSlingshotInfo(swap_acceptance_rate=step_swaps)
        
    return one_step

def tempered_slingshot(
    logdensity_fn: Callable,
    num_temperatures: int,
    proposals: int = 1000,
    target_accept: float = 0.65,
    target_swap_accept: float = 0.30,
    is_warmup: bool = False
) -> SamplingAlgorithm:
    """Exposes the standalone functional algorithm specification to the top-level API."""
    kernel = build_kernel(logdensity_fn, num_temperatures, proposals, target_accept, target_swap_accept, is_warmup)
    return SamplingAlgorithm(
        init=lambda init_pos: init(init_pos, logdensity_fn, num_temperatures),
        step=kernel
    )
