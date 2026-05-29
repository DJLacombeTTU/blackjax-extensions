from typing import Callable, NamedTuple, Any, Dict
import jax
import jax.numpy as jnp
import numpy as np
import blackjax
from blackjax.mcmc.slingshot import init_adaptation, dual_averaging_step
from blackjax.base import SamplingAlgorithm

class TemperedSlingshotState(NamedTuple):
    position_cont: Any
    position_disc: Any
    slingshot_states: Any
    da_states: Any
    logit_r: jnp.ndarray
    betas: jnp.ndarray

class TemperedSlingshotInfo(NamedTuple):
    position_cont: Any
    position_disc: Any
    swap_acceptance_rate: jnp.ndarray

def init(initial_positions: tuple, logdensity_fn: Callable, num_temperatures: int) -> TemperedSlingshotState:
    initial_cont, initial_disc = initial_positions
    leaves = jax.tree_util.tree_leaves(initial_cont)
    chains = leaves[0].shape[0] if len(leaves) > 0 else 1
    
    init_betas = jnp.array([1.0 / (2.0**i) for i in range(num_temperatures)])
    init_logit_r = jnp.zeros(num_temperatures - 1)
    
    # Extract a single chain's PyTree to build the flatten/unflatten utility
    single_c = jax.tree_util.tree_map(lambda x: x[0], initial_cont)
    flat_c_dummy, unravel_fn = jax.flatten_util.ravel_pytree(single_c)
    cont_dim = flat_c_dummy.shape[0]

    def init_temp_level(beta):
        def single_chain_init(c, d):
            flat_c, _ = jax.flatten_util.ravel_pytree(c)
            # Wrap the logdensity to accept flat 1D arrays from Slingshot and unflatten them
            def tempered_fn_flat(flat_c_in):
                c_tree = unravel_fn(flat_c_in)
                return beta * logdensity_fn({**c_tree, **d})
            return blackjax.slingshot(tempered_fn_flat, step_size=1.0, num_proposals=1000).init(flat_c)
        
        states_level = jax.vmap(single_chain_init)(initial_cont, initial_disc)
        da_states_level = jax.vmap(lambda ss: init_adaptation(ss, cont_dim))(jnp.ones(chains) * 0.1)
        return states_level, da_states_level
        
    slingshot_states, da_states = jax.vmap(init_temp_level)(init_betas)
    
    grid_cont = jax.tree_util.tree_map(lambda x: jnp.repeat(x[None, ...], num_temperatures, axis=0), initial_cont)
    grid_disc = jax.tree_util.tree_map(lambda x: jnp.repeat(x[None, ...], num_temperatures, axis=0), initial_disc)
    
    return TemperedSlingshotState(
        position_cont=grid_cont,
        position_disc=grid_disc,
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
    
    def one_step(rng_key: jax.Array, state: TemperedSlingshotState) -> tuple[TemperedSlingshotState, TemperedSlingshotInfo]:
        grid_cont = state.position_cont
        grid_disc = state.position_disc
        slingshot_states = state.slingshot_states
        da_states = state.da_states
        logit_r = state.logit_r
        betas = state.betas
        
        leaves = jax.tree_util.tree_leaves(grid_cont)
        chains = leaves[0].shape[1]
        sample_key, swap_key = jax.random.split(rng_key)
        
        # Dynamically reconstruct unravel function for this step
        single_c = jax.tree_util.tree_map(lambda x: x[0, 0], grid_cont)
        _, unravel_fn = jax.flatten_util.ravel_pytree(single_c)
        
        if is_warmup:
            r = jax.nn.sigmoid(logit_r)
            betas_list = [1.0]
            for idx in range(num_temperatures - 1):
                betas_list.append(betas_list[-1] * r[idx])
            betas = jnp.array(betas_list)

        def single_temp_step(beta, states_level, cont_level, disc_level, da_states_level, keys_level):
            def single_chain_step(key, s_cont_flat, p_cont, p_disc, da):
                key_disc, key_cont = jax.random.split(key)
                
                # --- PHASE 1: DISCRETE METROPOLIS RANDOM WALK ---
                def run_discrete_step():
                    proposal_delta = jax.tree_util.tree_map(
                        lambda x: jax.random.choice(key_disc, jnp.array([-1, 0, 1], dtype=jnp.int32), shape=x.shape),
                        p_disc
                    )
                    p_disc_prop = jax.tree_util.tree_map(lambda x, d: x + d, p_disc, proposal_delta)
                    
                    logp_curr = beta * logdensity_fn({**p_cont, **p_disc})
                    logp_prop = beta * logdensity_fn({**p_cont, **p_disc_prop})
                    logp_prop = jnp.nan_to_num(logp_prop, nan=-jnp.inf)
                    
                    accept_disc = jnp.log(jax.random.uniform(key_disc)) < (logp_prop - logp_curr)
                    return jax.tree_util.tree_map(lambda prop, curr: jnp.where(accept_disc, prop, curr), p_disc_prop, p_disc)
                
                has_discrete = len(jax.tree_util.tree_leaves(p_disc)) > 0
                next_p_disc = jax.lax.cond(has_discrete, run_discrete_step, lambda: p_disc)
                
                # --- PHASE 2: CONTINUOUS SLINGSHOT DRIFT ---
                def cond_logdensity_flat(flat_c_in):
                    c_tree = unravel_fn(flat_c_in)
                    return beta * logdensity_fn({**c_tree, **next_p_disc})
                
                step_size = jnp.exp(da.log_step_size) if is_warmup else jnp.exp(da.log_step_size_bar)
                algo = blackjax.slingshot(cond_logdensity_flat, step_size=step_size, num_proposals=proposals, cholesky=da.cholesky)
                
                # Slingshot processes the completely flat state
                next_s_cont_flat, info = algo.step(key_cont, s_cont_flat)
                
                # Unravel the position back into a PyTree dictionary for the global state
                next_p_cont = unravel_fn(next_s_cont_flat.position)
                
                if is_warmup:
                    acc_rate = getattr(info, "acceptance_rate", target_accept)
                    next_da = dual_averaging_step(da, acc_rate, next_s_cont_flat.position, target_rate=target_accept)
                    min_log_step = jnp.log(0.05)
                    next_da = next_da._replace(
                        log_step_size=jnp.maximum(next_da.log_step_size, min_log_step),
                        log_step_size_bar=jnp.maximum(next_da.log_step_size_bar, min_log_step)
                    )
                else:
                    next_da = da
                    
                return next_s_cont_flat, next_p_cont, next_p_disc, next_da
                
            return jax.vmap(single_chain_step)(keys_level, states_level, cont_level, disc_level, da_states_level)

        keys = jax.random.split(sample_key, num_temperatures * chains).reshape(num_temperatures, chains, 2)
        next_states, next_grid_cont, next_grid_disc, next_da_states = jax.vmap(single_temp_step)(betas, slingshot_states, grid_cont, grid_disc, da_states, keys)
        
        # --- PHASE 3: MASS MATRIX POOLING ---
        if is_warmup:
            cold_cholesky = next_da_states.cholesky[0]
            cold_cholesky_expanded = jnp.expand_dims(cold_cholesky, 0)
            betas_expanded = betas[:, None, None, None]
            pooled_cholesky = betas_expanded * next_da_states.cholesky + (1.0 - betas_expanded) * cold_cholesky_expanded
            next_da_states = next_da_states._replace(cholesky=pooled_cholesky)
            
        # --- PHASE 4: JOINT REPLICA EXCHANGE ---
        step_swaps = jnp.zeros(num_temperatures - 1)
        for i in range(num_temperatures - 1):
            j = i + 1
            state_i = jax.tree_util.tree_map(lambda x: x[i], next_states)
            state_j = jax.tree_util.tree_map(lambda x: x[j], next_states)
            cont_i = jax.tree_util.tree_map(lambda x: x[i], next_grid_cont)
            cont_j = jax.tree_util.tree_map(lambda x: x[j], next_grid_cont)
            disc_i = jax.tree_util.tree_map(lambda x: x[i], next_grid_disc)
            disc_j = jax.tree_util.tree_map(lambda x: x[j], next_grid_disc)
            
            logp_i = jax.vmap(lambda c, d: logdensity_fn({**c, **d}))(cont_i, disc_i)
            logp_j = jax.vmap(lambda c, d: logdensity_fn({**c, **d}))(cont_j, disc_j)
            
            log_alpha = (betas[i] - betas[j]) * (logp_j - logp_i)
            mean_p_accept = jnp.mean(jnp.minimum(1.0, jnp.exp(log_alpha)))
            
            swap_key, subkey = jax.random.split(swap_key)
            do_swap = jnp.log(jax.random.uniform(subkey, shape=(chains,))) < log_alpha
            step_swaps = step_swaps.at[i].set(jnp.mean(do_swap.astype(jnp.float32)))
            
            def update_full_tree(full_leaf, leaf_i, leaf_j):
                mask = jnp.reshape(do_swap, (chains,) + (1,) * (leaf_i.ndim - 1))
                return full_leaf.at[i].set(jnp.where(mask, leaf_j, leaf_i)).at[j].set(jnp.where(mask, leaf_i, leaf_j))
                
            next_states = jax.tree_util.tree_map(update_full_tree, next_states, state_i, state_j)
            next_grid_cont = jax.tree_util.tree_map(update_full_tree, next_grid_cont, cont_i, cont_j)
            next_grid_disc = jax.tree_util.tree_map(update_full_tree, next_grid_disc, disc_i, disc_j)
            
            if is_warmup:
                logit_r = logit_r.at[i].add(- (1.0 / jnp.power(100, 0.6)) * (mean_p_accept - target_swap_accept))
                
        return TemperedSlingshotState(
            position_cont=next_grid_cont,
            position_disc=next_grid_disc,
            slingshot_states=next_states,
            da_states=next_da_states,
            logit_r=logit_r,
            betas=betas
        ), TemperedSlingshotInfo(position_cont=next_grid_cont, position_disc=next_grid_disc, swap_acceptance_rate=step_swaps)
        
    return one_step

def tempered_slingshot(
    logdensity_fn: Callable,
    num_temperatures: int,
    proposals: int = 1000,
    target_accept: float = 0.65,
    target_swap_accept: float = 0.30,
    is_warmup: bool = False
) -> SamplingAlgorithm:
    kernel = build_kernel(logdensity_fn, num_temperatures, proposals, target_accept, target_swap_accept, is_warmup)
    return SamplingAlgorithm(
        init=lambda init_pos: init(init_pos, logdensity_fn, num_temperatures),
        step=kernel
    )
