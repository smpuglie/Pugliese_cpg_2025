from jax import random
import jax.numpy as jnp
from omegaconf import DictConfig, OmegaConf


def shuffle_W(W, key, idxs, independent: bool = False):
    
    return W.at[:, idxs].set(
        random.permutation(key, W[:, idxs], axis=1, independent=independent)
    )


def full_shuffle(
    W,
    base_key,
    excDnIdxs,
    inhDnIdxs,
    excInIdxs,
    inhInIdxs,
    mnIdxs,
    independent: bool = False,
):
    keys = random.split(base_key, 5)

    W = shuffle_W(W, keys[0], excDnIdxs, independent)
    W = shuffle_W(W, keys[1], inhDnIdxs, independent)
    W = shuffle_W(W, keys[2], excInIdxs, independent)
    W = shuffle_W(W, keys[3], inhInIdxs, independent)
    return shuffle_W(W, keys[4], mnIdxs, independent)


def extract_shuffle_indices(
    W_table,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    if 'predictedNt' in W_table.columns: 
        exc_dn_idxs = jnp.array(
            W_table.loc[
                (W_table["class"] == "descending neuron")
                & (W_table["predictedNt"] == "acetylcholine")
            ].index
        )
        inh_dn_idxs = jnp.array(
            W_table.loc[
                (W_table["class"] == "descending neuron")
                & ~(W_table["predictedNt"] == "acetylcholine")
            ].index
        )
        exc_in_idxs = jnp.array(
            W_table.loc[
                ~(W_table["class"].isin(["descending neuron", "motor neuron"]))
                & (W_table["predictedNt"] == "acetylcholine")
            ].index
        )
        inh_in_idxs = jnp.array(
            W_table.loc[
                ~(W_table["class"].isin(["descending neuron", "motor neuron"]))
                & ~(W_table["predictedNt"] == "acetylcholine")
            ].index
        )
        mn_idxs = jnp.array(W_table.loc[W_table["class"] == "motor neuron"].index)
    elif 'sign' in W_table.columns:
        exc_dn_idxs = jnp.array(
            W_table.loc[
                (W_table["class"] == "descending neuron")
                & (W_table["sign"] == 1)
            ].index
        )
        inh_dn_idxs = jnp.array(
            W_table.loc[
                (W_table["class"] == "descending neuron")
                & (W_table["sign"] == -1)
            ].index
        )
        exc_in_idxs = jnp.array(
            W_table.loc[
                (W_table["class"].isin(["intrinsic neuron"]))
                & (W_table["sign"] == 1)
            ].index
        )
        inh_in_idxs = jnp.array(
            W_table.loc[
                (W_table["class"].isin(["intrinsic neuron"]))
                & (W_table["sign"] == -1)
            ].index
        )
        mn_idxs = jnp.array(W_table.loc[W_table["class"] == "motor neuron"].index)
    else:
        print('No valid indices found')
        exc_dn_idxs = inh_dn_idxs = exc_in_idxs = inh_in_idxs = mn_idxs = jnp.array([])
    
    return exc_dn_idxs, inh_dn_idxs, exc_in_idxs, inh_in_idxs, mn_idxs
