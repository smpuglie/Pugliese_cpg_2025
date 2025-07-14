from jax import random
import jax.numpy as jnp
from omegaconf import DictConfig, OmegaConf


def shuffle_W(W, seed, idxs, independent: bool = False):
    key = random.key(seed)
    return W.at[:, idxs].set(
        random.permutation(key, W[:, idxs], axis=1, independent=independent)
    )


def full_shuffle(
    W,
    seedOfSeeds,
    excDnIdxs,
    inhDnIdxs,
    excInIdxs,
    inhInIdxs,
    mnIdxs,
    independent: bool = False,
):
    keyOfKeys = random.key(seedOfSeeds)
    seeds = random.randint(keyOfKeys, 5, 0, 100000)

    W = shuffle_W(W, seeds[0], excDnIdxs, independent)
    W = shuffle_W(W, seeds[1], inhDnIdxs, independent)
    W = shuffle_W(W, seeds[2], excInIdxs, independent)
    W = shuffle_W(W, seeds[3], inhInIdxs, independent)
    return shuffle_W(W, seeds[4], mnIdxs, independent)


def extract_shuffle_indices(
    W_table,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
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

    return exc_dn_idxs, inh_dn_idxs, exc_in_idxs, inh_in_idxs, mn_idxs
