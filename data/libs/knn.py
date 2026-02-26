import jax
import jax.numpy as jnp
import jsonargparse
import multiprocess as mp
import numpy as np
import tqdm

from src import chem


@jax.jit
def knn(X, idx, minsim=0.6, mink=100):  # (N D) int
    q = X[idx]  # (D)
    sum_mins = jnp.minimum(q, X).sum(axis=-1, dtype=jnp.float16)  # (N)
    sum_maxs = jnp.maximum(q, X).sum(axis=-1, dtype=jnp.float16)  # (N)
    sims = sum_mins / sum_maxs
    sims = sims.at[idx].set(-1.0)

    k = jnp.sum(sims >= minsim, dtype=jnp.int32).clip(min=mink)
    indices = jnp.argsort(sims, descending=True)
    return sims[indices], indices, k


def compute_neighbors(lib: str = "chemspace"):
    lib = chem.SynthesisLibrary(lib)
    X = chem.SynthesisLibrary.read_fingerprints(lib.name)

    sims, nbs, sizes = [], [], []
    for idx in tqdm.trange(X.shape[0], desc="KNN"):
        values, indices, k = knn(X, idx=idx)
        sims.append(values[:k])
        nbs.append(indices[:k])
        sizes.append(k)
    sims, nbs = np.concatenate(sims), np.concatenate(nbs)
    sizes = np.asarray(sizes)

    fpknn_path = lib.root / "block_knn.npz"
    np.savez(fpknn_path, fps=X, knn_sims=sims, knn_nbs=nbs, knn_sizes=sizes)


if __name__ == "__main__":
    mp.set_start_method("spawn")

    parser = jsonargparse.ArgumentParser()
    parser.add_function_arguments(compute_neighbors)
    args = parser.parse_args()

    compute_neighbors(**args.as_dict())
