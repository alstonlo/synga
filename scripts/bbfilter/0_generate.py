from collections import defaultdict

import jsonargparse
import multiprocess as mp
import numpy as np
import polars as ps
import tqdm

from src import chem, io, ops


def sample_init(lib):
    global _lib

    _lib = chem.SynthesisLibrary(lib)


def sample_tree(idx, seed):
    global _lib
    chem.silence_rdlogger()

    rng = np.random.default_rng(seed + idx)
    while True:
        T = _lib.sample(rng=rng)
        if not chem.check_mol(T.root.mol):
            continue
        return T.product, set(T.blocks)


def sample_dataset(
    lib: str = "chemspace",
    num_trees: int = 10000000,
    num_workers: int = 0,
    seed: int = 0,
    holdout: int = 10000,
):
    rng = np.random.default_rng(seed)

    # Sample trees
    mol2bbs = defaultdict(set)
    pbar = tqdm.tqdm(desc="Sampling trees", total=num_trees)

    with ops.ParallelMap(num_workers, init=sample_init, initargs=[lib]) as pmap:
        while len(mol2bbs) < num_trees:
            m = num_trees - len(mol2bbs)
            for smi, bbs in pmap(sample_tree, range(m), seed=seed, asiter=True):
                if smi not in mol2bbs:
                    pbar.update(1)
                mol2bbs[smi].update(bbs)
            seed += m  # so that we don't get same rng

    mol2bbs = [(m, sorted(mol2bbs[m])) for m in sorted(mol2bbs)]

    # Save as dataframe
    schema = [("mol", ps.String), ("bbs", ps.List(ps.UInt32))]
    df = ps.DataFrame(mol2bbs, schema=schema, orient="row")

    # Split molecules into train and val
    split = np.full(len(df), True, dtype=bool)
    split[:holdout] = False
    rng.shuffle(split)
    df = df.with_columns(ps.Series(name="train", values=split))

    # Finally, write the dataset to disk
    df.write_parquet(io.LIBS_ROOT / lib / f"block_recall.parquet")


if __name__ == "__main__":
    mp.set_start_method("spawn")

    parser = jsonargparse.ArgumentParser()
    parser.add_function_arguments(sample_dataset)
    args = parser.parse_args()

    sample_dataset(**args.as_dict())
