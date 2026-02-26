import jsonargparse
import multiprocess as mp
import numpy as np
import polars as ps
import tqdm

from scripts.utils import format_mean_pm_std
from src import chem, io, ops, oracle


def score_trees(lib, folds, rng):
    chem.silence_rdlogger()

    scores = []
    for _ in tqdm.trange(folds, desc="Sampling"):
        samples = set()
        while len(samples) < 100:
            T = lib.sample(rng=rng)
            if chem.check_mol(T.root.mol):
                samples.add(T.product)
        s = np.mean([oracle.call(smi) for smi in samples])
        scores.append(s)
    return scores


def run_baselines(
    lib: str = "chemspace",
    objective: str = "jnk3",
    num_workers: int = 30,
    filter_topk: int = 1000,
    seed: int = 0,
):
    lib = chem.SynthesisLibrary(lib)
    oracle.init(objective)
    rng = np.random.default_rng(seed)

    # Compute scores of each block
    cache_path = io.LIBS_ROOT / lib.name / f"block_nam_{objective}.npy"
    if not cache_path.exists():
        with ops.ParallelMap(num_workers, init=oracle.init, initargs=[objective]) as pmap:
            scores = pmap(oracle.call, lib.blocks)
            scores = np.asarray(scores, dtype=float)
        np.save(cache_path, scores)
    else:
        scores = np.load(cache_path)

    # Load dataframe
    df = ps.read_parquet(lib.root / "block_nam.parquet")
    df = df.filter((ps.col("objective") == objective) & (ps.col("split") == 1))
    assert len(df) > 0

    # Score
    reduce_fns = ["sum", "avg", "max"]
    df = (
        df.with_columns(
            ps.col("bbs").map_elements(
                lambda bbs: [scores[i] for i in bbs],
                return_dtype=ps.List(ps.Float64),
            )
        )
        .with_columns(
            ps.col("bbs").list.sum().alias("sum"),
            ps.col("bbs").list.mean().alias("avg"),
            ps.col("bbs").list.max().alias("max"),
        )
        .select("fold", "score", *reduce_fns)
    )

    # Compute correlations
    summary = df.group_by("fold").agg(*(
        ps.corr("score", fn, method="spearman").alias(f"{fn}_corr")
        for fn in reduce_fns
    ))
    nfolds = len(summary)

    # Sample trees
    base_scores = score_trees(lib, nfolds, rng=rng)

    k = filter_topk
    subset = np.argpartition(scores, -k)[-k:].tolist()
    lib.restrict(subset, eps=0.1)
    filt_scores = score_trees(lib, nfolds, rng=rng)

    summary = summary.with_columns(
        ps.Series(name="score_base", values=base_scores),
        ps.Series(name="score_filt", values=filt_scores),
    )

    # Summarize
    summary = (
        summary.drop("fold")
        .transpose(include_header=True, header_name="metric")
        .with_columns(ps.concat_list(f"column_{i}" for i in range(nfolds)).alias("value"))
        .with_columns(format_mean_pm_std("value", inagg=False))
        .select("metric", "value")
    )
    print(summary)


if __name__ == "__main__":
    mp.set_start_method("spawn")

    parser = jsonargparse.ArgumentParser()
    parser.add_function_arguments(run_baselines)
    args = parser.parse_args()

    run_baselines(**args.as_dict())
