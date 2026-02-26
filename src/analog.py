from typing import Any, Dict, Optional

import jsonargparse
import multiprocess as mp
import numpy as np
import pandas as pd

from src import chem, io, ops
from src.models.bbfilter import BlockFilter
from src.optim import SynthesisGA
from src.optimize import optimize


def early_stopping(history):
    return history.memory and (max(history.memory.values()) >= 1.0)


def analog_optimize(input, objective, optimizer_params, **kwargs):
    query, top_bbs, eps, seed = input
    query = chem.csmiles(query)

    # Kind of hacky because jsonargparse init is a class method
    parser = jsonargparse.ArgumentParser()
    parser.add_subclass_arguments(SynthesisGA, "optimizer")
    init = parser.instantiate_classes({"optimizer": optimizer_params})
    synga = init.optimizer

    # Restrict to top blocks
    synga.libconfig.update(subset=top_bbs, eps=eps)

    # Optimizer routine
    history = optimize(synga, dict(**objective, ref=query), seed=seed, **kwargs)

    # Take the top analogs
    df = history.table(kwargs["log_samples"], retro=synga.retro)
    df = df.rename(columns={"smiles": "analog"})
    df["query"] = query
    df = df[["query", "analog", "fitness", "route"]]

    # Return results
    best, _ = max(history.memory.items(), key=lambda kv: kv[1])
    return query, best, df


def analog_search(
    optimizer: SynthesisGA,
    logger: io.WandbLogger,
    dataset: str,
    objective: Dict[str, bool] = dict(count=True, murcko=False),
    bbfilter: Optional[BlockFilter] = None,
    log_analogs: int = 10,
    num_workers: int = 0,
    budget: int = 10000,
    seed: int = 0,
    optimizer_params: Optional[Dict[str, Any]] = None,  # don't set manually
):
    config = dict(locals())  # save important input arguments
    config["optimizer"] = optimizer.config
    config["bbfilter"] = bbfilter.config if (bbfilter is not None) else None
    del config["optimizer_params"]
    del config["logger"]

    # Load test set
    if dataset.endswith(".txt"):
        queries = io.readlines(dataset)
    else:
        queries = pd.read_csv(dataset)["smiles"].to_list()
    np.random.default_rng(seed).shuffle(queries)

    # Filter top blocks for each query
    if bbfilter is not None:
        top_bbs = bbfilter(optimizer.hparams.lib, queries)
        eps = bbfilter.eps
    else:
        top_bbs = [None] * len(queries)
        eps = 1.0
    eps = [eps] * len(top_bbs)

    # Analog search: either a single query or sweep over a dataset
    objective = dict(name="analog", **objective)

    logger.log_hyperparams(config)

    pmap = ops.ParallelMap(num_workers)
    results = pmap(
        f=analog_optimize,
        inputs=zip(queries, top_bbs, eps, range(seed, seed + len(queries))),
        objective=objective,
        optimizer_params=optimizer_params,
        logger=io.NoLogger(),
        log_samples=log_analogs,
        early_stopping=early_stopping,
        num_workers=-1,
        budget=budget,
        verbose=False,
        pbar="Searching analog",
        asiter=True,
    )

    samples = []
    metrics = ops.ListOfMetrics()

    for i, (query, analog, df) in enumerate(results):
        m1, m2 = [chem.rdmol(m) for m in [query, analog]]
        sims = {
            "recons": float(chem.csmiles(m1) == chem.csmiles(m2)),
            "morgan": chem.tanimoto_similarity(m1, m2, fp="morgan"),
            "murcko": chem.tanimoto_similarity(m1, m2, fp="morgan", murcko=True),
            "gobbi": chem.dice_similarity(m1, m2, fp="gobbi"),
        }
        m = ops.fix_keys(sims, pre="sim/")
        if top_bbs[i] is not None:
            m["top_bbs"] = len(top_bbs[i])
        metrics.update(m)

        logger.log_metrics(metrics.mean_and_std())
        samples.append(df)

    samples = pd.concat(samples, axis=0)
    logger.log_table("samples", dataframe=pd.DataFrame(samples))
    pmap.shutdown()


if __name__ == "__main__":
    mp.set_start_method("spawn")

    parser = jsonargparse.ArgumentParser()
    parser.add_function_arguments(analog_search)
    args = parser.parse_args()

    init = parser.instantiate_classes(args)
    init.optimizer_params = args.optimizer
    analog_search(**init.as_dict())
