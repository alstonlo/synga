import functools
from collections import OrderedDict
from typing import Callable, Literal, Optional, Union

import jsonargparse
import lightning.pytorch as pl
import multiprocess as mp
import numpy as np
import pandas as pd
import tdc
import tqdm
import wandb

from src import io, ops, oracle
from src.optim import MolecularOptimizer


class History:

    def __init__(self):
        self.memory = OrderedDict()
        self.prev_metrics = None

        self.proposals = 0
        self.has_updates = False
        self.num_repeats = 0

    def __len__(self):
        return len(self.memory)

    def __contains__(self, smiles):
        return smiles in self.memory

    def commit(self, smiles, score):
        if smiles in self:
            self.num_repeats += 1
        else:
            self.memory[smiles] = score
            self.has_updates = True

    def metrics(self, budget):
        sorted_items = sorted(self.memory.items(), key=(lambda kv: kv[1]), reverse=True)
        sorted_smiles, sorted_scores = zip(*sorted_items)

        metrics = {
            "oracle_calls": len(self),
            "proposals": self.proposals,
            "repeats": self.num_repeats,
            "1st": sorted_scores[0],
            "2nd": sorted_scores[1],
            "3rd": sorted_scores[2],
        }

        diversity = tdc.Evaluator(name="diversity")
        for k in [10, 100]:
            mean = np.mean(sorted_scores[:k]).item()

            if self.prev_metrics is None:
                auc = 0.5 * mean * len(self)
            else:
                prev_mean = self.prev_metrics[f"top{k}/mean"]
                prev_auc = self.prev_metrics[f"top{k}/auc"]
                prev_calls = self.prev_metrics["oracle_calls"]
                auc = prev_auc + 0.5 * (mean + prev_mean) * (len(self) - prev_calls)

            metrics[f"top{k}/mean"] = mean
            metrics[f"top{k}/auc"] = auc
            metrics[f"top{k}/diversity"] = diversity(sorted_smiles[:k])

        self.prev_metrics = dict(metrics)
        self.has_updates = False

        # Pessimistic AUC, i.e., if we never discover anything better
        for k in [10, 100]:
            assert len(self) <= budget
            metrics[f"top{k}/auc"] += metrics[f"top{k}/mean"] * (budget - len(self))
            metrics[f"top{k}/auc"] /= budget

        return metrics

    def table(self, topk, retro=None):
        columns = ["smiles", "idx", "fitness"]
        data = [[smi, i, y] for i, (smi, y) in enumerate(self.memory.items())]

        # Append synthesis routes if we are given them
        if retro is not None:
            columns.append("route")
            for row in data:
                row.append(retro[row[0]].postfix)

        df = pd.DataFrame(data, columns=columns)
        if topk is not None:
            df = df.sort_values(by="fitness", ascending=False).head(topk)
        return df


def optimize(
    optimizer: MolecularOptimizer,
    objective: Union[Literal[tuple(oracle.NAMES)], str],
    logger: io.WandbLogger,
    log_every_n_calls: int = 100,  # (!!) changes how AUC is computed, PMO uses 100
    log_samples: int = 0,
    early_stopping: Optional[Callable[[History], bool]] = None,
    num_workers: int = 5,
    budget: int = 10000,
    seed: int = 0,
    verbose: bool = True,
):
    config = dict(locals())  # save important input arguments
    config["optimizer"] = optimizer.config
    del config["logger"]
    logger.log_hyperparams(config)

    # We will use NumPy RNG
    rng = np.random.default_rng(seed)

    # Create a pool
    oracle_init = functools.partial(oracle.init, objective=objective)
    oracle_init(dry=True)  # prevents download race in pmap init

    pmap_init = ops.chain(oracle_init, optimizer.pmap_init)
    pmap = ops.ParallelMap(num_workers, init=pmap_init)

    # Optimization loop
    history = History()
    early_stopping = (lambda h: False) if (early_stopping is None) else early_stopping

    pbar = tqdm.tqdm(total=budget, desc="Optimizing", disable=(not verbose))
    while (len(history) < budget) and not early_stopping(history):
        proposed = optimizer.propose(pmap=pmap, rng=rng)
        scores = pmap(oracle.call, proposed, cache=dict(history.memory))
        scores = list(zip(proposed, scores))
        history.proposals += 1

        for smi, x in scores:
            if len(history) >= budget:
                break
            history.commit(smiles=smi, score=x)
            if history.has_updates and (len(history) % log_every_n_calls == 0):
                logger.log_metrics(history.metrics(budget))
        pbar.update(len(history) - pbar.n)

        optimizer.on_propose_end(scores, logger, pmap=pmap, rng=rng)
    pbar.close()

    pmap.shutdown()
    optimizer.on_end()

    if history.has_updates:
        logger.log_metrics(history.metrics(budget))
    if log_samples:
        df = history.table(topk=log_samples, retro=getattr(optimizer, "retro", None))
        logger.log_table("samples", dataframe=df)
    return history


def optimize_dispatcher(args, init_fn):
    trials, base_seed = args.trials, args.seed
    del args.trials
    del args.seed

    logger_init = args.logger.init_args
    if (trials > 1) and (logger_init.group is None):
        logger_init.group = f"{args.objective}-{wandb.util.generate_id()}"

    for offset in range(trials):
        seed = base_seed + offset
        pl.seed_everything(seed)  # needed to seed torch model init

        init = init_fn(args)
        optimize(**init.as_dict(), seed=seed)
        wandb.finish()


if __name__ == "__main__":
    mp.set_start_method("spawn")

    parser = jsonargparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=1)
    parser.add_function_arguments(optimize)
    args = parser.parse_args()

    optimize_dispatcher(args, init_fn=parser.instantiate_classes)
