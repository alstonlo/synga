from typing import Literal

import jsonargparse
import multiprocess as mp
import numpy as np
import pandas as pd
import tqdm
from rdkit.Chem import DataStructs

from src import chem, io, ops
from src.optim import MolecularOptimizer


def fitness(vina, qed):
    return 0.5 * (-0.1 * min(vina, 0)) + 0.5 * qed


class DockHistory:

    def __init__(self):
        self.memory = dict()
        self.prev_metrics = None

        self.proposals = 0
        self.has_updates = False
        self.num_repeats = 0

        self.fps = dict()

    def __len__(self):
        return len(self.memory)

    def __contains__(self, smiles):
        return smiles in self.memory

    def commit(self, smiles, vina, qed):
        if smiles in self:
            self.num_repeats += 1
        else:
            self.memory[smiles] = (vina, qed)
            self.fps[smiles] = chem.fingerprint(smiles, params="rdkit")
            self.has_updates = True

    def modes(self):
        sorted_smiles = sorted(self.memory, key=(lambda k: self.memory[k][0]))
        # (!) vina score should be minimized

        modes = []
        for smi in sorted_smiles:
            vina, qed = self.memory[smi]
            if len(modes) >= 100:
                break
            if qed <= 0.5:
                continue
            sims = DataStructs.BulkTanimotoSimilarity(self.fps[smi], [self.fps[m] for m in modes])
            if modes and max(sims) >= 0.5:
                continue
            modes.append(smi)

        self.has_updates = False
        return [(m, *self.memory[m]) for m in modes]

    def metrics(self):
        _, vinas, qeds = zip(*self.modes())

        # Also keep track of top-k fitness to make sure GA is working
        k = 1000
        scores = [fitness(v, q) for v, q in self.memory.values()]
        if len(scores) > k:
            scores = np.asarray(scores)
            scores = scores[np.argpartition(scores, -k)[-k:]]

        return {
            "oracle_calls": len(self),
            "proposals": self.proposals,
            "repeats": self.num_repeats,
            "fitness": np.mean(scores),
            "modes/vina": np.mean(vinas),
            "modes/qed": np.mean(qeds),
            "modes/size": len(vinas),
        }

    def table(self, retro=None):
        columns = ["smiles", "vina", "qed"]
        data = [[smi, ds, q] for smi, ds, q in self.modes()]

        # Append synthesis routes if we are given them
        if retro is not None:
            columns.append("route")
            for row in data:
                row.append(retro[row[0]].postfix)

        df = pd.DataFrame(data, columns=columns)
        df = df.sort_values(by="vina", ascending=False)
        return df


def optimize(
    optimizer: MolecularOptimizer,
    receptor: Literal[tuple(chem.LITPCBA_RECEPTORS)],
    logger: io.WandbLogger,
    log_every_n_calls: int = 500,
    num_workers: int = 5,
    budget: int = 64000,
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
    pmap = ops.ParallelMap(num_workers, init=optimizer.pmap_init)

    # Optimization loop
    unidock = chem.UniDocker(receptor)
    history = DockHistory()

    pbar = tqdm.tqdm(total=budget, desc="Optimizing", disable=(not verbose))
    while len(history) < budget:
        proposed = optimizer.propose(pmap=pmap, rng=rng)
        dockscores = unidock(proposed, pmap=pmap)
        qeds = [chem.qed(smi) for smi in proposed]
        history.proposals += 1

        scores = []
        for smi, v, q in zip(proposed, dockscores, qeds):
            x = fitness(vina=v, qed=q)
            if len(history) >= budget:
                break
            history.commit(smiles=smi, vina=v, qed=q)
            if history.has_updates and (len(history) % log_every_n_calls == 0):
                logger.log_metrics(history.metrics())
            scores.append(x)
        pbar.update(len(history) - pbar.n)

        scores = list(zip(proposed, scores))
        optimizer.on_propose_end(scores, logger, pmap=pmap, rng=rng)
    pbar.close()

    pmap.shutdown()
    optimizer.on_end()

    if history.has_updates:
        logger.log_metrics(history.metrics())
    df = history.table(retro=getattr(optimizer, "retro", None))
    logger.log_table("samples", dataframe=df)


if __name__ == "__main__":
    mp.set_start_method("spawn")

    parser = jsonargparse.ArgumentParser()
    parser.add_function_arguments(optimize)
    args = parser.parse_args()

    init = parser.instantiate_classes(args)
    optimize(**init.as_dict())
