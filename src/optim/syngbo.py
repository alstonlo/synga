import time
from typing import Literal

import gpytorch
import numpy as np
import torch
import torch_geometric as pyg
import tqdm
from torch.utils.data import random_split

from src import chem, ops
from src.models.modules import TanimotoGP
from src.models.nam import LitNAM, NAMDataset
from src.models.trainers import SimpleTrainer
from src.optim.base import MolecularOptimizer
from src.optim.synga import SynthesisGA


def subset(scores, top, rand, rng, keys=None):
    if keys is None:
        keys = scores
    if len(keys) <= top + rand:
        return sorted(keys)
    items = ops.rank_by_value(keys, scores.get, rng=rng)
    rand = ops.choices(items[top:], n=rand, rng=rng) if (rand > 0) else []
    return items[:top] + rand


class SynthesisGBO(MolecularOptimizer):

    def __init__(
        self,
        lib: str = "chemspace",
        initial_size: int = 10,
        propose_size: int = 10,
        nam_train_frequency: int = 25,
        nam_batch_size: int = 50,
        nam_subset_topk: int = 1000,
        gp_train_samples: int = 5000,
        synga_generations: int = 5,
        synga_founder_size: int = 500,
        synga_population_size: int = 1000,
        synga_offspring_size: int = 100,
        synga_offspring_pcross: float = 0.8,
        synga_offspring_pmut: float = 0.5,
        synga_sampling: str = "invrank",
        synga_num_workers: int = 5,
        maxatoms: int = 1000,  # default: disabled
        device: Literal["cpu", "cuda"] = "cpu"
    ):
        self.save_hyperparameters()
        super().__init__()

        self.history = dict()
        self.fpcache = dict()
        self.retro = dict()

        self.synga = None
        self.pmap = None

        self.gp = None
        self.gp_hparams = None
        self.ucb_beta = None

        self.proposals = 0
        self.device = torch.device(device)

    def propose_first(self, rng):
        hp = self.hparams
        chem.silence_rdlogger()
        lib = chem.SynthesisLibrary(name=hp.lib)
        while len(self.retro) < hp.initial_size:
            T = lib.sample(rng=rng)
            if chem.check_mol(T.root.mol, maxatoms=self.hparams.maxatoms):
                self.retro[T.product] = T
        return sorted(self.retro)

    def propose(self, pmap, rng):
        if not self.history:
            return self.propose_first(rng=rng)
        del pmap  # for safety

        hp = self.hparams
        synga, pmap = self.synga, self.pmap

        # Run SynGA to optimize UCB
        for gen in tqdm.trange(hp.synga_generations + 1, desc="Acquisition", leave=False):
            proposed = synga.propose(pmap=pmap, rng=rng)

            X = np.stack(pmap(chem.fingerprint, proposed, params="ml", asnumpy=True), axis=0)
            X = torch.from_numpy(X)
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                scores = self.gp.ucb(X, beta=self.ucb_beta).tolist()
            scores = list(zip(proposed, scores))

            # Add top-scoring SMILES to initial population
            if gen == 0:
                top = subset(self.history, top=hp.synga_population_size, rand=0, rng=rng)
                synga.retro.update({smi: self.retro[smi] for smi in top})
                scores.extend((smi, self.history[smi]) for smi in top)

            synga.on_propose_end(scores, None, pmap=pmap, rng=rng)

        # Take fittest unseen individuals as offspring
        unseen = set(synga.history) - set(self.history)
        children = subset(synga.history, top=hp.propose_size, rand=0, rng=rng, keys=unseen)
        self.retro.update({smi: synga.retro[smi] for smi in children})

        return children

    def on_propose_end(self, scores, logger, pmap, rng):
        hp = self.hparams

        for smi, x in scores:
            if smi in self.history:
                continue
            self.history[smi] = x
            self.fpcache[smi] = chem.fingerprint(smi, params="ml", asnumpy=True)

        # Re-initialize GA for next propose() step
        self.synga = SynthesisGA(
            lib=hp.lib,
            founder_size=hp.synga_founder_size,
            population_size=hp.synga_population_size,
            offspring_size=hp.synga_offspring_size,
            offspring_pcross=hp.synga_offspring_pcross,
            offspring_pmut=hp.synga_offspring_pmut,
            sampling=hp.synga_sampling,
            maxatoms=hp.maxatoms,
        )

        # Clear caches
        del self.gp
        if "cuda" in str(self.device):
            torch.cuda.empty_cache()

        # Refit the GP
        gp_metrics = self.fit_gp(rng=rng)

        # Maybe refit the NAM
        if (len(self.history) >= 500) and (self.proposals % hp.nam_train_frequency == 0):
            nam_metrics, S = self.fit_nam(rng=rng)
            self.synga.libconfig.update(subset=S, eps=0.1)
            self.pmap.shutdown()
            self.pmap = None
        else:
            nam_metrics = dict()

        # Initialize pmap
        if self.pmap is None:
            self.pmap = ops.ParallelMap(hp.synga_num_workers, init=self.synga.pmap_init)

        # Log metrics
        metrics = {**gp_metrics, **nam_metrics, "proposals": self.proposals}
        logger.log_metrics(metrics)
        self.proposals += 1

    def fit_gp(self, rng):
        hp = self.hparams

        n = hp.gp_train_samples // 2
        data = subset(self.history, top=n, rand=n, rng=rng)
        X = np.stack([self.fpcache[smi] for smi in data])
        y = np.array([self.history[smi] for smi in data])

        # Refit the GP
        self.gp = TanimotoGP(X, y, device=self.device)
        self.gp.manual_fit()

        if self.proposals * hp.propose_size < hp.gp_train_samples:
            self.ucb_beta = 10.0**rng.uniform(-2, 0)
        else:
            self.ucb_beta = 0.0
        gp_metrics = {"ucb_beta": self.ucb_beta, **self.gp.hparams}
        gp_metrics = ops.fix_keys(gp_metrics, pre="gp/")
        return gp_metrics

    def fit_nam(self, rng):
        hp = self.hparams

        dataset = NAMDataset([
            {"mol": smi, "bbs": self.retro[smi].blocks, "score": self.history[smi]}
            for smi in sorted(self.history)
        ])
        g = torch.Generator().manual_seed(int(rng.integers(2**32)))
        splits = random_split(dataset, [0.9, 0.1], generator=g)

        train_loader = pyg.loader.DataLoader(
            dataset=splits[0],
            batch_size=hp.nam_batch_size,
            num_workers=2,
            shuffle=True,
            drop_last=True,
            persistent_workers=True,
        )
        val_loader = pyg.loader.DataLoader(
            dataset=splits[1],
            batch_size=len(splits[1]),
            num_workers=2,
            shuffle=False,
            drop_last=False,
            persistent_workers=True,
        )

        trainer = SimpleTrainer(
            accelerator=("gpu" if ("cuda" in str(self.device)) else "cpu"),
            max_epochs=1000,
            early_stop=True,
            early_stop_on="corr",
            early_stop_patience=5,
            verbose=True,
        )

        start = time.time()
        nam = LitNAM(lib=hp.lib)
        trainer.fit(nam, train_dataloaders=train_loader, val_dataloaders=val_loader)
        metrics = trainer.validate(nam, dataloaders=val_loader, verbose=False)[0]
        subset = nam.score_blocks().topk(hp.nam_subset_topk).indices
        fit_time = time.time() - start

        nam_metrics = {"fit_time": fit_time, **metrics}
        nam_metrics = ops.fix_keys(nam_metrics, pre="nam/")
        return nam_metrics, subset.tolist()

    def on_end(self):
        self.pmap.shutdown()
