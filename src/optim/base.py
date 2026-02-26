import numpy as np
from lightning.pytorch.core.mixins import HyperparametersMixin

from src import ops


class MolecularOptimizer(HyperparametersMixin):

    @property
    def config(self):
        method = self.__class__.__name__
        return {"method": method, **self.hparams}

    @property
    def pmap_init(self):
        return lambda: None

    def propose(self, pmap, rng):
        raise NotImplementedError()

    def on_propose_end(self, scores, logger, pmap, rng):
        pass

    def on_end(self):
        pass


class GA(MolecularOptimizer):

    def __init__(
        self,
        founder_size,
        population_size,
        offspring_size,
        parents_per_offspring,
        sampling,
    ):
        super().__init__()

        # Update the hparams in case subclass doesn't have these init args
        assert self.hparams is not None
        kwargs = dict(locals())
        del kwargs["self"]
        del kwargs["__class__"]
        self.hparams.update(kwargs)

        # Let's actually store every single individual encountered
        self.history = dict()
        self.popheap = ops.Heap(population_size)

    def population(self, rng):
        return ops.rank_by_value(self.popheap.poll(), f=self.history.get, rng=rng)

    def propose_first(self, n, pmap, rng):
        raise NotImplementedError()

    def propose(self, pmap, rng):
        hp = self.hparams
        if not self.history:
            cands = self.propose_first(hp.founder_size, pmap=pmap, rng=rng)
        else:
            n = hp.offspring_size
            k = hp.parents_per_offspring
            parents = self.mating_pool(self.population(rng=rng), n=n, k=k, rng=rng)
            cands = self.offspring(parents, pmap=pmap, rng=rng)
        return ops.unique(cands)

    def on_propose_end(self, scores, logger, pmap, rng):
        for smi, x in scores:
            if smi in self.history:
                continue
            self.history[smi] = x
            self.popheap.update(smi, score=x)

    def mating_pool(self, population, n, k, rng):
        population = np.asarray(population)
        scores = np.asarray([self.history[smi] for smi in population])
        assert np.all(scores[:-1] >= scores[1:])  # check descending

        method = self.hparams.sampling
        if method == "fitness":
            p = scores
        elif method == "invrank":
            p = 1 / np.arange(1, len(scores) + 1)
        else:
            raise ValueError()
        p = ops.normalize(p)
        return [ops.choices(population, n=k, rng=rng, p=p) for _ in range(n)]

    def offspring(self, parents, pmap, rng):
        raise NotImplementedError()
