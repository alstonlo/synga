import functools
import itertools

from src import chem, ops
from src.optim.base import GA


class SynthesisGA(GA):

    def __init__(
        self,
        lib: str = "chemspace",
        founder_size: int = 100,
        population_size: int = 500,
        offspring_size: int = 5,
        offspring_pcross: float = 0.8,
        offspring_pmut: float = 0.5,
        sampling: str = "invrank",
        maxatoms: int = 1000,  # default: disabled
    ):
        self.save_hyperparameters()
        super().__init__(
            founder_size=founder_size,
            population_size=population_size,
            offspring_size=offspring_size,
            parents_per_offspring=2,
            sampling=sampling,
        )

        self.libconfig = dict(name=lib)
        self.retro = dict()

    def check(self, T):
        if T is None:
            return False
        return chem.check_mol(T.root.mol, maxatoms=self.hparams.maxatoms)

    @property
    def pmap_init(self):
        return functools.partial(synthesis_init, config=self.libconfig)

    def propose_first(self, n, pmap, rng):
        while len(self.retro) < n:
            b = n - len(self.retro)
            for T in pmap(synthesis_start, rng.spawn(b)):
                if self.check(T):
                    self.retro[T.product] = T
        return sorted(self.retro)

    def offspring(self, parents, pmap, rng):
        hp = self.hparams

        parents = [[self.retro[p] for p in kple] for kple in parents]
        syntrees = pmap(
            f=synthesis_crossmut,
            inputs=zip(parents, rng.spawn(len(parents))),
            pcross=hp.offspring_pcross,
            pmut=hp.offspring_pmut,
        )

        children = dict()
        for T in syntrees:
            if not self.check(T) or (T.product in self.history):
                continue
            children[T.product] = T
        self.retro.update(children)
        return sorted(children)


def synthesis_init(config):
    global _lib

    _lib = chem.SynthesisLibrary(**config)


def synthesis_start(rng):
    global _lib
    chem.silence_rdlogger()

    return _lib.sample(rng=rng)


def synthesis_crossmut(input, pcross, pmut, tries=10):
    global _lib
    chem.silence_rdlogger()

    parents, rng = input
    if rng.random() < pcross:
        T = synthesis_cross(_lib, parents, rng=rng)
        mutate = (rng.random() < pmut)
    else:
        T = ops.choice(parents, rng=rng)
        mutate = True
    if mutate and (T is not None):
        T = synthesis_mutate(_lib, T, rng=rng, tries=tries)

    return T


def synthesis_cross(lib, Ts, rng):
    assert len(Ts) == 2
    T1, T2 = Ts

    options = []
    for v1, v2 in itertools.product(T1.G, T2.G):
        if T1.steps(v1) + T2.steps(v2) + 1 > lib.max_steps:
            continue
        rs = [T1.node(v1), T2.node(v2)]
        cross_rxns = [i for i in range(len(lib.reactions)) if lib.can_react(i, rs)]
        if cross_rxns:
            options.append((v1, v2, cross_rxns))
    if not options:
        return None
    options = sorted(options)  # impose an order

    v1, v2, rxnids = ops.choice(options, rng=rng)
    rxnid = ops.choice(rxnids, rng=rng)
    S = [T1.subtree(v1), T2.subtree(v2)]
    return chem.SynthesisTree.join(lib, rxnid, S, rng=rng)


def synthesis_mutate(lib, T, rng, tries=10):
    F = [T.grow, T.shrink, T.rerun, T.change_internal, T.change_leaf]
    p = ops.normalize([1, 1, 2, 2, 2])

    actions = list(range(len(F)))
    attempt = 0
    while attempt < tries:
        if sum(p) == 0:
            return None
        act = ops.choice(actions, rng=rng, p=p)
        Tmut, retriable = F[act](lib, rng=rng)
        if Tmut is not None:
            return Tmut
        elif retriable:
            attempt += 1
        else:
            p[act] = 0  # so we don't sample it again
            p = ops.normalize(p)
    return None
