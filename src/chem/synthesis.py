import dataclasses
import functools
import itertools
from typing import List, Optional

import dill as pickle
import networkx as nx
import numpy as np
from rdkit.Chem import Mol as RDMol, SanitizeMol
from rdkit.Chem.rdChemReactions import ReactionFromSmarts

from src import io, ops
from src.chem.core import csmiles, rdmol


def read_reactions(path):
    templates = io.readlines(path)
    return [ReactionFromSmarts(sma) for sma in templates]


# Refactored to standalone function for multiprocessing
def argmatch(mol, reactions):
    mol = rdmol(mol)
    return [
        [pos for pos, s in enumerate(rxn.GetReactants()) if mol.HasSubstructMatch(s)]
        for rxn in reactions
    ]


def ary(rxn):
    return rxn.GetNumReactantTemplates()


@dataclasses.dataclass(frozen=True)
class SynthesisNode:

    mol: RDMol
    match: List[List[int]]
    rxn: Optional[int]
    bb: Optional[int]

    @functools.cached_property
    def smiles(self):
        return csmiles(self.mol)


class SynthesisTree:

    @classmethod
    def trivial(cls, lib, bbid):
        assert isinstance(bbid, int)
        mol = rdmol(lib.blocks[bbid])
        match = lib.matches[bbid]
        root = SynthesisNode(mol=mol, match=match, rxn=None, bb=bbid)

        T = nx.DiGraph()
        T.add_node(0, node=root)
        return cls(T)

    @classmethod
    def join(cls, lib, rxnid, syntrees, rng=None, product=None):
        P = sorted(lib.react(rxnid, [T.root for T in syntrees]))
        if product is None:
            if not P:
                return None
            product = ops.choice(P, rng=rng)
        else:
            assert product in P

        U = syntrees[0].G.copy()
        for T in syntrees[1:]:
            U = nx.disjoint_union(U, T.G)
        child_ids = [v for v, d in U.in_degree() if d == 0]  # save previous roots

        root_mol = rdmol(product)
        root_match = argmatch(root_mol, lib.reactions)
        root = SynthesisNode(mol=root_mol, match=root_match, rxn=rxnid, bb=None)
        root_id = max(U.nodes) + 1

        assert root_id not in U
        U.add_node(root_id, node=root)
        for v in child_ids:
            U.add_edge(root_id, v)
        return cls(U)

    def __init__(self, G):
        assert nx.is_tree(G)
        self.G = G
        self.root_id = next(n for n, d in self.G.in_degree() if d == 0)

        # Cache subtree steps
        steps = dict()
        for v in reversed(list(nx.topological_sort(G))):
            if G.out_degree(v) == 0:
                steps[v] = 0
            else:
                steps[v] = 1 + sum(steps[u] for u in G.successors(v))
        nx.set_node_attributes(G, steps, name="steps")

    def serialize(self, id=None, indent=""):
        if id is None:
            id = self.root_id
        v = self.node(id)
        toks = [f"R{v.rxn}:{v.smiles}" if (v.bb is None) else f"B{v.bb}"]
        for r in self.G.successors(id):
            subtoks = self.serialize(r, indent=indent)
            toks.extend([indent + s for s in subtoks])
        return toks

    def __repr__(self):
        return "\n".join(self.serialize(indent="  "))

    @property
    def postfix(self):
        return " ".join(reversed(self.serialize()))

    @classmethod
    def from_postfix(cls, lib, postfix):
        if isinstance(postfix, str):
            postfix = postfix.split(" ")
        stack = []
        for tok in postfix:
            cmd, tok = tok[0], tok[1:]
            if cmd == "R":
                rxnid, p = tok.split(":")
                rxnid = int(rxnid)
                k = ary(lib.reactions[rxnid])
                T = cls.join(lib, rxnid, stack[-k:], product=p)
                stack = stack[:-k]
            elif cmd == "B":
                T = cls.trivial(lib, int(tok))
            else:
                raise ValueError()
            stack.append(T)
        if len(stack) != 1:
            raise ValueError()
        return stack[0]

    @property
    def is_trivial(self):
        return len(self.G) == 1

    @property
    def root(self):
        return self.node(self.root_id)

    @property
    def product(self):
        return self.root.smiles

    @property
    def blocks(self):
        return [self.node(v).bb for v, d in self.G.out_degree() if d == 0]

    def node(self, id):
        return self.G.nodes[id]["node"]

    def replace_node(self, id, node):
        G = self.G.copy()
        G.nodes[id]["node"] = node
        return SynthesisTree(G)

    def steps(self, id=None):
        if id is None:
            id = self.root_id
        return self.G.nodes[id]["steps"]

    def subtree(self, id):
        assert id in self.G
        Gsub = self.G.subgraph(nx.descendants(self.G, id) | {id}).copy()
        return SynthesisTree(Gsub)

    def grow(self, lib, rng):
        if self.steps() >= lib.max_steps:
            return None, False
        has_partners = False
        for rxnid, _ in ops.randenumerate(lib.reactions, rng=rng):
            partners = lib.sample_partners(rxnid, [self.root], rng=rng)
            if partners is None:
                continue
            has_partners = True
            partners = [self.trivial(lib, bbid) for bbid in partners]
            T = self.join(lib, rxnid, [self, *partners], rng=rng)
            if T is not None:
                return T, True
        return None, has_partners

    def shrink(self, lib, rng):
        if self.is_trivial:
            return None, False
        child = ops.choice(sorted(self.G.successors(self.root_id)), rng=rng)
        return self.subtree(child), True

    @staticmethod
    def rerun_pipe(lib, rxnid, pipes, rng):
        seen = set()
        for r1, pf1 in pipes[0]:
            for r2, pf2 in (pipes[1] if len(pipes) > 1 else [(None, [])]):
                reactants = [r1] if r2 is None else [r1, r2]
                products = sorted(lib.react(rxnid, reactants))
                for _, p in ops.randenumerate(products, rng=rng):
                    if p in seen:
                        continue
                    yield rdmol(p), (*pf1, *pf2, f"R{rxnid}:{p}")
                    seen.add(p)

    def rerun(self, lib, rng):
        if self.is_trivial:
            return None, False
        curr_product = self.product

        pipes = dict()
        for id in reversed(list(nx.topological_sort(self.G))):
            v = self.node(id)
            if self.G.out_degree(id) == 0:
                pipes[id] = [(v.mol, [f"B{v.bb}"])]
            else:
                children = [pipes[u] for u in self.G.successors(id)]
                pipes[id] = self.rerun_pipe(lib, v.rxn, children, rng=rng)

        for p, pf in pipes[self.root_id]:
            if csmiles(p) == curr_product:
                continue
            return self.from_postfix(lib, pf), True
        return None, False

    def change_internal(self, lib, rng):
        if self.is_trivial:
            return None, False
        internal = ops.choice([v for v, d in self.G.out_degree() if d > 0], rng=rng)
        children = [self.node(u) for u in self.G.successors(internal)]
        v = self.node(internal)

        for rxnid, _ in ops.randenumerate(lib.reactions, rng=rng):
            if (rxnid == v.rxn) or not lib.can_react(rxnid, children):
                continue
            T = self.replace_node(internal, dataclasses.replace(v, rxn=rxnid))
            return T.rerun(lib, rng=rng)[0], True
        return None, True

    def change_leaf(self, lib, rng):
        if self.is_trivial:
            bbid = lib.sample_block(rng=rng)
            return self.trivial(lib, bbid), True

        leaf = ops.choice([v for v, d in self.G.out_degree() if d == 0], rng=rng)
        parent = next(self.G.predecessors(leaf))
        siblings = ops.unique(self.G.successors(parent), exclude={leaf})
        siblings = [self.node(u) for u in siblings]

        sub = lib.sample_partners(self.node(parent).rxn, siblings, rng=rng)
        if sub is None:
            return None, True
        sub = self.trivial(lib, sub[0])

        T = self.replace_node(leaf, sub.root)
        return T.rerun(lib, rng=rng)[0], True


class SynthesisLibrary:

    @classmethod
    def read_fingerprints(cls, name):
        return np.load(io.LIBS_ROOT / name / "block_fps.npz")["fps"]

    @classmethod
    def read_neighbors(cls, name, minsim=0.0, maxsim=1.0, maxk=None):
        assert 0.0 <= minsim < maxsim <= 1.0
        knn = np.load(io.LIBS_ROOT / name / "block_knn.npz")
        sims, nbs, sizes = [knn[f"knn_{s}"] for s in ["sims", "nbs", "sizes"]]
        k = sizes.max() if (maxk is None) else maxk

        N, i = [], 0
        for m in sizes:
            S, I = sims[i:(i + m)], nbs[i:(i + m)]
            N.append(I[(minsim <= S) & (S <= maxsim)][:k].tolist())
            i += m
        return N

    def __init__(
        self,
        name: str = "chemspace",
        subset: Optional[List[int]] = None,
        eps: float = 1.0,  # sample from subset with prob 1 - eps
    ):
        self.name = name
        self.root = io.LIBS_ROOT / name

        self.blocks = io.readlines(self.root / "blocks.txt")
        self.reactions = read_reactions(self.root / "reactions.txt")
        self.max_steps = 5

        # Invert matches (rxn, pos) -> [compatible block IDs]
        compat = [[[] for _ in range(ary(rxn))] for rxn in self.reactions]
        with open(self.root / "block_argmatch.pkl", "rb") as f:
            self.matches = pickle.load(f)
        for bbid, match in enumerate(self.matches):
            for rxnid, P in enumerate(match):
                for pos in P:
                    compat[rxnid][pos].append(bbid)
        self.compat = compat

        # Optionally restrict to a subset of blocks
        self.universe = list(range(len(self.blocks)))
        self.subset, self.subset_compat, self.eps = None, None, 1.0
        self.restrict(subset, eps=eps)  # sets the above

    def restrict(self, subset=None, eps=1.0):
        if subset is None:
            self.subset = self.universe
            self.subset_compat = self.compat
        else:
            S = set(subset)
            self.subset = sorted(S)
            self.subset_compat = [[sorted(S & set(x)) for x in X] for X in self.compat]
        self.eps = eps

    def can_react(self, rxnid, reactants):
        rxn = self.reactions[rxnid]
        if ary(rxn) != len(reactants):
            return False
        elif ary(rxn) == 1:
            return 0 in reactants[0].match[rxnid]
        elif ary(rxn) == 2:
            m1, m2 = [v.match[rxnid] for v in reactants]
            return (0 in m1 and 1 in m2) or (1 in m1 and 0 in m2)
        else:
            raise NotImplementedError()

    def react(self, rxnid, reactants):
        rxn = self.reactions[rxnid]
        reactants = [r.mol if isinstance(r, SynthesisNode) else r for r in reactants]
        assert len(reactants) == ary(rxn)

        products = set()
        for rs in itertools.permutations(reactants, r=ary(rxn)):
            for p, in rxn.RunReactants(rs):
                if SanitizeMol(p, catchErrors=True) == 0:
                    products.add(csmiles(p))
        return products

    def sample(self, rng):
        T = SynthesisTree.trivial(self, self.sample_block(rng=rng))
        steps = rng.integers(1, self.max_steps + 1).item()
        while T.steps() < steps:
            Tnext, _ = T.grow(self, rng=rng)
            if Tnext is None:  # cannot grow
                break
            T = Tnext
        return T

    def sample_partners(self, rxnid, nodes, rng):
        rxn = self.reactions[rxnid]

        # Try various assignments of each node to a reaction position
        assignments = list(itertools.product(*[v.match[rxnid] for v in nodes]))
        for _, assign in ops.randenumerate(assignments, rng=rng):
            assign = set(assign)
            if len(assign) != len(nodes):
                continue
            partners = [
                self.sample_block(rng=rng, rxnid=rxnid, pos=pos)
                for pos in range(ary(rxn)) if (pos not in assign)
            ]
            if None in partners:
                continue
            return partners

        return None

    def sample_block(self, rng, rxnid=None, pos=None):
        if rxnid is None:
            S = self.subset
            X = self.universe
        else:
            S = self.subset_compat[rxnid][pos]
            X = self.compat[rxnid][pos]
        bbids = S if (S and rng.random() >= self.eps) else X
        return ops.choice(bbids, rng=rng) if bbids else None
