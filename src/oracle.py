import numpy as np
import tdc

from src import chem

NAMES = set(tdc.oracles.oracle_names)


class AnalogOracle:

    def __init__(self, ref, count, murcko):
        self.ref = chem.rdmol(ref)
        self.fp = dict(name="morgan", bits=4096, radius=2, count=count)
        self.murcko = murcko

    def __call__(self, mol):
        s1 = chem.tanimoto_similarity(mol, self.ref, fp=self.fp)
        if self.murcko:
            s2 = chem.tanimoto_similarity(mol, self.ref, fp=self.fp, murcko=True)
            return (0.9 * s1) + (0.1 * s2)
        else:
            return s1


def init(objective, dry=False):
    if isinstance(objective, dict):
        objective = dict(objective)
        assert objective.pop("name") == "analog"
        oracles = {"analog": AnalogOracle(**objective)}
    else:
        if "+" in objective:
            names = objective.split("+")
        else:
            names = [objective]
        assert all(k in NAMES for k in names)
        oracles = {k: tdc.Oracle(name=k) for k in names}  # this may trigger a download

    if not dry:
        global _oracles
        _oracles = oracles


def call(smiles):
    global _oracles

    result = 0.0
    for k, fn in _oracles.items():
        try:
            score = fn(smiles)
            if not np.isfinite(score):
                print(f"Oracle {k} NaN on", smiles)
                return 0.0
        except Exception:
            print(f"Oracle {k} error on", smiles)
            return 0.0
        result += score
    return result / len(_oracles)
