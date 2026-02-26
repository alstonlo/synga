import functools

import numpy as np
from rdkit.Chem import DataStructs
from rdkit.Chem.Pharm2D import Generate, Gobbi_Pharm2D
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator, GetRDKitFPGenerator

from src.chem import rdmol


def fingerprint(mol, params, murcko=False, asnumpy=False):
    mol = rdmol(mol)

    if params == "ml":  # default fp for ML
        params = dict(name="morgan", bits=2048, radius=2, count=True)
    elif isinstance(params, str):
        params = dict(name=params)
    if murcko:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        if scaffold is not None:
            mol = scaffold

    name = params["name"]
    if name == "morgan":
        assert set(params) <= {"name", "bits", "radius", "count"}
        fpgen = GetMorganGenerator(fpSize=params.get("bits", 4096), radius=params.get("radius", 2))
        if params.get("count", False):
            fp = fpgen.GetCountFingerprint(mol)
        else:
            fp = fpgen.GetFingerprint(mol)

    elif name == "gobbi":
        fp = Generate.Gen2DFingerprint(mol, Gobbi_Pharm2D.factory)
        fp = DataStructs.cDataStructs.ConvertToExplicit(fp)

    elif name == "rdkit":
        fpgen = GetRDKitFPGenerator()
        fp = fpgen.GetFingerprint(mol)

    else:
        raise ValueError()

    if asnumpy:
        array = np.zeros((0,), dtype=np.uint32)
        DataStructs.ConvertToNumpyArray(fp, array)
        return array.clip(max=255).astype(np.uint8)
    else:
        return fp


def tanimoto_similarity(mol1, mol2, fp="morgan", murcko=False):
    fp = functools.partial(fingerprint, params=fp, murcko=murcko)
    return DataStructs.TanimotoSimilarity(fp(mol1), fp(mol2))


def dice_similarity(mol1, mol2, fp="gobbi"):
    fp = functools.partial(fingerprint, params=fp)
    return DataStructs.DiceSimilarity(fp(mol1), fp(mol2))
