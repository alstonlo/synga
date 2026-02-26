import pathlib
import shutil

import dill as pickle
import jsonargparse
import multiprocess as mp
import numpy as np
from rdkit.Chem.MolStandardize import rdMolStandardize

from src import chem, io, ops

ALLOWED_ATOMS = {"B", "Br", "C", "Cl", "F", "H", "I", "N", "O", "P", "S", "Se", "Si"}


def standardize(mol):
    mol = chem.rdmol(mol)
    for atom in mol.GetAtoms():
        if (atom.GetIsotope() != 0) or (atom.GetSymbol() not in ALLOWED_ATOMS):
            return None
    if chem.check_mol(mol):
        rdMolStandardize.ChargeParentInPlace(mol)
        return chem.csmiles(mol)
    return None


def match_init(reactions):
    global _R
    _R = chem.read_reactions(reactions)


def standardize_and_match(mol):
    global _R
    mol = standardize(mol)
    match = chem.argmatch(mol, _R) if (mol is not None) else None
    return mol, match


def setup(
    name: str,
    blocks: str,
    reactions: pathlib.Path = io.LIBS_ROOT / "hb.txt",
    num_workers: int = 0,
):
    # Create new synthesis library
    root = io.LIBS_ROOT / name
    root.mkdir(exist_ok=True)
    shutil.copy(reactions, root / "reactions.txt")

    # Standardize blocks. Also, try to match them to reactions.
    matches = dict()
    discard = []

    smiles = io.readlines(blocks)
    with ops.ParallelMap(num_workers, init=match_init, initargs=[reactions]) as pmap:
        results = pmap(standardize_and_match, smiles, pbar="Setup")
        for i, (smi, m) in enumerate(results):
            if (smi is None) or not any(m):
                discard.append(smiles[i])
            else:
                matches[smi] = m
    io.writelines(root / "errors.txt", discard)  # for debugging

    # Save blocks
    blocks = sorted(matches)
    io.writelines(root / "blocks.txt", blocks)

    # Save compatability matrix
    with open(root / "block_argmatch.pkl", "wb") as f:
        matches = [matches[smi] for smi in blocks]
        pickle.dump(matches, f)

    # Precompute fingerprints
    with ops.ParallelMap(num_workers) as pmap:
        fps = pmap(chem.fingerprint, blocks, params="ml", asnumpy=True, pbar="FPs")
        fps = np.stack(fps, axis=0)
    np.savez(root / "block_fps.npz", fps=fps)


if __name__ == "__main__":
    mp.set_start_method("spawn")

    parser = jsonargparse.ArgumentParser()
    parser.add_function_arguments(setup)
    args = parser.parse_args()

    setup(**args.as_dict())
