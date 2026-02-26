"""Spot-check TDC to check for discrepancies.

Dependencies:
$ mamba create --name=pmo python=3.7
$ mamba activate pmo
$ pip install fastparquet networkx pandas pyarrow requests PyTDC==0.3.6 scikit_learn==0.21.3
$ mamba install -c conda-forge rdkit
"""

import functools
import multiprocessing as mp
import pathlib
import shutil

import pandas as pd
import tdc
import tdc.version

PMO_ORACLES = sorted([
    "qed", "jnk3", "gsk3b", "drd2",
    "celecoxib_rediscovery", "troglitazone_rediscovery", "thiothixene_rediscovery",
    "albuterol_similarity", "mestranol_similarity",
    "isomers_c7h8n2o2", "isomers_c9h10n2o2pf2cl",
    "median1", "median2",
    "osimertinib_mpo", "fexofenadine_mpo", "ranolazine_mpo", "perindopril_mpo",
    "amlodipine_mpo", "sitagliptin_mpo", "zaleplon_mpo",
    "valsartan_smarts",
    "deco_hop", "scaffold_hop",
])


def score_smiles(oracle, smiles):
    fn = tdc.Oracle(oracle)
    scores = [fn(smi) for smi in smiles]
    return oracle, scores


def pmo_score_smiles():
    version = tdc.version.__version__.replace(".", "-")
    root = pathlib.Path(__file__).parent

    # Load test set
    smiles = pd.read_csv(root / "zinc.csv")["smiles"].to_list()

    # Refresh oracle cache
    cache_dir = pathlib.Path("./oracle")
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    [score_smiles(oracle, "C") for oracle in PMO_ORACLES]

    # Run against all PMO oracles
    data = dict()
    with mp.Pool(len(PMO_ORACLES)) as pool:
        worker_fn = functools.partial(score_smiles, smiles=smiles)
        for oracle, scores in pool.map(worker_fn, PMO_ORACLES):
            data[oracle] = scores
    data = pd.DataFrame(data).astype("float32")
    data.to_parquet(root / f"scores_{version}.parquet", index=False)

    # Clear oracle cache
    shutil.rmtree(cache_dir)


if __name__ == "__main__":
    pmo_score_smiles()
