import pickle

import multiprocess as mp
import polars as ps

from scripts.project_designs.qvina import QVinaDockingTask
from src import io, ops

NUM_WORKERS = 40


def dock(ligand, receptor, outroot):
    out = QVinaDockingTask(receptor, ligand, outroot=outroot)()
    if out is not None:
        assert out.smiles == ligand
        return -out.affinity
    else:
        return 0.0


def main():
    root = io.DATA_ROOT / "results" / "project_designs"

    receptors = ["ALDH1", "ESR1_ant", "TP53"]

    scores = dict()
    df = ps.read_csv(root / "analogs.csv")
    with ops.ParallelMap(NUM_WORKERS) as pmap:
        for k in receptors:
            outroot = root / "docking" / k
            outroot.mkdir(parents=True, exist_ok=True)
            dfk = df.filter(ps.col("property") == k)
            smiles = list(set(dfk["query"].to_list() + dfk["analog"].to_list()))
            naffns = pmap(dock, smiles, receptor=k, outroot=outroot, pbar=f"Docking {k}")
            scores[k] = dict(zip(smiles, naffns))
    with open(root / "dock_scores.pkl", "wb") as f:
        pickle.dump(scores, f)


if __name__ == "__main__":
    mp.set_start_method("spawn")

    main()
