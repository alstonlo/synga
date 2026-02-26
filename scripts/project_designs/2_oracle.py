import pickle

import polars as ps
import tdc

from src import io


def main():
    root = io.DATA_ROOT / "results" / "project_designs"

    oracles = ["scaffold_hop", "osimertinib_mpo", "perindopril_mpo"]
    oracles = {k: tdc.Oracle(k) for k in oracles}

    scores = dict()
    df = ps.read_csv(root / "analogs.csv")
    for k, fn in oracles.items():
        dfk = df.filter(ps.col("property") == k)
        smiles = set(dfk["query"].to_list() + dfk["analog"].to_list())
        scores[k] = {smi: fn(smi) for smi in smiles}
    with open(root / "oracle_scores.pkl", "wb") as f:
        pickle.dump(scores, f)


if __name__ == "__main__":
    main()
