import pickle

import polars as ps

from src import chem, io

from scripts.utils import format_mean_pm_std


def score_column(col, scores):
    return ps.struct(col, "property").map_elements(
        lambda s: scores[s["property"]][s[col]],
        return_dtype=ps.Float64,
    )


def main():
    root = io.DATA_ROOT / "results" / "project_designs"
    df = ps.read_csv(root / "analogs.csv")

    scores = dict()
    with open(root / "oracle_scores.pkl", "rb") as f:
        scores.update(pickle.load(f))
    with open(root / "dock_scores.pkl", "rb") as f:
        scores.update(pickle.load(f))

    for (prop,), dfp in df.partition_by(by="property", as_dict=True).items():
        smiles = set(dfp["query"].to_list() + dfp["analog"].to_list())
        assert smiles == set(scores[prop])  # sanity check

    prefix = ["property", "method"]
    fp = dict(name="morgan", bits=4096, radius=2, count=True)
    df = (
        df.with_columns(
            score_column("query", scores).alias("ref"),
            score_column("analog", scores).alias("ana"),
        )
        .group_by(*prefix, "query").map_groups(lambda G: G.top_k(1, by="ana"))
        .with_columns(
            (ps.col("ana") - ps.col("ref")).alias("del"),
            ps.struct("query", "analog")
            .map_elements(
                lambda x: chem.tanimoto_similarity(*x.values(), fp=fp),
                return_dtype=ps.Float64,
            )
            .alias("sim")
        )
    )

    summary = (
        df.select(*prefix, "sim", "del")
        .group_by(*prefix)
        .agg(
            ps.len().alias("valid"),
            format_mean_pm_std("sim"),
            format_mean_pm_std("del"),
        )
        .with_columns(ps.col("valid").max().over("property").alias("n"))
        .with_columns(ps.col("valid") / ps.col("n"))
        .select("method", "property", "n", "valid", "sim", "del")
        .sort(by=["method", "property"])
    )

    with ps.Config(tbl_rows=len(summary)):
        print(summary)


if __name__ == "__main__":
    main()
