import polars as ps
from rdkit.Chem.MolStandardize import rdMolStandardize

from src import chem, io


def standardize(smiles):
    mol = chem.rdmol(smiles)
    if "." in smiles:
        rdMolStandardize.FragmentParentInPlace(mol)
    return chem.csmiles(mol)


def main():
    root = io.DATA_ROOT / "results" / "project_designs"
    root.mkdir(parents=True, exist_ok=True)

    dfsf = (
        ps.read_csv(root / "synformer.csv")
        .rename({"target": "query", "smiles": "analog"})
        .select("query", "analog")
        .with_columns(
            ps.col("analog").map_elements(standardize, return_dtype=ps.String),
            ps.lit("synformer").alias("method"),
        )
    )

    dfga = (
        ps.read_csv(root / "synga.csv")  # obtained from WandB
        .select("query", "analog")
        .with_columns(ps.lit("synga").alias("method"))
    )

    # Some sanity checks
    dfq = ps.read_csv(io.DATA_ROOT / "test" / "designs.csv").drop("score")
    queries = set(dfq["smiles"].to_list())
    assert set(dfsf["query"].to_list()) <= queries
    assert set(dfga["query"].to_list()) == queries

    # Merge results
    df = ps.concat([dfsf, dfga], how="align")
    df = dfq.join(df, left_on="smiles", right_on="query", validate="1:m", maintain_order="left")
    assert len(df) == (len(dfsf) + len(dfga))

    # Exclude problematic atoms for docking
    docking = ps.col("property").is_in(["ALDH1", "ESR1_ant", "TP53"])
    df = df.filter((docking & ps.col("analog").str.contains(r"B[^r]|Si")).not_())

    # Retain top-5 closest analogs
    prefix = ["property", "method", "query"]
    fp = dict(name="morgan", bits=4096, radius=2, count=True)
    df = (
        df.rename({"smiles": "query"})
        .select(*prefix, "analog")
        .unique()  # in case ChemProjector output isomers
        .with_columns(
            ps.struct("query", "analog")
            .map_elements(
                lambda x: chem.tanimoto_similarity(*x.values(), fp=fp),
                return_dtype=ps.Float64,
            )
            .alias("sim")
        )
        .group_by(*prefix).map_groups(lambda G: G.top_k(5, by="sim"))
        .sort(by=[*prefix, "sim"])
    )

    df.write_csv(root / "analogs.csv")


if __name__ == "__main__":
    main()
