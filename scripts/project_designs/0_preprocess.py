import polars as ps
import tdc

from src import chem, io


def main():
    receptors = ["ALDH1", "ESR1_ant", "TP53"]
    oracles = ["Scaffold Hop", "Osimertinib MPO", "Perindopril MPO"]

    # Download from:
    # https://github.com/luost26/ChemProjector/blob/main/data/sbdd/pocket2mol.csv
    df_sb = (
        ps.read_csv("pocket2mol.csv")
        .filter(ps.col("receptor").is_in(receptors))
        .with_columns(
            ps.col("smiles").map_elements(chem.csmiles, return_dtype=ps.String),
            ps.col("vina").neg(),
        )
        .rename({"receptor": "property", "vina": "score"})
    )

    # Download from:
    # https://github.com/luost26/ChemProjector/blob/main/data/goal_directed/goal_hard_cwo.csv
    df_gd = (
        ps.read_csv("goal_hard_cwo.csv")
        .filter(
            ps.col("property").is_in(oracles)
            & (ps.col("method") != "best_from_chembl")
            & (ps.col("tb_synthesizability") <= 0)
        )
        .with_columns(
            ps.col("SMILES").map_elements(chem.csmiles, return_dtype=ps.String).alias("smiles"),
            ps.col("property").str.to_lowercase().str.replace_all(" ", "_"),
        )
        .select("property", "smiles")
        .with_columns(
            ps.struct("property", "smiles").map_elements(
                lambda s: tdc.Oracle(s["property"])(s["smiles"]),
                return_dtype=ps.Float64,
            ).alias("score")
        )
    )

    df = ps.concat([df_sb, df_gd])
    keep = set()

    for propdf in df.partition_by(by="property"):
        centroids = []
        for smi in propdf.sort(by="score", descending=True)["smiles"]:
            fp = dict(name="morgan", bits=4096, radius=2, count=True)
            if all(chem.tanimoto_similarity(smi, c, fp=fp) < 0.7 for c in centroids):
                centroids.append(smi)
        keep.update(centroids)

    df = df.filter(ps.col("smiles").is_in(keep))
    df = df.sort(by=["property", "score"], descending=[True, False])
    df.write_csv(io.DATA_ROOT / "test" / "designs.csv")
    print(df.group_by("property").count().sort(by="property"))


if __name__ == "__main__":
    main()
