import json

import polars as ps
import wandb
from rdkit.Chem.rdMolDescriptors import CalcNumHeavyAtoms

from src import chem, io


def heavy_atoms(mol):
    return CalcNumHeavyAtoms(chem.rdmol(mol))


def fetch_runs():
    project = "synga_dock"  # WandB project

    rows = []
    for run in wandb.Api().runs(project):
        config = {k: v for k, v in run.config.items() if not k.startswith("_")}
        summary = run.summary._json_dict

        samples = summary["samples"]
        table = run.file(samples["path"]).download(root=io.LOG_DIR, exist_ok=True)
        table = json.load(table)
        table = ps.DataFrame(data=table["data"], schema=table["columns"], orient="row")
        hac_col = ps.col("smiles").map_elements(heavy_atoms, return_dtype=ps.Int32)
        table = table.with_columns((-ps.col("vina") / hac_col).alias("eff"))
        assert (summary["modes/vina"] - table["vina"].mean()) <= 1e-4

        rows.append({
            "method": config["optimizer"]["method"],
            "receptor": config["receptor"],
            "seed": config["seed"],
            "vina": table["vina"].mean(),
            "eff": table["eff"].mean(),
        })
    return ps.from_dicts(rows)


def main():
    root = io.DATA_ROOT / "results" / "litpcba"
    df = fetch_runs()

    methods = {"synga": "SynthesisGA", "syngbo": "SynthesisGBO"}
    for k, name in methods.items():
        summary = (
            df.filter(ps.col("method") == name)
            .group_by("receptor")
            .agg(
                ps.col("vina").mean().round(2).alias("vina_mean"),
                ps.col("vina").std().round(2).alias("vina_stdev"),
                ps.col("eff").mean().round(3).alias("eff_mean"),
                ps.col("eff").std().round(3).alias("eff_stdev"),
            )
            .sort("receptor")
        )
        summary.write_csv(root / f"{k}.csv")
        print(summary)


if __name__ == "__main__":
    main()
