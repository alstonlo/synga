import json

import jsonargparse
import polars as ps
import tqdm
import wandb

from src import chem, io

"""
In a new empty WandB project, run: 

```
python -m src.optimize \
    --seed=00 --trials=5 --objective=jnk3 \
    --budget=1100 --log_samples=1100 \
    --optimizer=SynthesisGA --logger.project=[PROJECT]
python -m src.optimize \
    --seed=10 --trials=5 --objective=osimertinib_mpo \
    ... copy as above ...
```

If you change seeds, see (!!). The following downloads the samples into a table.
"""


def download_dataset(project: str, lib: str = "chemspace"):
    lib = chem.SynthesisLibrary(lib)
    chem.silence_rdlogger()

    def extract_blocks(product, postfix):
        T = chem.SynthesisTree.from_postfix(lib, postfix)
        if T.product != product:
            raise ValueError()
        return T.blocks

    df = []
    columns = ["objective", "fold", "split", "mol", "score", "bbs"]

    for run in tqdm.tqdm(wandb.Api().runs(project), desc="Processing"):
        config = {k: v for k, v in run.config.items() if not k.startswith("_")}
        assert config["optimizer"]["lib"] == lib.name

        samples = run.summary._json_dict["samples"]
        table = run.file(samples["path"]).download(root=io.LOG_DIR, exist_ok=True)
        table = json.load(table)
        table = ps.DataFrame(data=table["data"], schema=table["columns"], orient="row")
        assert len(table) == 1100

        table = (
            table.with_columns(
                ps.lit(config["seed"] % 10).alias("fold"),  # (!!) hack
                ps.lit(config["objective"]).alias("objective"),
                ps.when(ps.col("idx") < 1000).then(0).otherwise(1).alias("split"),
                ps.struct("smiles", "route").map_elements(
                    lambda x: extract_blocks(*x.values()),
                    return_dtype=ps.List(ps.Int32),
                )
                .alias("bbs")
            )
            .rename({"smiles": "mol", "fitness": "score"})
            .select(*columns)
            .sort(columns[:-1])
        )

        df.append(table)

    df = ps.concat(df)
    df.write_parquet(io.LIBS_ROOT / lib.name / "block_nam.parquet")


if __name__ == "__main__":
    parser = jsonargparse.ArgumentParser()
    parser.add_function_arguments(download_dataset)
    args = parser.parse_args()

    download_dataset(**args.as_dict())
