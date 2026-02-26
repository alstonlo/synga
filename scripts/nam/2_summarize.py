import polars as ps

from scripts.utils import format_mean_pm_std
from src import io


def main():
    root = io.DATA_ROOT / "results" / "nam"
    df = ps.read_csv(root / "results.csv")  # download from WandB

    summary = df.group_by(["Group", "model.objective"]).agg(
        ps.col("data.fold").n_unique().alias("n"),
        format_mean_pm_std("test/corr_nam").alias("corr_nam"),
        format_mean_pm_std("test/corr_mol").alias("corr_gp"),
        format_mean_pm_std("test/scores").alias("scores"),
    )

    for subsum in summary.partition_by("model.objective"):
        print(subsum.sort("Group"))


if __name__ == "__main__":
    main()
