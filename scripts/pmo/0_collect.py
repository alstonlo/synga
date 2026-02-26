import polars as ps

from src import io


def main():
    df = ps.read_csv("results.csv")  # download from WandB

    summary = (
        df.select("objective", "top10/auc")
        .group_by("objective").agg(
            ps.len().alias("n"),
            ps.col("top10/auc").mean().round(3).alias("auc_mean"),
            ps.col("top10/auc").std().round(3).alias("auc_std"),
        )
        .rename({"objective": "oracle"})
        .sort("oracle")
    )
    assert set(summary["n"].to_list()) == {5}  # sanity check
    summary = summary.drop("n")

    summary.write_csv(io.DATA_ROOT / "results" / "pmo" / "synga.csv")  # change for syngbo


if __name__ == "__main__":
    main()
