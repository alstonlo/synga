import polars as ps

from scripts.utils import pretty_round
from src import io

PRETTY_PMO = {
    "qed": "QED",
    "jnk3": "JNK3",
    "gsk3b": r"GSK3$\beta$",
    "drd2": "DRD2",
    "celecoxib_rediscovery": "Cele.~Redisc.",
    "troglitazone_rediscovery": "Trog.~Redisc.",
    "thiothixene_rediscovery": "Thio.~Redisc.",
    "albuterol_similarity": "Albu.~Sim.",
    "mestranol_similarity": "Mest.~Sim.",
    "isomers_c7h8n2o2": "Isom.~C7H8.",
    "isomers_c9h10n2o2pf2cl": "Isom.~C9H10.",
    "median1": "Median 1",
    "median2": "Median 2",
    "osimertinib_mpo": "Osim.~MPO",
    "fexofenadine_mpo": "Fexo.~MPO",
    "ranolazine_mpo": "Rano.~MPO",
    "perindopril_mpo": "Peri.~MPO",
    "amlodipine_mpo": "Amlo.~MPO",
    "sitagliptin_mpo": "Sita.~MPO",
    "zaleplon_mpo": "Zale.~MPO",
    "deco_hop": "Deco Hop",
    "scaffold_hop": "Scaffold Hop",
}


def main():
    summary = None

    methods = {
        "f-rag": "$f$-RAG",
        "gpbo": "GPBO",
        "genetic-gfn": "Genetic GFN",
        # "reinvent": "REINVENT",
        # "molga": "MolGA",
        # "synnet": "SynNet",
        "synga": r"\ours{}",  # LaTeX macros
        "syngbo": r"\oursbo{}",
    }

    for k, name in methods.items():
        df = ps.read_csv(io.DATA_ROOT / "results" / "pmo" / f"{k}.csv")
        df = df.filter(ps.col("oracle") != "valsartan_smarts")
        assert len(df) == 22
        aucsum = df["auc_mean"].sum()

        df = (
            df.with_columns(
                (
                    "\pmocell{"
                    + ps.concat_str([
                        pretty_round(ps.col("auc_mean")),
                        pretty_round(ps.col("auc_std")),
                    ], separator=" $\pm$ ")
                    + "}"
                ).alias(name)
            )
            .select("oracle", name)
        )
        sumrow = ps.DataFrame({"oracle": "Sum", name: f"{aucsum:.3f}"})
        df = ps.concat([df, sumrow], how="vertical")

        if summary is None:
            summary = df
        else:
            summary = summary.join(df, on="oracle", validate="1:1")

    summary = (
        summary.with_columns(ps.col("oracle").replace(PRETTY_PMO))
        .rename({"oracle": "Oracle"})
    )

    tab = summary.to_pandas()
    print(tab.to_latex(index=False, column_format=("l" + "c" * len(methods))))


if __name__ == "__main__":
    main()
