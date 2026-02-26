import polars as ps

from src import io
from src.chem import LITPCBA_RECEPTORS

methods = {
    "synnet": "SynNet",
    "bbar": "BBAR",
    "synflownet": "SynFlowNet",
    "rgfn": "RGFN",
    "rxnflow": "RxnFlow",
    "3dsynthflow": "3DSynthFlow",
    "synga": r"\ours{}",
    "syngbo": r"\oursbo{}",
}


def fmt_val(mean, stdev=None, bold=False, metric="vina"):
    prec = 2 if metric == "vina" else 3
    prefix = r"\minus" if mean < 0 else ""
    mean_str = f"{abs(mean):.{prec}f}"
    if bold:
        mean_str = rf"\textbf{{{mean_str}}}"
    stdev_str = f" \\pm {abs(stdev):.{prec}f}" if stdev is not None else ""
    return rf"${prefix}{mean_str}{stdev_str}$"


def get_stats(df, rec, metric):
    row = df.filter(ps.col("receptor") == rec)
    return row[f"{metric}_mean"][0], row[f"{metric}_stdev"][0]


def mean_stat(df, receptors, metric):
    valid = [df.filter(ps.col("receptor") == r) for r in receptors]
    valid = [r for r in valid if len(r) > 0]
    return sum(r[f"{metric}_mean"][0] for r in valid) / len(valid)


def find_best(data, cols, metric, all_receptors=None):
    cmp = min if metric == "vina" else max

    def score(k, col):
        if col == "Mean":
            return mean_stat(data[k], all_receptors, metric)
        return get_stats(data[k], col, metric)[0]

    return {col: cmp(data, key=lambda k, c=col: score(k, c)) for col in cols}


def col_header(cols):
    return " & ".join(rf"\multicolumn{{1}}{{c}}{{{c.replace('_', chr(92) + '_')}}}" for c in cols)


def make_table(data, receptors, metric):
    best = find_best(data, receptors, metric)
    col_spec = (
        r"lr@{\hspace{2mm}}r@{\hspace{2.5mm}}r@{\hspace{2mm}}r@{\hspace{2mm}}r"
        if metric == "vina"
        else "l" + "c" * len(receptors)
    )
    lines = [
        r"\begin{tabular}{" + col_spec + "}",
        r"\toprule",
        f"Method & {col_header(receptors)}" + r" \\",
        r"\midrule",
    ]
    for k, name in methods.items():
        if k == "synga":
            lines.append(r"\midrule")
        cells = [name] + [
            fmt_val(*get_stats(data[k], rec, metric), bold=(best[rec] == k), metric=metric)
            for rec in receptors
        ]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return rf"\resizebox{{\textwidth}}{{!}}{{{chr(10).join(lines)}}}"


def make_summary_table(data, display_receptors, all_receptors, metric):
    columns = display_receptors + ["Mean"]
    best = find_best(data, columns, metric, all_receptors)

    n = len(columns)
    col_spec = r"lc" + r"r@{\hspace{2mm}}" * (n - 1) + "r"
    lines = [
        r"\begin{tabular}{" + col_spec + "}",
        r"\toprule",
        f"Method & Calls & {col_header(columns)}" + r" \\",
        r"\midrule",
    ]

    def fmt_cell(df, col, k):
        if col == "Mean":
            return fmt_val(mean_stat(df, all_receptors, metric), bold=(best[col] == k), metric=metric)
        return fmt_val(*get_stats(df, col, metric), bold=(best[col] == k), metric=metric)

    def build_rows(group_keys, calls):
        rows = []
        for i, k in enumerate(group_keys):
            call_cell = rf"\multirow{{{len(group_keys)}}}{{*}}{{{calls}}}" if i == 0 else ""
            cells = [methods[k], call_cell] + [fmt_cell(data[k], col, k) for col in columns]
            rows.append(" & ".join(cells) + r" \\")
        return rows

    baseline_keys = [k for k in methods if k not in ("synga", "syngbo")]
    ours_keys = [k for k in methods if k in ("synga", "syngbo")]

    lines += build_rows(baseline_keys, 64000)
    lines.append(r"\midrule")
    lines += build_rows(ours_keys, 16000)
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def main():
    data = {k: ps.read_csv(io.DATA_ROOT / "results" / "litpcba" / f"{k}.csv") for k in methods}
    all_receptors = list(LITPCBA_RECEPTORS)
    chunks = [all_receptors[i:i + 5] for i in range(0, 15, 5)]

    for metric in ["vina", "eff"]:
        print(f"\n{'=' * 60}\n  Metric: {metric}\n{'=' * 60}\n")
        for i, chunk in enumerate(chunks):
            print(f"% Table {i + 1}: {', '.join(chunk)}")
            print(make_table(data, chunk, metric))
            if i < len(chunks) - 1:
                print(r"\vspace{\mytabskip}")
            print()

    display_receptors = ["ALDH1", "ESR_ant", "TP53"]
    print(f"\n{'=' * 60}\n  Summary: vina\n{'=' * 60}\n")
    print(make_summary_table(data, display_receptors, all_receptors, "vina"))


if __name__ == "__main__":
    main()