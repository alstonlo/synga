import polars as ps


# A lot of extra work just to keep the trailing zeros
def pretty_round(col, decimals=3):
    col = col.round(decimals).cast(str)
    whole = col.str.find(".", literal=True)
    col = ps.concat_str([col, ps.lit("0" * decimals)], separator="")
    return col.str.slice(0, length=(whole + 1 + decimals))


def format_mean_pm_std(col, sep=" \pm ", decimals=3, inagg=True):
    col = ps.col(col) if inagg else ps.col(col).list
    avg_col = pretty_round(col.mean(), decimals=decimals)
    std_col = pretty_round(col.std(), decimals=decimals)
    return ps.concat_str([avg_col, std_col], separator=sep)
