import numpy as np
import polars as ps
import tdc
import tdc.version

from data.test.pmo.score import PMO_ORACLES
from src.io import DATA_ROOT


def test_pmo(version):
    print(f"Testing discrepancy {version} -> {tdc.version.__version__}")
    version_old = version.replace(".", "-")
    version_new = tdc.version.__version__.replace(".", "-")
    root = DATA_ROOT / "test" / "pmo"

    # Load score sheets
    data_old = ps.read_parquet(root / f"scores_{version_old}.parquet")
    data_new = ps.read_parquet(root / f"scores_{version_new}.parquet")

    # Compare
    for oracle in PMO_ORACLES:
        scores_old = data_old[oracle].to_numpy()
        scores_new = data_new[oracle].to_numpy()
        error = np.abs(scores_new - scores_old).max()
        if np.isnan(error) or (error > 0):
            print(f"\tError {oracle}: {error}")


if __name__ == "__main__":
    test_pmo(version="0.3.6")  # PMO
    test_pmo(version="0.4.0")  # Genetic-GFN
