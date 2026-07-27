from pathlib import Path
import tempfile

import pandas as pd

from phishing_hybrid.data import load_dataset


def test_csv_label_mapping_once() -> None:
    df = pd.DataFrame(
        {
            "f1": [1, -1, 0],
            "f2": [0, 1, -1],
            "Result": [-1, 1, -1],
        }
    )

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "toy.csv"
        df.to_csv(path, index=False)
        x, y = load_dataset(path, target_column="Result", fillna_value=0.0)

    assert x.shape == (3, 2)
    assert set(y.tolist()) == {0, 1}
