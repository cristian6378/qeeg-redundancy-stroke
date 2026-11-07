"""Fake stand-in for the thesis synthetic feature table (2000 trials, 17 features, 4 labels)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common as K  # noqa: E402

out = Path(sys.argv[1])
rng = np.random.default_rng(0)
labels = np.array(["normal", "ischemic", "hemorrhagic", "tia"])[rng.integers(0, 4, 2000)]
code = pd.Series(labels).map({"normal": 0, "ischemic": 1, "hemorrhagic": 2, "tia": 3}).values
df = pd.DataFrame(np.exp(rng.normal(size=(2000, 17))), columns=K.F17)
df["delta"] *= 1 + 0.8 * (code == 1)
df["alpha"] *= 1 + 0.8 * (code == 0)
df["delta_alpha"] = df["delta"] / df["alpha"]
df["sample_entropy"] += 0.8 * (code == 2)
df["pdbsi_delta"] += 0.5 * (code == 3)
df = df.rename(columns={"higuchi_fd": "higuchi_fractal_dimension"})
df["label"] = labels
out.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out, index=False)
print("fake synthetic table ->", out)
