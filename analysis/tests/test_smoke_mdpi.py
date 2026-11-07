"""End-to-end smoke test of Studies A-C on simulated data (enabled with SMOKE=1)."""
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

HERE = Path(__file__).resolve().parents[1]
REPO = HERE.parent
pytestmark = pytest.mark.skipif(os.environ.get("SMOKE") != "1", reason="set SMOKE=1 to run the smoke test")


def run(args, env):
    r = subprocess.run([sys.executable, *args], cwd=HERE, env=env, capture_output=True, text=True)
    assert r.returncode == 0, f"{args[0]} failed:\n{r.stdout[-3000:]}\n{r.stderr[-3000:]}"


def test_studies_a_b_c_and_review(tmp_path):
    subprocess.run([sys.executable, str(REPO / "tests" / "make_fake_dataset.py"), str(tmp_path / "data"),
                    "--subjects", "14", "--montage", "full"], check=True)
    subprocess.run([sys.executable, str(HERE / "tests" / "make_fake_synthetic.py"), str(tmp_path / "synth.csv")], check=True)
    env = dict(os.environ, STROKE_EEG_ROOT=str(tmp_path / "data"), SYNTH_FEATURES=str(tmp_path / "synth.csv"),
               MDPI_WORK=str(tmp_path / "work"), MDPI_RESULTS=str(tmp_path / "res"), MPLBACKEND="Agg")
    run(["A1_selection_synthetic.py", "--fast", "--methods", "all", "panel", "corr_vif", "mrmr", "glasso", "pca"], env)
    run(["A2_analyze_synthetic.py"], env)
    run(["B1_extract_variants.py"], env)
    run(["B2_measurement_properties.py"], env)
    run(["C1_clinical_severity.py", "--repeats", "2", "--n-perm", "2", "--methods", "all", "panel", "stability"], env)
    # review-stage analyses
    run(["B3_taskmatched.py", "--boot", "5", "--n-jobs", "1"], env)
    run(["B4_hierarchy.py", "--boot", "5"], env)
    run(["B6_uncertainty.py", "--boot", "10", "--boot-redundancy", "3"], env)
    run(["B5_scaling.py", "--reps", "1", "--fast"], env)
    run(["A3_hierarchical_synthetic.py", "--fast", "--n-jobs", "1", "--methods", "all", "corr_vif", "mrmr"], env)
    res = tmp_path / "res"
    a = pd.read_csv(res / "studyA" / "summary.csv")
    assert set(a.method) == {"all", "panel", "corr_vif", "mrmr", "glasso", "pca"}
    rel = pd.read_csv(res / "studyB" / "reliability.csv")
    assert rel.icc1.between(-1, 1).all() and set(rel.length_s) == {1, 2, 4, 8}
    tm = pd.read_csv(res / "studyB" / "reliability_taskmatched.csv")
    assert {"imagery", "instruction", "rest", "full8"} <= set(tm.phase)
    assert (res / "studyB" / "hierarchy_summary.csv").exists() and (res / "studyB" / "uncertainty_taskmatched.csv").exists()
    assert set(pd.read_csv(res / "studyA3" / "results.csv").method) == {"all", "corr_vif", "mrmr"}
    assoc = pd.read_csv(res / "studyC" / "association.csv")
    top = assoc[assoc.target == "nihss"].assign(a=lambda d: d.rho.abs()).sort_values("a").feature.iloc[-1]
    assert top in {"delta", "delta_alpha", "dt_ab", "theta_alpha", "sample_entropy", "higuchi_fd",
                   "magnitude_squared_coherence", "theta"}   # simulated slowing scales with NIHSS
