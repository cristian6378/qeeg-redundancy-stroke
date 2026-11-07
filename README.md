# Beyond the Full Cap? qEEG redundancy, repeatability and montage dependence in stroke

[![CI](https://github.com/OWNER/qeeg-redundancy-stroke/actions/workflows/ci.yml/badge.svg)](https://github.com/OWNER/qeeg-redundancy-stroke/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

This repository holds the code and the published results for the article

> **Beyond the Full Cap? Feature Redundancy, Within-Session Repeatability and Montage Dependence of
> Quantitative EEG in Stroke: A Simulation Study and Exploratory Patient Analysis.**
> S. Valchev, H. Dela Cruz, C. Rodriguez Rivero, P. Rajaram, J. Pucheta, G. Rajan.
> *Big Data and Cognitive Computing* (2026).

It evaluates 17 interpretable qEEG features (band powers, slowing ratios, distribution, complexity,
connectivity and hemispheric-asymmetry measures, defined as in ELECTRA-STROKE) in three studies.

| Study | Question | Data |
|---|---|---|
| **A** | How do ten feature-selection strategies compare in accuracy, subset size, selection stability, residual redundancy and extraction cost? | Synthetic four-class EEG feature benchmark (algorithmic stress test), plus a hierarchical benchmark with graded difficulty and collinearity |
| **B** | How repeatable is each feature within a session, and how do task phase, window length, retrospective spatial subsampling and electrode loss change the features and their redundancy? | 47 stroke patients, 40 motor-imagery trials each (Liu et al., 2024) |
| **C** | Which features show exploratory associations with contemporaneous severity (NIHSS, modified Barthel Index, mRS)? | Same patients |

Everything runs on a CPU; no GPU or deep-learning library is needed.

## Repository layout

```
pipeline/                 signal processing and feature definitions (shared with the companion studies)
  config.py               paths and settings (python pipeline/config.py checks the paths)
  data.py                 loading, channel resolution, filtering
  features.py             the 17 qEEG features
  selection.py            correlation + VIF redundancy chain
  stats_tools.py          statistics helpers
  00_check_timing.py      prints the task markers (instruction 0 s, imagery 2 s, rest 6 s)
analysis/                 the experiments of the article
  common.py               shared settings, feature families, montages, output folders
  fs_methods.py           the ten selection strategies
  measures.py             ICC, Spearman-Brown, CCC, Nogueira stability, partial Spearman
  A0-A3_*.py              Study A (synthetic benchmarks)
  B1-B6_*.py              Study B (measurement properties on patient EEG)
  C1_clinical_severity.py Study C (clinical association and prediction)
  run_mdpi.sh / .bat      Studies A-C (first submission)
  run_review.bat          analyses added in review (B3-B6, A3)
  tests/                  unit tests and the end-to-end smoke test
tests/                    unit tests of the pipeline modules and the simulated-dataset generator
results/                  the results reported in the article (CSV, LaTeX tables, figures)
```

## Installation

```bash
conda env create -f environment.yml      # or: python -m venv .venv && pip install -r requirements.txt
conda activate qeeg-redundancy
```

## Data

The data are not included in this repository.

* **Patient EEG.** Liu et al. (2024), figshare, <https://doi.org/10.6084/m9.figshare.21679035>.
  Unpack the deposit into `data/` so that `data/sourcedata/sub-XX/*.mat`,
  `data/participants.tsv` and `data/code/channel_location.locs` exist, or point
  `STROKE_EEG_ROOT` to another folder. `python pipeline/config.py` checks the paths.
  Subjects 5, 15 and 49 are excluded as duplicate recordings, leaving 47 patients.
* **Synthetic benchmark (Study A).** The synthetic EEG stroke dataset on Kaggle (Wood, 2025).
  `python analysis/A0_make_synthetic_features.py` downloads it with `kagglehub` (or use `--kaggle-dir`)
  and computes the 17 features; alternatively set `SYNTH_FEATURES` to an existing
  `electra_features_realistic.csv`.

## Running the analyses

From the `analysis` folder:

```bash
# Studies A-C as first submitted (about 2-4 h on a laptop)
./run_mdpi.sh                    # Windows: run_mdpi.bat

# analyses added in review: task-matched windows (B3), between- and within-patient redundancy (B4),
# computational scaling (B5), paired bootstrap intervals (B6), hierarchical benchmark (A3)
run_review.bat check             # quick check, a few minutes
run_review.bat                   # full run, about 1-3 h
```

Each script can also be run alone; `python <script> --help` lists its options. Intermediate files go to
`work/` and new results to `output/` (override with `MDPI_WORK` and `MDPI_RESULTS`). B3 and B6 read the
features that B1 writes, so run B1 first.

| Script | Produces |
|---|---|
| `A1_selection_synthetic.py`, `A2_analyze_synthetic.py` | strategy comparison, Table A, Pareto and stability figures |
| `A3_hierarchical_synthetic.py` | hierarchical benchmark, 9 scenarios of difficulty and collinearity |
| `B1_extract_variants.py`, `B2_measurement_properties.py` | features per window length, montage and electrode loss; ICC, CCC and redundancy |
| `B3_taskmatched.py` | repeatability in windows inside one task phase, per hand, with a quality screen |
| `B4_hierarchy.py` | redundancy for pooled trials, patient means and within-patient deviations |
| `B5_scaling.py` | extraction time and memory against channel count and window length |
| `B6_uncertainty.py` | paired patient-bootstrap 95% intervals for repeatability differences, montage redundancy and electrode loss |
| `C1_clinical_severity.py` | association with NIHSS, MBI and mRS; out-of-fold NIHSS prediction |

## Published results

`results/` contains the outputs behind every table and figure of the article: `studyA`, `studyA3`,
`studyB` (including the task-matched, hierarchy, scaling and uncertainty files) and `studyC`. The
per-trial predictions of Study A (6.5 MB) and the copy of the synthetic table are not included; they
are regenerated by `A1_selection_synthetic.py`.

## Tests

```bash
pytest -q                                              # unit tests
SMOKE=1 pytest -q analysis/tests/test_smoke_mdpi.py    # every script on a simulated dataset (about 10 min)
```

The smoke test builds a simulated stand-in for the patient deposit (`tests/make_fake_dataset.py`) and
for the synthetic table (`analysis/tests/make_fake_synthetic.py`), then runs Studies A-C and the
review analyses end to end. GitHub Actions runs linting, the unit tests on Python 3.10-3.12 and the
smoke test on every push.

## Citation

See [`CITATION.cff`](CITATION.cff). If you use the patient data, also cite Liu et al. (2024).

## License

MIT, see [LICENSE](LICENSE).
