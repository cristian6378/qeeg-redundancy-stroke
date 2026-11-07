"""Study B, step 5 (review revision): extraction cost versus channel count and window length.

Measures, for each feature family, the wall-clock time of the Python implementation (single thread) for
n channels in (2, 4, 8, 19, 29) and windows of (1, 2, 4, 8) s at 500 Hz, with repeats (median, IQR), the peak
memory of one extraction (tracemalloc, Python allocations only), and the empirical scaling exponents
(log-log slope against channels and against samples). Hardware and library versions are written to
hardware.json. These are RELATIVE costs of a desktop CPU implementation; they do not establish
microcontroller, power or latency feasibility.

  python B5_scaling.py [--reps 7] [--fast]    (the signal is always coloured Gaussian noise; timing does not depend on content)
Outputs (MDPI_RESULTS/studyB/): scaling.csv, scaling_exponents.csv, hardware.json, fig_scaling.png
"""
import argparse
import json
import platform
import time
import tracemalloc

import numpy as np
import pandas as pd

import common as K

OUT = K.OUT / "studyB"
CHANNELS = (2, 4, 8, 19, 29)
LENGTHS = (1, 2, 4, 8)


def _signal(n_ch, n, rng):
    x = rng.standard_normal((n_ch, n))
    return np.cumsum(x, axis=1) * 0.05 + x          # 1/f-like mixture, enough for timing


def components(fs):
    import features as F

    def bands(x, names, pairs):
        return F.band_powers(x, fs)

    def sampen(x, names, pairs):
        return [F.sample_entropy(c) for c in x]

    def hfd(x, names, pairs):
        return [F.higuchi_fd(c) for c in x]

    def wpli(x, names, pairs):
        return F.wpli_all_pairs(x)

    def msc(x, names, pairs):
        return F.msc_all_pairs(x, fs)

    return {"band powers": bands, "sample entropy": sampen, "Higuchi FD": hfd, "wPLI": wpli, "MSC": msc}


def run(reps, fs, seed=K.SEED, channels=CHANNELS, lengths=LENGTHS):
    rng = np.random.default_rng(seed)
    comps = components(fs)
    rows = []
    for n_ch in channels:
        for L in lengths:
            x = _signal(n_ch, int(L * fs), rng)
            for name, fn in comps.items():
                fn(x, None, None)                                    # warm-up
                ts = []
                for _ in range(reps):
                    t0 = time.perf_counter(); fn(x, None, None); ts.append((time.perf_counter() - t0) * 1e3)
                tracemalloc.start(); fn(x, None, None); peak = tracemalloc.get_traced_memory()[1]; tracemalloc.stop()
                rows.append({"component": name, "n_channels": n_ch, "length_s": L, "n_samples": x.shape[1],
                             "median_ms": float(np.median(ts)), "q25_ms": float(np.percentile(ts, 25)),
                             "q75_ms": float(np.percentile(ts, 75)), "peak_mem_kb": peak / 1024})
    return pd.DataFrame(rows)


def exponents(df):
    rows = []
    for name, g in df.groupby("component"):
        gl = g[g.length_s == g.length_s.max() if 4 not in set(g.length_s) else g.length_s == 4]
        gc = g[g.n_channels == g.n_channels.max()]
        a = np.polyfit(np.log(gl.n_channels), np.log(gl.median_ms), 1)[0]
        b = np.polyfit(np.log(gc.n_samples), np.log(gc.median_ms), 1)[0]
        rows.append({"component": name, "exponent_channels": a, "exponent_samples": b})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--fast", action="store_true", help="2 and 4 channels, 1- and 2-s windows (tests)")
    args = ap.parse_args()
    import config as C
    K.ensure_dirs(); OUT.mkdir(parents=True, exist_ok=True)
    df = run(args.reps, C.FS_RAW, channels=(2, 4) if args.fast else CHANNELS, lengths=(1, 2) if args.fast else LENGTHS)
    df.to_csv(OUT / "scaling.csv", index=False)
    ex = exponents(df); ex.to_csv(OUT / "scaling_exponents.csv", index=False)
    import scipy, sklearn
    (OUT / "hardware.json").write_text(json.dumps({"platform": platform.platform(), "processor": platform.processor(),
        "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
        "sklearn": sklearn.__version__, "threads": "single (timing loop)", "fs_hz": C.FS_RAW}, indent=2))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for name, g in df[df.length_s == (4 if 4 in set(df.length_s) else df.length_s.max())].groupby("component"):
        ax[0].plot(g.n_channels, g.median_ms, "o-", label=name)
    for name, g in df[df.n_channels == df.n_channels.max()].groupby("component"):
        ax[1].plot(g.length_s, g.median_ms, "o-", label=name)
    for a_, xl in zip(ax, ("channels (4-s window)", "window length, s (29 channels)")):
        a_.set_xscale("log"); a_.set_yscale("log"); a_.set_xlabel(xl); a_.set_ylabel("median time, ms")
    ax[0].legend(fontsize=8); fig.tight_layout(); fig.savefig(OUT / "fig_scaling.png", dpi=200)
    print(ex.to_string()); print("done ->", OUT)


if __name__ == "__main__":
    main()
