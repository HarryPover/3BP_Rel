"""
Equipartition from many independent runs with the same masses (different starting
positions and momenta), compared with the block-average error from single runs.

If the spread across runs matches the block-average error inside one run, a single long
run explores phase space the same way many short ones do: direct evidence of ergodicity.

Usage (from the folder containing ThreeBodyProblem.py):
    python many_runs.py                               masses 2.3 5.6 9.5, 20 runs
    python many_runs.py --masses 1 2 1 --runs 20      isosceles case (spread may blow up)
"""
import argparse

import numpy as np

from ThreeBodyProblem import initialise, run

parser = argparse.ArgumentParser(description="Many seeds, same masses: equipartition spread.")
parser.add_argument("--masses", type=float, nargs=3, default=[2.3, 5.6, 9.5])
parser.add_argument("--runs", type=int, default=20)
parser.add_argument("--n", type=int, nargs="+", default=[100_000, 400_000],
                    help="collisions per run (one or more values)")
args = parser.parse_args()

m = np.array(args.masses)
M = m.sum()
theory = (M - m) / (2 * M)

for n_collisions in args.n:
    averages, block_errors = [], []
    for seed in range(args.runs):
        print(f"  seed {seed + 1}/{args.runs} done", flush=True)
        _, x, p = initialise(np.random.default_rng(seed), m=m)
        out = run(x, p, m, n_collisions)

        E = 0.5 * np.sum(out["p"][0] ** 2 / m)
        K = out["p"][:-1] ** 2 / (2 * m) / E        # row n lasts for dt[n+1]
        w = out["dt"][1:]
        averages.append(np.average(K, axis=0, weights=w))

        blocks = np.array_split(np.arange(len(w)), 20)
        block_avgs = np.array([np.average(K[b], axis=0, weights=w[b]) for b in blocks])
        block_errors.append(block_avgs.std(axis=0, ddof=1) / np.sqrt(20))

    averages, block_errors = np.array(averages), np.array(block_errors)
    print(f"\n{n_collisions:,} collisions per run, {args.runs} runs, masses {m}")
    print("  spread across runs:      ", averages.std(axis=0, ddof=1).round(4))
    print("  typical block error:     ", block_errors.mean(axis=0).round(4))
    print("  mean of runs:            ", averages.mean(axis=0).round(4),
          "+/-", (averages.std(axis=0, ddof=1) / np.sqrt(args.runs)).round(4))
    print("  theory (M - m_i) / 2M:   ", theory.round(4))