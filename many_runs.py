"""
Equipartition from many independent runs with the same masses (different starting
positions and momenta), compared with the block-average error from single runs.

If the spread across runs matches the block-average error inside one run, a single long
run explores phase space the same way many short ones do: direct evidence of ergodicity.

Usage (from the folder containing ThreeBodyProblem.py):
    python many_runs.py                               masses 2.3 5.6 9.5, 20 runs
    python many_runs.py --masses 1 2 1 --runs 20      isosceles case (spread may blow up)
    python many_runs.py --masses 1 2 3 --rel --K 1e-4 1 100 --runs 5 --n 100000
"""
import argparse

import numpy as np

from ThreeBodyProblem import initialise, run, pick_physics, set_energy

parser = argparse.ArgumentParser(description="Many seeds, same masses: equipartition spread.")
parser.add_argument("--masses", type=float, nargs=3, default=[2.3, 5.6, 9.5])
parser.add_argument("--runs", type=int, default=20)
parser.add_argument("--n", type=int, nargs="+", default=[100_000, 400_000],
                    help="collisions per run (one or more values)")
parser.add_argument("--rel", action="store_true", help="relativistic physics (default: Newtonian)")
parser.add_argument("--K", type=float, nargs='+', default=None, help="target kinetic energy (default: keep initialise own energy)")
args = parser.parse_args()

velocity, kinetic, _ = pick_physics(args.rel)

m = np.array(args.masses)
M = m.sum()
newt_theory = (M - m) / (2 * M)

for K in (args.K or [None]):
    for n_collisions in args.n:
        averages, block_errors = [], []
        for seed in range(args.runs):
            _, x, p = initialise(np.random.default_rng(seed), m=m)
            if K is not None:
                p = set_energy(p, m, K, kinetic)
            out = run(x, p, m, n_collisions, rel=args.rel)
            print(f"  seed {seed + 1}/{args.runs} done", flush=True)

            E = kinetic(out["p"][0], m).sum()
            K_values = kinetic(out["p"][:-1], m) / E
            w = out["dt"][1:]
            averages.append(np.average(K_values, axis=0, weights=w))

            blocks = np.array_split(np.arange(len(w)), 20)
            block_avgs = np.array([np.average(K_values[b], axis=0, weights=w[b]) for b in blocks])
            block_errors.append(block_avgs.std(axis=0, ddof=1) / np.sqrt(20))

        averages, block_errors = np.array(averages), np.array(block_errors)
        print(f"\n{n_collisions:,} collisions per run, {args.runs} runs, masses {m}, K={K}")
        print("  spread across runs:      ", averages.std(axis=0, ddof=1).round(4))
        print("  typical block error:     ", block_errors.mean(axis=0).round(4))
        print("  mean of runs:            ", averages.mean(axis=0).round(4),
              "+/-", (averages.std(axis=0, ddof=1) / np.sqrt(args.runs)).round(4))
        if args.rel and K is not None and K > 1e-3:
            print("  WARNING: relativistic, K > 1e-3: theory below is Newtonian")
        print("  Newtonian theory:        ", newt_theory.round(4))
            