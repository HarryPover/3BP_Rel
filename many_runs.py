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

from ThreeBodyProblem import initialise, run, pick_physics, set_energy, santalo_tau

import os

RESULTS_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scans")


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

#save results to .npz
saved = {
    "K": [],
    "n": [],
    "shares": [],        # energy shares (averages)
    "pv_shares": [],
    "fracs": [],
    "dt_ratios": [],
    "var_xs": [],
    "block_errors": [],
}



for K in (args.K or [None]):
    for n_collisions in args.n:
        averages, block_errors, fracs, dt_ratios, var_xs = [], [], [], [], []
        pv_shares = []
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
            pv = out["p"][:-1] * velocity(out["p"][:-1], m)
            time_avgs = np.average(pv, axis=0, weights=w)
            pv_shares.append(time_avgs / time_avgs.sum())

            #collision fractions for A, B, C
            frac = np.array([np.mean(out["label"] == k) for k in (1, 2, 3)])
            fracs.append(frac)

            #mean dt/tau
            tau, wall, perimeter = santalo_tau(M, m, E)
            
            mean_adjusted_time = out["dt"].mean() / tau
            dt_ratios.append(mean_adjusted_time)

            #Variancee of X_rest change at lag 1000
            X = out["X_rest"]
            var_x_rest = np.var(X[1000:] - X[:-1000]) if len(X) >= 10_000 else np.nan
            var_xs.append(var_x_rest)   

            blocks = np.array_split(np.arange(len(w)), 20)
            block_avgs = np.array([np.average(K_values[b], axis=0, weights=w[b]) for b in blocks])
            block_errors.append(block_avgs.std(axis=0, ddof=1) / np.sqrt(20))

        averages, block_errors, fracs, dt_ratios, var_xs = np.array(averages), np.array(block_errors), np.array(fracs), np.array(dt_ratios), np.array(var_xs)
        pv_shares = np.array(pv_shares)
        
        print(f"\n{n_collisions:,} collisions per run, {args.runs} runs, masses {m}, K={K}")
        if args.rel and K is not None and K > 1e-3:
            print("  WARNING: relativistic, K > 1e-3: 'Newtonian' values are for comparison only")
        print("  spread across runs:      ", averages.std(axis=0, ddof=1).round(4))
        print("  typical block error:     ", block_errors.mean(axis=0).round(4))
        print("  mean of runs:            ", averages.mean(axis=0).round(4),
              "+/-", (averages.std(axis=0, ddof=1) / np.sqrt(args.runs)).round(4),
              "  Newtonian:", newt_theory.round(4))
        print("  <p v> shares: ", pv_shares.mean(axis=0).round(4),
                "+/-", (pv_shares.std(axis=0, ddof=1) / np.sqrt(args.runs)).round(4),
                "  Newtonian:", newt_theory.round(4))
        print("  collision fractions:     ", fracs.mean(axis=0).round(4),
              "+/-", (fracs.std(axis=0, ddof=1) / np.sqrt(args.runs)).round(4),
              "  Newtonian:", (wall / perimeter).round(4))
        print("  mean dt/tau:             ", dt_ratios.mean().round(4),
              "+/-", (dt_ratios.std(ddof=1) / np.sqrt(args.runs)).round(4),
              "  Newtonian:", 1.0)
        print(f"  var dX_rest (lag 1000):   {np.nanmean(var_xs):.3e} +/- {np.nanstd(var_xs, ddof=1)/np.sqrt(args.runs):.3e}")

        #append blocks array
        saved["K"].append(np.nan if K is None else K)
        saved["n"].append(n_collisions)
        saved["shares"].append(averages)
        saved["pv_shares"].append(pv_shares)
        saved["fracs"].append(fracs)
        saved["dt_ratios"].append(dt_ratios)
        saved["var_xs"].append(var_xs)
        saved["block_errors"].append(block_errors)



os.makedirs(RESULTS_FOLDER, exist_ok=True)
name = f"scan_m{m[0]:g}-{m[1]:g}-{m[2]:g}_{'rel' if args.rel else 'newt'}.npz"
filename = os.path.join(RESULTS_FOLDER, name)
print(f"\nSaving results to {filename}")
np.savez(filename, m=m, rel=args.rel, runs=args.runs,
         **{key: np.array(value) for key, value in saved.items()})
print(f"saved {filename}")