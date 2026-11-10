"""
Publication-style figures for one saved run (.npz) from ThreeBodyProblem.py.

Usage (from the folder containing ThreeBodyProblem.py):
    python SHP_histograms.py                              newest run in runs/, shown on screen
    python SHP_histograms.py runs/run3_seed42.npz         a specific run
    python SHP_histograms.py --save                       also write PDF + PNG to figures/<run name>/

One figure per quantity. No titles: in a report the caption carries that information.
Every theory curve is parameter-free: it uses only the masses, E and L.

  fig1a-c_momentum_time     p_i / p_max,i weighted by time, one figure per particle + arcsine theory
  fig2a-c_momentum_coll.    the same momenta sampled once per collision, one per particle (not arcsine)
  fig3_equipartition        <K_i>/E against m_i/M, with block-average error bars + theory line
  fig4_collision_positions  where collisions happen, one colour per label + theory
  fig5_positions            where each particle is, weighted by time + theory
  fig6_free_time            time between collisions, in units of the billiard prediction

Run contents: see the docstring of ThreeBodyProblem.py (label 1, 2, 3 = A, B, C = the
mass that sat out; t, dt, x_c, x, p one row per collision, just AFTER it).
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

from ThreeBodyProblem import L, X_MIN, load_run   # same ring geometry and file format


COLOURS = ["#2a78d6", "#eb6834", "#1baf7a"]      # m1, m2, m3 / labels A, B, C
THEORY = dict(color="black", ls="--", lw=1.0)    # every theory curve looks the same


parser = argparse.ArgumentParser(description="Publication-style figures for one ring run.")
parser.add_argument("filename", nargs="?",
                    help="a .npz run saved by ThreeBodyProblem.py (default: newest in runs/)")
parser.add_argument("--bins", type=int, default=40)
parser.add_argument("--save", action="store_true",
                    help="write each figure as PDF and PNG to figures/<run name>/")
args = parser.parse_args()


# --- load the run -------------------------------------------------------------------
data = load_run(args.filename, need_state=True)
if data["labels_only"]:
    raise SystemExit("This run kept only the labels - use compression_test.py, "
                     "or make a normal run for the figures.")
m, E = data["m"], float(data["E"])
t, dt = data["t"], data["dt"]
label = data["label"].astype(int)    # 1, 2, 3 = the mass that sat out
x_c = data["x_c"]
x = data["x"]
p = data["p"]                        # just after each collision; row n lasts for dt[n+1]
w = dt[1:]                           # time weight for rows 0 .. N-2

M = m.sum()
p_max = np.sqrt(2 * E * m * (M - m) / M)         # largest momentum each particle can have


# --- the triangle (mass-weighted configuration space) ---------------------------------
# wall k = "mass k sits out". Its length sets how often label k happens.
wall = L * np.sqrt(m * (M - m) / M)
perimeter = wall.sum()
s = perimeter / 2
area = np.sqrt(s * np.prod(s - wall))                       # Heron's formula
tau = np.pi * area / (perimeter * np.sqrt(2 * E))           # Santalo mean free time


# --- uniform-over-the-triangle theory for positions ------------------------------------
# The configuration is set by the gaps a (1 -> 2) and b (2 -> 3), with c = L - a - b.
# Total momentum is zero, so the centre of mass X = sum(m x)/M never moves (on the
# unwrapped ring). Given X and the gaps, every position follows:
#     x1 = X - (m2 a + m3 (a + b)) / M,   x2 = x1 + a,   x3 = x2 + b.
# Ergodic theory says the gaps are uniform over the triangle a, b >= 0, a + b <= L.

# X from row 0: rebuild unwrapped positions relative to x1 (particles never pass, so the
# order round the ring is always 1 -> 2 -> 3). Any row gives the same X, mod L.
x2_unwrapped = x[0, 0] + (x[0, 1] - x[0, 0]) % L
x3_unwrapped = x2_unwrapped + (x[0, 2] - x[0, 1]) % L
X = (m[0] * x[0, 0] + m[1] * x2_unwrapped + m[2] * x3_unwrapped) / M

def positions_from_gaps(a, b):
    x1 = X - (m[1] * a + m[2] * (a + b)) / M
    x_all = np.column_stack([x1, x1 + a, x1 + a + b])
    return X_MIN + (x_all - X_MIN) % L

rng = np.random.default_rng(0)
n_theory = 1_000_000
u = np.sort(rng.uniform(0, L, size=(n_theory, 2)), axis=1)  # uniform over the triangle
x_theory = positions_from_gaps(u[:, 0], u[:, 1] - u[:, 0])

# collisions: one gap is zero and the other is uniform on [0, L]
g = rng.uniform(0, L, n_theory)
zero = np.zeros(n_theory)
xc_theory = [positions_from_gaps(g, zero)[:, 1],             # A: m2 meets m3 (b = 0)
             positions_from_gaps(g, L - g)[:, 2],            # B: m3 meets m1 (c = 0)
             positions_from_gaps(zero, g)[:, 0]]             # C: m1 meets m2 (a = 0)


# --- printout -------------------------------------------------------------------------
K = p[:-1] ** 2 / (2 * m) / E                    # kinetic energy fractions
K_avg = np.average(K, axis=0, weights=w)
K_theory = (M - m) / (2 * M)

def block_error(n_blocks):
    blocks = np.array_split(np.arange(len(w)), n_blocks)
    block_avgs = np.array([np.average(K[b], axis=0, weights=w[b]) for b in blocks])
    return block_avgs.std(axis=0, ddof=1) / np.sqrt(n_blocks)

print(f"\nmasses {m.round(3)}   E = {E:.4f}   {len(t):,} collisions")
print("\nblock-size check (error on <K_i>/E should level off as blocks get longer)")
for n_blocks in (640, 160, 40, 20, 10):
    print(f"  {n_blocks:>4} blocks of {len(w) // n_blocks:>6,}:  {block_error(n_blocks).round(5)}")
K_err = block_error(20)

print("\n                 simulation          theory")
for i in range(3):
    print(f"  <K{i + 1}>/E      {K_avg[i]:.4f} +/- {K_err[i]:.4f}    {K_theory[i]:.4f}")
for k in range(3):
    print(f"  fraction {'ABC'[k]}    {np.mean(label == k + 1):.4f}              "
          f"{wall[k] / perimeter:.4f}")
print(f"  mean dt       {dt.mean():.5f}             {tau:.5f}")


# --- figure style ---------------------------------------------------------------------
# Single-column journal width is ~3.4 in; text sized to match a 9-10 pt caption.
plt.rcParams.update({
    "figure.figsize": (3.4, 2.5), "figure.dpi": 150, "savefig.dpi": 600,
    "font.family": "serif", "font.serif": ["DejaVu Serif"], "mathtext.fontset": "dejavuserif",
    "font.size": 9, "axes.labelsize": 9, "legend.fontsize": 8,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True,
    "axes.linewidth": 0.6, "lines.linewidth": 1.2,
    "legend.frameon": False,
    "figure.constrained_layout.use": True,
})
figures = {}


# Fig 1a, 1b, 1c: momentum of each particle, time-weighted, scaled by its p_max.
# One figure per mass. Theory is the same arcsine for all three: 1 / (pi sqrt(1 - u^2)).
edges = np.linspace(-1, 1, args.bins + 1)
curve = np.linspace(-1, 1, 801)[1:-1]
for i in range(3):
    fig, ax = plt.subplots()
    ax.hist(p[:-1, i] / p_max[i], bins=edges, weights=w, density=True,
            histtype="step", color=COLOURS[i])
    ax.plot(curve, 1 / (np.pi * np.sqrt(1 - curve ** 2)), **THEORY)
    # legend entries drawn as lines (default for a step histogram would be a box)
    ax.plot([], [], color=COLOURS[i], label=f"simulation, $m_{i + 1} = {m[i]:.2f}$")
    ax.plot([], [], **THEORY, label="arcsine theory")
    ax.set(xlim=(-1, 1), ylim=(0, 2), xticks=[-1, -0.5, 0, 0.5, 1],
           yticks=[0, 0.5, 1, 1.5, 2],
           xlabel=rf"$p_{i + 1} \,/\, p_{{\max,{i + 1}}}$", ylabel="probability density")
    ax.legend(loc="upper center")
    figures[f"fig1{'abc'[i]}_momentum_time_m{i + 1}"] = fig


# Fig 2a, 2b, 2c: the same momenta sampled once per collision, one figure per mass
for i in range(3):
    fig, ax = plt.subplots()
    ax.hist(p[:, i] / p_max[i], bins=edges, density=True,
            histtype="step", color=COLOURS[i])
    ax.plot([], [], color=COLOURS[i], label=f"simulation, $m_{i + 1} = {m[i]:.2f}$")
    ax.set(xlim=(-1, 1), ylim=(0, 3), xticks=[-1, -0.5, 0, 0.5, 1],
           yticks=[0, 1, 2, 3],
           xlabel=rf"$p_{i + 1} \,/\, p_{{\max,{i + 1}}}$", ylabel="probability density")
    ax.legend(loc="upper center")
    figures[f"fig2{'abc'[i]}_momentum_collisions_m{i + 1}"] = fig


# Fig 3: equipartition. Theory is a straight line: <K_i>/E = (1 - m_i/M) / 2
fig, ax = plt.subplots()
ax.plot([0, 1], [0.5, 0], **THEORY)
ax.axhline(1 / 3, color="0.6", ls=":", lw=1.0)                      # naive equipartition
for i in range(3):
    ax.errorbar(m[i] / M, K_avg[i], yerr=K_err[i], fmt="o", ms=4, capsize=2,
                color=COLOURS[i])
ax.set(xlim=(0, 1), ylim=(0, 0.5), xlabel=r"$m_i \,/\, M$",
       ylabel=r"$\langle K_i \rangle \,/\, E$")
figures["fig3_equipartition"] = fig


# Fig 4: collision positions, one histogram per label, each normalised per collision
fig, ax = plt.subplots()
edges = np.linspace(X_MIN, X_MIN + L, 3 * args.bins // 2 + 1)
bin_width = edges[1] - edges[0]
for k in range(3):
    ax.hist(x_c[label == k + 1], bins=edges, color=COLOURS[k], alpha=0.5,
            weights=np.full(np.sum(label == k + 1), 1 / (len(x_c) * bin_width)),
            label="ABC"[k])
    ax.hist(xc_theory[k], bins=edges, histtype="step", **THEORY,
            weights=np.full(n_theory, wall[k] / perimeter / (n_theory * bin_width)))
ax.set(xlim=(X_MIN, X_MIN + L), ylim=(0, None), xlabel=r"collision position $x_c \,/\, L$",
       ylabel="probability density")
ax.set_ylim(0, 1.3 * ax.get_ylim()[1])                    # room for the legend
ax.legend(loc="upper center", ncol=3)
figures["fig4_collision_positions"] = fig


# Fig 5: particle positions, weighted by time. Sample random times uniformly over the
# run, find which collision interval each falls in, and move the particles there.
times = rng.uniform(t[0], t[-1], 200_000)
n = np.searchsorted(t, times, side="right") - 1
x_samples = X_MIN + (x[n] + p[n] / m * (times - t[n])[:, None] - X_MIN) % L

fig, ax = plt.subplots()
edges = np.linspace(X_MIN, X_MIN + L, args.bins + 1)
for i in range(3):
    ax.hist(x_samples[:, i], bins=edges, density=True, histtype="step",
            color=COLOURS[i], label=f"$m_{i + 1}$")
    ax.hist(x_theory[:, i], bins=edges, density=True, histtype="step", **THEORY)
ax.set(xlim=(X_MIN, X_MIN + L), ylim=(0, None), xlabel=r"position $x_i \,/\, L$",
       ylabel="probability density")
ax.set_ylim(0, 1.25 * ax.get_ylim()[1])                   # room for the legend
ax.legend(loc="upper center", ncol=3)
figures["fig5_positions"] = fig


# Fig 6: time between collisions in units of the billiard mean free time
fig, ax = plt.subplots()
ax.hist(dt / tau, bins=args.bins, density=True, histtype="step", color="0.2")
ax.axvline(1, **THEORY)
ax.set(xlim=(0, None), ylim=(0, None), xlabel=r"$\Delta t \,/\, \tau$",
       ylabel="probability density")
figures["fig6_free_time"] = fig


if args.save:
    run_name = os.path.splitext(os.path.basename(data["filename"]))[0]
    folder = os.path.join("figures", run_name)
    os.makedirs(folder, exist_ok=True)
    for name, fig in figures.items():
        # fig.savefig(os.path.join(folder, f"{name}.pdf"))      # vector, for the report - UNCOMMENT FOR FINAL VERSION
        fig.savefig(os.path.join(folder, f"{name}.png"))      # quick preview
    print(f"\nsaved {len(figures)} figures (PDF + PNG) to {folder}")
else:
    plt.show()