import numpy as np
import os
import glob

#geometry of ring
X_MIN = 0
L = 1.0

#saving runs
RUNS_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")


#physics
def collision(p, m, behind, ahead):
    # Elastic collision in 1D
    #conservation of momentum and energy

    p_new = p.copy()
    p1, p2 = p[behind], p[ahead]
    m1, m2 = m[behind], m[ahead]
    M = m1 + m2

    # Calculate new momenta after collision
    p_new[behind] = (m1-m2)/(M)*p1 + (2*m1)/(M)*p2
    p_new[ahead] = (2*m2)/(M)*p1 + (m2-m1)/(M)*p2 
    return p_new


#initialise system
def initialise(rng, m= None, m_range=(1.0, 10.0), x_range=(X_MIN, X_MIN + L), p_range=(-5.0, 5.0)):
    """
    Three particles on the ring. Returns m, x, p as length-3 NumPy arrays:
      x sorted so 0 -> 1 -> 2 runs anticlockwise; p sums to exactly zero.
    """
    
    if m is None:
        m = rng.uniform(*m_range, size=3)
    m = np.array(m, dtype=float)
    x = np.sort(rng.uniform(*x_range, size=3))  # Sort positions to ensure 0 -> 1 -> 2 anticlockwise

    p = np.empty(3)
    p[:2] = rng.uniform(*p_range, size=2)
    p[2] = -(p[0] + p[1])
    return m, x, p

def time_to_meet(i, x, v):

    # 2 masses either side of gap i
    behind, ahead = (i + 1) % 3, (i + 2) % 3
    gap = (x[ahead] - x[behind]) % L  # Distance to the next particle in a circular domain

    closing_velocity = v[behind] - v[ahead]  # Relative velocity

    if closing_velocity <= 0:
        return np.inf
    return gap / closing_velocity


#collision
def step(x, p, m):
    #Velocity
    v = p/m

    times = np.array([time_to_meet(i, x, v) for i in range(3)])
    i = int(np.argmin(times))  # Index of the next collision
    dt = times[i]
    if not np.isfinite(dt):
        raise RuntimeError("No more collisions will occur - all particles at rest?")

    x_new = X_MIN + ((x+v*dt) - X_MIN) % L   # Update positions with periodic boundary conditions

    behind, ahead = (i + 1) % 3, (i + 2) % 3
    x_new[behind] = x_new[ahead]  # Ensure particles meet at the same position
    x_c = x_new[behind]  # Collision position
    p_new = collision(p, m, behind, ahead)  

    return x_new, p_new, dt, x_c, i + 1


#RUN
def run(x, p, m, n_collisions):
    """
    Run n_collisions steps from the given state.
 
    Returns a dict of arrays, one row per collision (state recorded AFTER it):
      t (cumulative time), dt, k, x_c, section, x (n, 3), p (n, 3)
    """
    
    dts = np.empty(n_collisions)
    labels = np.empty(n_collisions, dtype= np.int8) 
    xcs = np.empty(n_collisions)
    xs = np.empty((n_collisions, 3))
    ps = np.empty((n_collisions, 3))
 
    for n in range(n_collisions):
        x, p, dt, x_c, label = step(x, p, m)
        dts[n], labels[n], xcs[n] = dt, label, x_c
        xs[n], ps[n] = x, p
 
    return {"dt": dts, "t": np.cumsum(dts), "label": labels, "x_c": xcs, "x": xs, "p": ps}
 

PAIR_OF = {1: "2-3", 2: "1-3", 3: "1-2"}


def print_collisions(out, n_rows=20):

    """Print the first n_rows collisions as a table."""

    print(f"{'n':>3} {'dt':>7} {'label':>8} {'pair':>5}"
          f" {'x1':>7} {'x2':>7} {'x3':>7}  {'p1':>7} {'p2':>7} {'p3':>7}")
    
    for n in range(min(n_rows, len(out["dt"]))):
        x1, x2, x3 = out["x"][n]
        p1, p2, p3 = out["p"][n]
        lab = int(out["label"][n])
        print(f"{n:>3} {out['dt'][n]:7.4f} {'ABC'[lab-1]:>8} {PAIR_OF[lab]:>5}"
              f" {x1:7.3f} {x2:7.3f} {x3:7.3f}  {p1:7.3f} {p2:7.3f} {p3:7.3f}")


def save_run(out, m, x0, p0, seed, labels_only=False):
    """Save one run as runs/runK_seedS.npz (K = next free run number)."""
    os.makedirs(RUNS_FOLDER, exist_ok=True)
    used = [int(f[3:].split("_")[0]) for f in os.listdir(RUNS_FOLDER)
                if f.startswith("run") and f.endswith(".npz")]
    k = max(used, default=0) + 1
    tag = "_labels" if labels_only else ""
    filename = os.path.join(RUNS_FOLDER, f"run{k}_seed{seed}{tag}.npz")
 
    E = 0.5 * np.sum(p0**2 / m)  # Initial energy
    np.savez(filename, m=m, seed=seed, E=E, L=L, X_MIN=X_MIN, x0=x0, p0=p0,
             labels_only=labels_only, **out)
    return filename
 
def load_run(filename=None, need_state=False):
    """
    Load a run saved by save_run as a dict of arrays (keys as in the module docstring,
    plus "filename"). With no filename, load the newest .npz in runs/ - skipping
    labels-only runs (names ending _labels.npz) if need_state is True.
    """
    if filename is None:
        found = glob.glob(os.path.join(RUNS_FOLDER, "*.npz"))
        if need_state:
            found = [f for f in found if not f.endswith("_labels.npz")]
        if not found:
            raise SystemExit(f"No .npz runs in {RUNS_FOLDER} - run ThreeBodyProblem.py first.")
        filename = max(found, key=os.path.getmtime)
        print(f"using newest run: {filename}")
    if not filename.endswith(".npz"):
        raise SystemExit(f"{filename}: runs are now saved as .npz. The old .csv runs were "
                         "preliminary tests - make a new run with ThreeBodyProblem.py.")
    with np.load(filename) as f:
        data = {key: f[key] for key in f.files}
    data["filename"] = filename
    return data
 
 
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Three particles colliding on a ring.")
    parser.add_argument("--masses", type=float, nargs=3, metavar=("M1", "M2", "M3"),
                        help="fix the masses (default: random in [1, 10])")
    parser.add_argument("--seed", type=int, help="random seed (default: a fresh one)")
    parser.add_argument("--n", type=int, default=100_000, help="number of collisions")
    parser.add_argument("--labels-only", action="store_true",
                        help="keep only the collision labels (for very long runs)")
    parser.add_argument("--show", metavar="FILE", help="print the first collisions of a saved run")
    #save file under chosen name
    parser.add_argument("--save", metavar="FILE", help="save the run under this name instead of the default")
    args = parser.parse_args()
 
    if args.show:
        data = load_run(args.show)
        print(f"masses {data['m']}   seed {data['seed']}   E = {data['E']:.6g}   "
              f"{len(data['label']):,} collisions")
        if data["labels_only"]:
            print("labels only:", "".join("ABC"[k - 1] for k in data["label"][:60]), "...")
        else:
            print_collisions(data, 20)
        raise SystemExit
 
    seed = args.seed if args.seed is not None else int(np.random.SeedSequence().entropy % 1_000_000_000)
    rng = np.random.default_rng(seed)
    m, x, p = initialise(rng, m=args.masses)
    print(f"seed = {seed}")
    print(f"masses: m1 = {m[0]:.3f}, m2 = {m[1]:.3f}, m3 = {m[2]:.3f}")
    print(f"start positions: x1 = {x[0]:.3f}, x2 = {x[1]:.3f}, x3 = {x[2]:.3f}\n")
    x0, p0 = x.copy(), p.copy()
 
    if args.labels_only:
        # run in chunks of 10^6 so memory stays small; keep only the labels
        chunks = []
        while sum(len(c) for c in chunks) < args.n:
            size = min(1_000_000, args.n - sum(len(c) for c in chunks))
            out = run(x, p, m, size)
            chunks.append(out["label"])
            x, p = out["x"][-1], out["p"][-1]     # carry the state into the next chunk
            print(f"  {sum(len(c) for c in chunks):,} / {args.n:,}")
        out = {"label": np.concatenate(chunks)}
    else:
        out = run(x, p, m, args.n)
        print_collisions(out, 20)
 
    labels, counts = np.unique(out["label"], return_counts=True)
    print("\ncollisions per label:", {"ABC"[k - 1]: int(c) for k, c in zip(labels, counts)})
 
    filename = save_run(out, m, x0, p0, seed, labels_only=args.labels_only)
    if args.save:
        filename = args.save
    print(f"\nsaved {filename}")
 