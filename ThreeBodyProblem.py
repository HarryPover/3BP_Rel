import numpy as np
import os
import glob

#geometry of ring
X_MIN = 0
L = 1.0

#saving runs
RUNS_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")


#physics

def pick_physics(rel):
    """Return the functions that are effected by relativistic dynamics.
    Chosen once per run so newtonian and relativistic effects dont get mixed up."""
    if rel:
        velocity = lambda p, m: p / np.hypot(m, p)
        kinetic = lambda p, m: p**2 / (np.hypot(m, p) + m)
        collision = collision_rel
    else:
        velocity = lambda p, m: p / m
        kinetic = lambda p, m: 0.5 * p**2 / m
        collision = collision_newton
    return velocity, kinetic, collision

def collision_newton(p, m, behind, ahead):
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

def collision_rel(p, m, behind, ahead):
    """
    Relativistic 1D elastic collision (c = 1).
    In the pair's zero-momentum frame (rapidity zeta, tanh zeta = P/E) both momenta reverse,
    so each rapidity reflects: eta' = 2*zeta - eta. Bystander unchanged.
    """
    # Relativistic elastic collision using rapidity.
    p_new = p.copy()
    pair = np.array([behind, ahead])
    E = np.hypot(m[pair], p[pair])
    eta = np.arcsinh(p[pair] / m[pair])
    P = np.sum(p[pair])
    E_pair = np.sum(E)
    zeta = np.arctanh(P / E_pair)
    eta_new = 2 * zeta - eta
    p_new[pair] = m[pair] * np.sinh(eta_new)
    return p_new

def set_energy(p, m, K, kinetic):
    """Scale p by one factor s so that sum(kinetic(s*p, m)) = K * sum(m).
    kinetic comes from pick_physics(rel).
    Looking for point of bisection between K(s) and target K * m"""

    #guard for momenta being all 0:
    if np.all(p == 0):
        raise ValueError("all momenta are zero: no scale factor can give energy K")

    target = K * np.sum(m)  #energy we want, in units where c=1

    def excess(s):
        return kinetic(s * p, m).sum() - target   #negative if s too small, positive if s too large
    
    lo = 0.0
    hi = 1.0

    #1. bracket: find lo and hi
    while excess(hi) < 0:   #p too small, need to scale up
        lo = hi
        hi *= 2

    #2. bisection: shrink bracket to answer
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if excess(mid) < 0:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-15 * hi:
            break

    s = 0.5 * (lo + hi)

    return s * p


    


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



def step(x, xu, p, m, velocity, collision):

    v = velocity(p, m)  # Calculate velocities based on momenta and masses

    
    times = np.array([time_to_meet(i, x, v) for i in range(3)])
    i = int(np.argmin(times))  # Index of the next collision
    dt = times[i]
    if not np.isfinite(dt):
        raise RuntimeError("No more collisions will occur - all particles at rest?")

    x_new = X_MIN + ((x+v*dt) - X_MIN) % L   # Update positions with periodic boundary conditions

    behind, ahead = (i + 1) % 3, (i + 2) % 3
    x_new[behind] = x_new[ahead]  # Ensure particles meet at the same position
    x_c = x_new[behind]  # Collision position
    p_new = collision(p, m, behind, ahead)  # Update momenta after collision

     # unwrapped positions: bookkeeping only, momenta untouched
    xu_new = xu + v * dt                  # same drift, no mod L
    if behind == 2 and ahead == 0:        # label B, across the seam
        xu_new[2] = xu_new[0] + L         # bead 3 sits on bead 1's image
    else:
        xu_new[behind] = xu_new[ahead]    # snap, like the wrapped version
    
    return x_new,xu_new, p_new, dt, x_c, i + 1,  # Return the time step, collision position, and label (1-based index)


#RUN
def run(x, p, m, n_collisions, rel=False):
    """
    Run n_collisions steps from the given state.
 
    Returns a dict of arrays, one row per collision (state recorded AFTER it):
      t (cumulative time), dt, k, x_c, section, x (n, 3), p (n, 3)
    """
    xu = x.copy()

    dts = np.empty(n_collisions)
    labels = np.empty(n_collisions, dtype= np.int8) 
    xcs = np.empty(n_collisions)
    xs = np.empty((n_collisions, 3))
    ps = np.empty((n_collisions, 3))
    xus = np.empty((n_collisions, 3))
    X_rest = np.empty(n_collisions)
    X_moller = np.empty(n_collisions)

    velocity, kinetic, collision = pick_physics(rel)
    X_rest = np.empty(n_collisions)

    for n in range(n_collisions):
        x, xu, p, dt, x_c, label = step(x, xu, p, m, velocity, collision)
        
        dts[n], labels[n], xcs[n] = dt, label, x_c
        xs[n], ps[n] = x, p
        xus[n] = xu
        X_rest[n] = np.sum(m * xu) / np.sum(m)  
        E = np.hypot(m,p)
        X_moller[n] = np.sum(E * xu) / np.sum(E) if rel else np.nan
 
    return {"dt": dts, "t": np.cumsum(dts), "label": labels, "x_c": xcs, "x": xs, "p": ps,
     "xu": xus, "X_rest": X_rest, "X_moller": X_moller}
 

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




def save_run(out, m, x0, p0, seed, labels_only=False, rel=False, name=None, K=None):
    """Save one run as runs/runK_seedS.npz (K = next free run number)."""
    os.makedirs(RUNS_FOLDER, exist_ok=True)
    if name is None:
        used = [int(f[3:].split("_")[0]) for f in os.listdir(RUNS_FOLDER)
                    if f.startswith("run") and f[3:].split("_")[0].isdigit() and f.endswith(".npz")]
        k = max(used, default=0) + 1
        mode = "_rel" if rel else "_newt"
        tag = "_labels" if labels_only else ""
        name = f"run{k}_seed{seed}{mode}{tag}.npz"
    if not name.endswith(".npz"):
        name += ".npz"
    filename = os.path.join(RUNS_FOLDER, name)

    velocity, kinetic, collision = pick_physics(rel)
    E = np.sum(kinetic(p0, m))
    
    np.savez(filename, m=m, seed=seed, E=E, L=L, X_MIN=X_MIN, x0=x0, p0=p0,
             labels_only=labels_only, relativistic = rel,K=np.nan if K is None else K,**out)
    return filename



def is_relativistic(filename):
    """Read the stored flag. Old runs have no flag: they were Newtonian."""
    with np.load(filename) as f:
        return bool(f["relativistic"]) if "relativistic" in f.files else False
 
def load_run(filename=None, need_state=False, rel=None):
    """
    Load a run saved by save_run as a dict of arrays (keys as in the module docstring,
    plus "filename"). With no filename, load the newest .npz in runs/ - skipping
    labels-only runs (names ending _labels.npz) if need_state is True, and keeping only relatistic (rel=True)
    or only Newtonian (rel=False) runs if rel is given. rel=None means either.
    """
    if filename is None:
        found = glob.glob(os.path.join(RUNS_FOLDER, "*.npz"))
        if need_state:
            found = [f for f in found if not f.endswith("_labels.npz")]
        if rel is not None:
            found = [f for f in found if is_relativistic(f) == rel]
        if not found:
            which = {None: "", True: "relativistic", False: "Newtonian"}[rel]
            raise SystemExit(f"No {which}.npz runs in {RUNS_FOLDER} - run ThreeBodyProblem.py first.")
        
        filename = max(found, key=os.path.getmtime)
        print(f"using newest run: {filename}")
    if not filename.endswith(".npz"):
        raise SystemExit(f"{filename}: runs are now saved as .npz. The old .csv runs were "
                         "preliminary tests - make a new run with ThreeBodyProblem.py.")
    with np.load(filename) as f:
        data = {key: f[key] for key in f.files}
    data["filename"] = filename
    data.setdefault("relativistic", np.array(False))     # old runs: Newtonian
 
    if rel is not None and bool(data["relativistic"]) != rel:
        raise SystemExit(f"{filename} is {'relativistic' if data['relativistic'] else 'Newtonian'}, "
                         f"but a {'relativistic' if rel else 'Newtonian'} run was asked for.")
 
    # one header line, so you can check which run a plot came from
    mode = "relativistic" if data["relativistic"] else "Newtonian"
    print(f"loaded {os.path.basename(filename)}: {mode}, masses {np.round(data['m'], 3)}, "
          f"E = {float(data['E']):.4g}, E/M = {float(data['E']) / data['m'].sum():.3g}, "
          f"{len(data['label']):,} collisions")
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
    parser.add_argument("--rel", action="store_true", help="use relativistic dynamics")
    parser.add_argument("--K", type=float, default=None, help="set totale kinetic energy to K * Mc^2 (default: leave initialse's energy)")
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
    if args.K is not None:
        velocity, kinetic, collision = pick_physics(args.rel)
        p = set_energy(p, m, args.K, kinetic)
    print(f"seed = {seed}")
    print(f"masses: m1 = {m[0]:.3f}, m2 = {m[1]:.3f}, m3 = {m[2]:.3f}")
    print(f"start positions: x1 = {x[0]:.3f}, x2 = {x[1]:.3f}, x3 = {x[2]:.3f}\n")
    x0, p0 = x.copy(), p.copy()
 
    if args.labels_only:
        # run in chunks of 10^6 so memory stays small; keep only the labels
        chunks = []
        while sum(len(c) for c in chunks) < args.n:
            size = min(1_000_000, args.n - sum(len(c) for c in chunks))
            out = run(x, p, m, size, rel=args.rel)
            chunks.append(out["label"])
            x, p = out["x"][-1], out["p"][-1]     # carry the state into the next chunk
            print(f"  {sum(len(c) for c in chunks):,} / {args.n:,}")
        out = {"label": np.concatenate(chunks)}
    else:
        out = run(x, p, m, args.n, rel=args.rel)
        print_collisions(out, 20)
 
    labels, counts = np.unique(out["label"], return_counts=True)
    print("\ncollisions per label:", {"ABC"[k - 1]: int(c) for k, c in zip(labels, counts)})
 
    filename = save_run(out, m, x0, p0, seed, labels_only=args.labels_only, rel=args.rel,
                        name=args.save, K=args.K)
    print(f"\nsaved {filename}")
 