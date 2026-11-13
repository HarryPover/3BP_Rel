"""Quick health check of one saved run: mode, energy target, conservation."""
import sys
import numpy as np
from ThreeBodyProblem import load_run


d = load_run(sys.argv[1] if len(sys.argv) > 1 else None, need_state=True)
m, p, p0 = d["m"], d["p"], d["p0"]
rel = bool(d["relativistic"])

if rel:
    E_tot = np.hypot(m, p).sum(axis=1)          # total energy incl. rest mass, every row
    E0 = np.hypot(m, p0).sum()
    K0 = (p0**2 / (np.hypot(m, p0) + m)).sum()
else:
    E_tot = (0.5 * p**2 / m).sum(axis=1)
    E0 = K0 = (0.5 * p0**2 / m).sum()

print(f"K target       {float(d['K']):.6g}   (nan = not set)")
print(f"K0 / M         {K0 / m.sum():.6g}")
print(f"max |sum p|    {np.abs(p.sum(axis=1)).max():.2e}   (scale: sum|p| = {np.abs(p0).sum():.3g})")
print(f"max |E - E0|/E0 {np.abs(E_tot - E0).max() / E0:.2e}")
print(f"fastest v/c    {np.max(np.abs(p) / np.hypot(m, p)):.4f}" if rel else "")

# --- unwrapped positions and centroids ---
if "xu" in d:
    L_ring = float(d["L"])
    # circular difference, so a bead sitting right on the seam (x = 0 vs xu % L = 0.999...) isn't flagged
    diff = (d["xu"] % L_ring - d["x"] + L_ring / 2) % L_ring - L_ring / 2
    print(f"max |xu mod L - x| {np.abs(diff).max():.2e}   (expect ~1e-12)")
    print(f"X_rest   spread   {np.ptp(d['X_rest']):.3e}   (Newton: ~1e-8 rounding floor; relativistic: clearly larger)")
    if rel: print(f"X_moller spread   {np.ptp(d['X_moller']):.3e}")