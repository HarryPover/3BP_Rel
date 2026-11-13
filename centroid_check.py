import sys
import numpy as np
from ThreeBodyProblem import load_run

d = load_run(sys.argv[1] if len(sys.argv) > 1 else None, need_state=True, rel=True)
m, p, p0, L = d["m"], d["p"], d["p0"], float(d["L"])
label = d["label"].astype(int)
XE = d["X_moller"]

#1. Jump test 

#XE after collision n vs before it(before collision 0 = start state)
E_all = np.hypot(m,np.vstack([p0, p]))
E_tot = E_all[0].sum()
xu0 = d["x0"]
XE_before =np.concatenate([[np.sum(E_all[0] * xu0) / E_tot], XE[:-1]])
dXE = XE - XE_before
dE3 = E_all[1:, 2] - E_all[:-1, 2]
predicted = np.where(label == 2, L * dE3 / E_tot, 0.0)

for k, name in ((1, "A",), (2, "B"), (3, "C")):
    sel = label == k
    print(f"{name} collisions: max |dX_E - prediction| = {np.abs(dXE[sel] - predicted[sel]).max():.2e}"
          f"   (typical |dX_E| = {np.median(np.abs(dXE[sel])):.2e})")


#2. Diffusive or Ballistic
print("\nlag      var[X_rest(n+lag) - X_rest(n)]   ratio to previous") 

X = d["X_rest"]
prev = None
for lag in (10, 100, 1000, 10_000, 100_000):
    if lag > len(X) // 10:          # fewer than ~10 independent windows: unreliable
        print(f"{lag:>6}   (skipped: run too short, need n > {10 * lag:,})")
        continue
    v = np.var(X[lag:] - X[:-lag])
    print(f"{lag:>6}   {v:.3e}" + (f"        {v / prev:.1f}" if prev else ""))
    prev = v

print("ratio ~10 per decade: diffusive (random walk);  ~100: ballistic (steady drift)")