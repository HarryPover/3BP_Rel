"""Tests for the relativistic collision."""
import numpy as np
from ThreeBodyProblem import collision_newton, collision_rel, run, load_run, RUNS_FOLDER
import os

# Reference values from an independent implementation, relativistic.
# Beads 0 (behind) and 1 (ahead) collide; bead 2 is a bystander.
TABLE = [
    # m_behind, m_ahead, p_behind, p_ahead,  p_behind_after, p_ahead_after
    (1, 2, 0.03, -0.01, -0.016665,  0.036665),
    (1, 2, 0.3,  -0.1,  -0.164926,  0.364926),
    (1, 2, 3.0,  -1.0,  -1.238643,  3.238643),
    (1, 2, 9.0,  -3.0,  -3.144812,  9.144812),
    (1, 1, 2.0,  -0.5,  -0.5,       2.0),
]


def test_reference_table():
    for m1, m2, p1, p2, p1_after, p2_after in TABLE:
        # set up: a 3-bead state; bystander has mass 5 and momentum 0.7
        m = np.array([m1, m2, 5.0])
        p = np.array([p1, p2, 0.7])
        # run
        p_new = collision_rel(p, m, behind=0, ahead=1)
        # assert: colliding pair matches the table to 6 decimals ...
        np.testing.assert_allclose(p_new[:2], [p1_after, p2_after], rtol=0, atol=1e-6,
                                   err_msg=f"table row m=({m1},{m2}) p=({p1},{p2})")
        # ... and the bystander is untouched (exactly: it should never be modified)
        assert p_new[2] == 0.7, "bystander momentum changed"
        # ... and the input array was not modified (a classic NumPy aliasing bug)
        assert p[0] == p1 and p[1] == p2, "collision_rel modified its input array"


PAIRS = [(1, 2), (2, 0), (0, 1)]          # (behind, ahead) exactly as step() uses them

def test_conservation():
    """Total momentum and energy are unchanged by any collision, for all three pairs."""
    rng = np.random.default_rng(0)
    for _ in range(1000):
        m = rng.uniform(1, 10, size=3)
        p = rng.uniform(-10, 10, size=3)
        for behind, ahead in PAIRS:
            p_new = collision_rel(p, m, behind, ahead)
            np.testing.assert_allclose(p_new.sum(), p.sum(), rtol=0,
                                       atol=1e-12 * np.abs(p).sum(),
                                       err_msg=f"momentum, m={m} p={p} pair={behind, ahead}")
            np.testing.assert_allclose(np.hypot(m, p_new).sum(), np.hypot(m, p).sum(),
                                       rtol=1e-12, atol=0,
                                       err_msg=f"energy, m={m} p={p} pair={behind, ahead}")
            bystander = 3 - behind - ahead
            assert p_new[bystander] == p[bystander], f"bystander changed, pair={behind, ahead}"
            assert not np.allclose(p_new[[behind, ahead]], p[[behind, ahead]]), \
                f"trivial solution returned, pair={behind, ahead}"


def test_equal_masses_swap():
    """Equal masses swap momenta at any energy (relative error grows with |p|)."""
    rng = np.random.default_rng(0)
    for _ in range(100):
        p = rng.uniform(-100, 100, size=3)
        m = np.array([1.0, 1.0, 5.0])
        p_new = collision_rel(p, m, behind=0, ahead=1)
        np.testing.assert_allclose(p_new[:2], p[[1, 0]], rtol=1e-10,
                                   err_msg=f"no swap for p={p}")


def test_newtonian_limit():
    """At K/Mc^2 ~ 1e-8 the results agree to ~1e-8, but are not identical."""
    rng = np.random.default_rng(0)
    p = rng.uniform(-0.001, 0.001, size=3)
    m = np.array([1.0, 2.0, 5.0])
    p_rel = collision_rel(p, m, behind=0, ahead=1)
    p_newt = collision_newton(p, m, behind=0, ahead=1)
    np.testing.assert_allclose(p_rel, p_newt, rtol=1e-6)        # agree to well within 1e-6
    rel_diff = np.abs(p_rel[:2] - p_newt[:2]) / np.abs(p_newt[:2])
    assert np.all(rel_diff > 1e-12), f"identical to Newton ({rel_diff}): relativity not applied?"


#regression tests for the new simulator, to ensure it reproduces the old one when rel=False

REF_FILE = os.path.join(RUNS_FOLDER, "ref_newton.npz")   # default; change with --ref

def test_newtonian_regression():
    """Replaying a saved run from its start state reproduces it exactly."""
    ref = load_run(REF_FILE)
    if ref.get("labels_only", False):
        raise SystemExit(f"{REF_FILE} is labels-only: no x, p to compare. Pick a full run.")
    rel = bool(ref.get("relativistic", False))   # old runs have no flag: they were Newtonian
    print(f"  replaying {os.path.basename(REF_FILE)} ({len(ref['label']):,} collisions, rel={rel})")
    out = run(ref["x0"].copy(), ref["p0"].copy(), ref["m"], len(ref["label"]), rel=rel)
    assert np.array_equal(out["label"], ref["label"]), "labels changed"
    assert np.array_equal(out["p"], ref["p"]), "momenta changed"
    assert np.array_equal(out["x"], ref["x"]), "positions changed"


TESTS = {"table": test_reference_table, "conservation": test_conservation,
         "swap": test_equal_masses_swap, "limit": test_newtonian_limit,
         "regression": test_newtonian_regression}

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Tests for the relativistic simulator.")
    parser.add_argument("tests", nargs="*", metavar="TEST",
                        help=f"which tests to run (default: all). Choices: {', '.join(TESTS)}")
    parser.add_argument("--ref", metavar="FILE",
                        help="run file for the regression test: a path, or a name inside runs/ "
                             "(default: runs/ref_newton.npz)")
    parser.add_argument("--rel", action="store_true", help="regression: replay the newest relativistic run (ignored with --ref)")
    parser.add_argument("--newt", action="store_true", help="lregression: replay the newest Newtonian run (ignored with --ref)") 
    args = parser.parse_args()
    unknown = [t for t in args.tests if t not in TESTS]
    if unknown:
        parser.error(f"unknown test(s) {unknown}; choose from {list(TESTS)}")

    if args.rel and args.newt:
        parser.error("choose --rel or --newt, not both")
    if args.ref:
        REF_FILE = args.ref if os.path.exists(args.ref) else os.path.join(RUNS_FOLDER, args.ref)
    elif args.rel or args.newt:
        # no file named: replay the newest full run of the chosen kind
        REF_FILE = load_run(need_state=True, rel=args.rel)["filename"]

    for name in (args.tests or TESTS):
        TESTS[name]()
        print(f"PASS  {name}")
    print("all selected tests passed")