"""
Compression test of the collision history from a ThreeBodyProblem.py run.

Usage (from the folder containing ThreeBodyProblem.py):
    python compression_test.py                              newest run in runs/
    python compression_test.py runs/run3_seed42.npz         a specific run (normal or --labels-only)
    python compression_test.py --save run3                  also save the plot and results table
    python compression_test.py --logistic 1000000           chaotic reference: logistic map, r = 4
References for the three-way comparison:
    chaotic   python compression_test.py --logistic 1000000
    periodic  python ThreeBodyProblem.py --masses 1 1 1 --n 1000000 --labels-only, then test it

What it does:
  1. Loads the collision labels (1, 2, 3 = A, B, C: the mass that sat out).
  2. Turns them into a binary string (EM1): one bit per pair of consecutive collisions.
       1 = next label is one step forward round A->B->C->A (A->B, B->C, C->A)
       0 = next label is one step back                      (A->C, B->A, C->B)
     N collisions give N - 1 bits. Each bit is stored as one ASCII character '0' or '1'.
  3. Compresses the first n bits with lzma for a range of lengths n, and does the same
     for a shuffled copy (same number of 0s and 1s, order destroyed).
  4. Reports, for each n:
       compressed size as % of original       (the measure used by Vedak & Ackland 2023)
       bits per collision = 8 * size / n      (estimate of information per collision)
       real / shuffled                         (how much of the compression comes from the
                                                ORDER of collisions: 1 = none, -> 0 = all)

Notes:
  - lzma adds ~32 bytes of header, so lengths below ~10^4 bits are dominated by overhead.
  - The shuffled copy uses a fixed seed so results are reproducible.
"""
import argparse
import lzma
import os

import matplotlib.pyplot as plt
import numpy as np

from ThreeBodyProblem import load_run


parser = argparse.ArgumentParser(description="lzma compression test of a three-particle ring run.")
parser.add_argument("filename", nargs="?",
                    help="a .npz run saved by ThreeBodyProblem.py (default: newest in runs/)")
parser.add_argument("--logistic", type=int, metavar="N",
                    help="instead of a run, test N bits from the logistic map x -> 4x(1-x)")
parser.add_argument("--seed", type=int, default=0, help="seed for the shuffled copy")
parser.add_argument("--save", help="save the plot and a results table using this filename stem")
parser.add_argument("--show", type=int, metavar="N", default=0, help ="print first N labels next to encoded bits and check")
parser.add_argument("--shuffles", type=int, default=5, help="shuffled copies averaged to length")
args = parser.parse_args()


# --- functions ----------------------------------------------------------------------

def labels_to_bits(labels):
    step = np.diff(labels) % 3                  # 1 = one step forward round A->B->C->A, 2 = one step back
    assert np.all(step != 0), "Consecutive labels should not be the same."   # physically impossible
    return (step == 1).astype(np.uint8)         # forward -> 1, back -> 0

def compressed_size(bits):
    return len(lzma.compress(np.packbits(bits).tobytes(), preset=9 | lzma.PRESET_EXTREME))

def show_encoding(labels, bits, n_rows):
    """Print the first labels with the bit between each pair + check bits to labels"""
    k = min(n_rows, len(bits))
    print("\nlabels: " + "   ".join("ABC"[l - 1] for l in labels[:k + 1]))
    print("bits:   " + "   ".join(str(b) for b in bits[:k]))
    #round trip
    steps = np.where(bits == 1, 1, 2)
    rebuilt = (labels[0] - 1 + np.concatenate([[0], np.cumsum(steps)])) % 3 + 1
    same = np.array_equal(rebuilt, labels)
    print(f"round trip: {len(labels):,} labels rebuilt from the first label + {len(bits):,} bits "
          f"-> {'Matches the file exactly' if same else 'Does not match - encoding bug'}")


# --- the string to compress ---------------------------------------------------------
if args.logistic:
    # chaotic reference: exactly 1 bit of new information per step, but fully deterministic
    x = 0.3141592653589793
    bits = np.empty(args.logistic, dtype=np.uint8)
    for i in range(args.logistic):
        x = 4 * x * (1 - x)
        bits[i] = x > 0.5
    name, source = "logistic map", "logistic map r = 4 (chaotic reference)"
else:
    data = load_run(args.filename)
    m = data["m"]
    labels = data["label"].astype(int)
    bits = labels_to_bits(labels)
    name, source = os.path.basename(data["filename"]), f"m = {m[0]:.2f}, {m[1]:.2f}, {m[2]:.2f}"
    if  args.show:
        show_encoding(labels, bits, args.show)


# --- compress the first n bits for a range of n --------------------------------------
rng = np.random.default_rng(args.seed)

print(f"\n{name}\n{source}     {len(bits):,} bits     "
      f"fraction of 1s = {bits.mean():.3f}     \n")



lengths = np.unique(np.append(np.logspace(3, np.log10(len(bits)), 12).astype(int)[:-1],
                              len(bits)))       # 12 log-spaced lengths, 10^3 ... the full string

results = []
print(f"{'n bits':>10} {'compressed':>11} {'bits/coll':>10}"
      f"{'shuffled':>9} {'bits/coll':>10} {'real/shuf':>10}")
for n in lengths:
    size  = compressed_size(bits[:n])
    size_shuffled = np.mean([compressed_size(rng.permutation(bits[:n])) for _ in range(args.shuffles)])
    results.append((n, size, size_shuffled))
    print(f"{n:>10,} {size:>11,} {8 * size / n:>10.3f}"
          f"{size_shuffled:>9.1f} {8 * size_shuffled / n:>10.3f} {size / size_shuffled:>10.3f}")
results = np.array(results)
n, size, size_shuffled = results[:, 0], results[:, 1], results[:, 2]


# --- plot ---------------------------------------------------------------------------
fig, (ax_bits, ax_ratio) = plt.subplots(1, 2, figsize=(9, 3.6), layout="constrained")
fig.suptitle(f"{name}     {source}     lzma (preset 9, extreme), 8 bits per byte")

ax_bits.semilogx(n, 8 * size_shuffled / n, "o-", color="grey", label="shuffled copy (mean)")
ax_bits.semilogx(n, 8 * size / n, "o-", color="#2a78d6", label="collision string (EM1)")
ax_bits.axhline(1, color="k", ls="--", lw=1, label="1 bit (coin flip)")
ax_bits.axvspan(n.min(), 1e4, color="0.9", label="header overhead dominates")
ax_bits.set_xlabel("string length n (bits)")
ax_bits.set_ylabel("compressed bits per collision")
ax_bits.set_ylim(0, 2)
ax_bits.legend(frameon=False, fontsize=8)

ax_ratio.semilogx(n, size / size_shuffled, "o-", color="#2a78d6")
ax_ratio.axhline(1, color="k", ls="--", lw=1, label="no structure in the order")
ax_ratio.axvspan(n.min(), 1e4, color="0.9")
ax_ratio.set_xlabel("string length n (bits)")
ax_ratio.set_ylabel("compressed size: real / shuffled")
ax_ratio.set_ylim(0, 1.1)
ax_ratio.legend(frameon=False, fontsize=8)

if args.save:
    fig.savefig(f"{args.save}_compression.png", dpi=300)
    np.savetxt(f"{args.save}_compression.csv", results, delimiter=",", fmt="%.1f",
               header=f"{source}\n"
                      f"source {name}, lzma preset 9 extreme, shuffle seed {args.seed}\n"
                      f"n_bits,lzma_bytes,lzma_bytes_shuffled")
    print(f"\nsaved {args.save}_compression.png and {args.save}_compression.csv")
else:
    plt.show()