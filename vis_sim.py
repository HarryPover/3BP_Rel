"""
Animate a saved run from ThreeBodyProblem.py.

Usage (from the folder containing ThreeBodyProblem.py):
    python vis_sim.py                                        newest run in runs/
    python vis_sim.py runs/run1_seed513400270.npz            a specific run
    python vis_sim.py --collisions 40 --speed 1              more collisions, faster
    python vis_sim.py --save run1.gif                        save instead of showing
    python vis_sim.py --collisions 500 --still unfold.png    final frame only: long-term pattern

Press SPACE in the window to pause / resume.

The animation freezes for a moment at every collision (--pause seconds), so you can check
that the two touching particles are the pair the label names, and that their velocity
arrows change in the way an elastic collision should.

Right: phase space. The configuration triangle (mass-weighted coordinates, centre of mass
removed), where the whole system is ONE ball bouncing off three walls - wall A, B, C is
hit exactly when that label's pair collides - and the momentum circle, showing which way
the ball moves. Together, triangle x circle is the full energy surface.

Top left: the ring, with the seam (where x = X_MIN and x = X_MIN + L meet) at the top.
Bottom: the same ring cut open into the line [X_MIN, X_MIN + L), i.e. [0, 1) with the
current ThreeBodyProblem.py; particles leaving one end reappear at the other.

How the motion is reconstructed: each saved row n holds the positions and momenta just
AFTER collision n, at time t[n]. Until the next collision every particle moves in a
straight line, so at any time tau between t[n] and t[n+1]:
    x(tau) = x[n] + (p[n] / m) * (tau - t[n]),   wrapped back onto the ring.
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from ThreeBodyProblem import L, X_MIN, load_run   # same ring geometry and file format

COLOURS = ["#2a78d6", "#eb6834", "#1baf7a"]   # m1, m2, m3
INK, MUTED, GRID, BG = "#0b0b0b", "#52514e", "#d9d8d3", "#fcfcfb"
PAIR_OF = {"A": "m2-m3", "B": "m1-m3", "C": "m1-m2"}
FADE_FRAMES = 8    # frames over which a collision burst fades after the pause


parser = argparse.ArgumentParser(description="Animate a three-particle ring run.")
parser.add_argument("filename", nargs="?",
                    help="a .npz run saved by ThreeBodyProblem.py (default: newest in runs/)")
parser.add_argument("--collisions", type=int, default=20, help="how many collisions to show")
parser.add_argument("--speed", type=float, default=0.6,
                    help="average collisions per second of animation (lower = slower)")
parser.add_argument("--pause", type=float, default=0.1,
                    help="seconds to freeze at each collision")
parser.add_argument("--fps", type=int, default=30)
parser.add_argument("--save", help="save to this .gif instead of opening a window")
parser.add_argument("--still", help="save only the final frame to this .png (fast, for long "
                                    "runs: e.g. --collisions 500 --still unfold.png)")
args = parser.parse_args()

# --- load the run -------------------------------------------------------------------
data = load_run(args.filename, need_state=True)             # newest run in runs/ if no filename given
if data["labels_only"]:
    raise SystemExit("This run kept only the labels - make a normal run to animate.")
m = data["m"]
t, x_c = data["t"], data["x_c"]
x, p = data["x"], data["p"]
label = np.array(list("ABC"))[data["label"].astype(int) - 1]  # letters for display
n_show = min(args.collisions, len(t) - 1)

# animation clock. Between collisions, simulation time runs at ONE constant rate, so
# speeds on screen are comparable across the whole animation. At each collision the
# clock stops for --pause seconds (repeated frames at exactly t[k]).
motion_frames = args.fps * n_show / args.speed           # frames spent moving, in total
frames_per_time = motion_frames / (t[n_show] - t[0])
pause_frames = int(round(args.pause * args.fps))
frame_times = []
for k in range(n_show):
    frame_times += [t[k]] * pause_frames                  # freeze at collision k
    steps = max(2, int(round((t[k + 1] - t[k]) * frames_per_time)))
    frame_times += list(np.linspace(t[k], t[k + 1], steps, endpoint=False)[1:])
frame_times += [t[n_show]] * pause_frames
frame_times = np.array(frame_times)
n_frames = len(frame_times)
position_history = [[] for _ in range(3)]


# --- phase space: the triangle (where the system is) x the circle (which way it moves) ---
# Mass-weighted coordinates q_i = sqrt(m_i) x_i turn the three particles into ONE point
# moving at constant speed sqrt(2E). Sliding all three particles round the ring together
# moves q along n_hat = sqrt(m)/|sqrt(m)| (the centre-of-mass direction), so we look at the
# plane perpendicular to n_hat. There, only the gaps matter:
#     a = gap 1 -> 2,  b = gap 2 -> 3,  so relative positions are (0, a, a + b).
# Walls: a = 0 is wall C (1-2 collide), b = 0 is wall A (2-3), a + b = L is wall B (3-1).
sqrt_m = np.sqrt(m)
n_hat = sqrt_m / np.linalg.norm(sqrt_m)
e1 = np.array([0, 0, 1.0]) - n_hat[2] * n_hat      # along wall C, so wall C is horizontal
e1 /= np.linalg.norm(e1)
e2 = np.cross(n_hat, e1)                           # (e1, e2) span the plane
if (sqrt_m * np.array([0, L, L])) @ e2 < 0:        # put the triangle above wall C
    e2 = -e2

def gaps_to_plane(a, b):
    q = sqrt_m * np.array([0, a, a + b])
    return np.array([q @ e1, q @ e2])

corner = {"all at 1": gaps_to_plane(0, 0), "C|A": gaps_to_plane(0, L), "A|B": gaps_to_plane(L, 0)}
wall_ends = {"A": (gaps_to_plane(0, 0), gaps_to_plane(L, 0)),     # b = 0
             "B": (gaps_to_plane(L, 0), gaps_to_plane(0, L)),     # a + b = L
             "C": (gaps_to_plane(0, 0), gaps_to_plane(0, L))}     # a = 0

# gaps just after each saved collision. The simulation sets the colliding pair to exactly
# the same position, so the closed gap is exactly 0 (or a + b exactly L) - no rounding.
gap_a = (x[:, 1] - x[:, 0]) % L
gap_b = (x[:, 2] - x[:, 1]) % L
p_tilde = p / sqrt_m                               # q-space velocity; |p_tilde| = sqrt(2E)
speed = np.sqrt(2 * data["E"])
bounce_points = np.array([gaps_to_plane(gap_a[k], gap_b[k]) for k in range(n_show + 1)])
base_corners = np.array([corner["all at 1"], corner["C|A"], corner["A|B"]])


# --- unfolding ------------------------------------------------------------------------
# Instead of reflecting the ball off a wall, reflect the TRIANGLE across that wall and let
# the ball carry straight on into the copy. The trajectory becomes one straight line
# through a chain of copies. Copy n is the original triangle moved by an isometry
#     z -> R_n z + s_n     (R_n a 2x2 rotation or reflection, s_n a shift),
# and each bounce composes one more reflection, across the CURRENT copy of the wall hit.
def reflection_across(A, B):
    """Reflection across the line through points A and B, as z -> F z + g."""
    u = (B - A) / np.linalg.norm(B - A)
    F = 2 * np.outer(u, u) - np.eye(2)
    return F, A - F @ A

R_copy, s_copy = [np.eye(2)], [np.zeros(2)]        # copy 0 = the original triangle
for k in range(1, n_show + 1):                     # row k is just after bounce k
    A, B = wall_ends["ABC"[int(data["label"][k]) - 1]]
    R, s = R_copy[-1], s_copy[-1]
    F, g = reflection_across(R @ A + s, R @ B + s)  # the wall as it sits in the current copy
    R_copy.append(F @ R)
    s_copy.append(F @ s + g)
copies = np.array([(R @ base_corners.T).T + s for R, s in zip(R_copy, s_copy)])
# which way each copy faces: an angle for the rotation part, and +1 / -1 for mirrored or not
# (rounded to 1e-6 degrees first, then wrapped, so 359.9999999 and 0 count as the same)
orientation = [(round(np.degrees(np.arctan2(R[1, 0], R[0, 0])), 6) % 360, round(np.linalg.det(R)))
               for R in R_copy]
# direction of motion after each bounce, on the momentum circle
directions = np.column_stack([p_tilde[: n_show + 1] @ e1, p_tilde[: n_show + 1] @ e2])


# --- figure layout --------------------------------------------------------------------
plt.rcParams.update({"font.size": 10, "text.color": INK, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED})
fig = plt.figure(figsize=(13.5, 8.6), facecolor=BG)
# left column: the ring and the cut-open line. Right column: phase space.
ax_ring = fig.add_axes([0.03, 0.31, 0.45, 0.60])
ax_line = fig.add_axes([0.05, 0.215, 0.41, 0.07])
ax_unf = fig.add_axes([0.52, 0.32, 0.46, 0.58])
ax_tri = fig.add_axes([0.53, 0.03, 0.20, 0.22])
ax_circ = fig.add_axes([0.77, 0.03, 0.20, 0.22])
for ax in (ax_ring, ax_line, ax_unf, ax_tri, ax_circ):
    ax.set_facecolor(BG)

title = f"masses  m1 = {m[0]:.2f}   m2 = {m[1]:.2f}   m3 = {m[2]:.2f}"
fig.text(0.255, 0.965, os.path.basename(data["filename"]), ha="center",
         fontsize=12, weight="bold")
fig.text(0.255, 0.937, title, ha="center", color=MUTED)

# ring: position x -> angle pi/2 + 2*pi*(x - X_MIN)/L, so the seam (x = 0 = 1) sits at
# the top and x increases anticlockwise
R = 1.0
th = np.linspace(0, 2 * np.pi, 400)
ax_ring.plot(R * np.cos(th), R * np.sin(th), color=GRID, lw=3, zorder=1)
for tick in X_MIN + L * np.array([0, 0.25, 0.5, 0.75]):
    a = np.pi / 2 + 2 * np.pi * (tick - X_MIN) / L   # position -> angle on the ring
    ax_ring.plot([0.94 * np.cos(a), 1.06 * np.cos(a)], [0.94 * np.sin(a), 1.06 * np.sin(a)],
                 color=MUTED, lw=1)
    text = f"x = {X_MIN:g} | {X_MIN + L:g}" if tick == X_MIN else f"x = {tick:g}"
    ax_ring.text(1.2 * np.cos(a), 1.2 * np.sin(a), text, ha="center", va="center",
                 color=MUTED, fontsize=9)
ax_ring.set_xlim(-1.45, 1.45)
ax_ring.set_ylim(-1.45, 1.45)
ax_ring.set_aspect("equal")
ax_ring.axis("off")

# unrolled line
ax_line.plot([X_MIN, X_MIN + L], [0, 0], color=GRID, lw=3, zorder=1)
ax_line.set_xlim(X_MIN - 0.05, X_MIN + L + 0.05)
ax_line.set_ylim(-1.2, 1.2)
ax_line.set_yticks([])
ax_line.set_xticks(X_MIN + L * np.array([0, 0.25, 0.5, 0.75, 1]))
for side in ("left", "right", "top"):
    ax_line.spines[side].set_visible(False)
ax_line.spines["bottom"].set_color(MUTED)
ax_line.set_xlabel("ring cut open at the seam: position x")

# triangle: configuration space in mass-weighted coordinates. Wall k is coloured like
# particle k, the one that sits out when the system hits that wall.
for k, name in enumerate("ABC"):
    (x0_, y0_), (x1_, y1_) = wall_ends[name]
    ax_tri.plot([x0_, x1_], [y0_, y1_], color=COLOURS[k], lw=3, solid_capstyle="round", zorder=1)
    mid = np.array([(x0_ + x1_) / 2, (y0_ + y1_) / 2])
    centre = sum(corner.values()) / 3
    out = mid + 0.08 * L * (mid - centre) / np.linalg.norm(mid - centre)   # just outside
    ax_tri.text(*out, name, ha="center", va="center", color=COLOURS[k], fontsize=9,
                weight="bold")
tri_trail, = ax_tri.plot([], [], color=MUTED, lw=0.5, alpha=0.6, zorder=2)
tri_dot = ax_tri.scatter([0], [0], s=60, color=INK, zorder=4)
tri_arrow = ax_tri.quiver([0], [0], [0], [0], color=INK, angles="xy", scale_units="xy",
                          scale=speed / (0.12 * L), width=0.006, zorder=3)
tri_burst = ax_tri.scatter([], [], s=400, facecolors="none", edgecolors=INK, lw=2, zorder=5)
all_corners = np.array(list(corner.values()))
pad = 0.15 * L
ax_tri.set_xlim(all_corners[:, 0].min() - pad, all_corners[:, 0].max() + pad)
ax_tri.set_ylim(all_corners[:, 1].min() - pad, all_corners[:, 1].max() + pad)
ax_tri.set_aspect("equal")
ax_tri.axis("off")
ax_tri.set_title("folded: one ball bouncing\nin the triangle", color=MUTED, fontsize=9)

# momentum circle: the same point's velocity. Fixed length sqrt(2E) (energy), and in the
# plane (total momentum = 0), so only its direction theta changes - at collisions only.
circ = np.linspace(0, 2 * np.pi, 300)
ax_circ.plot(speed * np.cos(circ), speed * np.sin(circ), color=GRID, lw=3, zorder=1)
circ_line, = ax_circ.plot([0, 0], [0, 0], color=INK, lw=1.5, zorder=2)
circ_dot = ax_circ.scatter([0], [0], s=50, color=INK, zorder=3)
circ_trail = ax_circ.scatter([], [], s=10, color="#d62728", alpha=0.6, zorder=2)
ax_circ.set_xlim(-1.3 * speed, 1.3 * speed)
ax_circ.set_ylim(-1.3 * speed, 1.3 * speed)
ax_circ.set_aspect("equal")
ax_circ.axis("off")
ax_circ.set_title("momentum circle: direction\nof motion (radius $\\sqrt{2E}$)",
                  color=MUTED, fontsize=9)

# unfolded view: every copy of the triangle the straight line passes through. Copies are
# shaded by orientation: mirrored copies (odd number of bounces) darker than unmirrored.
from matplotlib.collections import PolyCollection
unf_copies = PolyCollection([], facecolors=[], edgecolors=MUTED, linewidths=0.6, zorder=1)
ax_unf.add_collection(unf_copies)
unf_current = PolyCollection([], facecolors="none", edgecolors=INK, linewidths=1.6, zorder=3)
ax_unf.add_collection(unf_current)
unf_line, = ax_unf.plot([], [], color="#d62728", lw=1.6, zorder=4)
unf_dot = ax_unf.scatter([0], [0], s=40, color=INK, zorder=5)
unf_text = ax_unf.text(0.0, 1.0, "", transform=ax_unf.transAxes, va="top", color=MUTED,
                      fontsize=9, family="monospace")
span = np.ptp(copies.reshape(-1, 2), axis=0).max()
lo = copies.reshape(-1, 2).min(axis=0) - 0.04 * span
hi = copies.reshape(-1, 2).max(axis=0) + 0.04 * span
ax_unf.set_xlim(lo[0], hi[0])
ax_unf.set_ylim(lo[1], hi[1])
ax_unf.set_aspect("equal")
ax_unf.axis("off")
ax_unf.set_title(f"unfolded: reflect the triangle, not the ball  -  the trajectory is one "
                 f"straight line\n(mass-weighted coordinates, centre of mass removed;  "
                 f"{n_show} bounces = {n_show + 1} copies)", color=MUTED, fontsize=9)
copy_shade = ["#e8e6df" if det > 0 else "#c9c6bb" for _, det in orientation]

# particles: marker area proportional to mass, so heavier particles look bigger
sizes = 60 + 500 * m / m.max()
ring_dots = ax_ring.scatter([0, 0, 0], [0, 0, 0], s=sizes, c=COLOURS, zorder=3,
                            edgecolors=BG, linewidths=2)
line_dots = ax_line.scatter([0, 0, 0], [0, 0, 0], s=sizes * 0.45, c=COLOURS, zorder=3,
                            edgecolors=BG, linewidths=1.5)
# velocity arrows on the line: length proportional to velocity, same scale all run
v_max = np.max(np.abs(p[: n_show + 1] / m))
arrows = ax_line.quiver(np.zeros(3), np.full(3, 0.55), np.zeros(3), np.zeros(3),
                        color=COLOURS, angles="xy", scale_units="xy", scale=v_max / 0.45,
                        width=0.006, zorder=2)
ring_names = [ax_ring.text(0, 0, f"m{j + 1}", ha="center", va="center", fontsize=9,
                           color=COLOURS[j], weight="bold") for j in range(3)]

# collision burst (a ring around the collision point) and its label
burst_ring = ax_ring.scatter([], [], s=900, facecolors="none", edgecolors=INK, lw=2, zorder=4)
burst_line = ax_line.scatter([], [], s=300, facecolors="none", edgecolors=INK, lw=1.5, zorder=4)
burst_text = ax_ring.text(0, 0, "", ha="center", va="center", fontsize=11, weight="bold")

# read-outs in the centre of the ring and along the bottom
clock = ax_ring.text(0, 0.22, "", ha="center", fontsize=11)
counter = ax_ring.text(0, 0.02, "", ha="center", fontsize=11)
last = ax_ring.text(0, -0.2, "", ha="center", color=MUTED)
sequence = fig.text(0.255, 0.04, "", ha="center", family="monospace", fontsize=11)
fig.text(0.255, 0.075, "label = mass NOT involved:  A = m1 sat out,  B = m2,  C = m3",
         ha="center", color=MUTED, fontsize=9)


fig.text(0.255, 0.012, "SPACE = pause / resume    arrows = velocity", ha="center",
         color=MUTED, fontsize=8)

# --- one frame ------------------------------------------------------------------------
last_burst = {"frame": -10**9, "n": None}

def draw(frame):
    tau = frame_times[frame]

    # n = last collision at or before tau; reconstruct positions by straight-line motion
    n = int(np.searchsorted(t, tau, side="right")) - 1
    pos = X_MIN + (x[n] + p[n] / m * (tau - t[n]) - X_MIN) % L
    for j in range(3):
        position_history[j].append(pos[j])

    a = np.pi / 2 + 2 * np.pi * (pos - X_MIN) / L     # position -> angle on the ring
    ring_dots.set_offsets(np.column_stack([R * np.cos(a), R * np.sin(a)]))
    line_dots.set_offsets(np.column_stack([pos, np.zeros(3)]))
    arrows.set_offsets(np.column_stack([pos, np.full(3, 0.55)]))
    arrows.set_UVC(p[n] / m, np.zeros(3))
    for j in range(3):
        ring_names[j].set_position((0.80 * np.cos(a[j]), 0.80 * np.sin(a[j])))

    # phase space. Gaps change at the relative velocities (no wrapping needed).
    v = p[n] / m
    gap_now_a = gap_a[n] + (v[1] - v[0]) * (tau - t[n])
    gap_now_b = gap_b[n] + (v[2] - v[1]) * (tau - t[n])
    point = gaps_to_plane(gap_now_a, gap_now_b)
    velocity = np.array([p_tilde[n] @ e1, p_tilde[n] @ e2])
    tri_trail.set_data(*np.vstack([bounce_points[: n + 1], point]).T)

    # unfolded: the same point carried into copy n. All bounces so far become one line.
    unfolded_bounces = [R_copy[k] @ bounce_points[k] + s_copy[k] for k in range(n + 1)]
    unfolded_now = R_copy[n] @ point + s_copy[n]
    unf_line.set_data(*np.vstack(unfolded_bounces + [unfolded_now]).T)
    unf_dot.set_offsets([unfolded_now])
    unf_copies.set_verts(copies[: n + 1])
    unf_copies.set_facecolors(copy_shade[: n + 1])
    unf_current.set_verts([copies[n]])
    unf_text.set_text(f"copies visited          {n + 1:>6}\n"
                      f"different orientations  {len(set(orientation[: n + 1])):>6}\n"
                      f"directions of motion    {len(np.unique(directions[: n + 1].round(9), axis=0)):>6}")
    tri_dot.set_offsets([point])
    tri_arrow.set_offsets([point])
    tri_arrow.set_UVC([velocity[0]], [velocity[1]])
    circ_line.set_data([0, velocity[0]], [0, velocity[1]])
    circ_dot.set_offsets([velocity])
    circ_trail.set_offsets(directions[: n + 1])     # every direction visited so far

    # start a burst when a new collision has happened since the last frame
    if n != last_burst["n"]:
        last_burst.update(frame=frame, n=n)
    age = frame - last_burst["frame"]
    if age < pause_frames + FADE_FRAMES:
        ac = np.pi / 2 + 2 * np.pi * (x_c[n] - X_MIN) / L
        fade = min(1.0, 1 - (age - pause_frames) / FADE_FRAMES)
        burst_ring.set_offsets([[R * np.cos(ac), R * np.sin(ac)]])
        burst_line.set_offsets([[x_c[n], 0]])
        burst_ring.set_alpha(fade)
        burst_line.set_alpha(fade)
        burst_text.set_text(label[n])
        burst_text.set_position((1.32 * np.cos(ac), 1.32 * np.sin(ac)))
        burst_text.set_alpha(fade)
        tri_burst.set_offsets([gaps_to_plane(gap_a[n], gap_b[n])])   # the bounce point
        tri_burst.set_alpha(fade)
    else:
        burst_ring.set_offsets(np.empty((0, 2)))
        burst_line.set_offsets(np.empty((0, 2)))
        burst_text.set_text("")
        tri_burst.set_offsets(np.empty((0, 2)))

    clock.set_text(f"t = {tau:.3f}")
    counter.set_text(f"collision {n} of {n_show}")
    last.set_text(f"last: {label[n]}  ({PAIR_OF[label[n]]} collided)")
    recent = "".join(label[max(0, n - 29): n + 1])
    sequence.set_text(f"sequence:  {recent}")
    return ring_dots, line_dots, burst_ring, burst_line, burst_text, clock, counter, last, sequence, pos


if args.still:
    draw(n_frames - 1)                      # final frame only: the whole unfolding at once
    fig.savefig(args.still, dpi=150, facecolor=BG)
    print(f"saved {args.still}")
    raise SystemExit

anim = FuncAnimation(fig, draw, frames=n_frames, interval=1000 / args.fps, blit=False)

# SPACE pauses / resumes the window
paused = {"on": False}
def on_key(event):
    if event.key == " ":
        if paused["on"]:
            anim.resume()
        else:
            anim.pause()
        paused["on"] = not paused["on"]
fig.canvas.mpl_connect("key_press_event", on_key)

if args.save:
    anim.save(args.save, writer=PillowWriter(fps=args.fps))
    print(f"saved {args.save}")
else:
    plt.show()