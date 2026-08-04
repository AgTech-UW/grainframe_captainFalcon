"""Draw the leader's steering field, so you can SEE what the controller does.

Reads its parameters straight out of exp5_simple.py, so whatever you tune there
is what gets drawn here. Nothing is duplicated.

    python experiments/plot_fields.py            guidance + repulsion + sum
    python experiments/plot_fields.py law        just the one-number-in law
    python experiments/plot_fields.py sweep      what k and chi_inf actually do
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from matplotlib import pyplot as plt

import exp5_simple as E   # same folder, so this just works when run from the repo root

# Where to park the other leader for the picture. In the real sim he is moving,
# so the repulsion blob slides along with him and the field differs every step.
# Freezing him is just so the arrows hold still long enough to look at.
OTHER = (0.0, 0.0)
OTHER_PR = E.PR_RED              # the range of the leader we are drawing


# ---------------------------------------------------------------------
#  The two fields, as plain functions of position
# ---------------------------------------------------------------------

def guidanceField(x, y, pathHeading=0.0):
    """Vector field path following. Works on arrays."""
    e = y                                       # cross-track error for the line y=0
    chiD = pathHeading - E.CHI_INF * (2.0 / np.pi) * np.arctan(E.K_CROSS * e)
    return E.GUID_MAG * np.cos(chiD), E.GUID_MAG * np.sin(chiD)


def repulsionField(x, y, ox=OTHER[0], oy=OTHER[1], pr=OTHER_PR):
    """Khatib potential. Works on arrays. Zero outside pr."""
    dx, dy = x - ox, y - oy
    d = np.clip(np.hypot(dx, dy), 0.6, None)    # clip so the centre stays finite
    eta = E.REP_GAIN * pr ** 3 / 4.0
    w = np.where(d <= pr, eta * (1.0 / d - 1.0 / pr) / d ** 2, 0.0)
    return w * dx / d, w * dy / d


def flyIt(x0, y0, th0=0.0, steps=9000, xStop=26.0):
    """Release a leader into the summed field and let it steer, bang-bang."""
    v, dt, R = 1.0, 0.01, 5.0
    dpsiMax = (v / R) * dt
    x, y, th = x0, y0, th0
    px, py = [x], [y]
    for _ in range(steps):
        gx, gy = guidanceField(np.array(x), np.array(y))
        rx, ry = repulsionField(np.array(x), np.array(y))
        chiD = np.arctan2(gy + ry, gx + rx)
        err = np.arctan2(np.sin(chiD - th), np.cos(chiD - th))
        th += np.clip(err, -dpsiMax, dpsiMax)   # the turn-radius limit
        x += v * np.cos(th) * dt
        y += v * np.sin(th) * dt
        px.append(x); py.append(y)
        if x > xStop:
            break
    return np.array(px), np.array(py)


# ---------------------------------------------------------------------
#  Plots
# ---------------------------------------------------------------------

def fields():
    """Three panels: guidance, repulsion, and their sum."""
    X, Y = np.meshgrid(np.linspace(-22, 26, 25), np.linspace(-13, 13, 17))
    fig, ax = plt.subplots(3, 1, figsize=(11, 12))

    def draw(a, U, V, title, colorByStrength=False):
        M = np.hypot(U, V)
        M[M < 1e-9] = 1e-9                      # unit arrows: direction only
        if colorByStrength:
            a.quiver(X, Y, U / M, V / M, np.log10(M + 1), cmap='YlOrRd',
                     scale=32, width=0.004, pivot='mid')
        else:
            a.quiver(X, Y, U / M, V / M, color='steelblue',
                     scale=32, width=0.004, pivot='mid')
        a.axhline(0, color='k', lw=1.5, zorder=0)
        a.add_patch(plt.Circle(OTHER, OTHER_PR, fill=False, ls='--',
                               color='goldenrod', lw=1.5))
        a.plot(*OTHER, 'o', color='goldenrod', ms=13, zorder=5)
        a.set_title(title, fontsize=11)
        a.set_ylim(-13, 13)
        a.set_aspect('equal')

    gU, gV = guidanceField(X, Y)
    rU, rV = repulsionField(X, Y)

    draw(ax[0], gU, gV, '1. GUIDANCE alone  -  "get back on my line"')
    draw(ax[1], rU, rV, '2. REPULSION alone  -  "get away from him"   '
                        '(dashed = protected range %.0f)' % OTHER_PR,
         colorByStrength=True)
    draw(ax[2], gU + rU, gV + rV,
         '3. THE SUM  -  what the leader actually steers toward')

    px, py = flyIt(-22.0, 0.0)
    ax[2].plot(px, py, color='crimson', lw=2.5, zorder=6, label='leader flying it')
    ax[2].legend(fontsize=9, loc='lower right')

    plt.suptitle('chi_inf=%.0f deg   k=%.2f   GUID_MAG=%.0f   REP_GAIN=%.0f'
                 % (np.degrees(E.CHI_INF), E.K_CROSS, E.GUID_MAG, E.REP_GAIN),
                 fontsize=10, y=0.995)
    plt.tight_layout()
    plt.show()


def law():
    """The control law as a single curve: one number in, one angle out."""
    e = np.linspace(-12, 12, 600)
    chi = np.degrees(-E.CHI_INF * (2.0 / np.pi) * np.arctan(E.K_CROSS * e))
    deg = np.degrees(E.CHI_INF)

    plt.figure(figsize=(9, 5))
    plt.plot(e, chi, color='crimson', lw=2.5)
    plt.axhline(deg, ls=':', color='gray')
    plt.axhline(-deg, ls=':', color='gray')
    plt.axhline(0, color='k', lw=0.6)
    plt.axvline(0, color='k', lw=0.6)
    plt.axvspan(-1 / E.K_CROSS, 1 / E.K_CROSS, color='goldenrod', alpha=0.2)
    plt.text(0, -deg * 0.85, 'boundary layer\n|e| < 1/k = %.1f' % (1 / E.K_CROSS),
             ha='center', fontsize=9)
    plt.annotate('flattens out at chi_inf = %.0f deg' % deg, (6, deg * 0.93),
                 fontsize=9)
    plt.xlabel('e   (how far off the line you are)')
    plt.ylabel('angle to hold  (deg)')
    plt.title('THE WHOLE LAW: chi_d = chi_path - chi_inf * (2/pi) * atan(k*e)')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def sweep():
    """What each knob actually does. Guidance only, no other leader."""
    saveK, saveC = E.K_CROSS, E.CHI_INF
    e0 = -8.0
    fig, ax = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    E.CHI_INF = np.deg2rad(70.0)
    for k in [0.1, 0.2, 0.5, 1.5]:
        E.K_CROSS = k
        px, py = _flyGuidanceOnly(-22.0, e0)
        ax[0].plot(px, py, lw=2, label='k = %.1f  (1/k = %.1f)' % (k, 1 / k))
    ax[0].set_title('k  -  how EARLY you start straightening out\n'
                    '(turn radius R = 5, so 1/k near 5 is the clean choice)',
                    fontsize=10)

    E.K_CROSS = 0.2
    for c in [30, 50, 70, 85]:
        E.CHI_INF = np.deg2rad(c)
        px, py = _flyGuidanceOnly(-22.0, e0)
        ax[1].plot(px, py, lw=2, label='chi_inf = %d deg  (closes at %.2f v)'
                   % (c, np.sin(np.deg2rad(c))))
    ax[1].set_title('chi_inf  -  how HARD you cut across\n'
                    '(sets the closing rate, v * sin(chi_inf))', fontsize=10)

    for a in ax:
        a.axhline(0, color='k', lw=1.2)
        a.plot(-22, e0, 'ko', ms=6)
        a.legend(fontsize=8)
        a.grid(alpha=0.3)
        a.set_xlabel('x')
    ax[0].set_ylabel('e')
    E.K_CROSS, E.CHI_INF = saveK, saveC
    plt.tight_layout()
    plt.show()


def _flyGuidanceOnly(x0, y0, steps=9000, xStop=40.0):
    v, dt, R = 1.0, 0.01, 5.0
    dpsiMax = (v / R) * dt
    x, y, th = x0, y0, 0.0
    px, py = [x], [y]
    for _ in range(steps):
        gx, gy = guidanceField(np.array(x), np.array(y))
        chiD = np.arctan2(gy, gx)
        err = np.arctan2(np.sin(chiD - th), np.cos(chiD - th))
        th += np.clip(err, -dpsiMax, dpsiMax)
        x += v * np.cos(th) * dt
        y += v * np.sin(th) * dt
        px.append(x); py.append(y)
        if x > xStop:
            break
    return np.array(px), np.array(py)


if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'fields'
    {'law': law, 'sweep': sweep}.get(mode, fields)()