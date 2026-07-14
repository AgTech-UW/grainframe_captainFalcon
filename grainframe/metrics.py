"""Error metrics and reports."""
import numpy as np
from .utils import wrapToPi


def crossTrackError(px, py, pathXY):
    dxToPath = px[:, None] - pathXY[None, :, 0]
    dyToPath = py[:, None] - pathXY[None, :, 1]
    d = np.hypot(dxToPath, dyToPath)
    return d.min(axis=1)


# ---------------- Fanboid metrics (errors vs the Captain) ----------------

def fanFinalErrors(run):
    dx = run['fanX'][-1] - run['x'][-1]
    dy = run['fanY'][-1] - run['y'][-1]
    d = np.hypot(dx, dy)
    fanTheta = np.arctan2(run['fanVy'][-1], run['fanVx'][-1])
    dth = wrapToPi(fanTheta - run['theta'][-1])
    return dx, dy, d, dth


def fanChi2(run, opts):
    # Split position and heading errors
    dx, dy, _, dth = fanFinalErrors(run)
    chi2_pos = float(np.sum(dx**2 / opts.sigmaPos**2 + dy**2 / opts.sigmaPos**2))
    chi2_ang = float(np.sum(dth**2 / opts.sigmaTh**2))
    return chi2_pos, chi2_ang


# ---------------- Captain metrics (errors vs the GOAL pose) ----------------

def falconFinalErrors(run, data):
    """Captain Falcon's end pose vs the goal pose qf."""
    dx  = float(run['x'][-1] - data.qf[0])
    dy  = float(run['y'][-1] - data.qf[1])
    dth = float(wrapToPi(run['theta'][-1] - data.qf[2]))
    return dx, dy, np.hypot(dx, dy), dth


def falconChi2(run, data, opts):
    """chi^2 of Captain Falcon's end pose vs goal (per Adam's request)."""
    dx, dy, _, dth = falconFinalErrors(run, data)
    chi2_pos = (dx**2 + dy**2) / opts.sigmaPos**2
    chi2_ang = dth**2 / opts.sigmaTh**2
    return float(chi2_pos), float(chi2_ang)


# ---------------- Reports ----------------

def flockFinalReport(run, opts):
    # 1. Smoothness: how hard they were cranking the wheel
    omega = run['fanOmega']  # heading rate, not desired angle
    rms_omega = np.sqrt(np.mean(omega**2))
    max_omega = np.max(np.abs(omega))

    x = run['fanX'][-1]
    y = run['fanY'][-1]
    vx = run['fanVx'][-1]
    vy = run['fanVy'][-1]

    # 2. Cohesion: average distance from their own center of mass
    cm_x, cm_y = np.mean(x), np.mean(y)
    cohesion_dist = np.mean(np.hypot(x - cm_x, y - cm_y))

    # 3. Alignment: standard deviation of headings
    headings = np.arctan2(vy, vx)
    alignment_std = np.std(headings)

    # 4. Separation: minimum distance between any two boids
    n = len(x)
    min_sep = np.inf
    for i in range(n):
        for j in range(i + 1, n):
            d = np.hypot(x[i] - x[j], y[i] - y[j])
            min_sep = min(min_sep, d)

    print('\n--- Fanboid Final-State Report ---')
    print('Smoothness (Omega) : RMS = %.3f rad/s, Max = %.3f rad/s' % (rms_omega, max_omega))
    print('Cohesion (Spread)  : Mean distance from center of mass = %.3f' % cohesion_dist)
    print('Alignment (Heading): Std dev of headings = %.3f deg' % np.degrees(alignment_std))
    print('Separation (Crash) : Minimum distance between boids = %.3f (Protected Range = %.1f)'
          % (min_sep, opts.PR))
