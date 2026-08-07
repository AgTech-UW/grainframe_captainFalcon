"""Sandbox: two squads approach each other. One run, one plot.

ALL the knobs live in the PARAMETERS block below -- edit and rerun.

Each leader is a Dubins car (constant speed, bounded turn rate) doing:

    desired velocity  =  GUIDANCE (follow my path)  +  REPULSION (dodge the
                                                       other leader)

then bang-bang steering toward that vector. The two terms are deliberately
separate so you can change one without disturbing the other.

  GUIDANCE  -- vector field path following, Nelson/Barber/McLain/Beard,
               IEEE T-RO 23(3), 2007.
  REPULSION -- Khatib obstacle potential, IJRR 5(1), 1986.

Run it:
    python experiments/exp5_simple.py            static plot
    python experiments/exp5_simple.py anim       live playback
    python experiments/exp5_simple.py collide    collision report only
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from matplotlib import pyplot as plt
from grainframe import SimOptions
from grainframe.data_io import apply_dataset
from grainframe.paths import straightPath
from grainframe.dynamics import (falconSteering, ackermannClamp,
                                 WaypointTracker)


# =====================================================================
#  PARAMETERS: edit me
# =====================================================================

SEED  = 8            # change for a different random squadron layout
N_FAN = 3            # fanboids per squadron

GEOMETRY = 'cross'   # 'headon' = straight at each other on the same line
                     # 'cross'  = the X crossing at the origin

HALF = 40.0          # how far from the middle each leader starts
LANE_OFFSET = 0.0    # shove Gold's lane sideways. 0 = perfectly symmetric.

# Gain overrides from config.py. None = keep the stock value.
OVERRIDES = {
    # 'SF': 5.0,              # fanboid separation strength
    # 'CF': 0.8,              # fanboid cohesion
    # 'AF': 2.0,              # fanboid alignment
    # 'fanLeaderFactor': 1.0, # fanboid pull toward own leader
    # 'R': 5.0,               # leader turn radius (bigger = clumsier)
}

# ---------------------------------------------------------------------
#  GUIDANCE: how each leader follows his own path
# ---------------------------------------------------------------------
# 'vfield'   -- vector field. For every point in the plane it defines the
#               heading you SHOULD have. Far off the line you fly in at
#               CHI_INF; as you close, the desired heading rotates to match
#               the path. Cross-track error and heading error therefore null
#               at the same moment, which is what stops the oscillation.
#
# 'waypoint' -- the old behaviour: aim straight at the next waypoint. Kept
#               so you can reproduce the limit cycle for comparison. It
#               oscillates because it nulls POSITION but leaves you pointed
#               sideways: arriving at a waypoint from off-line means arriving
#               with ~90 deg of heading error, every single time.
GUIDANCE = 'vfield'

K_CROSS  = 0.5                   # how sharply the field bends back to the line.
                                 # Too small = lazy recovery. Too large = the
                                 # field demands more curvature than the car
                                 # has and you get chatter. 0.4-0.6 is the
                                 # sweet spot at R=5.
CHI_INF  = np.deg2rad(70.0)      # approach angle when far off the line (deg).
                                 # 90 would be "fly straight at it", which is
                                 # untrackable. 70-85 is the useful range.
GUID_MAG = 10.0                  # magnitude of the guidance vector. MUST be a
                                 # constant. The old code used the distance to
                                 # the waypoint, which slid from 10 down to 0
                                 # across each leg -- meaning the balance
                                 # between path-following and avoidance
                                 # silently changed depending on where in the
                                 # leg the encounter happened.

# Per-leader path pull. Scales GUID_MAG, so it sets how strongly each leader
# resists being pushed off his line relative to the repulsion.
LEADER_FACTOR_RED  = 1.0
LEADER_FACTOR_GOLD = 1.0

# ---------------------------------------------------------------------
#  REPULSION: how each leader dodges the other
# ---------------------------------------------------------------------
# Per-leader protected range: how early each leader reacts to the OTHER.
# Different values break the mirror symmetry -- the one with the bigger range
# gives way, the one with the smaller range mostly holds his line.
PR_RED  = 12.0
PR_GOLD = 4.0

# True  -- Khatib form. Force is zero at the range boundary and rises steeply
#          toward contact. Smooth, no discontinuity.
# False -- legacy flat form, SF * displacement. This was BACKWARDS: strongest
#          at the boundary (5 * 12 = 60, versus a path pull of ~7) and it
#          jumped from 0 to 60 the instant the other leader crossed PR.
# IMPORTANT: this flag applies to the LEADERS ONLY. The fanboids keep the
# stock Reynolds separation rule (see fanSeparation below) so the boids model
# under study is not altered by a change to the leader controller.
SMOOTH_REPULSION = True

REP_GAIN = 45.0      # repulsion force at HALF the protected range.
                     # NOTE: opts.SF does NOT transfer to the Khatib form --
                     # the two laws have wildly different scales. This is its
                     # own independent knob. 45 reproduces roughly the same
                     # swerve magnitude the old flat rule produced.

# ---------------------------------------------------------------------
#  INFEASIBILITY STOP (paper Section 4.1)
# ---------------------------------------------------------------------
# If a leader physically cannot turn hard enough to avoid the other, he STOPS
# rather than orbiting uselessly. Stopping is always feasible regardless of
# turn radius.
USE_INFEASIBILITY_STOP = True
STOP_RANGE     = 6.0    # start worrying about a stop inside this distance
STOP_LOOKAHEAD = 3200   # steps to roll forward when testing feasibility.
                        # This is a DISTANCE in disguise: steps * v * dt.
                        # 3200 * 1.0 * 0.01 = 32 units = one full 2*pi*R turn
                        # at R=5. The old value of 40 was 0.4 units of travel,
                        # i.e. no lookahead at all -- canAvoid() collapsed to
                        # "am I inside STOP_MARGIN right now".
STOP_MARGIN    = 1.5    # stop if the best-case future gap stays below this

FAN_SPREAD = 3.0     # how loosely each squadron spawns around its leader
CIRCLE_R   = 2.5     # radius of the drawn COLLISION circles
# =====================================================================

RED, GOLD = 'crimson', 'goldenrod'


# ---------------------------------------------------------------------
#  Guidance law
# ---------------------------------------------------------------------

def guidance(x, y, segA, segB, wpX, wpY, gain):
    """Desired velocity vector for a leader at (x, y).

    segA -> segB is the path segment currently being tracked.
    wpX, wpY is the active waypoint (only used by the 'waypoint' mode).
    """
    if GUIDANCE == 'waypoint':
        return (wpX - x) * gain, (wpY - y) * gain

    # --- vector field ---
    seg = segB - segA
    L = np.linalg.norm(seg)
    if L < 1e-9:
        return (wpX - x) * gain, (wpY - y) * gain

    u = seg / L                              # along-track unit vector
    n = np.array([-u[1], u[0]])              # left normal
    e = (x - segA[0]) * n[0] + (y - segA[1]) * n[1]   # signed cross-track error

    chiPath = np.arctan2(u[1], u[0])
    # Far off the line (|e| large) atan saturates and chiD = chiPath -+ CHI_INF.
    # On the line (e = 0) atan is 0 and chiD = chiPath exactly.
    chiD = chiPath - CHI_INF * (2.0 / np.pi) * np.arctan(K_CROSS * e)

    return GUID_MAG * gain * np.cos(chiD), GUID_MAG * gain * np.sin(chiD)


# ---------------------------------------------------------------------
#  Repulsion
# ---------------------------------------------------------------------

def fanSeparation(xb, yb, xOthers, yOthers, opts, range_=None):
    """STOCK Reynolds/SAC separation -- used by the FANBOIDS. Do not change.

    This is part of the boids model being studied, so it stays exactly as it
    was: a flat push proportional to displacement, cut off at the protected
    range. Note this is the same family of object as the Khatib potential
    below -- both are repulsive artificial potential fields -- they just have
    different distance profiles. Keeping them separate means a change to the
    leader controller cannot silently alter the swarm results.
    """
    if range_ is None:
        range_ = opts.PR
    dx = xb - xOthers
    dy = yb - yOthers
    D = np.hypot(dx, dy)
    tooClose = (D > 1e-9) & (D <= range_)
    return (float(np.sum(dx[tooClose]) * opts.SF),
            float(np.sum(dy[tooClose]) * opts.SF))


def leaderRepulsion(xb, yb, xOthers, yOthers, opts, range_=None):
    """Leader-vs-leader repulsion ONLY. Not used by the fanboids.

    Khatib (1986):  F = eta * (1/d - 1/range) / d^2, directed away.
    Zero at d = range, singular at d = 0. eta is scaled so that REP_GAIN is
    literally the force magnitude at d = range/2.
    """
    if range_ is None:
        range_ = opts.PR

    dx = xb - xOthers
    dy = yb - yOthers
    D = np.hypot(dx, dy)
    tooClose = (D > 1e-9) & (D <= range_)
    if not np.any(tooClose):
        return 0.0, 0.0

    if not SMOOTH_REPULSION:                      # legacy flat rule
        return (float(np.sum(dx[tooClose]) * opts.SF),
                float(np.sum(dy[tooClose]) * opts.SF))

    d = np.clip(D[tooClose], 0.25, None)          # clip so it stays finite
    eta = REP_GAIN * range_ ** 3 / 4.0
    w = eta * (1.0 / d - 1.0 / range_) / d ** 2   # magnitude
    ux, uy = dx[tooClose] / d, dy[tooClose] / d   # unit vector, away from them
    return float(np.sum(w * ux)), float(np.sum(w * uy))


_ROLLOUT = np.arange(1, STOP_LOOKAHEAD + 1)   # rebuilt once, not per call


def canAvoid(x, y, th, xOther, yOther, thOther, opts):
    """Can this leader clear the other by turning as hard as he legally can?

    Roll both forward STOP_LOOKAHEAD steps: the other holds heading, I turn at
    my maximum rate (v/R) in whichever direction opens the gap. If even that
    best case stays closer than STOP_MARGIN, no feasible avoidance exists and
    the caller should STOP.

    Vectorised: the whole rollout is closed-form under a constant turn rate,
    so it is one cumsum instead of a Python loop. Same answer, ~100x faster,
    which matters because STOP_LOOKAHEAD has to be a real horizon (thousands
    of steps) to mean anything.
    """
    dt, v, R = opts.dt, opts.v, opts.R
    dpsiMax = (v / R) * dt                  # most I can rotate per step

    # Which way opens the gap? Sign of cross(heading, toward-other).
    toX, toY = xOther - x, yOther - y
    cross = np.cos(th) * toY - np.sin(th) * toX
    turn = -np.sign(cross) * dpsiMax        # turn AWAY

    n = _ROLLOUT                            # hoisted; see module level
    thi = th + turn * n                     # my heading each step
    xi = x + np.cumsum(v * np.cos(thi) * dt)
    yi = y + np.cumsum(v * np.sin(thi) * dt)
    xo = xOther + v * np.cos(thOther) * dt * n   # he just keeps going straight
    yo = yOther + v * np.sin(thOther) * dt * n

    return float(np.min(np.hypot(xi - xo, yi - yo))) >= STOP_MARGIN


def makePaths():
    if GEOMETRY == 'headon':
        dataR = straightPath(-HALF, 0, +HALF, 0)                     # left -> right
        dataG = straightPath(+HALF, LANE_OFFSET, -HALF, LANE_OFFSET) # right -> left
    else:                                                            # 'cross'
        dataR = straightPath(-HALF, -HALF, +HALF, +HALF)
        dataG = straightPath(-HALF, +HALF + LANE_OFFSET, +HALF, -HALF + LANE_OFFSET)
    return dataR, dataG


# ---------------------------------------------------------------------
#  Simulation
# ---------------------------------------------------------------------

def run(plot=True):
    opts = SimOptions()
    dataR, dataG = makePaths()
    apply_dataset(opts, dataR)
    for name, val in OVERRIDES.items():
        if val is not None:
            setattr(opts, name, val)
    np.random.seed(SEED)

    # Leader state: index 0 = Red, index 1 = Gold
    X  = np.array([dataR.qi[0], dataG.qi[0]])
    Y  = np.array([dataR.qi[1], dataG.qi[1]])
    TH = np.array([dataR.qi[2], dataG.qi[2]])
    wayPts   = [dataR.wayPts, dataG.wayPts]
    trackers = [WaypointTracker(opts), WaypointTracker(opts)]
    idP  = [1, 1]
    done = [False, False]
    goalMin     = [np.inf, np.inf]
    stoppedFlag = [False, False]     # frozen this step?
    stopCount   = [0, 0]             # how many steps each spent frozen

    # Squadrons: N_FAN boids clumped near each leader's start
    fanX, fanY, fanVx, fanVy, squad = [], [], [], [], []
    for k in range(2):
        fanX.append(X[k] + np.random.normal(0, FAN_SPREAD, N_FAN))
        fanY.append(Y[k] + np.random.normal(0, FAN_SPREAD, N_FAN))
        ang = np.random.uniform(0, 2 * np.pi, N_FAN)
        spd = np.random.uniform(opts.minSpeed, opts.maxSpeed, N_FAN)
        fanVx.append(spd * np.cos(ang))
        fanVy.append(spd * np.sin(ang))
        squad.append(np.full(N_FAN, k))
    fanX, fanY   = np.concatenate(fanX),  np.concatenate(fanY)
    fanVx, fanVy = np.concatenate(fanVx), np.concatenate(fanVy)
    squad = np.concatenate(squad)
    # squad id for every column of the pairwise block: the fans, then the two
    # leaders. Built once -- it never changes.
    teamAll = np.concatenate((squad, np.array([0, 1])))

    dt, v = opts.dt, opts.v
    log = {'x': [[], []], 'y': [[], []], 'fanX': [], 'fanY': []}

    t = 0.0
    tMax = opts.tMax(dataR.refLen)
    while t <= tMax and not all(done):

        # ------------------------- LEADERS -------------------------
        newLead = []
        for k in range(2):
            if done[k]:
                newLead.append((X[k], Y[k], TH[k]))
                continue

            wp = wayPts[k]
            idP[k] = trackers[k].advance(X[k], Y[k], idP[k], wp)
            wpX, wpY = wp[idP[k], 0], wp[idP[k], 1]
            dGoal = np.hypot(wpX - X[k], wpY - Y[k])

            # done at closest approach to the last waypoint
            if idP[k] == wp.shape[0] - 1:
                goalMin[k] = min(goalMin[k], dGoal)
                if goalMin[k] < opts.passWindow and dGoal > goalMin[k] + opts.hysteresis:
                    done[k] = True
                    newLead.append((X[k], Y[k], TH[k]))
                    continue

            # --- term 1: follow my path ---
            segA = wp[max(idP[k] - 1, 0)]
            segB = wp[idP[k]]
            myGain = LEADER_FACTOR_RED if k == 0 else LEADER_FACTOR_GOLD
            vx, vy = guidance(X[k], Y[k], segA, segB, wpX, wpY, myGain)

            # --- term 2: dodge the other leader (leaders ignore fanboids) ---
            other = 1 - k
            myPR = PR_RED if k == 0 else PR_GOLD
            sx, sy = leaderRepulsion(X[k], Y[k],
                                np.array([X[other]]), np.array([Y[other]]),
                                opts, range_=myPR)
            vx += sx
            vy += sy

            # --- bang-bang Dubins steering toward the summed vector ---
            vTh = falconSteering(TH[k], vx, vy, opts)

            distOther = np.hypot(X[other] - X[k], Y[other] - Y[k])

            # DEADLOCK GUARD: if the other leader is already frozen he is not
            # closing on me, so freezing too would lock us both forever with
            # the gap constant and canAvoid() never able to recover. Only one
            # of us needs to yield.
            mustStop = (USE_INFEASIBILITY_STOP
                        and distOther < STOP_RANGE
                        and not stoppedFlag[other]
                        and not canAvoid(X[k], Y[k], TH[k],
                                         X[other], Y[other], TH[other], opts))

            if mustStop:
                newLead.append((X[k], Y[k], TH[k]))          # hold position
                stoppedFlag[k] = True
            else:
                newLead.append((X[k] + v * np.cos(TH[k]) * dt,
                                Y[k] + v * np.sin(TH[k]) * dt,
                                TH[k] + vTh * dt))
                stoppedFlag[k] = False

        # ------------------------- FANBOIDS -------------------------
        # Same rules as before, computed as ONE (n x n+2) pairwise block
        # instead of a Python loop. The old version called np.append six times
        # per fan per step; every one of those allocated a fresh array, and at
        # 8-element arrays NumPy's per-call overhead completely dominated the
        # arithmetic. Six boids cost 11.6 MILLION function calls per run.
        allX  = np.concatenate((fanX,  X))
        allY  = np.concatenate((fanY,  Y))
        allVx = np.concatenate((fanVx, v * np.cos(TH)))
        allVy = np.concatenate((fanVy, v * np.sin(TH)))

        dx = fanX[:, None] - allX[None, :]
        dy = fanY[:, None] - allY[None, :]
        D  = np.hypot(dx, dy)

        notSelf = D > 1e-9
        # Separation sees EVERYONE: both squadrons and both leaders.
        sep = notSelf & (D <= opts.PR)
        # Cohesion and alignment see my OWN squadron plus my OWN leader. The
        # team mask handles the leaders for free -- teamAll is 0 and 1 for
        # them, so a fan matches its own captain and not the other one.
        vis = notSelf & (D <= opts.VR) & (teamAll[None, :] == squad[:, None])

        sX = opts.SF * (dx * sep).sum(1)
        sY = opts.SF * (dy * sep).sum(1)

        cnt = vis.sum(1)
        safe = np.where(cnt > 0, cnt, 1)            # avoid 0/0; masked out below
        seen = cnt > 0
        cX = np.where(seen, ((allX  * vis).sum(1) / safe - fanX)  * opts.CF, 0.0)
        cY = np.where(seen, ((allY  * vis).sum(1) / safe - fanY)  * opts.CF, 0.0)
        aX = np.where(seen, ((allVx * vis).sum(1) / safe - fanVx) * opts.AF, 0.0)
        aY = np.where(seen, ((allVy * vis).sum(1) / safe - fanVy) * opts.AF, 0.0)

        # pull toward MY leader only
        lX = (X[squad] - fanX) * opts.fanLeaderFactor
        lY = (Y[squad] - fanY) * opts.fanLeaderFactor

        vxDes = fanVx + (sX + aX + cX + lX) * dt
        vyDes = fanVy + (sY + aY + cY + lY) * dt

        fanVx, fanVy, _ = ackermannClamp(fanVx, fanVy, vxDes, vyDes, opts)
        fanX = fanX + fanVx * dt
        fanY = fanY + fanVy * dt

        # ------------------------- commit + log -------------------------
        for k in range(2):
            X[k], Y[k], TH[k] = newLead[k]
            if stoppedFlag[k]:
                stopCount[k] += 1
            log['x'][k].append(X[k])
            log['y'][k].append(Y[k])
        log['fanX'].append(fanX.copy())
        log['fanY'].append(fanY.copy())
        t += dt

    # ------------------------- report -------------------------
    xR, yR = np.array(log['x'][0]), np.array(log['y'][0])
    xG, yG = np.array(log['x'][1]), np.array(log['y'][1])
    dLeaders = np.hypot(xR - xG, yR - yG)

    print('guidance: %s | leader repulsion: %s | fanboid rule: stock Reynolds'
          % (GUIDANCE, 'Khatib' if SMOOTH_REPULSION else 'flat SAC'))
    print('closest leader-leader approach %.3f  (collision radius %.2f)'
          % (dLeaders.min(), opts.collisionRadius))
    print('finished: Red %s, Gold %s' % (done[0], done[1]))
    if USE_INFEASIBILITY_STOP:
        print('infeasibility stops: Red %d steps, Gold %d steps'
              % (stopCount[0], stopCount[1]))

    # Path-tracking quality: cross-track error against each reference line.
    # This is the number the guidance law is supposed to improve.
    for k, (name, xs, ys, d) in enumerate([('Red', xR, yR, dataR),
                                           ('Gold', xG, yG, dataG)]):
        a = d.refPath[0]
        u = (d.refPath[-1] - a) / np.linalg.norm(d.refPath[-1] - a)
        n = np.array([-u[1], u[0]])
        e = (xs - a[0]) * n[0] + (ys - a[1]) * n[1]
        print('  %-5s cross-track: peak %.2f, RMS %.2f' % (name, np.abs(e).max(),
                                                           np.sqrt(np.mean(e ** 2))))

    # collision episodes: contiguous stretches below collisionRadius
    below = dLeaders < opts.collisionRadius
    circles = []
    inRun = False
    for tt in range(len(dLeaders)):
        if below[tt] and not inRun:
            circles.append([tt, tt]); inRun = True
        elif below[tt]:
            circles[-1][1] = tt
        elif inRun:
            inRun = False

    results = {'log': log, 'squad': squad, 'dataR': dataR, 'dataG': dataG,
               'opts': opts, 'dLeaders': dLeaders, 'circles': circles,
               'xR': xR, 'yR': yR, 'xG': xG, 'yG': yG,
               'stopCount': stopCount}
    if not plot:
        return results

    # ------------------------- plot -------------------------
    plt.figure(figsize=(9, 8))
    for k, (c, name, d) in enumerate([(RED, 'Red Leader', dataR),
                                      (GOLD, 'Gold Leader', dataG)]):
        plt.plot(d.refPath[:, 0], d.refPath[:, 1], ':', color=c, alpha=0.6,
                 label='%s ref path' % name)
        plt.plot(log['x'][k], log['y'][k], '-', color=c, linewidth=2,
                 label='%s actual' % name)

    fx = np.array(log['fanX'])
    fy = np.array(log['fanY'])
    for i in range(2 * N_FAN):
        c = RED if squad[i] == 0 else GOLD
        plt.plot(fx[:, i], fy[:, i], '-', color=c, linewidth=0.5, alpha=0.35,
                 zorder=1)

    for n_, (t0, t1) in enumerate(circles):
        tw = t0 + int(np.argmin(dLeaders[t0:t1 + 1]))
        cx = 0.5 * (xR[tw] + xG[tw])
        cy = 0.5 * (yR[tw] + yG[tw])
        plt.gca().add_patch(plt.Circle((cx, cy), CIRCLE_R, fill=False,
                                       color='red', lw=2.0, zorder=6,
                                       label='COLLISION' if n_ == 0 else None))

    plt.axis('equal')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Two squads, %s approach (seed %d, %d fans each, guidance=%s)'
              % (GEOMETRY, SEED, N_FAN, GUIDANCE))
    plt.legend(fontsize=8)
    plt.show()
    return results


# ---------------------------------------------------------------------
#  Extras
# ---------------------------------------------------------------------

def collisions(radius=None):
    """Run the sim, then report EVERY collision: leader-leader, leader-fan,
    fan-fan (same squad and cross squad)."""
    from grainframe.collisions import agents_from_arrays, collision_report

    r = run(plot=False)
    log, squad = r['log'], r['squad']
    fx = np.array(log['fanX'])
    fy = np.array(log['fanY'])

    leaders = [(r['xR'], r['yR']), (r['xG'], r['yG'])]
    fans = [(fx[:, i], fy[:, i]) for i in range(fx.shape[1])]

    agents = agents_from_arrays(leaders, fans, fanTeams=squad,
                                leaderNames=['Red Leader', 'Gold Leader'],
                                teamNames={0: 'Red', 1: 'Gold'})
    if radius is None:
        radius = r['opts'].collisionRadius
    return collision_report(agents, radius, dt=r['opts'].dt)


def animate(frameSkip=50, tail=400, save=None):
    """Run the sim, then play it back using the shared animation utility."""
    from grainframe.animate import tracks_from_arrays, animate_tracks

    r = run(plot=False)
    log, squad = r['log'], r['squad']
    fx = np.array(log['fanX'])
    fy = np.array(log['fanY'])

    leaders = [(r['xR'], r['yR']), (r['xG'], r['yG'])]
    fans = [(fx[:, i], fy[:, i]) for i in range(fx.shape[1])]
    fanColors = [RED if squad[i] == 0 else GOLD for i in range(fx.shape[1])]

    dLeaders = r['dLeaders']
    dt = r['opts'].dt
    subtitle = lambda t: 't = %.1f s   leader gap = %.2f' % (t * dt, dLeaders[t])
    refs = [(r['dataR'].refPath[:, 0], r['dataR'].refPath[:, 1], RED),
            (r['dataG'].refPath[:, 0], r['dataG'].refPath[:, 1], GOLD)]

    tracks = tracks_from_arrays(leaders, fans, [RED, GOLD], fanColors,
                                leaderLabels=['Red Leader', 'Gold Leader'])
    return animate_tracks(tracks, dt=dt, frameSkip=frameSkip, tail=tail,
                          title='Two squads, %s (seed %d)' % (GEOMETRY, SEED),
                          subtitleFn=subtitle, refPaths=refs, save=save)


def compare():
    """Side by side: old waypoint chasing vs the vector field. For the paper."""
    global GUIDANCE, SMOOTH_REPULSION
    saveG, saveS = GUIDANCE, SMOOTH_REPULSION

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    settings = [('waypoint', False, 'BEFORE: chase next waypoint, flat repulsion'),
                ('vfield',   True,  'AFTER: vector field, Khatib repulsion')]
    for ax, (g, sm, label) in zip(axes, settings):
        GUIDANCE, SMOOTH_REPULSION = g, sm
        r = run(plot=False)
        ax.axhline(0, color='k', lw=0.5, ls=':')
        fx, fy = np.array(r['log']['fanX']), np.array(r['log']['fanY'])
        for i in range(fx.shape[1]):
            ax.plot(fx[:, i], fy[:, i], lw=0.4, alpha=0.3,
                    color=RED if r['squad'][i] == 0 else GOLD)
        ax.plot(r['xR'], r['yR'], color=RED,  lw=2, label='Red')
        ax.plot(r['xG'], r['yG'], color=GOLD, lw=2, label='Gold')
        ax.set_title('%s   |  closest gap %.2f' % (label, r['dLeaders'].min()),
                     fontsize=10)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc='upper left')

    GUIDANCE, SMOOTH_REPULSION = saveG, saveS
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'plot'
    if mode.startswith('anim'):
        animate()
    elif mode.startswith('coll'):
        collisions()
    elif mode.startswith('comp'):
        compare()
    else:
        run()