"""Sandbox: two squads approach each other. One run, one plot.

ALL the knobs live right below in the PARAMETERS block -- edit and rerun.
Everything else is the plain vanilla rules (flat SAC at stock PR, no
right-of-way, no enlarged ranges).
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root on the path
import numpy as np
from matplotlib import pyplot as plt
from grainframe import SimOptions
from grainframe.data_io import apply_dataset
from grainframe.paths import straightPath
from grainframe.dynamics import (boidsRules, falconSteering, ackermannClamp,
                                 WaypointTracker)

# ======================= PARAMETERS: edit me =======================
SEED  = 8            # change for a different random squadron layout
N_FAN = 6            # fanboids per squadron

GEOMETRY = 'headon'  # 'headon' = straight at each other on the same line
                     # 'cross'  = the X crossing at the origin

# Path endpoints (used by whichever GEOMETRY is picked)
HALF = 40.0          # how far from the middle each leader starts
LANE_OFFSET = 0.0    # shove Gold's lane sideways by this much. 0 = perfectly
                     # symmetric head-on (pure SAC CAN'T break the tie, they
                     # bulldoze through). Try 2-5 to let them actually veer past.

# Gain overrides: set to None to use the stock value from config.py
OVERRIDES = {
    # 'SF': 5.0,             # separation strength
    # 'CF': 0.8,             # cohesion
    # 'AF': 2.0,             # alignment
    # 'fanLeaderFactor': 1.0,# pull toward own leader
    # 'leaderFactor': 1.0,   # leader's pull toward his waypoints
    # 'R': 5.0,              # leader turn radius (bigger = clumsier)
}

# Per-leader protected range: how early each leader reacts to the OTHER leader.
# Make them DIFFERENT to break the mirror symmetry -- the one with the bigger
# range gives way, the one with the smaller range mostly holds his line.
PR_RED  = 12.0
PR_GOLD = 4.0

# Per-leader waypoint pull: how hard each leader's path drags him home.
# Crank RED up and he snaps back on course sooner (smaller avoidance bulge);
# the swerve size is set by the tug-of-war between SF (shove) and this (pull).
LEADER_FACTOR_RED  = 1.0
LEADER_FACTOR_GOLD = 1.0

# Infeasibility stop (paper Section 4.1): if a leader physically cannot turn
# hard enough to avoid the other, he STOPS instead of orbiting uselessly.
# Stopping is always feasible regardless of turn radius.
USE_INFEASIBILITY_STOP = True
STOP_RANGE   = 6.0    # start worrying about a stop inside this distance
STOP_LOOKAHEAD = 40   # look this many steps ahead under the turn clamp
STOP_MARGIN  = 1.5    # stop if best-case future gap stays below this

FAN_SPREAD = 3.0     # how loosely the squadron spawns around its leader
CIRCLE_R   = 3.0     # radius of the drawn COLLISION circles
# ===================================================================

RED, GOLD = 'crimson', 'goldenrod'


def canAvoid(x, y, th, xOther, yOther, thOther, opts):
    """Can this leader clear the other by turning as hard as he legally can?

    Roll both leaders forward STOP_LOOKAHEAD steps: the other holds heading,
    and I turn at my maximum rate (v/R) in the direction that opens the gap.
    If even that best-case future keeps us closer than STOP_MARGIN, no feasible
    avoidance exists -> caller should STOP.
    """
    dt, v, R = opts.dt, opts.v, opts.R
    dpsiMax = (v / R) * dt                  # most I can rotate per step

    # which way opens the gap? sign of the cross product of "toward other" and heading
    toX, toY = xOther - x, yOther - y
    cross = np.cos(th) * toY - np.sin(th) * toX
    turn = -np.sign(cross) * dpsiMax        # turn AWAY from the other leader

    xi, yi, thi = x, y, th
    xo, yo = xOther, yOther
    best = np.inf
    for _ in range(STOP_LOOKAHEAD):
        thi += turn
        xi += v * np.cos(thi) * dt
        yi += v * np.sin(thi) * dt
        xo += v * np.cos(thOther) * dt      # other just keeps going straight
        yo += v * np.sin(thOther) * dt
        best = min(best, np.hypot(xi - xo, yi - yo))
    return best >= STOP_MARGIN


def separation(xb, yb, xOthers, yOthers, opts, range_=None):
    # plain SAC: flat push away from anyone inside my protected range
    if range_ is None:
        range_ = opts.PR
    dx = xb - xOthers
    dy = yb - yOthers
    D = np.hypot(dx, dy)
    tooClose = (D > 1e-9) & (D <= range_)
    return (float(np.sum(dx[tooClose]) * opts.SF),
            float(np.sum(dy[tooClose]) * opts.SF))


def makePaths():
    if GEOMETRY == 'headon':
        dataR = straightPath(-HALF, 0, +HALF, 0)                    # Red:  left -> right
        dataG = straightPath(+HALF, LANE_OFFSET, -HALF, LANE_OFFSET) # Gold: right -> left
    else:                                            # 'cross'
        dataR = straightPath(-HALF, -HALF, +HALF, +HALF)
        dataG = straightPath(-HALF, +HALF + LANE_OFFSET, +HALF, -HALF + LANE_OFFSET)
    return dataR, dataG


def run(plot=True):
    opts = SimOptions()
    dataR, dataG = makePaths()
    apply_dataset(opts, dataR)
    for name, val in OVERRIDES.items():              # apply your tweaks
        if val is not None:
            setattr(opts, name, val)
    np.random.seed(SEED)

    # Leader state: [Red, Gold]
    X  = np.array([dataR.qi[0], dataG.qi[0]])
    Y  = np.array([dataR.qi[1], dataG.qi[1]])
    TH = np.array([dataR.qi[2], dataG.qi[2]])
    wayPts = [dataR.wayPts, dataG.wayPts]
    trackers = [WaypointTracker(opts), WaypointTracker(opts)]
    idP = [1, 1]
    done = [False, False]
    goalMin = [np.inf, np.inf]
    stoppedFlag = [False, False]      # is each leader frozen this step?
    stopCount = [0, 0]                # how many steps each spent stopped

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
    fanX, fanY = np.concatenate(fanX), np.concatenate(fanY)
    fanVx, fanVy = np.concatenate(fanVx), np.concatenate(fanVy)
    squad = np.concatenate(squad)

    dt, v = opts.dt, opts.v
    log = {'x': [[], []], 'y': [[], []], 'fanX': [], 'fanY': []}

    t = 0.0
    tMax = opts.tMax(dataR.refLen)
    while t <= tMax and not all(done):

        # ---------------- LEADERS ----------------
        newLead = []
        for k in range(2):
            if done[k]:
                newLead.append((X[k], Y[k], TH[k]))
                continue

            wp = wayPts[k]
            idP[k] = trackers[k].advance(X[k], Y[k], idP[k], wp)
            xL, yL = wp[idP[k], 0], wp[idP[k], 1]
            dL = np.hypot(xL - X[k], yL - Y[k])

            # done at closest approach to the last waypoint
            if idP[k] == wp.shape[0] - 1:
                goalMin[k] = min(goalMin[k], dL)
                if goalMin[k] < opts.passWindow and dL > goalMin[k] + opts.hysteresis:
                    done[k] = True
                    newLead.append((X[k], Y[k], TH[k]))
                    continue

            # follow the path... (each leader has his OWN pull strength)
            myPull = LEADER_FACTOR_RED if k == 0 else LEADER_FACTOR_GOLD
            vx = (xL - X[k]) * myPull
            vy = (yL - Y[k]) * myPull

            # ...plus SAC against ONLY the other leader (leaders ignore fanboids)
            # Each leader uses his OWN protected range -> asymmetry breaks the mirror
            other = 1 - k
            myPR = PR_RED if k == 0 else PR_GOLD
            sx, sy = separation(X[k], Y[k],
                                np.array([X[other]]), np.array([Y[other]]),
                                opts, range_=myPR)
            vx += sx
            vy += sy

            # Dubins steering
            vTh = falconSteering(TH[k], vx, vy, opts)

            # Infeasibility stop: if he can't turn hard enough to avoid the
            # other leader, freeze in place instead of carving useless circles.
            other = 1 - k
            distOther = np.hypot(X[other] - X[k], Y[other] - Y[k])
            if (USE_INFEASIBILITY_STOP and distOther < STOP_RANGE and
                    not canAvoid(X[k], Y[k], TH[k],
                                 X[other], Y[other], TH[other], opts)):
                newLead.append((X[k], Y[k], TH[k]))       # STOP: hold position
                stoppedFlag[k] = True
            else:
                newLead.append((X[k] + v * np.cos(TH[k]) * dt,
                                Y[k] + v * np.sin(TH[k]) * dt,
                                TH[k] + vTh * dt))
                stoppedFlag[k] = False

        # ---------------- FANBOIDS ----------------
        vxDes = np.empty(2 * N_FAN)
        vyDes = np.empty(2 * N_FAN)
        for i in range(2 * N_FAN):
            k = squad[i]                       # my squadron / my leader
            mine = (squad == k)

            # cohesion + alignment with my OWN squadron (+ my leader)
            sameX = np.append(fanX[mine], X[k])
            sameY = np.append(fanY[mine], Y[k])
            sameVx = np.append(fanVx[mine], v * np.cos(TH[k]))
            sameVy = np.append(fanVy[mine], v * np.sin(TH[k]))
            (_, _, aX, aY, cX, cY) = boidsRules(
                fanX[i], fanY[i], fanVx[i], fanVy[i],
                sameX, sameY, sameVx, sameVy, opts)

            # SAC against everyone, both squadrons and both leaders
            everyX = np.append(fanX, X)
            everyY = np.append(fanY, Y)
            sX, sY = separation(fanX[i], fanY[i], everyX, everyY, opts)

            # pull toward MY leader only
            lX = (X[k] - fanX[i]) * opts.fanLeaderFactor
            lY = (Y[k] - fanY[i]) * opts.fanLeaderFactor

            vxDes[i] = fanVx[i] + (sX + aX + cX + lX) * dt
            vyDes[i] = fanVy[i] + (sY + aY + cY + lY) * dt

        fanVx, fanVy, _ = ackermannClamp(fanVx, fanVy, vxDes, vyDes, opts)
        fanX = fanX + fanVx * dt
        fanY = fanY + fanVy * dt

        # ---------------- commit + log ----------------
        for k in range(2):
            X[k], Y[k], TH[k] = newLead[k]
            if stoppedFlag[k]:
                stopCount[k] += 1
            log['x'][k].append(X[k])
            log['y'][k].append(Y[k])
        log['fanX'].append(fanX.copy())
        log['fanY'].append(fanY.copy())
        t += dt

    # ---------------- report ----------------
    xR, yR = np.array(log['x'][0]), np.array(log['y'][0])
    xG, yG = np.array(log['x'][1]), np.array(log['y'][1])
    dLeaders = np.hypot(xR - xG, yR - yG)
    print('closest leader-leader approach %.3f  (collision radius %.2f)'
          % (dLeaders.min(), opts.collisionRadius))
    print('finished: Red %s, Gold %s' % (done[0], done[1]))
    if USE_INFEASIBILITY_STOP:
        print('infeasibility stops: Red %d steps, Gold %d steps'
              % (stopCount[0], stopCount[1]))

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
               'xR': xR, 'yR': yR, 'xG': xG, 'yG': yG}
    if not plot:
        return results

    # ---------------- plot ----------------
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

    # red circle at the deepest point of every collision episode
    for n, (t0, t1) in enumerate(circles):
        tw = t0 + int(np.argmin(dLeaders[t0:t1 + 1]))
        cx = 0.5 * (xR[tw] + xG[tw])
        cy = 0.5 * (yR[tw] + yG[tw])
        plt.gca().add_patch(plt.Circle((cx, cy), CIRCLE_R, fill=False,
                                       color='red', lw=2.0, zorder=6,
                                       label='COLLISION' if n == 0 else None))

    plt.axis('equal')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Two squads, %s approach (seed %d, %d fans each)'
              % (GEOMETRY, SEED, N_FAN))
    plt.legend(fontsize=8)
    plt.show()
    return results


def collisions(radius=None):
    """Run the sim, then report EVERY collision: leader-leader, leader-fan,
    fan-fan (same squad and cross squad). Prints a breakdown to the terminal.
    """
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


def animate(frameSkip=25, tail=400, save=None):
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


if __name__ == '__main__':
    # Pick a mode from the command line:
    #   python experiments/exp5_sandbox.py            -> static plot
    #   python experiments/exp5_sandbox.py anim       -> live playback
    #   python experiments/exp5_sandbox.py collide    -> collision report only
    mode = sys.argv[1] if len(sys.argv) > 1 else 'plot'
    if mode.startswith('anim'):
        animate()
    elif mode.startswith('coll'):
        collisions()
    else:
        run()