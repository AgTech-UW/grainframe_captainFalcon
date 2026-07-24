"""Barebones version of Adam's crossing experiment. No fixes, no extras.

The spec, as requested:
  1. Two leaders on straight paths that cross in an X. Each follows his own
     path. When one shows up in the other's protected range, the ordinary
     SAC (separation) rule engages between the leaders, on top of the
     follow-the-path pull.
  2. Same, but each leader brings his own squadron of fanboids. Leaders only
     attract THEIR OWN squadron; SAC applies to ALL fanboids across
     squadrons.

Everything below uses the stock boid rules and stock parameters. The
separation push is the flat-strength one from boidsRules, triggered at the
ordinary protected range PR.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root on the path
import numpy as np
from matplotlib import pyplot as plt
from grainframe import SimOptions, make_arena
from grainframe.data_io import apply_dataset
from grainframe.paths import straightPath
from grainframe.dynamics import (boidsRules, falconSteering, ackermannClamp,
                                 WaypointTracker)

RED, GOLD = 'crimson', 'goldenrod'


def separation(xb, yb, xOthers, yOthers, opts):
    # The plain SAC rule, same math as boidsRules: flat push away from
    # anyone inside my protected range.
    dx = xb - xOthers
    dy = yb - yOthers
    D = np.hypot(dx, dy)
    tooClose = (D > 1e-9) & (D <= opts.PR)
    return (float(np.sum(dx[tooClose]) * opts.SF),
            float(np.sum(dy[tooClose]) * opts.SF))


def run(nFan=0, seed=None, show=False, title=''):
    opts = SimOptions()
    if seed is None:
        seed = opts.seed
    np.random.seed(seed)

    # Two straight paths crossing in an X at the origin
    dataR = straightPath(-40, -40, +40, +40)   # Red:  SW -> NE
    dataG = straightPath(-40, +40, +40, -40)   # Gold: NW -> SE
    apply_dataset(opts, dataR)
    arena = make_arena(dataR, opts)            # (box around Red's path; Gold's is the mirror)

    # Leader state: [Red, Gold]
    X  = np.array([dataR.qi[0], dataG.qi[0]])
    Y  = np.array([dataR.qi[1], dataG.qi[1]])
    TH = np.array([dataR.qi[2], dataG.qi[2]])
    wayPts = [dataR.wayPts, dataG.wayPts]
    trackers = [WaypointTracker(opts), WaypointTracker(opts)]
    idP = [1, 1]
    done = [False, False]
    goalMin = [np.inf, np.inf]

    # Squadron state: each leader gets nFan boids clumped near his start
    fanX, fanY, fanVx, fanVy, squad = [], [], [], [], []
    for k in range(2):
        fanX.append(X[k] + np.random.normal(0, 3.0, nFan))
        fanY.append(Y[k] + np.random.normal(0, 3.0, nFan))
        ang = np.random.uniform(0, 2 * np.pi, nFan)
        spd = np.random.uniform(opts.minSpeed, opts.maxSpeed, nFan)
        fanVx.append(spd * np.cos(ang))
        fanVy.append(spd * np.sin(ang))
        squad.append(np.full(nFan, k))         # which squadron each boid belongs to
    fanX, fanY = np.concatenate(fanX), np.concatenate(fanY)
    fanVx, fanVy = np.concatenate(fanVx), np.concatenate(fanVy)
    squad = np.concatenate(squad)

    dt, v = opts.dt, opts.v
    log = {'x': [[], []], 'y': [[], []],
           'fanX': [], 'fanY': []}

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

            # follow the path...
            vx = (xL - X[k]) * opts.leaderFactor
            vy = (yL - Y[k]) * opts.leaderFactor

            # ...plus SAC against the OTHER leader (and any fanboids nearby)
            other = 1 - k
            nearX = np.append(fanX, X[other])
            nearY = np.append(fanY, Y[other])
            sx, sy = separation(X[k], Y[k], nearX, nearY, opts)
            vx += sx
            vy += sy

            # Dubins steering
            vTh = falconSteering(TH[k], vx, vy, opts)
            newLead.append((X[k] + v * np.cos(TH[k]) * dt,
                            Y[k] + v * np.sin(TH[k]) * dt,
                            TH[k] + vTh * dt))

        # ---------------- FANBOIDS ----------------
        if nFan > 0:
            vxDes = np.empty(2 * nFan)
            vyDes = np.empty(2 * nFan)
            for i in range(2 * nFan):
                k = squad[i]                    # my squadron / my leader
                mine = (squad == k)

                # cohesion + alignment with my OWN squadron (+ my leader)
                sameX = np.append(fanX[mine], X[k])
                sameY = np.append(fanY[mine], Y[k])
                sameVx = np.append(fanVx[mine], v * np.cos(TH[k]))
                sameVy = np.append(fanVy[mine], v * np.sin(TH[k]))
                (_, _, aX, aY, cX, cY) = boidsRules(
                    fanX[i], fanY[i], fanVx[i], fanVy[i],
                    sameX, sameY, sameVx, sameVy, opts)

                # SAC against EVERYONE, both squadrons and both leaders
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
            log['x'][k].append(X[k])
            log['y'][k].append(Y[k])
        if nFan > 0:
            log['fanX'].append(fanX.copy())
            log['fanY'].append(fanY.copy())
        t += dt

    # ---------------- report ----------------
    xR, yR = np.array(log['x'][0]), np.array(log['y'][0])
    xG, yG = np.array(log['x'][1]), np.array(log['y'][1])
    dLeaders = np.hypot(xR - xG, yR - yG)
    print('%s : closest leader-leader approach %.3f  (finished: Red %s, Gold %s)'
          % (title, dLeaders.min(), done[0], done[1]))

    # ---------------- plot ----------------
    plt.figure(figsize=(9, 8))
    for k, (c, name, d) in enumerate([(RED, 'Red Leader', dataR),
                                      (GOLD, 'Gold Leader', dataG)]):
        plt.plot(d.refPath[:, 0], d.refPath[:, 1], ':', color=c, alpha=0.6,
                 label='%s ref path' % name)
        plt.plot(log['x'][k], log['y'][k], '-', color=c, linewidth=2,
                 label='%s actual' % name)
    if nFan > 0:
        fx = np.array(log['fanX'])
        fy = np.array(log['fanY'])
        for i in range(2 * nFan):
            c = RED if squad[i] == 0 else GOLD
            plt.plot(fx[:, i], fy[:, i], '-', color=c, linewidth=0.5, alpha=0.35,
                     zorder=1)
    plt.axis('equal')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(title)
    plt.legend(fontsize=8)
    if show:
        plt.show()


if __name__ == '__main__':
    run(nFan=0, title='Barebones 1: two leaders, plain SAC')
    run(nFan=6, title='Barebones 2: Red vs Gold squadrons, plain SAC')
    plt.show()