import numpy as np
from matplotlib import pyplot as plt

# =========================================================================
# CAPTAIN FALCON & THE FANBOIDS
# -------------------------------------------------------------------------
# Three kinds of agent share the field now:
#
#   CAPTAIN FALCON - Dubins-car boid. Rules 1-3 (Sep/Align/Coh) + his own
#                    unique rule 4 "leader": chase the ACTIVE WAYPOINT of a
#                    precomputed Dubins path. Bang-bang steering rate in
#                    {-v/R, 0, +v/R}. Unchanged.
#
#   DISTRACTORS    - free-flying riff-raff. Rules 1-3 + wall turnback.
#                    NEW: Ackermann-style turn limit (see ackermannClamp),
#                    so they can no longer pirouette on the spot.
#
#   FANBOIDS       - the Captain's groupies. Rules 1-3 (their neighborhood
#                    includes each other AND the Captain) + their own rule 4
#                    "leader": chase FALCON'S LIVE POSITION. They know
#                    nothing about waypoints or the Dubins plan - they just
#                    chase the man himself. Same Ackermann turn limit.
#
# Experiments at the bottom:
#   (1) A/B      : falcon clean vs falcon + distractors (as before; note the
#                  B numbers shift because distractors are now turn-limited)
#   (2) showcase : falcon + nFanShowcase fanboids, NO distractors. Final-
#                  state Separation/Alignment/Cohesion report, path
#                  smoothness stats, and per-follower final-pose error plot
#                  (dx, dy, distance, heading) vs the unperturbed captain.
#   (3) sweep    : n in fanSweepN, fanTrials random clumped starts per n,
#                  chi^2 of final fan poses vs falcon's final pose,
#                  evaluated JUST AT THE END -> plot n vs chi^2.
#
# Because opts.falconSeesFanboids defaults to False, the captain's path is
# IDENTICAL in every fan run (he is deterministic when unperturbed), so
# chi^2 differences are purely follower dynamics. Flip the flag to True to
# instead study how hard an entourage of size n perturbs the captain.
#
# GOAL SEMANTICS (CPA ending): a run ends at the captain's closest approach
# to the final waypoint - the same pass logic advanceWaypoint uses - and
# every log is rewound to that moment. collectRadius is a GRADING threshold
# (reachedGoal = closest approach < collectRadius), not a termination
# condition, and out['goalMiss'] reports the miss distance. A deflected
# captain therefore never orbits the goal until tMax.
#
# This file is the ANSWER KEY for captain_falcon_skeleton.py: same
# structure, function for function (incl. fanLeaderRule), stubs filled in.
# =========================================================================

# -------------------------------------------------------------------------
# Load the MATLAB-exported Dubins data (CSVs live next to this script)
# -------------------------------------------------------------------------
dataTag = '1'
refPath = np.loadtxt('saved_dubins_path_%s.csv' % dataTag, delimiter=',')       # dense [x, y, theta]
wayPts  = np.loadtxt('saved_dubins_waypoints_%s.csv' % dataTag, delimiter=',')  # switching waypoints [x, y, theta]
qi      = np.loadtxt('saved_qi_%s.csv' % dataTag, delimiter=',')                # initial pose
qf      = np.loadtxt('saved_qf_%s.csv' % dataTag, delimiter=',')                # goal pose
pR, pV, pDt = np.loadtxt('saved_params_%s.csv' % dataTag, delimiter=',')        # [R, v, dt]

refPath = refPath[~np.isnan(refPath).any(axis=1)]  # writematrix can leave NaN rows
refLen  = np.sum(np.hypot(np.diff(refPath[:,0]), np.diff(refPath[:,1])))

# -------------------------------------------------------------------------
# Config object (taken from gridStruct pattern as boidsDemo.py)
# -------------------------------------------------------------------------
class gridStruct:
    def __init__(opts, TF, VR, PR, CF, SF, AF, maxS, minS, safetyF):
        opts.TF = TF # Turning factor
        opts.VR = VR # Visual range
        opts.PR = PR # Protected range
        opts.CF = CF # Cohesion factor
        opts.SF = SF # Separation factor
        opts.AF = AF # Alignment factor
        opts.maxSpeed=maxS # Maximum speed
        opts.minSpeed=minS # Minimum speed
        opts.safetyF=safetyF # Safety factor

opts = gridStruct(2.0,   # TF
                  8.0,   # VR
                  2.5,   # PR
                  0.8,   # CF
                  5.0,   # SF
                  2.0,   # AF
                  2.0,   # maxS
                  0.8,   # minS
                  5.0)   # safetyF

# Captain Falcon / experiment additions
opts.R  = float(pR)
opts.v  = float(pV)
opts.dt = float(pDt)
opts.N  = 10                 # number of distractor boids
opts.leaderFactor = 1.0
opts.falconBoidsWeight = 0.4 # how hard the flock's forces hit Falcon
opts.dAng = 0.005            # steering deadband [rad]
opts.collectRadius = 0.75    # capture window at the FINAL waypoint
opts.passWindow = 5          # a pass only counts within this range of the waypoint
opts.hysteresis = 2.5*opts.v*opts.dt
opts.seed = 8
opts.tMax = 1.75*refLen/opts.v + 10.0

# --- NEW: Ackermann-style constraint for the small boids -----------------
# Minimum turning radius for distractors AND fanboids. Their heading can
# change at most (speed/Rboid) rad/s, i.e. curvature is capped at 1/Rboid,
# same abstraction that makes Falcon a Dubins car - just with continuous
# steering instead of bang-bang, and a variable (clamped) speed.
# Keep Rboid comfortably below safetyF or wall turnbacks start overshooting.
opts.Rboid = 0.5*opts.R

# --- NEW: fanboid parameters ----------------------------------------------
opts.fanLeaderFactor   = 1.0    # rule-4 pull toward the captain's live position
opts.falconSeesFanboids = False # captain stays unperturbed (clean chi^2 reference)
opts.nFanShowcase = 10          # flock size for the showcase run + error plot
opts.fanSweepN    = [1, 2, 3, 5, 7, 12, 20]
opts.fanTrials    = 5           # random clumped restarts per n
opts.sigmaPos = opts.collectRadius      # chi^2 position scale: "parked within
opts.sigmaTh  = np.deg2rad(15.0)        #  capture tolerance, aligned to 15deg"
                                        #  gives reduced chi^2 ~ 1

# Field walls sized from the path (safety boundary inset by safetyF)
pad = 2.0*opts.R
xFMin=min(refPath[:,0].min(), qi[0], qf[0]) - pad
xFMax=max(refPath[:,0].max(), qi[0], qf[0]) + pad
yFMin=min(refPath[:,1].min(), qi[1], qf[1]) - pad
yFMax=max(refPath[:,1].max(), qi[1], qf[1]) + pad
xSafeMin=xFMin+opts.safetyF; xSafeMax=xFMax-opts.safetyF
ySafeMin=yFMin+opts.safetyF; ySafeMax=yFMax-opts.safetyF


def wrapToPi(a):
    return (a + np.pi) % (2*np.pi) - np.pi


# =========================================================================
# The Boids Rules
# =========================================================================
def boidsRules(xb, yb, vxb, vyb, xAll, yAll, vxAll, vyAll, opts):
    dx = xb - xAll
    dy = yb - yAll
    D = np.sqrt(dx**2 + dy**2)
    # D > 0 excludes SELF, so agents may safely appear in their own
    # neighbor arrays (the fanboids' arrays include themselves + falcon).
    idVisual    = (D > 1e-9) & (D <= opts.VR)
    idProtected = (D > 1e-9) & (D <= opts.PR)
    closeDX = np.sum(dx[idProtected])
    closeDY = np.sum(dy[idProtected])
    neighboringBoids = np.count_nonzero(idVisual)
    if neighboringBoids > 0:
        xPosAvg = np.mean(xAll[idVisual])
        yPosAvg = np.mean(yAll[idVisual])
        xVelAvg = np.mean(vxAll[idVisual])
        yVelAvg = np.mean(vyAll[idVisual])
        # 3 Cohesion - steer toward center of mass
        vxCohesion = (xPosAvg-xb)*opts.CF
        vyCohesion = (yPosAvg-yb)*opts.CF
        # 2 Alignment
        vxAlignment = (xVelAvg - vxb)*opts.AF
        vyAlignment = (yVelAvg - vyb)*opts.AF
    else:
        vxCohesion = 0.0
        vyCohesion = 0.0
        vxAlignment = 0.0
        vyAlignment = 0.0

    # 1 Separation
    vxSeparation = closeDX*opts.SF
    vySeparation = closeDY*opts.SF

    return (float(vxSeparation), float(vySeparation),
            float(vxAlignment),  float(vyAlignment),
            float(vxCohesion),   float(vyCohesion))


# =========================================================================
# Rule 4, fan edition: chase the captain's LIVE position
# -------------------------------------------------------------------------
# Mirrors the captain's own leader rule, but the target is HIM, not a
# waypoint - the fans know nothing about the Dubins plan. Returned as a
# per-second rate; the loop scales it by dt like every other kick.
# =========================================================================
def fanLeaderRule(xb, yb, xCap, yCap, opts):
    kx = (xCap - xb)*opts.fanLeaderFactor
    ky = (yCap - yb)*opts.fanLeaderFactor
    return kx, ky


# =========================================================================
# Dubins Steering (Captain Falcon only)
# =========================================================================
def falconSteering(thetai, vxTotal, vyTotal, opts):
    """Given the desired-direction vector (vxTotal, vyTotal), return the
    heading rate vThetai in {-v/R, 0, +v/R}.
    """
    nrm = np.hypot(vxTotal, vyTotal)
    if nrm < 1e-12:
        return 0.0  # no preference -> hold course (avoids 0/0 -> NaN)
    vi = (opts.v * np.cos(thetai), opts.v * np.sin(thetai))
    cosAng = np.clip(np.dot(vi, (vxTotal, vyTotal)) / (opts.v * nrm), -1, 1)
    totalAngle = np.arccos(cosAng)
    crossZ = vi[0]*vyTotal - vi[1]*vxTotal
    if totalAngle < opts.dAng:
        controlAngle = 0
    elif crossZ > 0:
        controlAngle = -1
    else:
        controlAngle = 1
    vThetai = -opts.v*controlAngle/opts.R
    return vThetai


# =========================================================================
# Ackermann-style update for the small boids (distractors + fanboids)
# =========================================================================
def ackermannClamp(vx, vy, vxDes, vyDes, opts):
    """Point boids used to teleport their velocity vector to wherever the
    rule kicks pointed - a 180 in one dt was legal. A car cannot do that:
    heading changes only at rate |omega| <= speed/Rboid, and since speed is
    clamped >= minSpeed the boid must roll forward to turn - no spinning on
    the spot. The kicks now shape a DESIRED velocity; this clamp limits how
    far the actual velocity may swing toward it in one step.
    Vectorized over all boids. Returns (vxNew, vyNew, omega)."""
    sDes = np.hypot(vxDes, vyDes)
    psi  = np.arctan2(vy, vx)
    psiDes = np.where(sDes > 1e-12, np.arctan2(vyDes, vxDes), psi)
    s = np.clip(sDes, opts.minSpeed, opts.maxSpeed)
    dpsiMax = (s/opts.Rboid)*opts.dt          # curvature cap 1/Rboid
    dpsi = np.clip(wrapToPi(psiDes - psi), -dpsiMax, dpsiMax)
    psiNew = psi + dpsi
    return s*np.cos(psiNew), s*np.sin(psiNew), dpsi/opts.dt


# =========================================================================
# Waypoint Handover (Captain Falcon)
# =========================================================================
def advanceWaypoint(xi, yi, idP, wayPts, opts):
    """
    Bullseye / Closest-Point-of-Approach handover.
    Advances when the distance to the waypoint stops decreasing,
    meaning the car has passed it.
    """
    nP = wayPts.shape[0]
    if idP >= nP - 1:
        return idP

    wp_x, wp_y = wayPts[idP, 0], wayPts[idP, 1]
    d = np.hypot(wp_x - xi, wp_y - yi)

    if not hasattr(opts, 'minDist'):
        opts.minDist = np.inf
    opts.minDist = min(opts.minDist, d)

    if opts.minDist < opts.passWindow and d > (opts.minDist + opts.hysteresis):
        idP += 1
        opts.minDist = np.inf
        while idP < nP - 1 and np.hypot(wayPts[idP,0]-xi, wayPts[idP,1]-yi) < 1e-5:
            idP += 1

    return idP


# =========================================================================
# Initializers for the small boids
# =========================================================================
def clampSpeeds(vx, vy, opts):
    s = np.maximum(np.hypot(vx, vy), 1e-9)
    sNew = np.clip(s, opts.minSpeed, opts.maxSpeed)
    return vx/s*sNew, vy/s*sNew

def initUniform(n, opts):
    """Distractors: uniform over the path's bounding box (as before)."""
    x = refPath[:,0].min() + (refPath[:,0].max()-refPath[:,0].min())*np.random.rand(n)
    y = refPath[:,1].min() + (refPath[:,1].max()-refPath[:,1].min())*np.random.rand(n)
    vx = (np.random.rand(n)-0.5)*2*opts.maxSpeed
    vy = (np.random.rand(n)-0.5)*2*opts.maxSpeed
    vx, vy = clampSpeeds(vx, vy, opts)
    return x, y, vx, vy

def initClumped(n, opts):
    """Fanboids: 'start random, position, and velocities, different clumps'.
    1-3 random clump centers inside the path bbox, gaussian spread around
    each, random headings, speeds uniform in [minSpeed, maxSpeed]."""
    k = np.random.randint(1, min(3, n) + 1)
    cx = refPath[:,0].min() + (refPath[:,0].max()-refPath[:,0].min())*np.random.rand(k)
    cy = refPath[:,1].min() + (refPath[:,1].max()-refPath[:,1].min())*np.random.rand(k)
    who = np.random.randint(0, k, n)
    x = cx[who] + 2.0*np.random.randn(n)
    y = cy[who] + 2.0*np.random.randn(n)
    ang = 2*np.pi*np.random.rand(n)
    spd = opts.minSpeed + (opts.maxSpeed-opts.minSpeed)*np.random.rand(n)
    return x, y, spd*np.cos(ang), spd*np.sin(ang)


# =========================================================================
# One simulation run
# =========================================================================
def runSimulation(opts, wayPts, qiPose, nDistract=0, nFan=0, seed=None):
    np.random.seed(opts.seed if seed is None else seed)
    opts.minDist = np.inf   # reset CPA memory - runs must not leak into each other

    xDis, yDis, vxDis, vyDis = initUniform(nDistract, opts)
    xFan, yFan, vxFan, vyFan = initClumped(nFan, opts) if nFan > 0 else \
                               (np.array([]),)*4

    xi, yi, thetai = float(qiPose[0]), float(qiPose[1]), float(qiPose[2])
    idP, nP = 1, wayPts.shape[0]
    t, dt, v = 0.0, opts.dt, opts.v
    log = {'t':[], 'x':[], 'y':[], 'theta':[],
           'boidX':[], 'boidY':[],
           'fanX':[], 'fanY':[], 'fanVx':[], 'fanVy':[], 'fanOmega':[]}
    fanX0, fanY0 = xFan.copy(), yFan.copy()
    reachedGoal = False
    goalMin, cpaIdx = np.inf, -1   # closest approach to the FINAL waypoint

    while t <= opts.tMax:
        # --- waypoint bookkeeping ---
        idP = advanceWaypoint(xi, yi, idP, wayPts, opts)
        xLeader, yLeader = wayPts[idP,0], wayPts[idP,1]
        distLeader = np.hypot(xLeader-xi, yLeader-yi)
        # --- CPA ending at the FINAL waypoint ---------------------------
        # Don't demand physical capture (a deflected captain would orbit
        # until tMax); instead track his closest approach and stop once he
        # is demonstrably receding after a genuine pass - the same logic
        # advanceWaypoint uses, applied to the goal.
        if idP == nP-1:
            if distLeader < goalMin:
                goalMin = distLeader
                cpaIdx = len(log['x']) - 1   # log row of the closest pose
            if goalMin < opts.passWindow and distLeader > goalMin + opts.hysteresis:
                break                        # passed the goal - stop now

        # --- Captain Falcon: rule 4 (waypoint leader) + rules 1-3 ---
        vxi, vyi = v*np.cos(thetai), v*np.sin(thetai)
        vxLeader = (xLeader - xi)*opts.leaderFactor
        vyLeader = (yLeader - yi)*opts.leaderFactor
        if opts.falconSeesFanboids and nFan > 0:
            xNb  = np.concatenate([xDis,  xFan]);  yNb  = np.concatenate([yDis,  yFan])
            vxNb = np.concatenate([vxDis, vxFan]); vyNb = np.concatenate([vyDis, vyFan])
        else:
            xNb, yNb, vxNb, vyNb = xDis, yDis, vxDis, vyDis
        fS_x,fS_y,fA_x,fA_y,fC_x,fC_y = boidsRules(xi,yi,vxi,vyi,xNb,yNb,vxNb,vyNb,opts) \
                                        if xNb.size > 0 else (0,0,0,0,0,0)
        vxTotal = vxLeader + opts.falconBoidsWeight*(fS_x + fA_x + fC_x)
        vyTotal = vyLeader + opts.falconBoidsWeight*(fS_y + fA_y + fC_y)

        vThetai = falconSteering(thetai, vxTotal, vyTotal, opts)
        t += dt
        xi += vxi*dt
        yi += vyi*dt
        thetai += vThetai*dt
        log['t'].append(t); log['x'].append(xi); log['y'].append(yi); log['theta'].append(thetai)

        # --- small boids: synchronous two-pass update -------------------
        nSmall = nDistract + nFan
        if nSmall > 0:
            # snapshot of everyone BEFORE anybody moves this step
            xAll  = np.concatenate([xDis,  xFan]);  yAll  = np.concatenate([yDis,  yFan])
            vxAll = np.concatenate([vxDis, vxFan]); vyAll = np.concatenate([vyDis, vyFan])
            # fanboids also see the captain (his fresh, ACTIVE position)
            xAllF  = np.append(xAll,  xi);            yAllF  = np.append(yAll,  yi)
            vxAllF = np.append(vxAll, v*np.cos(thetai)); vyAllF = np.append(vyAll, v*np.sin(thetai))

            vxDes = np.empty(nSmall); vyDes = np.empty(nSmall)
            for i in range(nSmall):
                isFan = i >= nDistract
                if isFan:
                    fS_x,fS_y,fA_x,fA_y,fC_x,fC_y = boidsRules(xAll[i],yAll[i],vxAll[i],vyAll[i],
                                                               xAllF,yAllF,vxAllF,vyAllF,opts)
                else:
                    fS_x,fS_y,fA_x,fA_y,fC_x,fC_y = boidsRules(xAll[i],yAll[i],vxAll[i],vyAll[i],
                                                               xAll,yAll,vxAll,vyAll,opts)
                kx = fS_x + fA_x + fC_x
                ky = fS_y + fA_y + fC_y
                if isFan:
                    # rule 4 (fan edition): chase the captain's live position
                    kx4, ky4 = fanLeaderRule(xAll[i], yAll[i], xi, yi, opts)
                    kx += kx4
                    ky += ky4
                # kicks are per-second rates -> scale by dt
                vxDes[i] = vxAll[i] + kx*dt
                vyDes[i] = vyAll[i] + ky*dt

            # wall turnback (also just a kick on the desired velocity)
            vxDes[xAll <= xSafeMin] += opts.TF*dt
            vxDes[xAll >= xSafeMax] -= opts.TF*dt
            vyDes[yAll <= ySafeMin] += opts.TF*dt
            vyDes[yAll >= ySafeMax] -= opts.TF*dt

            # Ackermann clamp: bounded curvature, clamped speed, then move
            vxNew, vyNew, omega = ackermannClamp(vxAll, vyAll, vxDes, vyDes, opts)
            xAll = xAll + vxNew*dt
            yAll = yAll + vyNew*dt

            xDis,  xFan  = xAll[:nDistract],  xAll[nDistract:]
            yDis,  yFan  = yAll[:nDistract],  yAll[nDistract:]
            vxDis, vxFan = vxNew[:nDistract], vxNew[nDistract:]
            vyDis, vyFan = vyNew[:nDistract], vyNew[nDistract:]

            if nDistract > 0:
                log['boidX'].append(xDis.copy()); log['boidY'].append(yDis.copy())
            if nFan > 0:
                log['fanX'].append(xFan.copy());   log['fanY'].append(yFan.copy())
                log['fanVx'].append(vxFan.copy()); log['fanVy'].append(vyFan.copy())
                log['fanOmega'].append(omega[nDistract:].copy())

    # Rewind every log (falcon AND boids stay time-aligned) to the CPA
    # moment, whether the loop broke on a pass or expired at tMax. Success
    # is graded by the closest approach: collectRadius is now a grading
    # threshold, not a termination condition.
    if cpaIdx >= 0:
        for k in log:
            log[k] = log[k][:cpaIdx+1]
        reachedGoal = goalMin < opts.collectRadius

    out = {k: np.array(vv) for k, vv in log.items()}
    out['goalMiss'] = float(goalMin)
    out['fanX0'], out['fanY0'] = fanX0, fanY0
    out['reachedGoal'] = reachedGoal
    return out


# =========================================================================
# Metrics
# =========================================================================
def crossTrackError(px, py, pathXY):
    d = np.hypot(px[:,None]-pathXY[None,:,0], py[:,None]-pathXY[None,:,1])
    return d.min(axis=1)

def fanFinalErrors(run):
    """Per-fanboid final-pose error vs the captain's final pose:
    (dx, dy, distance, dtheta). Evaluated JUST AT THE END."""
    dx = run['fanX'][-1] - run['x'][-1]
    dy = run['fanY'][-1] - run['y'][-1]
    psi = np.arctan2(run['fanVy'][-1], run['fanVx'][-1])
    dth = wrapToPi(psi - run['theta'][-1])
    return dx, dy, np.hypot(dx, dy), dth

def fanChi2(run, opts):
    dx, dy, _, dth = fanFinalErrors(run)
    return float(np.sum(dx**2/opts.sigmaPos**2 + dy**2/opts.sigmaPos**2
                        + dth**2/opts.sigmaTh**2))

def flockFinalReport(run, opts):
    """Final positions in terms of Separation / Alignment / Cohesion,
    plus path smoothness."""
    n = run['fanX'].shape[1]
    x, y = run['fanX'][-1], run['fanY'][-1]
    vx, vy = run['fanVx'][-1], run['fanVy'][-1]
    psi = np.arctan2(vy, vx)
    print('--- Fanboid final-state report (n = %d) ---' % n)
    # SEPARATION: nearest-neighbor spacing vs the protected range
    if n > 1:
        D = np.hypot(x[:,None]-x[None,:], y[:,None]-y[None,:])
        np.fill_diagonal(D, np.inf)
        nn = D.min(axis=1)
        print('Separation: nearest-neighbor dist  min %.2f  mean %.2f   (PR = %.2f, %d boid(s) inside PR)'
              % (nn.min(), nn.mean(), opts.PR, int(np.sum(nn < opts.PR))))
    else:
        print('Separation: n = 1, no neighbors to keep distance from')
    # ALIGNMENT: polar order parameter (1 = all noses parallel) + heading vs captain
    u = np.hypot(vx, vy); phi = np.hypot(vx.sum(), vy.sum()) / u.sum()
    dpsiF = wrapToPi(psi - run['theta'][-1])
    print('Alignment : order parameter %.3f (1 = parallel)   mean |heading - captain| %.1f deg'
          % (phi, np.degrees(np.abs(dpsiF)).mean()))
    # COHESION: spread about the flock centroid + centroid offset from captain
    cx, cy = x.mean(), y.mean()
    r = np.hypot(x-cx, y-cy)
    print('Cohesion  : radius about centroid  mean %.2f  max %.2f   centroid is %.2f from the captain'
          % (r.mean(), r.max(), np.hypot(cx-run['x'][-1], cy-run['y'][-1])))
    # SMOOTHNESS: turn-rate stats vs the Ackermann cap
    om = run['fanOmega']                                   # (T, n)
    s  = np.hypot(run['fanVx'], run['fanVy'])              # (T, n)
    cap = s/opts.Rboid
    sat = np.mean(np.abs(om) >= 0.98*cap)
    kappa = np.abs(om)/np.maximum(s, 1e-9)
    print('Smoothness: |omega| mean %.3f  p95 %.3f rad/s (cap s/Rboid, saturated %.0f%% of steps)'
          % (np.abs(om).mean(), np.percentile(np.abs(om), 95), 100*sat))
    print('            curvature mean %.3f 1/u  (hard cap 1/Rboid = %.3f;  captain rides 0 or 1/R = %.3f)'
          % (kappa.mean(), 1/opts.Rboid, 1/opts.R))


# =========================================================================
# Experiment 1: the original A/B (distractors are now turn-limited)
# =========================================================================
runA = runSimulation(opts, wayPts, qi, nDistract=0)
runB = runSimulation(opts, wayPts, qi, nDistract=opts.N)

errB  = crossTrackError(runB['x'], runB['y'], np.column_stack([runA['x'], runA['y']]))
errAr = crossTrackError(runA['x'], runA['y'], refPath[:,:2])

print('(A) reached goal: %s (miss %.3f)   baseline-vs-reference: mean %.4f  max %.4f'
      % (runA['reachedGoal'], runA['goalMiss'], errAr.mean(), errAr.max()))
print('(B) reached goal: %s (miss %.3f)   perturbed-vs-baseline: mean %.4f  RMS %.4f  max %.4f'
      % (runB['reachedGoal'], runB['goalMiss'], errB.mean(), np.sqrt(np.mean(errB**2)), errB.max()))
print('(A) endpoint vs qf: pos %.4f  heading %.4f rad'
      % (np.hypot(runA['x'][-1]-qf[0], runA['y'][-1]-qf[1]), abs(wrapToPi(runA['theta'][-1]-qf[2]))))
print('(B) endpoint vs qf: pos %.4f  heading %.4f rad'
      % (np.hypot(runB['x'][-1]-qf[0], runB['y'][-1]-qf[1]), abs(wrapToPi(runB['theta'][-1]-qf[2]))))

plt.figure(figsize=(10,6))
plt.plot(refPath[:,0], refPath[:,1], 'b-', linewidth=2, label='Reference Dubins path')
plt.plot(runA['x'], runA['y'], 'k--', linewidth=1.5, label='(A) Baseline')
plt.plot(runB['x'], runB['y'], 'g-', linewidth=1.5, label='(B) Perturbed')
plt.plot(wayPts[:,0], wayPts[:,1], 'rs', markersize=7, mfc='none', label='Waypoints')
plt.quiver(qi[0],qi[1],5*np.cos(qi[2]),5*np.sin(qi[2]),color='g',angles='xy',scale_units='xy',scale=1)
plt.quiver(qf[0],qf[1],5*np.cos(qf[2]),5*np.sin(qf[2]),color='r',angles='xy',scale_units='xy',scale=1)
if runB['boidX'].size > 0:
    for b in range(runB['boidX'].shape[1]):
        plt.plot(runB['boidX'][:,b], runB['boidY'][:,b], '-', color='0.8',
                 linewidth=0.6, zorder=1,
                 label='Distractor boids' if b == 0 else None)
    plt.plot(runB['boidX'][-1,:], runB['boidY'][-1,:], '.', color='0.5',
             markersize=6, zorder=1)
plt.axis('equal'); plt.xlabel('x'); plt.ylabel('y'); plt.legend()
plt.title('Captain Falcon vs the (turn-limited) distractors')

# =========================================================================
# Experiment 2: the Fanboids showcase (NO distractors)
# =========================================================================
print()
runC = runSimulation(opts, wayPts, qi, nFan=opts.nFanShowcase)
errCa = crossTrackError(runC['x'], runC['y'], np.column_stack([runA['x'], runA['y']]))
print('(C) captain reached goal: %s   captain-vs-baseline max %.2e  (unperturbed check)'
      % (runC['reachedGoal'], errCa.max()))
flockFinalReport(runC, opts)
print('(C) chi^2 = %.1f   reduced chi^2/(3n) = %.1f' % (fanChi2(runC, opts),
      fanChi2(runC, opts)/(3*opts.nFanShowcase)))

plt.figure(figsize=(10,6))
plt.plot(refPath[:,0], refPath[:,1], 'b-', linewidth=2, label='Reference Dubins path')
plt.plot(runC['x'], runC['y'], 'k--', linewidth=1.5, label='Captain Falcon')
cols = plt.cm.viridis(np.linspace(0.15, 0.9, opts.nFanShowcase))
for b in range(opts.nFanShowcase):
    plt.plot(runC['fanX'][:,b], runC['fanY'][:,b], '-', color=cols[b],
             linewidth=0.8, zorder=1, label='Fanboids' if b == 0 else None)
plt.plot(runC['fanX0'], runC['fanY0'], 'o', color='0.4', mfc='none',
         markersize=6, label='Fan start clumps')
psiEnd = np.arctan2(runC['fanVy'][-1], runC['fanVx'][-1])
plt.quiver(runC['fanX'][-1], runC['fanY'][-1],
           2*np.cos(psiEnd), 2*np.sin(psiEnd),
           color=cols, angles='xy', scale_units='xy', scale=1, width=0.004)
plt.quiver(qf[0],qf[1],5*np.cos(qf[2]),5*np.sin(qf[2]),color='r',angles='xy',scale_units='xy',scale=1)
plt.axis('equal'); plt.xlabel('x'); plt.ylabel('y'); plt.legend()
plt.title('Captain Falcon & the Fanboids - "Show me your moves!"')

# per-follower final-pose errors (x and y, distance, heading) vs unperturbed
dxE, dyE, dE, dthE = fanFinalErrors(runC)
fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
ax[0].scatter(dxE, dyE, c=cols, s=45, zorder=3)
ax[0].add_patch(plt.Circle((0,0), opts.sigmaPos, fill=False, ls='--', color='r',
                           label='capture tol. (sigmaPos)'))
ax[0].axhline(0, color='0.8', lw=0.8); ax[0].axvline(0, color='0.8', lw=0.8)
ax[0].set_xlabel('x error'); ax[0].set_ylabel('y error'); ax[0].axis('equal')
ax[0].legend(); ax[0].set_title('final position error vs captain')
idx = np.arange(opts.nFanShowcase)
ax[1].bar(idx, dE, color=cols, label='|position error|')
ax[1].set_xlabel('fanboid #'); ax[1].set_ylabel('distance error')
ax2 = ax[1].twinx()
ax2.plot(idx, np.degrees(dthE), 'k^', label='heading error')
ax2.set_ylabel('heading error [deg]')
ax[1].set_title('per-fanboid final errors')
ax[1].legend(loc='upper left'); ax2.legend(loc='upper right')
fig.suptitle('Fanboid final-pose errors, evaluated just at the end')

# =========================================================================
# Experiment 3: n vs chi^2 sweep (random clumped starts, fanTrials per n)
# =========================================================================
print('\n--- n vs chi^2 sweep (%d random clumped starts per n) ---' % opts.fanTrials)
sweep = {n: [] for n in opts.fanSweepN}
for n in opts.fanSweepN:
    for k in range(opts.fanTrials):
        r = runSimulation(opts, wayPts, qi, nFan=n, seed=opts.seed + 101*k + 7*n)
        sweep[n].append(fanChi2(r, opts))
    m = np.mean(sweep[n])
    print('n = %2d : chi^2 mean %8.1f  std %7.1f   reduced chi^2/(3n) = %6.1f'
          % (n, m, np.std(sweep[n]), m/(3*n)))

nArr   = np.array(opts.fanSweepN, dtype=float)
chiAvg = np.array([np.mean(sweep[n]) for n in opts.fanSweepN])
chiStd = np.array([np.std(sweep[n])  for n in opts.fanSweepN])

fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
for j, n in enumerate(opts.fanSweepN):
    ax[0].plot([n]*len(sweep[n]), sweep[n], '.', color='0.6', zorder=1)
ax[0].errorbar(nArr, chiAvg, yerr=chiStd, fmt='o-', color='C0', capsize=3, zorder=2)
ax[0].set_xlabel('n fanboids'); ax[0].set_ylabel(r'$\chi^2$')
ax[0].set_title(r'$n$ vs $\chi^2$ (dots = individual trials)')
ax[1].errorbar(nArr, chiAvg/(3*nArr), yerr=chiStd/(3*nArr), fmt='s-', color='C3', capsize=3)
ax[1].axhline(1.0, color='0.6', ls='--', lw=0.8)
ax[1].set_xlabel('n fanboids'); ax[1].set_ylabel(r'$\chi^2 / 3n$')
ax[1].set_title('reduced: per-boid badness (1 = within tolerance)')
fig.suptitle(r'Fanboid tracking fidelity vs flock size, $\chi^2$ just at the end')

plt.show()