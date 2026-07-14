import numpy as np
from matplotlib import pyplot as plt

# =========================================================================
# CAPTAIN FALCON & THE FANBOIDS -- SKELETON / STUDY VERSION (readable edit)
# -------------------------------------------------------------------------
# Same program as before, rewritten so that (almost) every line does ONE
# thing. Behavior is unchanged: same random-number call order, same math,
# same TODO stubs for you to fill in.
#
# YOUR TODOs, in suggested order:
#   [7] flockFinalReport - Separation/Alignment/Cohesion + smoothness
# =========================================================================


# -------------------------------------------------------------------------
# Load the MATLAB-exported Dubins data (CSVs live next to this script)
# -------------------------------------------------------------------------
dataTag = '1'

refPath = np.loadtxt('saved_dubins_path_%s.csv' % dataTag, delimiter=',')
wayPts  = np.loadtxt('saved_dubins_waypoints_%s.csv' % dataTag, delimiter=',')
qi      = np.loadtxt('saved_qi_%s.csv' % dataTag, delimiter=',')   # start pose (x, y, heading)
qf      = np.loadtxt('saved_qf_%s.csv' % dataTag, delimiter=',')   # goal  pose (x, y, heading)
pR, pV, pDt = np.loadtxt('saved_params_%s.csv' % dataTag, delimiter=',')
# pR = captain's turning radius, pV = his speed, pDt = timestep

# Remove any rows of the path that contain NaN.
rowHasNan = np.isnan(refPath).any(axis=1)   # True for each row with a NaN in it
refPath = refPath[~rowHasNan]               # keep only the clean rows

# Total arc length of the path: sum of the lengths of all its segments.
segDx = np.diff(refPath[:, 0])              # x step between consecutive points
segDy = np.diff(refPath[:, 1])              # y step between consecutive points
segLen = np.hypot(segDx, segDy)             # length of each segment
refLen = np.sum(segLen)

# Bounding box of the path (used by the initializers). "lo/hi" of each axis.
pathXMin = refPath[:, 0].min()
pathXMax = refPath[:, 0].max()
pathYMin = refPath[:, 1].min()
pathYMax = refPath[:, 1].max()


# -------------------------------------------------------------------------
# Config
# -------------------------------------------------------------------------
class gridStruct:
    def __init__(opts, TF, VR, PR, CF, SF, AF, maxS, minS, safetyF):
        opts.TF = TF          # Turning factor (wall push strength)
        opts.VR = VR          # Visual range
        opts.PR = PR          # Protected range
        opts.CF = CF          # Cohesion factor
        opts.SF = SF          # Separation factor
        opts.AF = AF          # Alignment factor
        opts.maxSpeed = maxS
        opts.minSpeed = minS
        opts.safetyF  = safetyF   # margin from the walls

opts = gridStruct(2.0, 8.0, 2.5, 0.8, 5.0, 2.0, 2.0, 0.8, 5.0)

opts.R  = float(pR)
opts.v  = float(pV)
opts.dt = float(pDt)
opts.N  = 10                    # number of distractor boids in run B
opts.leaderFactor = 1.0
opts.falconBoidsWeight = 0.4
opts.dAng = 0.005
opts.collectRadius = 0.75
opts.passWindow = 5
opts.hysteresis = 2.5 * opts.v * opts.dt
opts.seed = 8 # random seed for reproducibility
opts.tMax = 1.75 * refLen / opts.v + 10.0
opts.maxClumps = 3

# Ackermann-style constraint for the small boids (distractors + fanboids):
# minimum turning radius. Heading rate is capped at speed/Rboid, i.e.
# curvature is capped at 1/Rboid. Keep Rboid comfortably below safetyF or
# wall turnbacks start overshooting.
opts.Rboid = 0.5 * opts.R

# Fanboid parameters
opts.fanLeaderFactor    = 1.0    # rule-4 pull toward the captain's live position
opts.falconSeesFanboids = False  # keep him unperturbed -> clean chi^2 reference
opts.nFanShowcase = 10
opts.fanSweepN    = [1, 2, 3, 5, 7, 12, 20]
opts.fanTrials    = 5
#opts.sigmaPos = opts.collectRadius     # chi^2 position scale
#opts.sigmaTh  = np.deg2rad(15.0)       # chi^2 heading scale
opts.sigmaPos = 1.0
opts.sigmaTh  = 1.0

# The arena: the path's bounding box, grown by a padding, must also contain
# the start and goal points. Walls sit at xF/yF; the "safe" box is inset by
# safetyF -- boids get pushed back once they cross the safe box.
pad = 2.0 * opts.R
xFMin = min(pathXMin, qi[0], qf[0]) - pad
xFMax = max(pathXMax, qi[0], qf[0]) + pad
yFMin = min(pathYMin, qi[1], qf[1]) - pad
yFMax = max(pathYMax, qi[1], qf[1]) + pad

xSafeMin = xFMin + opts.safetyF
xSafeMax = xFMax - opts.safetyF
ySafeMin = yFMin + opts.safetyF
ySafeMax = yFMax - opts.safetyF


def wrapToPi(a):
    """Wrap an angle (or array of angles) into [-pi, pi)."""
    return (a + np.pi) % (2 * np.pi) - np.pi


# =========================================================================
# The Boids Rules
# -------------------------------------------------------------------------
# Computes the three classic boids "kicks" for ONE boid at (xb, yb) with
# velocity (vxb, vyb), given the positions/velocities of ALL agents it can
# potentially see. Returns six numbers: the x and y components of the
# Separation, Alignment, and Cohesion kicks.
# =========================================================================
def boidsRules(xb, yb, vxb, vyb, xAll, yAll, vxAll, vyAll, opts):
    # Vector FROM every other agent TO this boid, and the distances.
    dx = xb - xAll
    dy = yb - yAll
    D = np.sqrt(dx**2 + dy**2)

    # D > 0 excludes SELF, so agents may safely appear in their own
    # neighbor arrays (the fanboids' arrays include themselves + falcon).
    idVisual    = (D > 1e-9) & (D <= opts.VR)   # neighbors I can see
    idProtected = (D > 1e-9) & (D <= opts.PR)   # neighbors that are TOO CLOSE

    # 1 Separation: push away from too-close neighbors. dx/dy already point
    # away from them, so summing them gives the net "get away" direction.
    closeDX = np.sum(dx[idProtected])
    closeDY = np.sum(dy[idProtected])
    vxSeparation = closeDX * opts.SF
    vySeparation = closeDY * opts.SF

    neighboringBoids = np.count_nonzero(idVisual)
    if neighboringBoids > 0:
        # Averages over the visible neighbors.
        xPosAvg = np.mean(xAll[idVisual])
        yPosAvg = np.mean(yAll[idVisual])
        xVelAvg = np.mean(vxAll[idVisual])
        yVelAvg = np.mean(vyAll[idVisual])

        # 3 Cohesion: steer toward the neighbors' center of mass.
        vxCohesion = (xPosAvg - xb) * opts.CF
        vyCohesion = (yPosAvg - yb) * opts.CF

        # 2 Alignment: steer toward the neighbors' average velocity.
        vxAlignment = (xVelAvg - vxb) * opts.AF
        vyAlignment = (yVelAvg - vyb) * opts.AF
    else:
        # Nobody in sight: no cohesion, no alignment.
        vxCohesion = 0.0
        vyCohesion = 0.0
        vxAlignment = 0.0
        vyAlignment = 0.0

    return (float(vxSeparation), float(vySeparation),
            float(vxAlignment),  float(vyAlignment),
            float(vxCohesion),   float(vyCohesion))


# =========================================================================
# Rule 4, fan edition: chase the captain's LIVE position
# =========================================================================
def fanLeaderRule(xb, yb, xCap, yCap, opts):
    kx = (xCap - xb) * opts.fanLeaderFactor
    ky = (yCap - yb) * opts.fanLeaderFactor
    return kx, ky


# =========================================================================
# Dubins Steering (Captain Falcon only -- provided)
# -------------------------------------------------------------------------
# The captain is a Dubins car: constant speed, and his only control is
# "turn left at max rate / go straight / turn right at max rate".
# This picks one of those three based on which side the desired velocity
# (vxTotal, vyTotal) lies on. Returns his heading rate (rad/s).
# =========================================================================
def falconSteering(thetai, vxTotal, vyTotal, opts):
    nrm = np.hypot(vxTotal, vyTotal)
    if nrm < 1e-12:
        # Desired direction is ~zero: hold course instead of dividing 0/0
        # into a NaN that silently reads as "turn hard right".
        return 0.0

    # His current velocity vector.
    vi = (opts.v * np.cos(thetai), opts.v * np.sin(thetai))

    # Angle between current velocity and desired velocity.
    cosAng = np.dot(vi, (vxTotal, vyTotal)) / (opts.v * nrm)
    cosAng = np.clip(cosAng, -1, 1)      # guard against rounding past +-1
    totalAngle = np.arccos(cosAng)

    # Sign of the z component of the cross product tells LEFT vs RIGHT.
    crossZ = vi[0] * vyTotal - vi[1] * vxTotal

    if totalAngle < opts.dAng:
        controlAngle = 0        # close enough: go straight
    elif crossZ > 0:
        controlAngle = -1       # desired direction is to the left
    else:
        controlAngle = 1        # desired direction is to the right

    return -opts.v * controlAngle / opts.R


# =========================================================================
# Ackermann-style update for the small boids (distractors + fanboids)
# -------------------------------------------------------------------------
#    Point boids used to teleport their velocity vector to wherever the
#    rule kicks pointed - a 180 in one dt was legal. A car cannot do that:
#    heading changes only at rate |omega| <= speed/Rboid, and since speed is
#    clamped >= minSpeed the boid must roll forward to turn - no spinning on
#    the spot. The kicks now shape a DESIRED velocity; this clamp limits how
#    far the actual velocity may swing toward it in one step.
#    Vectorized over all boids. Returns (vxNew, vyNew, omega).
# =========================================================================
def ackermannClamp(vx, vy, vxDes, vyDes, opts):
    # Desired speed = length of the desired velocity vector.
    sDes = np.hypot(vxDes, vyDes)

    # Current heading of each boid.
    psi = np.arctan2(vy, vx)

    # Actual speed: the desired speed, clamped into [minSpeed, maxSpeed].
    s = np.clip(sDes, opts.minSpeed, opts.maxSpeed)

    # Maximum heading change allowed this step. THIS is the car constraint:
    # heading rate <= speed / turning radius, times dt.
    dpsiMax = (s / opts.Rboid) * opts.dt

    # How far the desired heading is from the current heading, wrapped to
    # [-pi, pi), then clamped to what the car can actually do this step.
    psiDes = np.arctan2(vyDes, vxDes)
    dpsi = wrapToPi(psiDes - psi)
    dpsi = np.clip(dpsi, -dpsiMax, +dpsiMax)

    # New heading and the resulting velocity components.
    psiNew = psi + dpsi
    vxNew = s * np.cos(psiNew)
    vyNew = s * np.sin(psiNew)

    omega = dpsi / opts.dt     # realized heading rate, for the smoothness report
    return vxNew, vyNew, omega


# =========================================================================
# Waypoint Handover (provided -- your CPA/bullseye logic, unchanged)
# -------------------------------------------------------------------------
# Advance to the next waypoint when the captain has PASSED the current one:
# we watch his distance to it, remember the minimum (closest approach), and
# hand over once he is clearly moving away again (min + hysteresis).
# =========================================================================
def advanceWaypoint(xi, yi, idP, wayPts, opts):
    nP = wayPts.shape[0]
    if idP >= nP - 1:
        return idP                     # already at the last waypoint

    wpX = wayPts[idP, 0]
    wpY = wayPts[idP, 1]
    d = np.hypot(wpX - xi, wpY - yi)   # distance to the current waypoint

    if not hasattr(opts, 'minDist'):
        opts.minDist = np.inf
    opts.minDist = min(opts.minDist, d)

    closeEnough  = opts.minDist < opts.passWindow
    movingAwayBy = d > (opts.minDist + opts.hysteresis)
    if closeEnough and movingAwayBy:
        idP += 1
        opts.minDist = np.inf
        # Skip any duplicate waypoints sitting on top of this position.
        while idP < nP - 1 and np.hypot(wayPts[idP, 0] - xi, wayPts[idP, 1] - yi) < 1e-5:
            idP += 1
    return idP


# =========================================================================
# Initializers
# =========================================================================
def clampSpeeds(vx, vy, opts):
    """Rescale each velocity vector so its LENGTH lies in
    [minSpeed, maxSpeed], keeping its direction."""
    s = np.hypot(vx, vy)               # current speed of each boid
    s = np.maximum(s, 1e-9)            # avoid dividing by zero
    sNew = np.clip(s, opts.minSpeed, opts.maxSpeed)
    scale = sNew / s
    return vx * scale, vy * scale


def initUniform(n, opts):
    """Distractors: positions uniform over the path's bounding box,
    velocities random with clamped speed."""
    # Random positions inside the bounding box.
    # uniform(lo, hi, n) = n random numbers between lo and hi.
    x = np.random.uniform(pathXMin, pathXMax, n)
    y = np.random.uniform(pathYMin, pathYMax, n)

    # Random velocity components in [-maxSpeed, +maxSpeed):
    # rand(n) is [0,1); minus 0.5 centers it; times 2*maxSpeed scales it.
    vx = (np.random.rand(n) - 0.5) * 2 * opts.maxSpeed
    vy = (np.random.rand(n) - 0.5) * 2 * opts.maxSpeed

    # Force each speed into [minSpeed, maxSpeed].
    vx, vy = clampSpeeds(vx, vy, opts)
    return x, y, vx, vy

def initClumped(n, opts):
    """Fanboids: 1-3 random clump centers inside the path bbox, 
    gaussian spread around each, random headings, speeds uniform 
    in [minSpeed, maxSpeed]."""
    # random clump centers inside the path's bounding box.
    k = np.random.randint(1, min(opts.maxClumps, n) + 1)
    # random value between the beginning and end of the path
    clumpCentersX = np.random.uniform(pathXMin, pathXMax, k)
    clumpCentersY = np.random.uniform(pathYMin, pathYMax, k)

    # Each boid picks a random clump; its position is that clump's
    # center plus Gaussian noise (the fuzzy blob).
    clumpAssignments = np.random.randint(0, k, n)
    x = clumpCentersX[clumpAssignments] + np.random.normal(0, 2.0, n)
    y = clumpCentersY[clumpAssignments] + np.random.normal(0, 2.0, n)

    # Random heading and speed -> velocity components.
    headings = np.random.uniform(0, 2 * np.pi, n)
    speeds = np.random.uniform(opts.minSpeed, opts.maxSpeed, n)
    vx = speeds * np.cos(headings)
    vy = speeds * np.sin(headings)

    return x, y, vx, vy


# =========================================================================
# One simulation run
# -------------------------------------------------------------------------
# The main loop. Each timestep:
#   captain : waypoint bookkeeping -> his 4 rules -> Dubins step
#   boids   : pass 1 computes desired velocities from a frozen snapshot,
#             pass 2 clamps them (car constraint) and moves everyone
# Logs everything; ends at the captain's closest approach to the goal.
# =========================================================================
def runSimulation(opts, wayPts, qiPose, nDistract=0, nFan=0, seed=None):
    # Initialize everything for the simulation
    if seed is None:
        seed = opts.seed
    np.random.seed(seed)

    # (i) Reset the CPA memory EVERY run. advanceWaypoint keeps its minimum
    #     on the shared opts object; without this reset, run 2 inherits run
    #     1's minimum and can fire a spurious early handover.
    opts.minDist = np.inf

    # Spawn the two flocks.
    xDis, yDis, vxDis, vyDis = initUniform(nDistract, opts)
    if nFan > 0:
        xFan, yFan, vxFan, vyFan = initClumped(nFan, opts)
    else:
        empty = np.array([])
        xFan, yFan, vxFan, vyFan = empty, empty, empty, empty

    # Captain's state: position + heading.
    xi = float(qiPose[0])
    yi = float(qiPose[1])
    thetai = float(qiPose[2])

    idP = 1                     # index of his current target waypoint
    nP = wayPts.shape[0]
    t  = 0.0
    dt = opts.dt
    v  = opts.v

    log = {'t': [], 'x': [], 'y': [], 'theta': [],
           'boidX': [], 'boidY': [],
           'fanX': [], 'fanY': [], 'fanVx': [], 'fanVy': [], 'fanOmega': []}
    fanX0 = xFan.copy()         # remember the start positions for Fig 2
    fanY0 = yFan.copy()
    reachedGoal = False
    goalMin = np.inf            # closest approach to the FINAL waypoint...
    cpaIdx = -1                 # ...and the log index where it happened

    # Start the main simulation loop
    while t <= opts.tMax:

        # ================= CAPTAIN =================

        # Which waypoint is he chasing right now?
        idP = advanceWaypoint(xi, yi, idP, wayPts, opts)
        xLeader = wayPts[idP, 0]
        yLeader = wayPts[idP, 1]
        distLeader = np.hypot(xLeader - xi, yLeader - yi)

        # Closest Point Approach ending at the FINAL waypoint: stop at his closest approach
        # instead of demanding capture, so a deflected captain never orbits
        # the goal until tMax. collectRadius grades success afterwards.
        atFinalWaypoint = (idP == nP - 1)
        if atFinalWaypoint:
            if distLeader < goalMin:
                goalMin = distLeader
                cpaIdx = len(log['x']) - 1 # marks closest point
            passedClosestPoint = (goalMin < opts.passWindow and
                                  distLeader > goalMin + opts.hysteresis)
            if passedClosestPoint:
                break

        # His current velocity.
        vxi = v * np.cos(thetai)
        vyi = v * np.sin(thetai)

        # Rule 4 (leader rule): pull toward the current waypoint.
        vxLeader = (xLeader - xi) * opts.leaderFactor
        vyLeader = (yLeader - yi) * opts.leaderFactor

        # Which small boids does the captain react to?
        if opts.falconSeesFanboids and nFan > 0:
            xNb  = np.concatenate([xDis,  xFan])
            yNb  = np.concatenate([yDis,  yFan])
            vxNb = np.concatenate([vxDis, vxFan])
            vyNb = np.concatenate([vyDis, vyFan])
        else:
            # Default: he never notices his fans -> stays unperturbed.
            xNb, yNb, vxNb, vyNb = xDis, yDis, vxDis, vyDis

        if xNb.size > 0:
            (fS_x, fS_y,
             fA_x, fA_y,
             fC_x, fC_y) = boidsRules(xi, yi, vxi, vyi, xNb, yNb, vxNb, vyNb, opts)
        else:
            fS_x = fS_y = fA_x = fA_y = fC_x = fC_y = 0

        # Total desired velocity = leader pull + weighted boids kicks.
        vxTotal = vxLeader + opts.falconBoidsWeight * (fS_x + fA_x + fC_x)
        vyTotal = vyLeader + opts.falconBoidsWeight * (fS_y + fA_y + fC_y)

        # Dubins step: pick a turn rate, then integrate his pose.
        vThetai = falconSteering(thetai, vxTotal, vyTotal, opts)
        t += dt
        xi += vxi * dt
        yi += vyi * dt
        thetai += vThetai * dt

        log['t'].append(t)
        log['x'].append(xi)
        log['y'].append(yi)
        log['theta'].append(thetai)

        # ================= SMALL BOIDS (synchronous two-pass) =================
        # (ii) Pass 1 computes every boid's desired velocity from a frozen
        #      SNAPSHOT of the flock; pass 2 moves everyone at once.
        nSmall = nDistract + nFan # total number of small boids
        if nSmall > 0:
            # The frozen snapshot: distractors first, then fanboids.
            # Combine the Distractors and the Fanboids into one big master list
            xAll  = np.concatenate([xDis,  xFan]) 
            yAll  = np.concatenate([yDis,  yFan])
            vxAll = np.concatenate([vxDis, vxFan])
            vyAll = np.concatenate([vyDis, vyFan])

            # (iii) Fanboids' neighborhood = all small boids AND the captain
            #       (his fresh, ACTIVE pose from this very step). This lets
            #       alignment velocity-match him (damping!) and separation
            #       hold them off him. Distractors never see him -- so in
            #       the A/B experiment the flock stays an exogenous
            #       disturbance, replayable from the seed.
            # special version of this list just for the Fanboids
            xAllF  = np.append(xAll,  xi)
            yAllF  = np.append(yAll,  yi)
            vxAllF = np.append(vxAll, v * np.cos(thetai))
            vyAllF = np.append(vyAll, v * np.sin(thetai))

            # ----- pass 1: desired velocities -----
            vxDes = np.empty(nSmall)
            vyDes = np.empty(nSmall)
            for i in range(nSmall):
                isFan = i >= nDistract   # fans sit after the distractors

                if isFan:
                    neighborhood = (xAllF, yAllF, vxAllF, vyAllF)  # sees captain (fans only see each other + captain)
                else:
                    neighborhood = (xAll, yAll, vxAll, vyAll)      # doesn't (distractors only see each other)

                (fS_x, fS_y,
                 fA_x, fA_y,
                 fC_x, fC_y) = boidsRules(xAll[i], yAll[i], vxAll[i], vyAll[i],
                                          *neighborhood, opts) # calculate the forces

                kx = fS_x + fA_x + fC_x # total kick in x
                ky = fS_y + fA_y + fC_y # total kick in y

                if isFan: 
                    kx4, ky4 = fanLeaderRule(xAll[i], yAll[i], xi, yi, opts) # fanboids also have a leader rule, and an additional kick
                    kx += kx4
                    ky += ky4

                # Kicks are per-second rates -> scale by dt.
                vxDes[i] = vxAll[i] + kx * dt
                vyDes[i] = vyAll[i] + ky * dt

            # (iv) Wall turnback is just another kick on the DESIRED
            #      velocity; the clamp below decides what actually happens.
            vxDes[xAll <= xSafeMin] += opts.TF * dt   # too far left  -> push right
            vxDes[xAll >= xSafeMax] -= opts.TF * dt   # too far right -> push left
            vyDes[yAll <= ySafeMin] += opts.TF * dt   # too low  -> push up
            vyDes[yAll >= ySafeMax] -= opts.TF * dt   # too high -> push down

            # ----- pass 2: desire -> motion, through the car constraint -----
            vxNew, vyNew, omega = ackermannClamp(vxAll, vyAll, vxDes, vyDes, opts)
            xAll = xAll + vxNew * dt
            yAll = yAll + vyNew * dt

            # Split the combined arrays back into the two flocks.
            xDis,  xFan  = xAll[:nDistract],  xAll[nDistract:]
            yDis,  yFan  = yAll[:nDistract],  yAll[nDistract:]
            vxDis, vxFan = vxNew[:nDistract], vxNew[nDistract:]
            vyDis, vyFan = vyNew[:nDistract], vyNew[nDistract:]

            if nDistract > 0:
                log['boidX'].append(xDis.copy())
                log['boidY'].append(yDis.copy())
            if nFan > 0:
                log['fanX'].append(xFan.copy())
                log['fanY'].append(yFan.copy())
                log['fanVx'].append(vxFan.copy())
                log['fanVy'].append(vyFan.copy())
                log['fanOmega'].append(omega[nDistract:].copy())

    # Rewind all logs to the CPA moment (drop the "moving away" tail).
    if cpaIdx >= 0:
        for key in log:
            log[key] = log[key][:cpaIdx + 1]
        reachedGoal = goalMin < opts.collectRadius

    out = {key: np.array(val) for key, val in log.items()}
    out['goalMiss'] = float(goalMin)
    out['fanX0'] = fanX0
    out['fanY0'] = fanY0
    out['reachedGoal'] = reachedGoal
    return out


# =========================================================================
# Metrics
# =========================================================================
def crossTrackError(px, py, pathXY):
    """For each point of a trajectory, its distance to the NEAREST point
    of a reference path. px, py: (T,) arrays. pathXY: (M, 2) array."""
    # d[i, j] = distance from trajectory point i to path point j.
    dxToPath = px[:, None] - pathXY[None, :, 0]
    dyToPath = py[:, None] - pathXY[None, :, 1]
    d = np.hypot(dxToPath, dyToPath)
    # Minimum over the path points -> one error per trajectory point.
    return d.min(axis=1)

def fanFinalErrors(run):
    dx = run['fanX'][-1] - run['x'][-1]
    dy = run['fanY'][-1] - run['y'][-1]
    d = np.hypot(dx, dy)
    fanTheta = np.arctan2(run['fanVy'][-1], run['fanVx'][-1])
    dth = wrapToPi(fanTheta - run['theta'][-1])
    return dx, dy, d, dth

def fanChi2(run, opts):
    dx, dy, _, dth = fanFinalErrors(run)
    posTermX = dx**2 / opts.sigmaPos**2    # tolerances-squared off, in x
    posTermY = dy**2 / opts.sigmaPos**2    # ... in y
    angTerm  = dth**2 / opts.sigmaTh**2    # ... in heading
    return float(np.sum(posTermX + posTermY + angTerm))



def flockFinalReport(run, opts):
    # --- placeholder ---
    print('--- Fanboid final-state report: TODO [7] not implemented yet ---')


# =========================================================================
# Experiment 1: the original A/B
# -------------------------------------------------------------------------
# Run A: captain alone. Run B: captain + N distractor boids. Compare his
# trajectory in B against his own clean trajectory from A.
# =========================================================================
runA = runSimulation(opts, wayPts, qi, nDistract=0)
runB = runSimulation(opts, wayPts, qi, nDistract=opts.N)

baselineXY = np.column_stack([runA['x'], runA['y']])
errB  = crossTrackError(runB['x'], runB['y'], baselineXY)   # B vs A
errAr = crossTrackError(runA['x'], runA['y'], refPath[:, :2])  # A vs reference

print('(A) reached goal: %s (miss %.3f)   baseline-vs-reference: mean %.4f  max %.4f'
      % (runA['reachedGoal'], runA['goalMiss'], errAr.mean(), errAr.max()))
print('(B) reached goal: %s (miss %.3f)   perturbed-vs-baseline: mean %.4f  RMS %.4f  max %.4f'
      % (runB['reachedGoal'], runB['goalMiss'],
         errB.mean(), np.sqrt(np.mean(errB**2)), errB.max()))

# How far did each run end from the goal pose qf?
posErrA = np.hypot(runA['x'][-1] - qf[0], runA['y'][-1] - qf[1])
angErrA = abs(wrapToPi(runA['theta'][-1] - qf[2]))
posErrB = np.hypot(runB['x'][-1] - qf[0], runB['y'][-1] - qf[1])
angErrB = abs(wrapToPi(runB['theta'][-1] - qf[2]))
print('(A) endpoint vs qf: pos %.4f  heading %.4f rad' % (posErrA, angErrA))
print('(B) endpoint vs qf: pos %.4f  heading %.4f rad' % (posErrB, angErrB))

plt.figure(figsize=(10, 6))
plt.plot(refPath[:, 0], refPath[:, 1], 'b-', linewidth=2, label='Reference Dubins path')
plt.plot(runA['x'], runA['y'], 'k--', linewidth=1.5, label='(A) Baseline')
plt.plot(runB['x'], runB['y'], 'g-', linewidth=1.5, label='(B) Perturbed')
plt.plot(wayPts[:, 0], wayPts[:, 1], 'rs', markersize=7, mfc='none', label='Waypoints')
plt.quiver(qi[0], qi[1], 5*np.cos(qi[2]), 5*np.sin(qi[2]),
           color='g', angles='xy', scale_units='xy', scale=1)
plt.quiver(qf[0], qf[1], 5*np.cos(qf[2]), 5*np.sin(qf[2]),
           color='r', angles='xy', scale_units='xy', scale=1)
if runB['boidX'].size > 0:
    nBoids = runB['boidX'].shape[1]
    for b in range(nBoids):
        plt.plot(runB['boidX'][:, b], runB['boidY'][:, b], '-', color='0.8',
                 linewidth=0.6, zorder=1,
                 label='Distractor boids' if b == 0 else None)
    plt.plot(runB['boidX'][-1, :], runB['boidY'][-1, :], '.', color='0.5',
             markersize=6, zorder=1)
plt.axis('equal')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Captain Falcon vs the (turn-limited) distractors')


# =========================================================================
# Experiment 2: the Fanboids showcase (NO distractors)
# =========================================================================
print()
runC = runSimulation(opts, wayPts, qi, nFan=opts.nFanShowcase)

# Sanity check: since falconSeesFanboids is False, his path in C should be
# numerically identical to run A.
errCa = crossTrackError(runC['x'], runC['y'], baselineXY)
print('(C) captain reached goal: %s   captain-vs-baseline max %.2e  (unperturbed check)'
      % (runC['reachedGoal'], errCa.max()))

flockFinalReport(runC, opts)
chi2C = fanChi2(runC, opts)
print('(C) chi^2 = %.1f   reduced chi^2/(3n) = %.1f'
      % (chi2C, chi2C / (3 * opts.nFanShowcase)))

plt.figure(figsize=(10, 6))
plt.plot(refPath[:, 0], refPath[:, 1], 'b-', linewidth=2, label='Reference Dubins path')
plt.plot(runC['x'], runC['y'], 'k--', linewidth=1.5, label='Captain Falcon')
cols = plt.cm.viridis(np.linspace(0.15, 0.9, opts.nFanShowcase))
for b in range(opts.nFanShowcase):
    plt.plot(runC['fanX'][:, b], runC['fanY'][:, b], '-', color=cols[b],
             linewidth=0.8, zorder=1, label='Fanboids' if b == 0 else None)
plt.plot(runC['fanX0'], runC['fanY0'], 'o', color='0.4', mfc='none',
         markersize=6, label='Fan start clumps')

# Final heading of each fan, from its final velocity.
psiEnd = np.arctan2(runC['fanVy'][-1], runC['fanVx'][-1])
plt.quiver(runC['fanX'][-1], runC['fanY'][-1],
           2*np.cos(psiEnd), 2*np.sin(psiEnd),
           color=cols, angles='xy', scale_units='xy', scale=1, width=0.004)
plt.quiver(qf[0], qf[1], 5*np.cos(qf[2]), 5*np.sin(qf[2]),
           color='r', angles='xy', scale_units='xy', scale=1)
plt.axis('equal')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Captain Falcon & the Fanboids - "Show me your moves!"')

# Fig 3: final-pose errors.
dxE, dyE, dE, dthE = fanFinalErrors(runC)
fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))

ax[0].scatter(dxE, dyE, c=cols, s=45, zorder=3)
ax[0].add_patch(plt.Circle((0, 0), opts.sigmaPos, fill=False, ls='--', color='r',
                           label='capture tol. (sigmaPos)'))
ax[0].axhline(0, color='0.8', lw=0.8)
ax[0].axvline(0, color='0.8', lw=0.8)
ax[0].set_xlabel('x error')
ax[0].set_ylabel('y error')
ax[0].axis('equal')
ax[0].legend()
ax[0].set_title('final position error vs captain')

idx = np.arange(opts.nFanShowcase)
ax[1].bar(idx, dE, color=cols, label='|position error|')
ax[1].set_xlabel('fanboid #')
ax[1].set_ylabel('distance error')
ax2 = ax[1].twinx()
ax2.plot(idx, np.degrees(dthE), 'k^', label='heading error')
ax2.set_ylabel('heading error [deg]')
ax[1].set_title('per-fanboid final errors')
ax[1].legend(loc='upper left')
ax2.legend(loc='upper right')
fig.suptitle('Fanboid final-pose errors, evaluated just at the end')


# =========================================================================
# Experiment 3: n vs chi^2 sweep (random clumped starts, fanTrials per n)
# =========================================================================
print('\n--- n vs chi^2 sweep (%d random clumped starts per n) ---' % opts.fanTrials)
sweep = {n: [] for n in opts.fanSweepN}
for n in opts.fanSweepN:
    for k in range(opts.fanTrials):
        trialSeed = opts.seed + 101*k + 7*n     # different but reproducible
        r = runSimulation(opts, wayPts, qi, nFan=n, seed=trialSeed)
        sweep[n].append(fanChi2(r, opts))
    m = np.mean(sweep[n])
    print('n = %2d : chi^2 mean %8.1f  std %7.1f   reduced chi^2/(3n) = %6.1f'
          % (n, m, np.std(sweep[n]), m / (3*n)))

nArr   = np.array(opts.fanSweepN, dtype=float)
chiAvg = np.array([np.mean(sweep[n]) for n in opts.fanSweepN])
chiStd = np.array([np.std(sweep[n])  for n in opts.fanSweepN])

fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
for j, n in enumerate(opts.fanSweepN):
    ax[0].plot([n]*len(sweep[n]), sweep[n], '.', color='0.6', zorder=1)
ax[0].errorbar(nArr, chiAvg, yerr=chiStd, fmt='o-', color='C0', capsize=3, zorder=2)
ax[0].set_xlabel('n fanboids')
ax[0].set_ylabel(r'$\chi^2$')
ax[0].set_title(r'$n$ vs $\chi^2$ (dots = individual trials)')

ax[1].errorbar(nArr, chiAvg/(3*nArr), yerr=chiStd/(3*nArr), fmt='s-', color='C3', capsize=3)
ax[1].axhline(1.0, color='0.6', ls='--', lw=0.8)
ax[1].set_xlabel('n fanboids')
ax[1].set_ylabel(r'$\chi^2 / 3n$')
ax[1].set_title('reduced: per-boid badness (1 = within tolerance)')
fig.suptitle(r'Fanboid tracking fidelity vs flock size, $\chi^2$ just at the end')

plt.show()
