import numpy as np
from matplotlib import pyplot as plt

# =========================================================================
# CAPTAIN FALCON & THE FANBOIDS
# =========================================================================

# Load the MATLAB-exported Dubins data (CSVs live next to this script)
dataTag = '1'

refPath = np.loadtxt('saved_dubins_path_%s.csv' % dataTag, delimiter=',')
wayPts  = np.loadtxt('saved_dubins_waypoints_%s.csv' % dataTag, delimiter=',')
qi      = np.loadtxt('saved_qi_%s.csv' % dataTag, delimiter=',')   # start pose (x, y, heading)
qf      = np.loadtxt('saved_qf_%s.csv' % dataTag, delimiter=',')   # goal pose (x, y, heading)
pR, pV, pDt = np.loadtxt('saved_params_%s.csv' % dataTag, delimiter=',')

# Remove any rows of the path that contain NaN.
rowHasNan = np.isnan(refPath).any(axis=1)
refPath = refPath[~rowHasNan] 

# Total arc length of the path
segDx = np.diff(refPath[:, 0])
segDy = np.diff(refPath[:, 1])
segLen = np.hypot(segDx, segDy)
refLen = np.sum(segLen)

# Bounding box of the path (used to spawn boids inside)
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
        opts.VR = VR          # Visual range (how far they see neighbors)
        opts.PR = PR          # Protected range (crash avoidance zone)
        opts.CF = CF          # Cohesion factor (pull to group center)
        opts.SF = SF          # Separation factor (push away from close neighbors)
        opts.AF = AF          # Alignment factor (match group speed/heading)
        opts.maxSpeed = maxS
        opts.minSpeed = minS
        opts.safetyF  = safetyF # Margin from the walls

opts = gridStruct(2.0, 8.0, 2.5, 0.8, 5.0, 2.0, 2.0, 0.8, 5.0)

opts.R  = float(pR)
opts.v  = float(pV)
opts.dt = float(pDt)
opts.N  = 10                  # number of distractor boids
opts.leaderFactor = 1.0
opts.falconBoidsWeight = 0.4
opts.dAng = 0.005
opts.collectRadius = 0.75     # bullseye size for goal
opts.passWindow = 5
opts.hysteresis = 2.5 * opts.v * opts.dt
opts.seed = 8 
opts.tMax = 1.75 * refLen / opts.v + 10.0
opts.maxClumps = 3            # random starting groups for fanboids

# Ackermann constraint: forces boids to steer like cars
opts.Rboid = 0.5 * opts.R

# Fanboid parameters
opts.fanLeaderFactor    = 1.0    # pull toward the captain
opts.falconSeesFanboids = False  # keep captain unperturbed
opts.nFanShowcase = 10
opts.fanSweepN    = [1, 2, 3, 5, 7, 12, 20]
opts.fanTrials    = 5
opts.sigmaPos = 1.0              # pos error capture tolerance
opts.sigmaTh  = 1.0              # heading error capture tolerance

# The arena: the path's bounding box, grown by a padding
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
    # Keeps angles between -pi and pi
    return (a + np.pi) % (2 * np.pi) - np.pi


def boidsRules(xb, yb, vxb, vyb, xAll, yAll, vxAll, vyAll, opts):
    # Vector FROM every other agent TO this boid, and the distances.
    dx = xb - xAll
    dy = yb - yAll
    D = np.sqrt(dx**2 + dy**2)

    idVisual    = (D > 1e-9) & (D <= opts.VR)   # neighbors I can see
    idProtected = (D > 1e-9) & (D <= opts.PR)   # neighbors that are TOO CLOSE

    # 1. Separation: push away from too-close neighbors.
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

        # 3. Cohesion: steer toward the neighbors' center of mass.
        vxCohesion = (xPosAvg - xb) * opts.CF
        vyCohesion = (yPosAvg - yb) * opts.CF

        # 2. Alignment: steer toward the neighbors' average velocity.
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


def fanLeaderRule(xb, yb, xCap, yCap, opts):
    # Extra pull toward the Captain for fanboids
    kx = (xCap - xb) * opts.fanLeaderFactor
    ky = (yCap - yb) * opts.fanLeaderFactor
    return kx, ky


def falconSteering(thetai, vxTotal, vyTotal, opts):
    # Dubins car physics... straight, max-left, or max-right
    nrm = np.hypot(vxTotal, vyTotal)
    if nrm < 1e-12:
        return 0.0 # prevent dividing by zero

    # His current velocity vector.
    vi = (opts.v * np.cos(thetai), opts.v * np.sin(thetai))

    # Angle between current velocity and desired velocity.
    cosAng = np.dot(vi, (vxTotal, vyTotal)) / (opts.v * nrm)
    cosAng = np.clip(cosAng, -1, 1) 
    totalAngle = np.arccos(cosAng)

    # Sign of the z component of the cross product tells LEFT vs RIGHT.
    crossZ = vi[0] * vyTotal - vi[1] * vxTotal

    if totalAngle < opts.dAng:
        controlAngle = 0        # go straight
    elif crossZ > 0:
        controlAngle = -1       # turn left
    else:
        controlAngle = 1        # turn right

    return -opts.v * controlAngle / opts.R


def ackermannClamp(vx, vy, vxDes, vyDes, opts):
    # The car constraint.. limits point-mass turning to a physical turning radius
    
    sDes = np.hypot(vxDes, vyDes)
    psi = np.arctan2(vy, vx)

    # Actual speed: clamped into [minSpeed, maxSpeed].
    s = np.clip(sDes, opts.minSpeed, opts.maxSpeed)

    # Maximum heading change allowed this step.
    dpsiMax = (s / opts.Rboid) * opts.dt

    # How far they want to turn, wrapped and clamped to what the car can do
    psiDes = np.arctan2(vyDes, vxDes)
    dpsi = wrapToPi(psiDes - psi)
    dpsi = np.clip(dpsi, -dpsiMax, +dpsiMax)

    # New heading and resulting velocities
    psiNew = psi + dpsi
    vxNew = s * np.cos(psiNew)
    vyNew = s * np.sin(psiNew)

    omega = dpsi / opts.dt     # heading rate for the final report
    return vxNew, vyNew, omega


def advanceWaypoint(xi, yi, idP, wayPts, opts):
    # Closest Point Approach (CPA) handover logic
    nP = wayPts.shape[0]
    if idP >= nP - 1:
        return idP

    wpX = wayPts[idP, 0]
    wpY = wayPts[idP, 1]
    d = np.hypot(wpX - xi, wpY - yi)

    if not hasattr(opts, 'minDist'):
        opts.minDist = np.inf
    opts.minDist = min(opts.minDist, d)

    closeEnough  = opts.minDist < opts.passWindow
    movingAwayBy = d > (opts.minDist + opts.hysteresis)
    
    # Hand over once he physically passes it and moves away
    if closeEnough and movingAwayBy:
        idP += 1
        opts.minDist = np.inf
        # Skip duplicate waypoints
        while idP < nP - 1 and np.hypot(wayPts[idP, 0] - xi, wayPts[idP, 1] - yi) < 1e-5:
            idP += 1
    return idP


def clampSpeeds(vx, vy, opts):
    # Rescale vector so its length is in [minSpeed, maxSpeed]
    s = np.hypot(vx, vy)
    s = np.maximum(s, 1e-9)
    sNew = np.clip(s, opts.minSpeed, opts.maxSpeed)
    scale = sNew / s
    return vx * scale, vy * scale


def initUniform(n, opts):
    # Distractors: random positions all over the map
    x = np.random.uniform(pathXMin, pathXMax, n)
    y = np.random.uniform(pathYMin, pathYMax, n)

    vx = (np.random.rand(n) - 0.5) * 2 * opts.maxSpeed
    vy = (np.random.rand(n) - 0.5) * 2 * opts.maxSpeed

    vx, vy = clampSpeeds(vx, vy, opts)
    return x, y, vx, vy


def initClumped(n, opts):
    # Fanboids: random clumps with gaussian noise
    k = np.random.randint(1, min(opts.maxClumps, n) + 1)
    
    clumpCentersX = np.random.uniform(pathXMin, pathXMax, k)
    clumpCentersY = np.random.uniform(pathYMin, pathYMax, k)

    clumpAssignments = np.random.randint(0, k, n)
    x = clumpCentersX[clumpAssignments] + np.random.normal(0, 2.0, n)
    y = clumpCentersY[clumpAssignments] + np.random.normal(0, 2.0, n)

    headings = np.random.uniform(0, 2 * np.pi, n)
    speeds = np.random.uniform(opts.minSpeed, opts.maxSpeed, n)
    vx = speeds * np.cos(headings)
    vy = speeds * np.sin(headings)

    return x, y, vx, vy


def runSimulation(opts, wayPts, qiPose, nDistract=0, nFan=0, seed=None):
    if seed is None:
        seed = opts.seed
    np.random.seed(seed)

    # Reset CPA memory every run
    opts.minDist = np.inf

    xDis, yDis, vxDis, vyDis = initUniform(nDistract, opts)
    if nFan > 0:
        xFan, yFan, vxFan, vyFan = initClumped(nFan, opts)
    else:
        empty = np.array([])
        xFan, yFan, vxFan, vyFan = empty, empty, empty, empty

    xi = float(qiPose[0])
    yi = float(qiPose[1])
    thetai = float(qiPose[2])

    idP = 1
    nP = wayPts.shape[0]
    t  = 0.0
    dt = opts.dt
    v  = opts.v

    log = {'t': [], 'x': [], 'y': [], 'theta': [],
           'boidX': [], 'boidY': [],
           'fanX': [], 'fanY': [], 'fanVx': [], 'fanVy': [], 'fanOmega': []}
           
    fanX0 = xFan.copy()
    fanY0 = yFan.copy()
    reachedGoal = False
    goalMin = np.inf
    cpaIdx = -1

    while t <= opts.tMax:

        # ================= CAPTAIN =================

        idP = advanceWaypoint(xi, yi, idP, wayPts, opts)
        xLeader = wayPts[idP, 0]
        yLeader = wayPts[idP, 1]
        distLeader = np.hypot(xLeader - xi, yLeader - yi)

        # Stop at closest approach to the final waypoint
        atFinalWaypoint = (idP == nP - 1)
        if atFinalWaypoint:
            if distLeader < goalMin:
                goalMin = distLeader
                cpaIdx = len(log['x']) - 1 # marks closest point
            passedClosestPoint = (goalMin < opts.passWindow and
                                  distLeader > goalMin + opts.hysteresis)
            if passedClosestPoint:
                break

        vxi = v * np.cos(thetai)
        vyi = v * np.sin(thetai)

        # Pull toward current waypoint
        vxLeader = (xLeader - xi) * opts.leaderFactor
        vyLeader = (yLeader - yi) * opts.leaderFactor

        if opts.falconSeesFanboids and nFan > 0:
            xNb  = np.concatenate([xDis,  xFan])
            yNb  = np.concatenate([yDis,  yFan])
            vxNb = np.concatenate([vxDis, vxFan])
            vyNb = np.concatenate([vyDis, vyFan])
        else:
            xNb, yNb, vxNb, vyNb = xDis, yDis, vxDis, vyDis

        if xNb.size > 0:
            (fS_x, fS_y,
             fA_x, fA_y,
             fC_x, fC_y) = boidsRules(xi, yi, vxi, vyi, xNb, yNb, vxNb, vyNb, opts)
        else:
            fS_x = fS_y = fA_x = fA_y = fC_x = fC_y = 0

        vxTotal = vxLeader + opts.falconBoidsWeight * (fS_x + fA_x + fC_x)
        vyTotal = vyLeader + opts.falconBoidsWeight * (fS_y + fA_y + fC_y)

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
        nSmall = nDistract + nFan
        if nSmall > 0:
            # Combine into master list for the snapshot
            xAll  = np.concatenate([xDis,  xFan]) 
            yAll  = np.concatenate([yDis,  yFan])
            vxAll = np.concatenate([vxDis, vxFan])
            vyAll = np.concatenate([vyDis, vyFan])

            # Fanboids see captain, distractors do not
            xAllF  = np.append(xAll,  xi)
            yAllF  = np.append(yAll,  yi)
            vxAllF = np.append(vxAll, v * np.cos(thetai))
            vyAllF = np.append(vyAll, v * np.sin(thetai))

            # ----- Pass 1: desired velocities (no movement yet) -----
            vxDes = np.empty(nSmall)
            vyDes = np.empty(nSmall)
            
            for i in range(nSmall):
                isFan = i >= nDistract

                if isFan:
                    neighborhood = (xAllF, yAllF, vxAllF, vyAllF)
                else:
                    neighborhood = (xAll, yAll, vxAll, vyAll)

                (fS_x, fS_y,
                 fA_x, fA_y,
                 fC_x, fC_y) = boidsRules(xAll[i], yAll[i], vxAll[i], vyAll[i],
                                          *neighborhood, opts)

                kx = fS_x + fA_x + fC_x
                ky = fS_y + fA_y + fC_y

                if isFan:
                    kx4, ky4 = fanLeaderRule(xAll[i], yAll[i], xi, yi, opts)
                    kx += kx4
                    ky += ky4

                vxDes[i] = vxAll[i] + kx * dt
                vyDes[i] = vyAll[i] + ky * dt

            # Wall turnback kicks
            vxDes[xAll <= xSafeMin] += opts.TF * dt
            vxDes[xAll >= xSafeMax] -= opts.TF * dt
            vyDes[yAll <= ySafeMin] += opts.TF * dt
            vyDes[yAll >= ySafeMax] -= opts.TF * dt

            # ----- Pass 2: desire -> motion through Ackermann constraint -----
            vxNew, vyNew, omega = ackermannClamp(vxAll, vyAll, vxDes, vyDes, opts)
            xAll = xAll + vxNew * dt
            yAll = yAll + vyNew * dt

            # Split back into flocks
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

    # Rewind logs to exact CPA frame
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
    dxToPath = px[:, None] - pathXY[None, :, 0]
    dyToPath = py[:, None] - pathXY[None, :, 1]
    d = np.hypot(dxToPath, dyToPath)
    return d.min(axis=1)

def fanFinalErrors(run):
    dx = run['fanX'][-1] - run['x'][-1]
    dy = run['fanY'][-1] - run['y'][-1]
    d = np.hypot(dx, dy)
    fanTheta = np.arctan2(run['fanVy'][-1], run['fanVx'][-1])
    dth = wrapToPi(fanTheta - run['theta'][-1])
    return dx, dy, d, dth

def fanChi2(run, opts):
    # Split position and heading errors for Adam's request
    dx, dy, _, dth = fanFinalErrors(run)
    
    posTermX = dx**2 / opts.sigmaPos**2
    posTermY = dy**2 / opts.sigmaPos**2
    angTerm  = dth**2 / opts.sigmaTh**2
    
    chi2_pos = float(np.sum(posTermX + posTermY))
    chi2_ang = float(np.sum(angTerm))
    
    return chi2_pos, chi2_ang


def flockFinalReport(run, opts):
    # --- placeholder ---
    print('--- Fanboid final-state report: TODO [7] not implemented yet ---')


# =========================================================================
# Experiment 1: the original A/B
# =========================================================================
runA = runSimulation(opts, wayPts, qi, nDistract=0)
runB = runSimulation(opts, wayPts, qi, nDistract=opts.N)

baselineXY = np.column_stack([runA['x'], runA['y']])
errB  = crossTrackError(runB['x'], runB['y'], baselineXY)
errAr = crossTrackError(runA['x'], runA['y'], refPath[:, :2])

print('(A) reached goal: %s (miss %.3f)   baseline-vs-reference: mean %.4f  max %.4f'
      % (runA['reachedGoal'], runA['goalMiss'], errAr.mean(), errAr.max()))
print('(B) reached goal: %s (miss %.3f)   perturbed-vs-baseline: mean %.4f  RMS %.4f  max %.4f'
      % (runB['reachedGoal'], runB['goalMiss'],
         errB.mean(), np.sqrt(np.mean(errB**2)), errB.max()))

posErrA = np.hypot(runA['x'][-1] - qf[0], runA['y'][-1] - qf[1])
angErrA = abs(wrapToPi(runA['theta'][-1] - qf[2]))
posErrB = np.hypot(runB['x'][-1] - qf[0], runB['y'][-1] - qf[1])
angErrB = abs(wrapToPi(runB['theta'][-1] - qf[2]))

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
# Experiment 2: the Fanboids showcase
# =========================================================================
print()
runC = runSimulation(opts, wayPts, qi, nFan=opts.nFanShowcase)

errCa = crossTrackError(runC['x'], runC['y'], baselineXY)
print('(C) captain reached goal: %s   captain-vs-baseline max %.2e  (unperturbed check)'
      % (runC['reachedGoal'], errCa.max()))

flockFinalReport(runC, opts)
chi2_pos, chi2_ang = fanChi2(runC, opts)
chi2C = chi2_pos + chi2_ang
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
# Experiment 3: n vs chi^2 sweep (Updated to track Pos and Heading separately)
# =========================================================================
print('\n--- n vs chi^2 sweep (%d random clumped starts per n) ---' % opts.fanTrials)

sweepPos = {n: [] for n in opts.fanSweepN}
sweepAng = {n: [] for n in opts.fanSweepN}

for n in opts.fanSweepN:
    for k in range(opts.fanTrials):
        trialSeed = opts.seed + 101*k + 7*n     
        r = runSimulation(opts, wayPts, qi, nFan=n, seed=trialSeed)
        
        # Grab both numbers from the newly updated function
        c2p, c2a = fanChi2(r, opts)
        sweepPos[n].append(c2p)
        sweepAng[n].append(c2a)
        
    mPos = np.mean(sweepPos[n])
    mAng = np.mean(sweepAng[n])
    print('n = %2d : chi^2(pos) mean %8.1f   chi^2(ang) mean %8.1f'
          % (n, mPos, mAng))

nArr = np.array(opts.fanSweepN, dtype=float)

# Calculate stats for position
chiPosAvg = np.array([np.mean(sweepPos[n]) for n in opts.fanSweepN])
chiPosStd = np.array([np.std(sweepPos[n])  for n in opts.fanSweepN])

# Calculate stats for heading
chiAngAvg = np.array([np.mean(sweepAng[n]) for n in opts.fanSweepN])
chiAngStd = np.array([np.std(sweepAng[n])  for n in opts.fanSweepN])

fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))

# Left plot: split position and heading total error
ax[0].errorbar(nArr, chiPosAvg, yerr=chiPosStd, fmt='o-', color='C0', capsize=3, label='Pos chi^2')
ax[0].errorbar(nArr, chiAngAvg, yerr=chiAngStd, fmt='s-', color='C1', capsize=3, label='Heading chi^2')
ax[0].set_xlabel('n fanboids')
ax[0].set_ylabel('chi^2')
ax[0].set_title('N vs chi^2 (dots = individual trials)')
ax[0].legend()

# Right plot: reduced error per boid
# Dividing position by 2N (since it has x and y) and heading by N
ax[1].errorbar(nArr, chiPosAvg/(2*nArr), yerr=chiPosStd/(2*nArr), fmt='o-', color='C0', capsize=3, label='Pos (reduced)')
ax[1].errorbar(nArr, chiAngAvg/(nArr), yerr=chiAngStd/(nArr), fmt='s-', color='C1', capsize=3, label='Heading (reduced)')
ax[1].axhline(1.0, color='0.6', ls='--', lw=0.8)
ax[1].set_xlabel('n fanboids')
ax[1].set_ylabel('reduced chi^2')
ax[1].set_title('Reduced: per-boid badness (1 = within tolerance)')
ax[1].legend()

plt.show()
