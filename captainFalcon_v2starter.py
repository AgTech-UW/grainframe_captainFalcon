import numpy as np
from matplotlib import pyplot as plt

# =========================================================================
# CAPTAIN FALCON & THE FANBOIDS -- SKELETON / STUDY VERSION
# -------------------------------------------------------------------------
# Same harness as the finished captain_falcon.py (that file is your ANSWER
# KEY - don't peek until stuck). All the NEW machinery is stubbed out here
# for you to implement yourself.
#
# WHAT ALREADY WORKS in this file (runs end-to-end from minute one):
#   - CSV loading, config, walls, the falcon (steering + CPA handover)
#   - the full simulation loop wiring: synchronous two-pass small-boid
#     update, fan neighborhoods, logging, all three experiments, all plots
#
# WHAT IS YOURS TO IMPLEMENT (marked with TODO, in suggested order):
#   [3] ackermannClamp  - turn-rate limit (minimum turning radius Rboid)
#   [4] initClumped     - random clumped starts for the fan flock
#   [5] fanFinalErrors  - (dx, dy, dist, dtheta) at the FINAL step only
#   [6] fanChi2         - chi^2 over the final fan poses
#   [7] flockFinalReport- Separation/Alignment/Cohesion + smoothness report
#
# AS SHIPPED (all stubs in place):
#   - fanboids spawn but IGNORE the captain: they just flock and mill
#   - no turn limit: small boids may still snap their velocity anywhere
#   - chi^2 reads 0.0 everywhere, so Fig 4 is a flat line
# Each TODO you land visibly changes a printout or a figure. Watch for it.
# =========================================================================

# -------------------------------------------------------------------------
# Load the MATLAB-exported Dubins data (CSVs live next to this script)
# -------------------------------------------------------------------------
dataTag = '1'
refPath = np.loadtxt('saved_dubins_path_%s.csv' % dataTag, delimiter=',')
wayPts  = np.loadtxt('saved_dubins_waypoints_%s.csv' % dataTag, delimiter=',')
qi      = np.loadtxt('saved_qi_%s.csv' % dataTag, delimiter=',')
qf      = np.loadtxt('saved_qf_%s.csv' % dataTag, delimiter=',')
pR, pV, pDt = np.loadtxt('saved_params_%s.csv' % dataTag, delimiter=',')

refPath = refPath[~np.isnan(refPath).any(axis=1)]
refLen  = np.sum(np.hypot(np.diff(refPath[:,0]), np.diff(refPath[:,1])))

# -------------------------------------------------------------------------
# Config
# -------------------------------------------------------------------------
class gridStruct:
    def __init__(opts, TF, VR, PR, CF, SF, AF, maxS, minS, safetyF):
        opts.TF = TF # Turning factor
        opts.VR = VR # Visual range
        opts.PR = PR # Protected range
        opts.CF = CF # Cohesion factor
        opts.SF = SF # Separation factor
        opts.AF = AF # Alignment factor
        opts.maxSpeed=maxS
        opts.minSpeed=minS
        opts.safetyF=safetyF

opts = gridStruct(2.0, 8.0, 2.5, 0.8, 5.0, 2.0, 2.0, 0.8, 5.0)

opts.R  = float(pR)
opts.v  = float(pV)
opts.dt = float(pDt)
opts.N  = 10
opts.leaderFactor = 1.0
opts.falconBoidsWeight = 0.4
opts.dAng = 0.005
opts.collectRadius = 0.75
opts.passWindow = 5
opts.hysteresis = 2.5*opts.v*opts.dt
opts.seed = 8
opts.tMax = 1.75*refLen/opts.v + 10.0

# Ackermann-style constraint for the small boids (distractors + fanboids):
# minimum turning radius. Heading rate is capped at speed/Rboid, i.e.
# curvature is capped at 1/Rboid. Keep Rboid comfortably below safetyF or
# wall turnbacks start overshooting.
opts.Rboid = 0.5*opts.R

# Fanboid parameters
opts.fanLeaderFactor    = 1.0   # rule-4 pull toward the captain's live position
opts.falconSeesFanboids = False # keep him unperturbed -> clean chi^2 reference
opts.nFanShowcase = 10
opts.fanSweepN    = [1, 2, 3, 5, 7, 12, 20]
opts.fanTrials    = 5
opts.sigmaPos = opts.collectRadius   # chi^2 position scale
opts.sigmaTh  = np.deg2rad(15.0)     # chi^2 heading scale

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
    idVisual    = (D > 1e-9) & (D <= opts.VR) # visual neighbors
    idProtected = (D > 1e-9) & (D <= opts.PR) # protected neighbors
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
# =========================================================================
def fanLeaderRule(xb, yb, xCap, yCap, opts):
    kx = (xCap - xb)*opts.fanLeaderFactor
    ky = (yCap - yb)*opts.fanLeaderFactor
    return kx, ky


# =========================================================================
# Dubins Steering (Captain Falcon only -- provided, unchanged except the
# zero-vector guard: if the desired direction is ~zero, hold course instead
# of dividing 0/0 into a NaN that silently reads as "turn hard right")
# =========================================================================
def falconSteering(thetai, vxTotal, vyTotal, opts):
    nrm = np.hypot(vxTotal, vyTotal)
    if nrm < 1e-12:
        return 0.0
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
    return -opts.v*controlAngle/opts.R


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

    sDes = np.hypot(vxDes, vyDes) # desired speed
    psi = np.arctan2(vy, vx) # current heading
    s = np.clip(sDes, opts.minSpeed, opts.maxSpeed)
    dpsiMax = (s/opts.Rboid)*opts.dt # (holonomic) this IS the heading change constraint
    dpsi = np.clip(wrapToPi(np.arctan2(vyDes, vxDes) - psi), -dpsiMax, +dpsiMax) # heading change this step
    psiNew = psi + dpsi
    return s*np.cos(psiNew), s*np.sin(psiNew), dpsi/opts.dt


# =========================================================================
# Waypoint Handover (provided -- your CPA/bullseye logic, unchanged)
# =========================================================================
def advanceWaypoint(xi, yi, idP, wayPts, opts):
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
# Initializers
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

# =========================================================================
# TODO [4] -- initClumped: "start random, position, and velocities,
#             different clumps"
# -------------------------------------------------------------------------
#   1. pick k = random integer in [1, min(3, n)] clump centers, uniform
#      inside the path's bounding box (same box initUniform uses)
#   2. assign each of the n fanboids to a random clump; position = its
#      clump center + gaussian noise (sigma ~ 2.0 works)
#   3. random heading in [0, 2pi); speed uniform in
#      [opts.minSpeed, opts.maxSpeed]; velocity from (speed, heading)
#   4. return x, y, vx, vy
#
# When it works: the gray start circles in Fig 2 gather into 1-3 tight
# knots instead of confetti, and the sweep is honestly randomized.
# =========================================================================
def initClumped(n, opts):
    k= np.random.randint(1, min(3, n)+1)  # number of clumps
    
    return initUniform(n, opts)


# =========================================================================
# One simulation run (provided -- but READ IT, this is the logic you asked
# about; the numbered comments below are the map)
# =========================================================================
def runSimulation(opts, wayPts, qiPose, nDistract=0, nFan=0, seed=None):
    np.random.seed(opts.seed if seed is None else seed)
    # (i) reset the CPA memory EVERY run. advanceWaypoint keeps its minimum
    #     on the shared opts object; without this reset, run 2 inherits run
    #     1's minimum and can fire a spurious early handover. With ~37 runs
    #     per execution (the sweep) this fix is load-bearing.
    opts.minDist = np.inf

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
        # --- captain: waypoint bookkeeping + his 4 rules + Dubins step ---
        idP = advanceWaypoint(xi, yi, idP, wayPts, opts)
        xLeader, yLeader = wayPts[idP,0], wayPts[idP,1]
        distLeader = np.hypot(xLeader-xi, yLeader-yi)
        # CPA ending at the FINAL waypoint (provided): stop at his closest
        # approach instead of demanding capture, so a deflected captain
        # never orbits the goal until tMax. collectRadius grades success.
        if idP == nP-1:
            if distLeader < goalMin:
                goalMin = distLeader
                cpaIdx = len(log['x']) - 1
            if goalMin < opts.passWindow and distLeader > goalMin + opts.hysteresis:
                break

        vxi, vyi = v*np.cos(thetai), v*np.sin(thetai)
        vxLeader = (xLeader - xi)*opts.leaderFactor
        vyLeader = (yLeader - yi)*opts.leaderFactor
        if opts.falconSeesFanboids and nFan > 0:
            xNb  = np.concatenate([xDis,  xFan]);  yNb  = np.concatenate([yDis,  yFan])
            vxNb = np.concatenate([vxDis, vxFan]); vyNb = np.concatenate([vyDis, vyFan])
        else:   # default: he never notices his fans -> stays unperturbed
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

        # --- small boids: SYNCHRONOUS two-pass update --------------------
        # (ii) pass 1 computes every boid's desired velocity from a frozen
        #      SNAPSHOT of the flock; pass 2 moves everyone at once. The old
        #      demo moved boid i before computing forces on boid i+1
        #      (order-dependent, Gauss-Seidel style); this way the update
        #      doesn't depend on array order.
        nSmall = nDistract + nFan
        if nSmall > 0:
            xAll  = np.concatenate([xDis,  xFan]);  yAll  = np.concatenate([yDis,  yFan])
            vxAll = np.concatenate([vxDis, vxFan]); vyAll = np.concatenate([vyDis, vyFan])
            # (iii) fanboids' neighborhood = all small boids AND the captain
            #       (his fresh, ACTIVE pose from this very step). This lets
            #       alignment velocity-match him (damping!) and separation
            #       hold them off him. Distractors never see him -- so in
            #       the A/B experiment the flock stays an exogenous
            #       disturbance, replayable from the seed.
            xAllF  = np.append(xAll,  xi);               yAllF  = np.append(yAll,  yi)
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
                    kx4, ky4 = fanLeaderRule(xAll[i], yAll[i], xi, yi, opts)   # TODO [2]
                    kx += kx4
                    ky += ky4
                # kicks are per-second rates -> scale by dt
                vxDes[i] = vxAll[i] + kx*dt
                vyDes[i] = vyAll[i] + ky*dt

            # (iv) wall turnback is just another kick on the DESIRED
            #      velocity; the clamp below decides what actually happens
            vxDes[xAll <= xSafeMin] += opts.TF*dt
            vxDes[xAll >= xSafeMax] -= opts.TF*dt
            vyDes[yAll <= ySafeMin] += opts.TF*dt
            vyDes[yAll >= ySafeMax] -= opts.TF*dt

            # (v) desire -> motion, through the car constraint
            vxNew, vyNew, omega = ackermannClamp(vxAll, vyAll, vxDes, vyDes, opts)  # TODO [3]
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

    if cpaIdx >= 0:                # rewind all logs to the CPA moment
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

# =========================================================================
# TODO [5] -- fanFinalErrors: per-fanboid final-pose error, "just at the
#             end" (index -1 of the logs, nothing along the way)
# -------------------------------------------------------------------------
#   dx  = final fan x - captain's final x          (array, one per fan)
#   dy  = same in y
#   d   = hypot(dx, dy)
#   dth = fan heading - captain heading, WRAPPED to [-pi, pi].
#         A fan has no theta state: get its heading from its final
#         velocity, atan2(vy, vx). Captain's is run['theta'][-1].
#   return dx, dy, d, dth
#
# When it works: Fig 3 stops being a pile of zeros at the origin.
# =========================================================================
def fanFinalErrors(run):
    # --- placeholder: everything reads as a perfect landing ---
    n = run['fanX'].shape[1]
    return np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)

# =========================================================================
# TODO [6] -- fanChi2: one number for "how badly did the flock end up"
# -------------------------------------------------------------------------
#   chi^2 = sum over fans of:
#             dx^2/sigmaPos^2 + dy^2/sigmaPos^2 + dth^2/sigmaTh^2
#   using fanFinalErrors. Each fan contributes 3 standardized squared
#   residuals, so chi^2/(3n) ~ 1 would mean "parked within capture
#   tolerance, aligned within ~15 degrees" -- note that with sigmaPos <
#   PR that is unreachable by construction (separation forbids parking
#   that close). Return a float.
#
# When it works: Fig 4 stops being a flat zero line and the sweep table
# prints real numbers.
# =========================================================================
def fanChi2(run, opts):
    # --- placeholder ---
    return 0.0

# =========================================================================
# TODO [7] -- flockFinalReport: the final positions "in terms of
#             Separation Alignment Cohesion", plus path smoothness
# -------------------------------------------------------------------------
# All from the LAST logged step (plus the fanOmega history for smoothness):
#   SEPARATION : pairwise distance matrix of the fans; per-fan nearest-
#                neighbor distance (hint: np.fill_diagonal(D, np.inf));
#                print min, mean, and how many fans sit inside opts.PR.
#   ALIGNMENT  : polar order parameter
#                phi = |sum of velocity vectors| / sum of speeds
#                (1 = all noses parallel, ~0 = disordered); plus mean
#                |heading - captain's heading| in degrees.
#   COHESION   : flock centroid; mean and max distance of fans from it;
#                distance from centroid to the captain.
#   SMOOTHNESS : from run['fanOmega'] (T x n) and speeds |v| (T x n):
#                mean and 95th-percentile |omega|; fraction of steps
#                SATURATED at the cap (|omega| >= 0.98 * s/Rboid); mean
#                curvature |omega|/s vs the hard cap 1/Rboid, and compare
#                with the captain (he rides curvature 0 or 1/R, bang-bang).
#
# When it works: run C prints a real report instead of the TODO line, and
# you can narrate every number in your writeup.
# =========================================================================
def flockFinalReport(run, opts):
    # --- placeholder ---
    print('--- Fanboid final-state report: TODO [7] not implemented yet ---')


# =========================================================================
# Experiment 1: the original A/B
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