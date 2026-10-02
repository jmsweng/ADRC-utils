"""
Plant identification and ADRC bandwidth-suggestion helpers for Betaflight
blackbox logs. Imported by the Fit_plant notebook -- see show_fit_summary()
for the main entry point.
"""

import io, sys, contextlib
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import csd, welch, coherence, TransferFunction, lsim, impulse
from scipy.optimize import least_squares
from glob import glob
from IPython.display import display

# Define axis names in order they're used
AXIS_NAMES = ['roll', 'pitch', 'yaw']

# Stuff to load in data
def load_blackbox_csv(path):
    '''
    Load in blackbox CSV, skips over header
    Returns blackbox as pandas dataframe
    '''
    with open(path, 'r', newline='') as f:
        lines = f.readlines()
    header_idx = None
    for i, line in enumerate(lines):
        if line.startswith('"loopIteration"') or line.startswith('loopIteration'):
            header_idx = i
            break
    if header_idx is None:
        raise ValueError(f"Could not find column header row in {path}")
    df = pd.read_csv(path, skiprows=header_idx)
    df.columns = [c.strip().strip("'\"") for c in df.columns]
    return df

def read_blackbox_header(path, max_lines=400):
    '''
    Read key/value pairs from the blackbox header, before the CSV column row.
    '''
    hdr = {}
    with open(path, 'r', newline='') as fh:
        for i, line in enumerate(fh):
            if i > max_lines or line.startswith('"loopIteration"'):
                break
            parts = line.rstrip('\n').split(',', 1)
            if len(parts) == 2:
                hdr[parts[0].strip().strip('"')] = parts[1].strip().strip('"')
    return hdr

def _num(hdr, key, default=0.0):
    '''
    Wrapper for reading numbers out of blackbox header
    '''
    try:
        return float(str(hdr.get(key, default)).split(',')[0])
    except ValueError:
        return float(default)

def get_axis_signals(df, axis='roll'):
    '''
    Gets signals for specified axis, defaults to roll axis.
    Takes 'roll', 'pitch', or 'yaw' as inputs
    
    Returns t (s), r = setpoint (deg/s), y = gyro (deg/s), u = axisSum (PID units).
    '''
    i = AXIS_NAMES.index(axis)
    t = df['time'].to_numpy(float) * 1e-6
    r = df[f'setpoint[{i}]'].to_numpy(float)
    y = df[f'gyroADC[{i}]'].to_numpy(float)
    u = df[f'axisSum[{i}]'].to_numpy(float) if f'axisSum[{i}]' in df.columns else None
    return t, r, y, u
# Frequency response stuff
def estimate_frf(r, x, fs, nperseg=8192, overlap=0.75):
    '''
    Computes frequency response function and magnitude-squared coherence between some 
    input/reference (r) and output signal (x).
    
    Input signal must be exogenous (not part of a feedback loop) or frequency response estimator
    becomes biased. Frequency response estimator defined as H_1(f) = S_xr(f) / S_rr(f)
    Returns:
        f -> Array of sample frequencies
        S_xr/S_rr -> Frequency response function estimator
        gam -> Magnitude squared coherence
        
    '''
    nov = int(nperseg * overlap) # Computes overlapping samples between eighboring segments
    f, Srr = welch(r, fs, nperseg=nperseg, noverlap=nov, detrend='linear') # Compute PSD
    _, Sxr = csd(r, x, fs, nperseg=nperseg, noverlap=nov, detrend='linear') # Cross spectral density
    _, gam = coherence(r, x, fs, nperseg=nperseg, noverlap=nov) # Compute coherence, values near 1 -> high SNR, values near 0 -> noisy
    return f, Sxr / Srr, gam

def ladrc_controller_frf(w, wc, wo, b0, order=2):
    '''
    Define a LADRC controller frequency response function.
    Control law: u(jw) = C_r(jw)*R(jw) + C_y(jw)*Y(jw)
        R -> Reference/setpoint signal
        Y -> Plant output
        C_r -> setpoint prefilter transfer function
        C_y -> Feedback controler transfer function
    Inputs: w: array of frequencies, wc: Controller bandwidth, wo: Observer bandwidth
            b0: nominal control gain, order: system order (defaults to 2nd order)
            
    order=2 (default): 3-state ESO, u = (wc^2 r - wc^2 z1 - 2 wc z2 - z3)/b0
    order=1          : 2-state ESO, u = (wc (r - z1) - z2)/b0
    '''
    if order == 2: # Second order LADRC
        b1, b2, b3 = 3 * wo, 3 * wo ** 2, wo ** 3
        A = np.array([[-b1, 1, 0],
                      [-(wc ** 2 + b2), -2 * wc, 0],
                      [-b3, 0, 0]])
        B = np.array([[0, b1], [wc ** 2, b2], [0, b3]])
        C = np.array([-wc ** 2 / b0, -2 * wc / b0, -1 / b0])
        D = np.array([wc ** 2 / b0, 0.0])
    elif order == 1:
        b1, b2 = 2 * wo, wo ** 2
        A = np.array([[-b1, 1], [-b2, 0]])
        B = np.array([[0, b1], [0, b2]])
        C = np.array([-wc / b0, -1 / b0])
        D = np.array([wc / b0, 0.0])
    else:
        raise ValueError("order must be 1 or 2")

    n = A.shape[0]
    I = np.eye(n)
    ws = np.maximum(np.asarray(w, dtype=float), 1e-6)
    # Batched over all frequencies at once: the margin sweep calls this a few
    # hundred times, and a Python loop over frequencies dominates the cost.
    M = 1j * ws[:, None, None] * I[None, :, :] - A[None, :, :]
    X = np.linalg.solve(M, np.broadcast_to(B, (len(ws), n, 2)))
    Cr = X[:, :, 0] @ C + D[0]
    Cy = X[:, :, 1] @ C + D[1]
    return Cr, Cy

def recover_plant_frf(r, y, u, fs, wc=None, wo=None, b0=None, order=2,
                      nperseg=8192, method='adrc'):
    '''
    Estimates the open loop plant frequency response function.
    Two methods: 'adrc' and 'controller_free'
    
    r (Exogenous) ───► [ C_r(s) ] ──(+)───────► [ Plant G(s) ] ──┬──► y (Measured)
                                   ▲                           │
                                   │                           │
                                [ C_y(s) ] ◄───────────────────┘
    
    U = C_r*R + C_y*Y    ,    Y = G*U
    Y * (1 - G*C_y) = (G*C_r) * R
    T = Y/R (closed loop transfer function from diagram)
    T = (G*C_r)/(1-G*C_y) 
    G = Y/U (open loop transfer function)

    adrc: Needs wc, wo, and b0. Performs model-based de-embedding
        G = T/(C_r + T * C_y)
        - Estimate closed loop (T) using estimate_frf
        - Estimates Cr and Cy using ladrc_controller_frf)
        - Denom of open loop transfer function (C_r + T * C_y) is calculated
        - Returns recovered plant G = T/denom
        - Conditioning ratio cond = np.abs(denom) / np.abs(C_r)
          Rule of thumb cond > 0.1, useful signal, discard anything below
                  
    controller_free: 
        G = (y/r) / (u/r), needs logged axisSum
        Use it to check the 'adrc' result.
    '''
    f, T, gam_y = estimate_frf(r, y, fs, nperseg)
    w = 2 * np.pi * f
    if method == 'controller_free':
        if u is None:
            raise ValueError("controller_free needs axisSum in the log")
        _, Tur, _ = estimate_frf(r, u, fs, nperseg)
        return f, T / Tur, gam_y, np.ones_like(f)
    Cr, Cy = ladrc_controller_frf(w, wc, wo, b0, order=order)
    denom = Cr + T * Cy
    cond = np.abs(denom) / np.maximum(np.abs(Cr), 1e-30)
    return f, T / denom, gam_y, cond

# Model fitting stuff
def plant_model_frf(w, p):
    '''
    Frequency response function of a second order plant
    Transfer function: G(s) = K * exp(-s*tau) / ((s + a) * (s/wm + 1))
    With a 5th parameter (zero), the yaw form with a lead zero:
                       G(s) = K * (1 + s/zero) * exp(-s*tau) / ((s + a) * (s/wm + 1))
    Used for model fitting
    '''
    K, a, wm, tau = p[:4]
    s = 1j * w
    G = K * np.exp(-s * tau) / ((s + a) * (s / wm + 1))
    if len(p) > 4:
        G = G * (1 + s / p[4])
    return G

def fit_plant_frf(f, G, gam, f_lo=0.8, f_hi=15.0, coh_min=0.85, cond=None,
                  cond_min=0.05, lead=False):
    '''
    Complex least squares of the rate-plant model to G(jw), over the band
    where the setpoint actually excites craft
    f_lo, f_hi: Frequency limits, defaults from 0.8 to 15 Hz
    coh_min: Minimum coherence threshold, default 0.85
    cond, cond_min: Conditioning ratio values, discard values below minimum value

    Returns:
    K, a, wm, tau: Identified physical model parameters
    b0: Nominal control gain, used for ADRC parameter tuning (b0 = K * w_m)
    rms_rel: Fit error
    band, nbins: Mask and count of frequency bins used
    wm_at_bound: Flag to show if wm hit upper bound, actuator is too fast to be resolved
    cov_log: Covariance matrix

    lead=True fits the yaw form with a lead zero (see plant_model_frf). Yaw
    torque is rotor drag (proportional to w^2, so it comes through the motor
    lag) PLUS the reaction torque from spinning the rotors up and down
    (proportional to dw/dt, so it arrives as fast as the motor responds). The
    second term puts a zero in the plant, at the ratio of the two, typically
    a few rad/s. Above that zero the yaw plant is close to K_hf/s -- a FIRST-
    order plant -- and the no-zero model cannot represent it: it bends 'a'
    and 'wm' to fake the flat |G|*w, wm lands on its bound, and the fit error
    is ~4x worse. Also returns b_hf = K*wm/zero, the high-frequency gain
    (the |G|*w plateau), which is the wc-independent yaw gain and the b0 a
    1st-order LADRC would use.
    '''
    # Create mask, filter by frequency band, coherence, remove any NaN or Inf
    band = (f >= f_lo) & (f <= f_hi) & (gam >= coh_min) & np.isfinite(G) 
    if cond is not None: # Remove any values below conditioning threshold if passed
        band &= cond >= cond_min
    if band.sum() < 8: # Make sure there's at least 8 valid frequency bins (arbitrary)
        raise ValueError(
            f"Only {band.sum()} usable frequency bins between {f_lo}-{f_hi} Hz. "
            "Input did not excite the plant enough, or coh_min is too strict."
        )
    w = 2 * np.pi * f[band]
    Gb = G[band]

    def res(lp): # Relative complex error normaized by magnitude
        e = (plant_model_frf(w, np.exp(lp)) - Gb) / np.abs(Gb)
        return np.concatenate([e.real, e.imag])
    # Parameter boundaries
    lo = np.log([1e-2, 1e-3, 10.0, 1e-6])
    hi = np.log([1e7, 50.0, 2000.0, 0.05])
    p0 = np.log([np.abs(Gb[0]) * w[0], 1.0, 60.0, 4e-3]) # Initial guess
    if not lead:
        out = least_squares(res, p0, bounds=(lo, hi)) # Least squares fit
    else:
        # zero bounded to 0.5-500 rad/s; multi-start on it, since the zero
        # and the 'a' pole can trade off from a poor start
        lo, hi = np.r_[lo, np.log(0.5)], np.r_[hi, np.log(500.0)]
        out = min((least_squares(res, np.r_[p0, np.log(z0)], bounds=(lo, hi))
                   for z0 in (3.0, 10.0, 30.0)), key=lambda o: o.cost)
    p = np.exp(out.x)
    rms = float(np.sqrt(2 * out.cost / band.sum())) # RMS values
    zero = float(p[4]) if lead else None
    return dict(K=p[0], a=p[1], wm=p[2], tau=p[3], b0=p[0] * p[2],
                zero=zero, b_hf=(p[0] * p[2] / zero if lead else float('nan')),
                param_keys=(('K', 'a', 'wm', 'tau', 'zero') if lead
                            else ('K', 'a', 'wm', 'tau')),
                rms_rel=rms, band=band, n_bins=int(band.sum()),
                f_valid_hz=float(f[band].max()),
                wm_at_bound=bool(p[2] > 0.99 * 2000.0),
                logp=out.x.copy(), cov_log=_cov_from_jac(out, lo, hi))

def hf_gain(fit):
    '''
    High-frequency gain K_hf = lim |G(jw)|*w of a fit with a lead zero
    (yaw), in deg/s^2 per unit u. For the closed-loop fits K*wm/zero; for the
    eRPM fit k_a*pole/zero. nan for fits without a zero.
    '''
    z = fit.get('zero')
    if not z:
        return float('nan')
    return float(fit['K'] * fit['wm'] / z)

def hf_gain_uncertainty(fit, n=400, seed=0):
    '''(K_hf, 1-sigma) from the fitted parameter covariance.'''
    central = hf_gain(fit)
    if central != central:
        return central, float('nan')
    vals = np.array([hf_gain(d) for d in sample_fits(fit, n=n, seed=seed)])
    vals = vals[np.isfinite(vals)]
    return central, (float(np.std(vals)) if len(vals) > 8 else float('nan'))

def _cov_from_jac(out, lo=None, hi=None, tol=1e-6):
    '''
    Parameter covariance from a least_squares result, in LOG parameter space,
    so the square roots read directly as relative (fractional) uncertainties.

    cov = s2 * inv(J^T J) with s2 the residual variance per degree of freedom.
    '''
    J = np.asarray(out.jac, dtype=float)
    m, npar = J.shape
    dof = max(m - npar, 1)
    s2 = 2.0 * out.cost / dof
    try:
        cov = s2 * np.linalg.pinv(J.T @ J)
    except np.linalg.LinAlgError:
        return np.full((npar, npar), np.nan)

    # A parameter sitting on a bound is not identified by the data -- its
    # Jacobian column is near-degenerate and pinv hands back an enormous
    # variance that is an artifact, not a measurement. Freeze those instead
    # of letting them dominate the error bars.
    if lo is not None and hi is not None:
        x = np.asarray(out.x, dtype=float)
        pinned = (np.abs(x - np.asarray(lo)) < tol) | (np.abs(x - np.asarray(hi)) < tol)
        if pinned.any():
            cov[pinned, :] = 0.0
            cov[:, pinned] = 0.0
    return cov

def sample_fits(fit, n=400, seed=0):
    '''
    Draw n parameter sets from the fitted covariance, as fit dicts ready for
    plant_frf_from_fit / b0_effective. Returns [] if the fit carries no
    covariance.
    '''
    if fit.get('cov_log') is None or fit.get('logp') is None:
        return []
    cov = np.asarray(fit['cov_log'], dtype=float)
    if not np.all(np.isfinite(cov)):
        return []
    rng = np.random.default_rng(seed)
    try:
        draws = rng.multivariate_normal(np.asarray(fit['logp'], dtype=float),
                                        cov, size=n, method='eigh')
    except (np.linalg.LinAlgError, ValueError):
        return []
    keys = fit.get('param_keys', ('K', 'a', 'wm', 'tau'))
    # Keep draws inside a sane envelope; a heavy tail in log space becomes an
    # overflow in linear space and poisons the statistics.
    sd = np.sqrt(np.clip(np.diag(cov), 0, None))
    mu = np.asarray(fit['logp'], dtype=float)
    half = np.minimum(4 * sd, 2.0)   # at most a factor e^2 either way
    draws = np.clip(draws, mu - half, mu + half)
    out = []
    for row in draws:
        d = dict(fit)
        for k, v in zip(keys, np.exp(row)):
            d[k] = float(v)
        out.append(d)
    return out

def b0_uncertainty(fit, wc, n=400, seed=0):
    """
    (b0_eff, 1-sigma) from the fitted parameter covariance. The sigma is
    statistical only -- see the note in _cov_from_jac.
    """
    central = b0_effective(fit, wc)
    draws = sample_fits(fit, n=n, seed=seed)
    if not draws:
        return central, float('nan')
    vals = np.array([b0_effective(d, wc) for d in draws])
    vals = vals[np.isfinite(vals)]
    return central, (float(np.std(vals)) if len(vals) > 8 else float('nan'))

def plant_tf(fit):
    '''
    scipy TransferFunction for the fitted plant (delay not included).
    '''
    K, a, wm = fit['K'], fit['a'], fit['wm']
    z = fit.get('zero')
    num = [K * wm / z, K * wm] if z else [K * wm]
    return TransferFunction(num, [1, a + wm, a * wm])


def plant_time_responses(fit, t_end=0.5, n=800):
    '''
    Returns t, impulse response, step response of the fitted plant.
    '''
    sys_ = plant_tf(fit)
    t = np.linspace(0, t_end, n)
    _, h = impulse((sys_.num, sys_.den), T=t)
    _, ystep, _ = lsim(sys_, U=np.ones_like(t), T=t)
    return t + fit['tau'], h, ystep

# Betaflight QUAD X motor order: M1 rear-right, M2 front-right, M3 rear-left, M4 front-left.
#  4   2
#  3   1
MIXER_QUADX = {
    'roll':  np.array([-1.0, -1.0,  1.0,  1.0]),
    'pitch': np.array([ 1.0, -1.0,  1.0, -1.0]),
    'yaw':   np.array([-1.0,  1.0,  1.0, -1.0]),
}


def rotor_speeds(df, motor_poles=12, erpm_scale=100.0):
    '''
    eRPM columns -> rotor angular speed (rad/s)
    '''
    e = df[[f'eRPM[{i}]' for i in range(4)]].to_numpy(float)
    rpm = e * erpm_scale / (motor_poles / 2.0)
    return rpm * 2 * np.pi / 60.0

def torque_basis(w_rotor, t, axis, mixer=None):
    '''
    Regressors for body torque / inertia, in units of angular acceleration.

    Roll and pitch come from thrust differential, which goes as w^2.
    Yaw is different: rotor drag (w^2) PLUS rotor angular momentum change
    (wdot). 
    
    Rotor Dynamics ──┬──> Differential Thrust (∝ ω²)  ──────> Roll / Pitch Torque Basis
                 │
                 ├──> Blade Profile Drag  (∝ ω²)  ──┬───> Yaw Torque Basis
                 └──> Rotor Inertia Accel (∝ dω/dt) ─┘
    '''
    mix = (mixer or MIXER_QUADX)[axis]
    T2 = (w_rotor ** 2) @ mix
    if axis != 'yaw':
        return [T2], ['w^2']
    Td = np.gradient(w_rotor, t, axis=0) @ mix
    return [T2, Td], ['w^2', 'wdot']

def _actuator_model(w, p, lead=False):
    '''
    Computes complex frequency response of an actuator transfer function across an
    array of frequencies (w)
    p: (k_a -> DC gain, pole -> Cutoff frequency, tau -> Time delay)
    Returns an array of complex numbers describing gain and phase shift across input frequencies
    '''
    if lead:
        k_a, pole, tau, zero = p
        return k_a * (1 + 1j * w / zero) / (1 + 1j * w / pole) * np.exp(-1j * w * tau)
    k_a, pole, tau = p
    return k_a / (1 + 1j * w / pole) * np.exp(-1j * w * tau)

def recover_plant_rpm(df, axis, motor_poles=12, erpm_scale=100.0, mixer=None,
                      nperseg=8192, f_lo=0.8, f_hi=25.0, coh_min=0.7):
    '''
    Identify the plant from the logged rotor speeds instead of inverting the closed loop.

    Splits the plant into two legs, each better conditioned than the setpoint-based inversion:

        u --[actuator: k_a/(1+s/pole)]--> body torque --[rigid body: 1/s]--> gyro

    The rigid-body leg is a set of constants, recovered by complex least squares of
    jw*Y(w) = sum_k c_k * T_k(w)

    Needs bidirectional DShot (dshot_bidir=1) so eRPM is populated.

    erpm_scale: multiplier from the logged eRPM column to rev/min before dividing by pole pairs. 
    This cancels out of b0, but shifts k_a and the rigid-body coefficients. Set it to match your 
    decoder if you want those to be physically meaningful.

    f_lo, f_hi: Frequency limits, defaults from 0.8 to 15 Hz
    coh_min: Minimum coherence threshold, default 0.85
    cond, cond_min: Conditioning ratio values, discard values below minimum value

    Returns dict with K, wm, tau, b0, the rigid-body coefficients, and the per-leg fit residuals.
    '''
    # Get data 
    i = AXIS_NAMES.index(axis)
    t = df['time'].to_numpy(float) * 1e-6
    fs = 1.0 / np.median(np.diff(t))
    u = df[f'axisSum[{i}]'].to_numpy(float)
    y = df[f'gyroADC[{i}]'].to_numpy(float)
    
    w_rotor = rotor_speeds(df, motor_poles, erpm_scale) # eRPM to rad/s
    basis, names = torque_basis(w_rotor, t, axis, mixer) # Generate torque basis regressors

    # --- FRFs of everything with respect to u (u is the instrument) -----
    f, Hy, coh_y = estimate_frf(u, y, fs, nperseg) # FRF and coherence for u -> y
    Hk, coh_k = [], []
    for T in basis:
        fk, Hkk, ck = estimate_frf(u, T, fs, nperseg) # FRF and coherence for u -> each torque regressor
        Hk.append(Hkk)
        coh_k.append(ck)
    Hk = np.array(Hk) # Array of all regressors (n_regressors, n_freqs)
    w = 2 * np.pi * f # Convert frequencies from Hz to rad/s

    band = (f >= f_lo) & (f <= f_hi) & (coh_y >= coh_min) # Create mask, filter by frequency band, coherence, remove any NaN or Inf
    for ck in coh_k:
        band &= ck >= coh_min
    if band.sum() < 8:
        raise ValueError(f"Only {band.sum()} usable bins for {axis}; relax coh_min or f_hi.")

    # --- rigid-body leg: jw*Hy = sum_k c_k * Hk  (real coefficients) ----
    lhs = 1j * w[band] * Hy[band]
    A = Hk[:, band].T
    Ar = np.vstack([A.real, A.imag])
    br = np.concatenate([lhs.real, lhs.imag])
    c, *_ = np.linalg.lstsq(Ar, br, rcond=None)
    resid_rb = float(np.linalg.norm(Ar @ c - br) / np.linalg.norm(br))

    # --- actuator leg: u -> effective torque, in angular-accel units ----
    H_eff = (c[:, None] * Hk).sum(axis=0) # Sum torque FRF

    # Yaw contains angular momentum term with phas elead
    lead = len(basis) > 1
    
    def res(lp): # Normalized residual
        e = (_actuator_model(w[band], np.exp(lp), lead) - H_eff[band]) / np.abs(H_eff[band])
        return np.concatenate([e.real, e.imag])
    # Parameter bounds
    lo = [1e-2, 5.0, 1e-6]
    hi = [1e7, 3000.0, 0.05]
    p0 = [np.abs(H_eff[band]).max(), 40.0, 3e-3] # Initial guess
    if lead:
        lo += [5.0]; hi += [5000.0]; p0 += [200.0] # Adjust bounds for lead model
    lo_l, hi_l = np.log(lo), np.log(hi)
    out = least_squares(res, np.log(p0), bounds=(lo_l, hi_l))
    pars = np.exp(out.x)
    k_a, pole, tau = pars[:3]
    zero = pars[3] if lead else None
    resid_act = float(np.sqrt(2 * out.cost / band.sum()))

    keys = ('K', 'wm', 'tau', 'zero') if lead else ('K', 'wm', 'tau')
    return dict(K=k_a, a=1e-3, wm=pole, tau=tau, b0=k_a * pole,
                rigid_coeffs=dict(zip(names, c)), rms_rigid=resid_rb,
                zero=zero, rms_act=resid_act, n_bins=int(band.sum()), f=f,
                H_eff=H_eff, coh=coh_y, band=band,
                f_valid_hz=float(f[band].max()),
                pole_at_bound=bool(pole > 0.99 * 3000.0),
                logp=out.x.copy(), cov_log=_cov_from_jac(out, lo_l, hi_l),
                param_keys=keys)

def plant_frf_from_fit(fit, w):
    '''
    G(jw) for a fit dict from any of the three recovery paths.
    '''
    s = 1j * np.asarray(w, dtype=float)
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        num = fit['K'] * np.exp(-s * fit['tau'])
        if fit.get('zero'):
            num = num * (1 + s / fit['zero'])
        return num / ((s + fit.get('a', 1e-3)) * (1 + s / fit['wm']))


def b0_effective(fit, wc):
    '''
    b0 evaluated as b0_eff = wc^2 * |G(j*wc)|.
    '''
    return float(wc ** 2 * np.abs(plant_frf_from_fit(fit, [wc])[0]))

# Motor lag
#
# The rate plant on roll/pitch is an integrator behind one first-order motor
# lag plus a small transport delay (ADRC_MATH_REVIEW.md, sections 2 and 2c):
#
#     G(s) = b_acc * exp(-s*d) / ( s * (1 + s*tau) )
#
# tau is the motor lag. Its corner, 1/(2*pi*tau), is ~8 Hz for a 20 ms motor,
# so it can only be told apart from the delay d when the fit band reaches well
# past that corner (the delay's phase keeps growing with frequency, the lag's
# levels off at -90 deg). The b0 fits above stop at f_hi = 15 Hz, which is why
# their 1/wm does not give the lag: the data trades tau against the delay
# along a ridge. A chirp flight stays coherent out to ~60 Hz, but only inside
# the chirp windows -- the rest of the log adds response with no setpoint
# energy behind it and drags the coherence down -- so the lag is fitted on
# those windows only.

BOXCHIRP_BIT = 6   # rc_modes.h: BOXARM=0, BOXANGLE, BOXHORIZON, BOXMAG,
                   # BOXALTHOLD, BOXHEADFREE, BOXCHIRP. Blackbox logs the
                   # rcModeActivationMask as 'flightModeFlags'.

def _filled(df, col):
    '''Column as float array with gaps (partial frames) filled from neighbours.'''
    return df[col].astype(float).ffill().bfill().to_numpy()

def find_excitation_segments(df, min_len_s=3.0, min_rms=5.0, share=0.3):
    '''
    Sample-index windows to identify each axis over, as
    ({axis: [(i0, i1), ...]}, source).

    Uses the chirp windows when the log has them (CHIRP mode bit set in
    flightModeFlags). Each window is assigned to every axis whose setpoint RMS
    in it is at least min_rms deg/s and at least `share` of the largest axis --
    upstream chirps one axis per window, but this also handles simultaneous
    ones. Falls back to the whole log for every axis (source 'whole log') when
    there are no chirp windows; stick moves carry little energy above ~5 Hz,
    so the lag is then poorly separated from the delay.
    '''
    t = _filled(df, 'time') * 1e-6
    fs = 1.0 / np.median(np.diff(t))
    whole = {a: [(0, len(df))] for a in AXIS_NAMES}
    if 'flightModeFlags' not in df.columns:
        return whole, 'whole log'
    fm = df['flightModeFlags']
    if fm.dtype == object:     # some decoders write mode names, not the mask
        on = fm.astype(str).str.upper().str.contains('CHIRP').to_numpy()
    else:
        v = fm.ffill().fillna(0).to_numpy().astype(np.int64)
        on = ((v >> BOXCHIRP_BIT) & 1).astype(bool)
    d = np.diff(np.r_[0, on.astype(int), 0])
    starts, ends = np.where(d == 1)[0], np.where(d == -1)[0]
    segs = {a: [] for a in AXIS_NAMES}
    for i0, i1 in zip(starts, ends):
        if (i1 - i0) / fs < min_len_s:
            continue
        rms = np.array([np.std(_filled(df, f'setpoint[{k}]')[i0:i1]) for k in range(3)])
        for k, a in enumerate(AXIS_NAMES):
            if rms[k] >= min_rms and rms[k] >= share * rms.max():
                segs[a].append((int(i0), int(i1)))
    if not any(segs.values()):
        return whole, 'whole log'
    # An axis that was never chirped falls back to the whole log on its own.
    src = {a: ('chirp' if segs[a] else 'whole log') for a in AXIS_NAMES}
    for a in AXIS_NAMES:
        if not segs[a]:
            segs[a] = whole[a]
    return segs, src

def _pooled_spectra(segs, r, outs, fs, nperseg, overlap=0.75):
    '''
    Welch auto/cross spectra pooled over several windows, weighted by the
    number of averages each contributes. r is the instrument (exogenous
    setpoint); outs is {name: signal}.
    Returns f, Srr, {name: (S_rx, S_xx)}, n_averages.
    '''
    lens = [b - a for a, b in segs]
    nps = int(min(nperseg, 2 ** int(np.floor(np.log2(max(min(lens) / 2, 64))))))
    nov = int(nps * overlap)
    acc, wsum, f = None, 0.0, None
    for a, b in segs:
        if b - a < nps:
            continue
        nav = 1 + (b - a - nps) // (nps - nov)
        f, Srr = welch(r[a:b], fs, nperseg=nps, noverlap=nov, detrend='linear')
        cur = {'_rr': Srr}
        for k, x in outs.items():
            _, Srx = csd(r[a:b], x[a:b], fs, nperseg=nps, noverlap=nov, detrend='linear')
            _, Sxx = welch(x[a:b], fs, nperseg=nps, noverlap=nov, detrend='linear')
            cur[k] = (Srx, Sxx)
        if acc is None:
            acc = {k: (v * nav if k == '_rr' else (v[0] * nav, v[1] * nav))
                   for k, v in cur.items()}
        else:
            for k, v in cur.items():
                acc[k] = (acc[k] + v * nav if k == '_rr'
                          else (acc[k][0] + v[0] * nav, acc[k][1] + v[1] * nav))
        wsum += nav
    if acc is None:
        raise ValueError("no excitation window is long enough for one Welch segment")
    Srr = acc.pop('_rr') / wsum
    return f, Srr, {k: (v[0] / wsum, v[1] / wsum) for k, v in acc.items()}, int(wsum), nps

def _fit_lag(f, H, coh, f_lo, f_hi, coh_min, integrator=True, lead=False):
    '''
    Complex least squares of
        k * (1 + s*Tz)^lead * exp(-s*d) / ( s^integrator * (1 + s*tau) )
    to H over the coherent band. Relative error, weighted by sqrt(coherence).
    Multi-start, because tau and d trade off and a single start can land on
    the wrong side of the ridge.
    '''
    band = (f >= f_lo) & (f <= f_hi) & (coh >= coh_min) & np.isfinite(H) & (np.abs(H) > 0)
    if band.sum() < 8:
        raise ValueError(f"only {band.sum()} coherent bins between {f_lo}-{f_hi} Hz")
    w = 2 * np.pi * f[band]
    Hb = H[band]
    wgt = np.sqrt(coh[band])
    s = 1j * w

    def model(p):
        g = p[0] * np.exp(-s * p[2]) / (1 + s * p[1])
        if integrator:
            g = g / s
        if lead:
            g = g * (1 + s * p[3])
        return g

    def res(lp):
        e = (model(np.exp(lp)) - Hb) / np.abs(Hb) * wgt
        return np.concatenate([e.real, e.imag])

    k0 = float(np.median(np.abs(Hb) * (w if integrator else 1.0)))
    lo = [1e-6, 1e-4, 1e-5] + ([1e-4] if lead else [])
    hi = [1e9, 0.3, 0.05] + ([2.0] if lead else [])
    lo_l, hi_l = np.log(lo), np.log(hi)
    best = None
    for tau0 in (0.005, 0.02, 0.05):
        for d0 in (5e-4, 3e-3):
            for tz0 in ((0.02, 0.15) if lead else (None,)):
                p0 = [k0 * (1 + 2 * np.pi * 8 * tau0), tau0, d0] + ([tz0] if lead else [])
                out = least_squares(res, np.clip(np.log(p0), lo_l + 1e-9, hi_l - 1e-9),
                                    bounds=(lo_l, hi_l))
                if best is None or out.cost < best.cost:
                    best = out
    p = np.exp(best.x)
    cov = _cov_from_jac(best, lo_l, hi_l)
    sd = np.sqrt(np.clip(np.diag(cov), 0, None))   # relative, log space
    return dict(k=float(p[0]), tau=float(p[1]), delay=float(p[2]),
                Tz=(float(p[3]) if lead else None),
                tau_sd=float(p[1] * sd[1]), delay_sd=float(p[2] * sd[2]),
                err=float(np.sqrt(2 * best.cost / band.sum())),
                n_bins=int(band.sum()), f_lo_hz=float(f[band].min()),
                f_hi_hz=float(f[band].max()),
                tau_at_bound=bool(abs(best.x[1] - lo_l[1]) < 1e-6
                                  or abs(best.x[1] - hi_l[1]) < 1e-6),
                band=band)

def estimate_motor_lag(df, axis, nperseg=4096, f_lo=1.5, f_hi=60.0, coh_min=0.8,
                       segments=None, use_rpm=True, motor_poles=12,
                       erpm_scale=100.0, mixer=None):
    '''
    Motor lag tau (seconds) for one axis, with the transport delay fitted
    alongside it so the two are not confused.

    Primary (gyro) path: controller-free plant H = S_ry / S_ru, with the
    setpoint r as the instrument (unbiased in closed loop, and independent of
    the ADRC parameters), fitted to
        roll/pitch: b_acc * exp(-s d) / (s (1 + s tau))
        yaw:        b_acc * (1 + s Tz) * exp(-s d) / (s (1 + s tau))
    Yaw gets the lead zero from rotor reaction torque; without it the yaw fit
    is wrong (ADRC_MATH_REVIEW.md 2c).

    Check (eRPM) path, if bidirectional DShot is logged: the motors' own
    response, u -> mixer-weighted rotor speed, fitted to k exp(-s d)/(1 + s tau)
    with the same instrument and windows. It never sees the airframe, so it is
    independent of the rigid-body model.

    f_hi defaults to 60 Hz: the lag only separates from the delay well past its
    corner (~8 Hz for 20 ms). On a chirp log both paths stay coherent that far.

    segments: {axis: [(i0, i1), ...]} sample windows, or None to use the chirp
    windows from flightModeFlags (whole log if there are none).

    Returns a dict; tau/delay in seconds. b0_lag = b_acc/tau is the ADRC b0
    this plant implies (roll/pitch only -- it has no meaning on yaw).
    '''
    i = AXIS_NAMES.index(axis)
    t = _filled(df, 'time') * 1e-6
    fs = 1.0 / np.median(np.diff(t))
    if segments is None:
        segments, src = find_excitation_segments(df)
        src = src if isinstance(src, str) else src[axis]
    else:
        src = 'user windows'
    segs = segments[axis]

    r = _filled(df, f'setpoint[{i}]')
    y = _filled(df, f'gyroADC[{i}]')
    u = _filled(df, f'axisSum[{i}]')
    outs = {'y': y, 'u': u}
    have_rpm = use_rpm and 'eRPM[0]' in df.columns
    if have_rpm:
        e = df[[f'eRPM[{k}]' for k in range(4)]].astype(float).ffill().bfill()
        w_rotor = rotor_speeds(e, motor_poles, erpm_scale)
        outs['rpm'] = w_rotor @ (mixer or MIXER_QUADX)[axis]

    f, Srr, S, n_avg, nps = _pooled_spectra(segs, r, outs, fs, nperseg)
    Sry, Syy = S['y']
    Sru, Suu = S['u']
    coh_y = np.abs(Sry) ** 2 / np.maximum(Srr * Syy, 1e-300)
    coh_u = np.abs(Sru) ** 2 / np.maximum(Srr * Suu, 1e-300)
    H = Sry / Sru
    coh = np.minimum(coh_y, coh_u)

    lead = (axis == 'yaw')
    fit = _fit_lag(f, H, coh, f_lo, f_hi, coh_min, integrator=True, lead=lead)

    rpm = None
    if have_rpm:
        Srw, Sww = S['rpm']
        coh_w = np.abs(Srw) ** 2 / np.maximum(Srr * Sww, 1e-300)
        try:
            fr = _fit_lag(f, Srw / Sru, np.minimum(coh_w, coh_u), f_lo, f_hi,
                          coh_min, integrator=False, lead=False)
            rpm = dict(tau=fr['tau'], tau_sd=fr['tau_sd'], delay=fr['delay'],
                       err=fr['err'], n_bins=fr['n_bins'], f_hi_hz=fr['f_hi_hz'])
        except ValueError as exc:
            rpm = dict(error=str(exc))

    tau = fit['tau']
    f_corner = 1.0 / (2 * np.pi * tau)
    return dict(tau=tau, tau_sd=fit['tau_sd'], delay=fit['delay'],
                delay_sd=fit['delay_sd'], b_acc=fit['k'],
                b0_lag=(fit['k'] / tau if not lead else float('nan')),
                zero_rad_s=(1.0 / fit['Tz'] if lead else None),
                err=fit['err'], n_bins=fit['n_bins'],
                f_lo_hz=fit['f_lo_hz'], f_hi_hz=fit['f_hi_hz'],
                f_corner_hz=f_corner,
                # The lag is only pinned when the coherent band runs well past
                # its corner; below ~3x the corner tau and the delay trade off.
                resolved=bool(fit['f_hi_hz'] >= 3.0 * f_corner and not fit['tau_at_bound']),
                tau_at_bound=fit['tau_at_bound'],
                source=src, n_windows=len(segs),
                duration_s=float(sum(b - a for a, b in segs) / fs),
                nperseg=nps, rpm=rpm, f=f, H=H, coh=coh, band=fit['band'])

def motor_lag_warnings(lag):
    '''[WARNING: ...] lines for a motor-lag result, as a list of strings.'''
    if lag is None:
        return []
    if lag.get('error'):
        return [f"motor lag not estimated ({lag['error']})"]
    flags = []
    if lag['source'] != 'chirp':
        flags.append("no chirp window for this axis -- motor lag fitted on the "
                     f"{lag['source']}")
    if lag['tau_at_bound']:
        flags.append("motor lag hit its fit bound -- not identified")
    elif not lag['resolved']:
        flags.append(f"motor lag corner ({lag['f_corner_hz']:.0f} Hz) is not well inside "
                     f"the coherent band (to {lag['f_hi_hz']:.0f} Hz) -- lag and delay "
                     "trade off, treat tau as rough")
    rpm = lag.get('rpm')
    if rpm and 'tau' in rpm:
        gap = abs(rpm['tau'] - lag['tau']) / max(0.5 * (rpm['tau'] + lag['tau']), 1e-9)
        if gap > 0.3:
            # The rotor-to-gyro path can only add lag. A shorter gyro value is
            # the gyro fit trading lag against delay; a longer one is lag the
            # controller really sees (gyro filtering, frame, props).
            which = ("trust the eRPM value; a shorter gyro value is fit bias"
                     if lag['tau'] < rpm['tau'] else
                     "trust the gyro value; the extra lag (gyro filtering, frame, "
                     "props) is real and the eRPM can't see it")
            flags.append(f"gyro and eRPM motor lag disagree by {gap*100:.0f}% "
                         f"({lag['tau']*1e3:.1f} vs {rpm['tau']*1e3:.1f} ms) -- {which}")
    return flags

def _axis_dict(v):
    '''Scalar -> same value on every axis; dict -> per-axis; None -> all None.'''
    if isinstance(v, dict):
        return {a: v.get(a) for a in AXIS_NAMES}
    return {a: v for a in AXIS_NAMES}

def wc_label(wc):
    '''"80 rad/s", or "roll 80 / pitch 80 / yaw 60 rad/s" when the axes differ.'''
    d = _axis_dict(wc)
    vals = [d[a] for a in AXIS_NAMES]
    if all(v == vals[0] for v in vals):
        return f"{vals[0]:g} rad/s"
    return " / ".join(f"{a} {d[a]:g}" for a in AXIS_NAMES) + " rad/s"

# Detail columns that show_fit_summary(verbose=False) leaves out
VERBOSE_COLUMNS = ('yaw HF gain', 'band', 'source', 'wc*(tau+d)')

def drop_verbose_columns(table):
    '''table without the VERBOSE_COLUMNS (whichever of them it has).'''
    return table.drop(columns=[c for c in VERBOSE_COLUMNS if c in table.columns])

def motor_lag_table(results, wc=None, wo=None):
    '''
    Per-axis motor lag table (DataFrame) from fit_plant_from_csv_indirect()
    results. With wc it adds wc*(tau+d), and with wo it adds wo*tau: the two
    products ADRC_MATH_REVIEW.md 2-3 tunes against (stability ceiling near
    wc*tau ~ 2). wc/wo may be scalars or per-axis dicts.
    '''
    wc = _axis_dict(wc) if wc is not None else None
    wo = _axis_dict(wo) if wo is not None else None
    rows = []
    for axis in AXIS_NAMES:
        lag = results[axis].get('motor_lag')
        row = {'axis': axis}
        if lag is None or lag.get('error'):
            row['motor lag (gyro)'] = 'n/a'
            row['motor lag (eRPM)'] = 'n/a'
            rows.append(row)
            continue
        # gyro: u -> gyro, what the controller sees. eRPM: u -> rotor speed,
        # the motors alone; the more direct number when the two differ.
        row['motor lag (gyro)'] = f"{lag['tau']*1e3:.1f} +/- {lag['tau_sd']*1e3:.1f} ms"
        rpm = lag.get('rpm')
        row['motor lag (eRPM)'] = (f"{rpm['tau']*1e3:.1f} +/- {rpm['tau_sd']*1e3:.1f} ms"
                                   if rpm and 'tau' in rpm else 'n/a')
        row['delay (gyro)'] = f"{lag['delay']*1e3:.1f} ms"
        row['band'] = f"{lag['f_lo_hz']:.1f}-{lag['f_hi_hz']:.0f} Hz"
        row['fit err'] = f"{lag['err']:.3f}"
        row['source'] = lag['source']
        if wo is not None and wo[axis] is not None:
            row['wo*tau'] = f"{wo[axis] * lag['tau']:.2f}"
        if wc is not None and wc[axis] is not None:
            row['wc*(tau+d)'] = f"{wc[axis] * (lag['tau'] + lag['delay']):.2f}"
        rows.append(row)
    return pd.DataFrame(rows).fillna('n/a')

###################################################################
#####  Stuff below this needs to be reviewed for correctness  #####
###################################################################
#
# The fit band tops out around 25 Hz, so everything rolling off above it is invisible 
# to the identification even though the aircraft flies with it: the gyro lowpasses, 
# and the scheduling plus actuator delay. 
# 
# feedback_path_frf rebuilds the missing lag from the log header and multiplies it 
# into the loop before the sweep. 

FILTER_PT1, FILTER_BIQUAD = 0, 1

def _lowpass_frf(w, f_hz, ftype):
    """PT1 or 2nd-order Butterworth biquad. f_hz <= 0 means the filter is off."""
    if f_hz <= 0:
        return np.ones_like(w, dtype=complex)
    wc = 2 * np.pi * f_hz
    s = 1j * w
    if int(ftype) == FILTER_BIQUAD:
        return 1.0 / (1 + np.sqrt(2) * s / wc + (s / wc) ** 2)
    return 1.0 / (1 + s / wc)


def feedback_path_frf(w, hdr, include_dterm=False, extra_delay_samples=1.5,
                      dyn_lpf_at_min=False, axis=None):
    """
    Lag that sits in the loop but NOT in the identified plant.

    The fit band tops out around 25 Hz, so anything rolling off above it is
    invisible to the identification even though the aircraft flies with it.
    That missing lag is what makes a phase-margin sweep on the bare plant
    optimistic -- and on yaw, whose fitted model is nearly first-order, it is
    the difference between a real wc ceiling and no ceiling at all.

    Included:
      * gyro lowpass 1 and 2 (PT1 or biquad, per gyro_soft_type / _soft2_type)
      * the scheduling + actuator delay, as extra_delay_samples * looptime
      * D-term lowpasses, only if include_dterm=True -- they sit in the PID
        D path, so they do not apply to an LADRC loop where the ESO does the
        differentiating

    NOT included: the dynamic notch and RPM notches. Their phase contribution
    depends on where they are tracking at the time, which is not recoverable
    from the header alone. They add lag near their center frequencies, so
    treat the result as still slightly optimistic.

    dyn_lpf_at_min: gyro_lowpass_dyn_hz gives a [min, max] sweep. Passing True
    evaluates at the min (the worst case, low throttle); default uses the
    static gyro_lowpass_hz, which is what applies when dynamic is disabled.
    """
    H = np.ones_like(w, dtype=complex)

    g1 = _num(hdr, 'gyro_lowpass_hz')
    if dyn_lpf_at_min:
        dyn = str(hdr.get('gyro_lowpass_dyn_hz', '0,0')).split(',')
        try:
            g1 = max(g1, float(dyn[0]))
        except (ValueError, IndexError):
            pass
    H *= _lowpass_frf(w, g1, _num(hdr, 'gyro_soft_type'))
    H *= _lowpass_frf(w, _num(hdr, 'gyro_lowpass2_hz'), _num(hdr, 'gyro_soft2_type'))

    if axis == 'yaw':
        # Betaflight's yaw-only PT1 on the yaw pidSum. It sits in the loop and
        # is far above the fit band, so the identification never sees it.
        H *= _lowpass_frf(w, _num(hdr, 'yaw_lpf_hz'), FILTER_PT1)

    if include_dterm:
        H *= _lowpass_frf(w, _num(hdr, 'dterm_lpf_hz'), _num(hdr, 'dterm_filter_type'))
        H *= _lowpass_frf(w, _num(hdr, 'dterm_lpf2_hz'), _num(hdr, 'dterm_filter2_type'))

    looptime = _num(hdr, 'looptime', 125.0) * 1e-6
    denom = max(_num(hdr, 'pid_process_denom', 1.0), 1.0)
    tau = extra_delay_samples * looptime * denom
    H *= np.exp(-1j * w * tau)

    return H, dict(gyro_lpf1=g1, gyro_lpf2=_num(hdr, 'gyro_lowpass2_hz'),
                   yaw_lpf=(_num(hdr, 'yaw_lpf_hz') if axis == 'yaw' else 0.0),
                   delay_s=tau, include_dterm=bool(include_dterm))
# Stuff to suggest bandwidth for ADRC
def _phase_margin(fit, wc, wo, b0, order=2, w=None, extra=None):
    '''
    Phase margin of the loop closed around the identified plant with an LADRC
    of the given (wc, wo, b0). Returns (pm_deg, crossover_rad_s), or
    (nan, nan) if |L| never reaches 1.

    extra: complex FRF over w for loop dynamics the identification could not
        see (gyro filters, scheduling delay). See filters.feedback_path_frf.
        Leaving it out makes the margin optimistic.
    '''
    if w is None:
        w = np.logspace(-1, 3.3, 1200)
    G = plant_frf_from_fit(fit, w)
    if extra is not None:
        G = G * extra
    _, Cy = ladrc_controller_frf(w, wc, wo, b0, order=order)
    L = -G * Cy
    m = np.abs(L)
    if m.max() < 1.0:
        return float('nan'), float('nan')
    i = int(np.argmin(np.abs(m - 1.0)))
    pm = (180 + np.angle(L[i], deg=True) + 180) % 360 - 180
    return float(pm), float(w[i])


def suggest_bandwidth(r, y, fs, fit, wc_cfg, wo_cfg, b0_cfg, order=2,
                      nperseg=8192, pm_target=45.0, coh_min=0.8,
                      wc_max_search=600.0, hdr=None, include_dterm=False,
                      extra_delay_samples=1.5, n_blocks=6, n_mc=24,
                      axis=None, w_valid=None, extrapolation_factor=1.0):
    '''
    What bandwidth the loop is actually delivering, and what the identified
    plant can support.

      f_3db     : the achieved closed-loop bandwidth, the -3 dB point of
                  T = setpoint -> gyro. Roughly half of wc is normal.
      wc_max    : the largest wc that still leaves pm_target of phase margin
                  against the identified plant, holding wo/wc fixed
      pm_cfg    : phase margin at the wc actually being used

    A wc_cfg above wc_max means you are running with less margin than
    pm_target, which is the usual source of a wobble that tuning b0 alone
    will not fix.
    '''
    def _cl_shape(rr, yy, nps):
        """
        (-3 dB point, peak in dB, frequency of peak) of the closed loop r -> y.

        Three things this does that a bare "first bin under ref/sqrt(2)" does
        not, all of which matter on chirp logs where |T| is spiky:
          - the reference is the median of the low-frequency plateau, not a
            single (noisy) bin, so one bad bin can't shift the whole answer;
          - |T| is median-smoothed over coherent bins before thresholding;
          - the drop has to hold for n_hold consecutive coherent bins, so a
            momentary notch is not mistaken for the bandwidth.
        It returns nan for f_3db when the loop never sustains a drop below
        -3 dB inside the coherent band. That is a real result, not a failure:
        a loop that peaks and then loses coherence has no measurable -3 dB
        point, and peak_db is the number that describes it.
        """
        nan3 = (float('nan'),) * 3
        f, T, coh = estimate_frf(rr, yy, fs, nps)
        ok = (f > 0.5) & (coh > coh_min)
        if ok.sum() <= 8:
            return nan3
        fo, mo = f[ok], np.abs(T[ok])
        k = 5
        ms = np.array([np.median(mo[max(0, i - k // 2):i + k // 2 + 1])
                       for i in range(len(mo))])
        ref = float(np.median(mo[fo <= max(fo[0] * 3.0, 2.0)]))
        if not np.isfinite(ref) or ref <= 0:
            return nan3
        ipk = int(np.argmax(ms))
        peak_db = float(20 * np.log10(ms[ipk] / ref))
        f_pk = float(fo[ipk])

        below = ms < ref / np.sqrt(2)
        n_hold = 3
        for i in range(len(below) - n_hold + 1):
            if below[i:i + n_hold].all():
                return float(fo[i]), peak_db, f_pk
        return float('nan'), peak_db, f_pk

    def _f3db(rr, yy, nps):
        return _cl_shape(rr, yy, nps)[0]

    f3, cl_peak_db, cl_peak_hz = _cl_shape(r, y, nperseg)

    # Spread of f_3db across independent stretches of the flight. This is the
    # honest error bar: it captures how much the answer moves with what the
    # pilot happened to be doing, which dominates the Welch bin width.
    f3_sd = float('nan')
    if n_blocks > 1:
        edges = np.linspace(0, len(r), n_blocks + 1).astype(int)
        blk_n = min(nperseg, max(256, (edges[1] - edges[0]) // 3))
        vals = [_f3db(r[a:b], y[a:b], blk_n) for a, b in zip(edges[:-1], edges[1:])]
        vals = np.array([v for v in vals if np.isfinite(v)])
        if len(vals) >= 3:
            f3_sd = float(np.std(vals, ddof=1))

    w_grid = np.logspace(-1, 3.3, 1200)
    extra, extra_info = (None, None)
    if hdr is not None:
        extra, extra_info = feedback_path_frf(
            w_grid, hdr, include_dterm=include_dterm,
            extra_delay_samples=extra_delay_samples, axis=axis)

    # Highest frequency the plant model is actually supported by data. Beyond
    # it the model is pure extrapolation, so a phase margin computed at a
    # crossover above it is not a measurement of anything. Yaw is the axis
    # that hits this: its rotor-momentum zero flattens the model to a -20
    # dB/dec, -90 deg asymptote, so without this guard the sweep happily
    # places the crossover a decade past the last excited bin.
    if w_valid is None and fit.get('f_valid_hz'):
        w_valid = 2 * np.pi * float(fit['f_valid_hz']) * extrapolation_factor

    def _in_band(wx):
        return w_valid is None or not np.isfinite(wx) or wx <= w_valid

    ratio = wo_cfg / wc_cfg
    pm_cfg, w_cross_cfg = _phase_margin(fit, wc_cfg, wo_cfg, b0_cfg, order,
                                        w_grid, extra)
    cfg_extrapolated = bool(np.isfinite(w_cross_cfg) and not _in_band(w_cross_cfg))

    # Coarse scan for the highest passing wc, then bisect into the gap.
    # Same answer as a flat 400-point sweep for ~15% of the evaluations.
    def _passes(wc):
        pm, wx = _phase_margin(fit, wc, ratio * wc, b0_cfg, order, w_grid, extra)
        ok = (not np.isnan(pm)) and pm >= pm_target and _in_band(wx)
        return ok, wx

    coarse = np.linspace(5.0, wc_max_search, 48)
    wc_max = float('nan'); w_cross_max = float('nan'); idx = -1
    for k, wc in enumerate(coarse):
        ok_k, wx = _passes(wc)
        if ok_k:
            wc_max, w_cross_max, idx = wc, wx, k
    if idx >= 0 and idx + 1 < len(coarse):
        lo_wc, hi_wc = coarse[idx], coarse[idx + 1]
        for _ in range(12):
            mid = 0.5 * (lo_wc + hi_wc)
            ok_m, wx = _passes(mid)
            if ok_m:
                lo_wc, wc_max, w_cross_max = mid, mid, wx
            else:
                hi_wc = mid
    at_bound = bool(wc_max >= 0.99 * wc_max_search)

    # Propagate the plant-fit covariance through the whole sweep.
    wc_max_sd = float('nan'); pm_cfg_sd = float('nan')
    draws = sample_fits(fit, n=n_mc, seed=1) if n_mc else []
    if draws:
        wm_s, pm_s = [], []
        for d in draws:
            pm_d, _ = _phase_margin(d, wc_cfg, wo_cfg, b0_cfg, order, w_grid, extra)
            if np.isfinite(pm_d):
                pm_s.append(pm_d)
            lo_i, hi_i = 5.0, wc_max_search
            for _ in range(14):
                mid = 0.5 * (lo_i + hi_i)
                pm_m, wx_m = _phase_margin(d, mid, ratio * mid, b0_cfg, order, w_grid, extra)
                if np.isfinite(pm_m) and pm_m >= pm_target and _in_band(wx_m):
                    lo_i = mid
                else:
                    hi_i = mid
            wm_s.append(lo_i)
        if len(pm_s) >= 8:
            pm_cfg_sd = float(np.std(pm_s, ddof=1))
        if len(wm_s) >= 8:
            wc_max_sd = float(np.std(wm_s, ddof=1))

    return dict(f_3db_hz=f3, f_3db_sd=f3_sd,
                w_3db=2 * np.pi * f3 if f3 == f3 else float('nan'),
                w_3db_sd=2 * np.pi * f3_sd if f3_sd == f3_sd else float('nan'),
                wc_cfg=wc_cfg, wc_max=wc_max, wc_max_sd=wc_max_sd,
                pm_cfg=pm_cfg, pm_cfg_sd=pm_cfg_sd,
                w_cross_cfg=w_cross_cfg, w_cross_max=w_cross_max,
                pm_target=pm_target, wc_max_at_bound=at_bound,
                cl_peak_db=cl_peak_db, cl_peak_hz=cl_peak_hz,
                w_valid=w_valid, cfg_extrapolated=cfg_extrapolated,
                wc_max_band_limited=bool(np.isfinite(wc_max) and w_valid is not None
                                         and np.isfinite(w_cross_max)
                                         and w_cross_max > 0.95 * w_valid),
                extra_lag=extra_info)

###################################################################
#####  Stuff above this needs to be reviewed for correctness  #####
###################################################################



def controller_consistency(r, y, u, fs, wc, wo, b0, order=2, nperseg=8192,
                           coh_min=0.5, f_lo=1.0, f_hi=15.0):
    """
    How well the assumed LADRC controller explains the logged output.

    The 'adrc' plant recovery never looks at u: it gets G by inverting the
    controller you say was flying, G = T/(Cr + T*Cy). So if the real controller
    had a term this model does not (feedforward is the usual one, since the
    LADRC Cr/Cy pair has no FF path), the recovery inverts the wrong thing and
    returns a wrong G -- silently, and with no bad-looking coherence.

    The logged u gives an independent check the recovery itself cannot do:
    u/r must equal Cr + Cy*(y/r) if the model is right. Returns the median
    magnitude ratio (1.0 is perfect) and median phase error in degrees over
    the coherent band.

    The error is amplified in G by 1/cond, where cond is the sensitivity |S|,
    so an axis with high loop gain turns a small controller mismatch into a
    large plant error. That is why this can look fine on roll and pitch and
    be badly wrong on yaw in the same log.
    """
    if u is None:
        return dict(ratio=float('nan'), phase_deg=float('nan'), n_bins=0)
    f, T, coh = estimate_frf(r, y, fs, nperseg)
    _, Tur, _ = estimate_frf(r, u, fs, nperseg)
    Cr, Cy = ladrc_controller_frf(2 * np.pi * f, wc, wo, b0, order=order)
    pred = Cr + Cy * T
    m = (f >= f_lo) & (f <= f_hi) & (coh >= coh_min) & (np.abs(pred) > 0)
    if m.sum() < 8:
        return dict(ratio=float('nan'), phase_deg=float('nan'), n_bins=int(m.sum()))
    e = Tur[m] / pred[m]
    return dict(ratio=float(np.median(np.abs(e))),
                phase_deg=float(np.median(np.angle(e, deg=True))),
                n_bins=int(m.sum()))


def axis_warnings(fit, bw, bw_fit, b0_values=None, adrc_flight=True, lag=None):
    """
    The [WARNING: ...] lines for one axis, as a list of strings.

    Shared by the per-axis console footer in fit_plant_from_csv_indirect() and
    the summary tables in show_fit_summary(), so both report the same checks
    from the same numbers.

    fit       : the ADRC (closed-loop) fit dict, for wm_at_bound
    bw        : a suggest_bandwidth() result dict, or None when the bandwidth
                analysis was skipped (suggest_wc=False). Then only the checks
                on the b0 fit itself are made: controller consistency,
                cross-method b0 spread, and an unresolved 2nd pole.
    bw_fit    : the fit the sweep was run against (eRPM path when available)
    b0_values : b0_eff from each method, for the cross-method spread check
    adrc_flight : False drops the checks that only mean something when the log
        was actually flown with the given LADRC controller -- the achieved
        bandwidth ratio and the phase margin at the configured wc, both of
        which compare a measured closed loop against controller settings that
        were never in the loop.
    lag       : an estimate_motor_lag() result dict, or None to skip the motor
                lag checks
    """
    f_val = bw_fit.get('f_valid_hz', float('nan')) if bw_fit else float('nan')
    flags = []
    if bw is not None:
        if bw['wc_max_at_bound']:
            flags.append("identified plant has too little phase lag to bound wc")
        if bw.get('wc_max_band_limited'):
            flags.append(
                f"wc ceiling is set by the identification band ({f_val:.0f} Hz), "
                "not by measured phase lag -- treat it as an upper bound only")
        if bw.get('cfg_extrapolated'):
            flags.append(
                f"wc={bw['wc_cfg']:.0f} puts the gain crossover at {bw['w_cross_cfg']/(2*np.pi):.0f} Hz, "
                f"above the {f_val:.0f} Hz the plant was identified over "
                "-- the PM above is extrapolation, not measurement")
    cc = fit.get('ctrl_check') if hasattr(fit, 'get') else None
    if adrc_flight and cc and cc['ratio'] == cc['ratio']:
        if not (0.8 <= cc['ratio'] <= 1.25) or abs(cc['phase_deg']) > 15.0:
            flags.append(
                f"assumed ADRC controller does not explain the logged output "
                f"(u/r is {cc['ratio']:.2f}x predicted, {cc['phase_deg']:+.0f} deg off) "
                "-- the adrc-recovered G is unreliable here; trust ctrl-free/eRPM")

    pk = float('nan')
    if bw is not None:
        pk = bw.get('cl_peak_db', float('nan'))
        if pk == pk and pk > 6.0:
            flags.append(
                f"closed loop peaks {pk:+.0f} dB at {bw['cl_peak_hz']:.1f} Hz "
                "-- the loop is resonant, not just slow")
        if adrc_flight:
            if bw['pm_cfg'] == bw['pm_cfg'] and bw['pm_cfg'] < bw['pm_target']:
                flags.append(f"configured wc leaves only {bw['pm_cfg']:.0f} deg PM")
            f3 = bw['f_3db_hz']
            if f3 == f3 and bw['w_3db'] > 0 and bw['wc_cfg'] / bw['w_3db'] > 3.0:
                flags.append("achieved bandwidth far below wc -- check b0")
    if b0_values:
        _v = np.array([v for v in b0_values if v == v])
        if len(_v) > 1 and (_v.max() - _v.min()) / max(_v.mean(), 1e-9) > 0.25:
            flags.append("methods disagree on b0 by more than the fit error bars")
    wm_flag = bool(fit.get('wm_at_bound'))
    if wm_flag:
        flags.append("2nd pole above the excited band")

    pk_flag = bool(pk == pk and pk > 6.0)
    # Either symptom alone means the model is missing dynamics somewhere
    # near or above the fitted band: an unresolved 2nd pole means the model
    # can't see that region at all, and a resonant peak means the real loop
    # has something there the model never captured in the first place.
    # wc_max is an extrapolation from the fitted model, so either symptom
    # is reason to treat it as an unverified ceiling rather than a target.
    if (wm_flag or pk_flag) and bw is not None:
        flags.append("wc estimates may be inaccurate -- "
                      + ("2nd pole unresolved" if wm_flag else "")
                      + (" and " if wm_flag and pk_flag else "")
                      + ("closed loop is resonant" if pk_flag else ""))
    z = fit.get('zero')
    if z:
        wc_used = fit.get('wc_used')
        if wc_used is None and bw is not None:
            wc_used = bw.get('wc_cfg')
        if wc_used and z < 0.5 * wc_used:
            flags.append(
                f"yaw is ~1st order at wc (zero at {z:.1f} rad/s): b0_eff ~ wc x HF gain "
                f"({hf_gain(fit):.0f}), so re-fit b0 whenever yaw wc changes")
        if fit.get('f_valid_hz', np.inf) < 10.0:
            flags.append(
                f"yaw coherent only to {fit['f_valid_hz']:.0f} Hz -- too low to pin the "
                "reaction-torque zero and the motor lag; fly a yaw chirp")
    flags += motor_lag_warnings(lag)
    return flags


def _sweep_fit(fit, fit_rpm):
    '''
    Fit the bandwidth sweep runs against. Normally the eRPM fit (best-
    conditioned b0). On yaw (lead-zero fit) the eRPM leg loses coherence
    early, while the closed-loop yaw fit stays coherent far wider on a chirp,
    so use whichever lead fit is backed by data to the higher frequency --
    the yaw crossover sits well above 15 Hz.
    '''
    if fit_rpm is None:
        return fit
    if fit.get('zero') and fit.get('f_valid_hz', 0) > fit_rpm.get('f_valid_hz', 0):
        return fit
    return fit_rpm

# Function that does all the fitting and plotting. 
def fit_plant_from_csv_indirect(csv_path, wo=None, b0=None, wc=None, order=2, nperseg=8192,
                                f_lo=0.8, f_hi=15.0, coh_min=0.85,
                                cross_check=True, use_rpm=True,
                                include_dterm=False, out_dir='Output metrics',
                                full_output=False, adrc_flight=True,
                                suggest_wc=True, save_outputs=True,
                                motor_lag=True, lag_f_hi=60.0, lag_coh_min=0.8,
                                lag_nperseg=4096, yaw_lead=True, yaw_f_hi=60.0,
                                wc_proposed=80.0):
    '''
    Recovers and fits the open-loop plant for roll/pitch/yaw from a blackbox
    log, given the (wc, wo, b0) the flight was flown with.

    wc_proposed: the wc you intend to fly (float, or dict keyed by axis
        name), default 80. Every b0 reported -- console footer, b0 table,
        results[axis]['b0_eff'] -- is b0_eff = wc_proposed^2 * |G(j*wc_proposed)|,
        i.e. the b0 that matches the plant at that wc. The plant is not
        exactly b0/s^2, so this number depends on wc (strongly on yaw): set it
        to the wc you will actually fly, and re-run if you change it.
    wo/b0/wc: what the log was flown with, float (all axes) or dict keyed by
        axis name. Needed only when adrc_flight=True. With adrc_flight=False
        they may be None; wo/b0 are then only used, as the proposed values,
        by the bandwidth analysis (suggest_wc=True).
    cross_check: also run the controller-free recovery (needs axisSum) and
        report it alongside. If the two disagree by more than ~20% the
        assumed controller parameters are wrong, not the data.
    adrc_flight: True if the log was flown with LADRC using the given
        (wc, wo, b0). The closed-loop 'adrc' recovery inverts that controller
        to get the plant, so it is only valid when this is True. Set False for
        a PID (or any non-ADRC) log: the adrc fit, its b0 column and its
        warnings are dropped, and the controller-free recovery becomes the
        primary path. wc/wo/b0 are then read as the ADRC values you are
        *considering*, and the bandwidth table answers whether they would fly.
    suggest_wc: False skips the whole bandwidth / wc analysis
        (suggest_bandwidth): no achieved -3dB, closed-loop peak, phase margin
        or max wc in the figure footer or metrics log, and only the b0-fit
        warnings are kept. results[axis]['bandwidth'] is None.
    save_outputs: False shows the figure but writes nothing to out_dir
        (no Plant fit .png / .txt, and the folder is not created).
    use_rpm: also run the eRPM decomposition (needs dshot_bidir=1). This is
        the best-conditioned path for b0, since the u -> rotor-speed leg
        stays coherent past 25 Hz instead of dying with the sticks at 15 Hz.

    motor_lag: also estimate the motor lag (estimate_motor_lag) on each axis.
        It is fitted on its own wider band (up to lag_f_hi, coherence >=
        lag_coh_min) over the chirp windows only, independent of f_hi/coh_min
        above, because the b0 band is too narrow to separate lag from delay.
        Result in results[axis]['motor_lag'] (tau and delay in seconds);
        motor_lag_table(results) tabulates it.

    All three paths report b0 as b0_eff at wc_proposed, so the numbers are
    directly comparable to each other.

    yaw_lead: fit yaw with the lead-zero model (fit_plant_frf(lead=True)).
        The yaw plant has a zero from rotor reaction torque, and above it is
        close to first order, so the no-zero model misfits it. Also reports
        the yaw high-frequency gain (hf_gain), the wc-independent yaw number.
    yaw_f_hi: upper fit frequency for the yaw closed-loop fits when the log
        has a yaw chirp window (otherwise f_hi is used). Yaw stays coherent
        far past 15 Hz on a chirp, and the wider band is what lets the
        bandwidth sweep check the yaw crossover against data instead of
        extrapolating.

    Note: 'tau' in the fit dicts is the fitted transport delay and 1/wm is the
    second pole of the b0 model; neither is the motor lag, because the b0 band
    (f_hi) is too narrow to tell the two apart. Use results[axis]['motor_lag'].

    Returns {axis: {K, a, wm, tau, b0, rms_rel, f, G, coh, tf, motor_lag, ...}}
    '''
    df = load_blackbox_csv(csv_path)
    hdr = read_blackbox_header(csv_path)
    out_dir = Path(out_dir)
    if save_outputs:
        out_dir.mkdir(parents=True, exist_ok=True)
    base = Path(csv_path).name.split('.')[0]
    out_png = out_dir / f'{base} Plant fit.png'
    out_txt = out_dir / f'{base} Plant fit.txt'

    wo_a, b0_a, wc_a = _axis_dict(wo), _axis_dict(b0), _axis_dict(wc)
    wp_a = _axis_dict(wc_proposed)
    if any(wp_a[a] is None for a in AXIS_NAMES):
        raise ValueError("wc_proposed must be given for every axis")
    if adrc_flight and any(v[a] is None for v in (wc_a, wo_a, b0_a) for a in AXIS_NAMES):
        raise ValueError("adrc_flight=True needs the flown wc, wo and b0 for every axis")
    if suggest_wc and not adrc_flight and any(v[a] is None for v in (wo_a, b0_a)
                                              for a in AXIS_NAMES):
        raise ValueError("suggest_wc=True on a non-ADRC log needs a proposed wo and b0")
    # wc the bandwidth analysis evaluates: the flown one on an ADRC log, the
    # proposed one otherwise
    wcb_a = wc_a if adrc_flight else wp_a

    buf = io.StringIO()
    class Tee(io.TextIOBase):
        def write(self, s):
            sys.__stdout__.write(s); buf.write(s); return len(s)

    results = {}
    _segs, _src = find_excitation_segments(df)
    yaw_chirp = (_src if isinstance(_src, str) else _src['yaw']) == 'chirp'
    fig, axes = plt.subplots(1, 3, figsize=(16, 8))

    with contextlib.redirect_stdout(Tee()):
        _, _lag = feedback_path_frf(np.array([1.0]), hdr, include_dterm=include_dterm)
        print("+/- is 1-sigma from the fit covariance: statistical only, i.e. how well this data "
              "pins down\nthis model. The gap BETWEEN methods is model error and is the real "
              "uncertainty on b0.\n"
              + ("adrc and ctrl-free share the same data and are not independent "
                 "of each other; eRPM is.\n" if adrc_flight else
                 "Log not flown with ADRC: the adrc recovery is skipped and ctrl-free is the "
                 "primary path.\n")
              + f"b0 is reported at the proposed wc = {wc_label(wc_proposed)}.\n"
              + ("Margin sweep includes header lag "
                 f"(gyro lpf2={_lag['gyro_lpf2']:.0f} Hz, delay={_lag['delay_s']*1e3:.2f} ms); "
                 "notches excluded, so margins stay slightly optimistic."
                 if suggest_wc else "").rstrip())

        for i, axis in enumerate(AXIS_NAMES):
            t, r, y, u = get_axis_signals(df, axis)
            fs = 1.0 / np.median(np.diff(t))
            lead = bool(yaw_lead and axis == 'yaw')
            ax_f_hi = max(f_hi, yaw_f_hi) if (lead and yaw_chirp) else f_hi

            if adrc_flight:
                f, G, coh, cond = recover_plant_frf(
                    r, y, u, fs, wc=wc_a[axis], wo=wo_a[axis], b0=b0_a[axis],
                    order=order, nperseg=nperseg, method='adrc')
                fit = fit_plant_frf(f, G, coh, f_lo=f_lo, f_hi=ax_f_hi,
                                    coh_min=coh_min, cond=cond, lead=lead)
            else:
                # No ADRC controller to invert, so the controller-free
                # recovery (r as instrument, u measured) is the primary path.
                if u is None:
                    raise ValueError(
                        f"{axis}: adrc_flight=False needs axisSum in the log "
                        "for the controller-free recovery")
                f, G, coh, cond = recover_plant_frf(
                    r, y, u, fs, nperseg=nperseg, method='controller_free')
                fit = fit_plant_frf(f, G, coh, f_lo=f_lo, f_hi=ax_f_hi,
                                    coh_min=coh_min, lead=lead)
            band = fit['band']

            fit_rpm = None
            if use_rpm and 'eRPM[0]' in df.columns:
                try:
                    fit_rpm = recover_plant_rpm(df, axis, nperseg=nperseg)
                except Exception as exc:
                    print(f"{axis:>6}: eRPM path unavailable ({exc})")

            G_cf = fit_cf = None
            if adrc_flight and cross_check and u is not None:
                _, G_cf, _, _ = recover_plant_frf(r, y, u, fs, nperseg=nperseg,
                                                  method='controller_free')
                fit_cf = fit_plant_frf(f, G_cf, coh, f_lo=f_lo, f_hi=ax_f_hi,
                                       coh_min=coh_min, lead=lead)

            ctrl_chk = None
            if adrc_flight:
                ctrl_chk = controller_consistency(
                    r, y, u, fs, wc_a[axis], wo_a[axis], b0_a[axis],
                    order=order, nperseg=nperseg)

            lag = None
            if motor_lag:
                try:
                    lag = estimate_motor_lag(df, axis, nperseg=lag_nperseg,
                                             f_hi=lag_f_hi, coh_min=lag_coh_min,
                                             use_rpm=use_rpm)
                except Exception as exc:
                    lag = dict(error=str(exc))

            bw_fit = _sweep_fit(fit, fit_rpm)
            bw = None
            if suggest_wc:
                bw = suggest_bandwidth(r, y, fs, bw_fit, wcb_a[axis], wo_a[axis],
                                       b0_a[axis], order=order, nperseg=nperseg,
                                       hdr=hdr, include_dterm=include_dterm,
                                       axis=axis, coh_min=coh_min)

            tf = plant_tf(fit)
            tt, h, st = plant_time_responses(fit)

            results[axis] = dict(fit, f=f, G=G, coh=coh, cond=cond, tf=tf,
                                 t=tt, impulse=h, step=st,
                                 G_cf=G_cf, fit_cf=fit_cf, fit_rpm=fit_rpm,
                                 b0_eff=b0_effective(fit, wp_a[axis]),
                                 bandwidth=bw, ctrl_check=ctrl_chk,
                                 motor_lag=lag, axis=axis, wc_used=wp_a[axis])

            # --- footer text: b0 and bandwidth only, with error bars -----
            b0_c, b0_sd = b0_uncertainty(fit, wp_a[axis])
            rows = [(('closed-loop / adrc' if adrc_flight else '        ctrl-free'),
                     b0_c, b0_sd)]
            if fit_cf is not None:
                v, sd = b0_uncertainty(fit_cf, wp_a[axis])
                rows.append(('  ctrl-free check', v, sd))
            if fit_rpm is not None:
                v, sd = b0_uncertainty(fit_rpm, wp_a[axis])
                rows.append(('eRPM decomposition', v, sd))

            _cfgword = 'configured' if adrc_flight else 'proposed'
            msg = f"{axis:>6}  b0_eff at proposed wc={wp_a[axis]:g}"
            if adrc_flight:
                msg += f"   (flown with wc={wc_a[axis]:g}, b0={b0_a[axis]:.0f})"
            for name, v, sd in rows:
                msg += f"\n           {name:<20s} {v:7.0f} +/- {sd:5.0f}"
                if adrc_flight:
                    msg += f"   [{v/b0_a[axis]*100:3.0f}% of configured]"

            # The two families are independent; ctrl-free shares data with adrc.
            indep = [r for r in rows if not r[0].startswith('  ')]
            if len(indep) == 2:
                a_, b_ = indep[0][1], indep[1][1]
                gap = abs(a_ - b_) / max(0.5 * (a_ + b_), 1e-9)
                pooled = np.hypot(indep[0][2], indep[1][2])
                msg += (f"\n           the two independent methods differ by {gap*100:3.0f}%"
                        f" ({abs(a_-b_):.0f}), against a pooled fit error of {pooled:.0f}")
                if abs(a_ - b_) > 2 * pooled:
                    msg += " -- model error dominates, use the range"

            if suggest_wc:
                f3, f3sd = bw['f_3db_hz'], bw['f_3db_sd']
                _pk = bw['cl_peak_db']
                msg += ("\n         bandwidth: achieved -3dB = "
                        + (f"{f3:5.2f} +/- {f3sd:4.2f} Hz ({bw['w_3db']:5.1f} rad/s)"
                           if f3 == f3 else
                           " n/a (never sustains -3dB inside the coherent band)")
                        + f"   vs wc = {bw['wc_cfg']:.0f}"
                        + (f"  [{bw['wc_cfg']/bw['w_3db']:.1f}x]" if f3 == f3 else "")
                        + (f"\n         closed-loop peak = {_pk:+.1f} dB at "
                           f"{bw['cl_peak_hz']:.1f} Hz" if _pk == _pk else ""))
                msg += ("\n         max wc for "
                        + f"{bw['pm_target']:.0f} deg PM = "
                        + ("not bounded by this model" if bw['wc_max_at_bound']
                           else f"{bw['wc_max']:5.0f} +/- {bw['wc_max_sd']:4.1f} rad/s")
                        + f";   PM at {_cfgword} wc = "
                        + (f"{bw['pm_cfg']:5.1f} +/- {bw['pm_cfg_sd']:4.2f} deg"
                           if bw['pm_cfg'] == bw['pm_cfg'] else "n/a (|L|<1)"))

            if fit.get('zero'):
                _hf, _hf_sd = hf_gain_uncertainty(fit)
                _hfs = [f"{_hf:.0f} +/- {_hf_sd:.1f}"]
                if fit_cf is not None and fit_cf.get('zero'):
                    _hfs.append(f"ctrl-free {hf_gain(fit_cf):.0f}")
                if fit_rpm is not None and fit_rpm.get('zero'):
                    _hfs.append(f"eRPM {hf_gain(fit_rpm):.0f}")
                msg += (f"\n         1st-order yaw: zero at {fit['zero']:.1f} rad/s, "
                        f"HF gain = {', '.join(_hfs)}  (b0 for a 1st-order ADRC)"
                        f"\n           b0_eff above is ~ wc x HF gain, so it moves with wc"
                        f"   [fit {f_lo:.1f}-{fit['f_valid_hz']:.0f} Hz, rms {fit['rms_rel']:.3f}]")

            if lag is not None and not lag.get('error'):
                _rpm = lag.get('rpm') or {}
                msg += (f"\n         motor lag = {lag['tau']*1e3:.1f} +/- {lag['tau_sd']*1e3:.1f} ms"
                        f", delay {lag['delay']*1e3:.1f} ms"
                        + (f"   (eRPM check {_rpm['tau']*1e3:.1f} ms)" if 'tau' in _rpm else "")
                        + f"   [{lag['f_lo_hz']:.1f}-{lag['f_hi_hz']:.0f} Hz, {lag['source']}]"
                        + "\n           "
                        + (f"wo*tau = {wo_a[axis]*lag['tau']:.2f} (flown wo), "
                           if adrc_flight else "")
                        + f"wc*(tau+d) = {wp_a[axis]*(lag['tau']+lag['delay']):.2f} (proposed wc)"
                        + (f",  b_acc/tau = {lag['b0_lag']:.0f}" if lag['b0_lag'] == lag['b0_lag']
                           else f",  lead zero at {lag['zero_rad_s']:.1f} rad/s"))

            if ctrl_chk and ctrl_chk['ratio'] == ctrl_chk['ratio']:
                msg += (f"\n         controller check: u/r is "
                        f"{ctrl_chk['ratio']:.2f}x predicted, "
                        f"{ctrl_chk['phase_deg']:+.0f} deg off "
                        f"({ctrl_chk['n_bins']} bins)")

            flags = axis_warnings(dict(fit, ctrl_check=ctrl_chk, wc_used=wp_a[axis]),
                                  bw, bw_fit,
                                  b0_values=[r[1] for r in rows],
                                  adrc_flight=adrc_flight, lag=lag)
            for fl in flags:
                msg += f"\n         [WARNING: {fl}]"
            print(msg)

            # --- plots -------------------------------------------------
            ax0 = axes[i]
            ax0.loglog(f[1:], np.abs(G[1:]), lw=.8, alpha=.5, label='recovered G (all bins)')
            _plabel = 'ADRC' if adrc_flight else 'controller-free'
            ax0.loglog(f[band], np.abs(G[band]), 'C0.', ms=4, label=f'{_plabel} fit band')
            if G_cf is not None:
                band_cf = fit_cf['band'] if fit_cf is not None else band
                ax0.loglog(f[band_cf], np.abs(G_cf[band_cf]), 'C2.', ms=3, alpha=.6,
                           label='controller-free check')
                if fit_cf is not None:
                    ax0.loglog(f[band_cf], np.abs(plant_frf_from_fit(fit_cf, 2*np.pi*f[band_cf])),
                               'C5-', lw=1.6, alpha=.9, label='controller-free fit')
            ax0.loglog(f[band], np.abs(plant_frf_from_fit(fit, 2*np.pi*f[band])), 'C3-', lw=2,
                       label=f'{_plabel} fit')
            if fit_rpm is not None:
                fr_ = fit_rpm['f']; br_ = fit_rpm['band']
                ax0.loglog(fr_[br_], np.abs(plant_frf_from_fit(fit_rpm, 2*np.pi*fr_[br_])),
                           'C4-', lw=1.6, alpha=.9, label='eRPM path')
            ax0.set_xlim(0.3, min(fs/2, 200)); ax0.set_ylim(1e-2, 1e3)
            ax0.set_title(axis.capitalize()); ax0.set_ylabel('|G|  (deg/s per u)')
            ax0.set_xlabel('Hz')
            ax0.grid(True, which='both', alpha=.3); ax0.legend(fontsize=7)

        fig.suptitle(f"{base} — plant identified from closed loop",
                     fontsize=13, fontweight='bold')

        # Plots only -- the per-axis metrics text is printed below the
        # figure (and written to the metrics log), not drawn into it.
        plot_h_in, title_h_in, margin_in = 5.0, 0.3, 0.1
        total_h_in = plot_h_in + title_h_in + margin_in
        fig.set_size_inches(16, total_h_in)
        plt.tight_layout(rect=[0, margin_in / total_h_in, 1,
                               1 - title_h_in / total_h_in])
        if save_outputs:
            plt.savefig(out_png, dpi=150)

    if save_outputs:
        with open(out_txt, 'w') as fh:
            fh.write(buf.getvalue())
    plt.show(); plt.close()
    if save_outputs:
        print(f"Saved plot to {out_png}")
        print(f"Saved metrics log to {out_txt}")
    return results

def _txt_table(df):
    '''
    Plain-text, fixed-width rendering of a DataFrame with aligned columns
    and a header rule -- readable in any text editor, unlike a raw
    to_string() dump where columns don't line up once values vary in width.
    '''
    cols = [str(c) for c in df.columns]
    rows = [[str(v) for v in row] for row in df.itertuples(index=False)]
    widths = [max(len(c), *(len(r[i]) for r in rows)) if rows else len(c)
              for i, c in enumerate(cols)]

    def fmt_row(vals):
        return '  '.join(v.ljust(w) for v, w in zip(vals, widths))

    lines = [fmt_row(cols), '  '.join('-' * w for w in widths)]
    lines += [fmt_row(r) for r in rows]
    return '\n'.join(lines)


def show_fit_summary(csv_path, wc=None, wo=None, b0=None, order=2, nperseg=8192, f_lo=0.8,
                     f_hi=15.0, coh_min=0.85, cross_check=True, use_rpm=True,
                     include_dterm=False, out_dir='Output metrics', results=None,
                     adrc_flight=True, suggest_wc=True, save_outputs=True,
                     motor_lag=True, lag_f_hi=60.0, lag_coh_min=0.8,
                     lag_nperseg=4096, yaw_lead=True, yaw_f_hi=60.0,
                     wc_proposed=80.0, verbose=False):
    '''
    Runs fit_plant_from_csv_indirect() (unless a precomputed `results` dict
    is passed in) and displays the b0 and bandwidth summary tables built
    from it, plus the wc/wo/b0 dicts the flight was flown with.

    wc_proposed (default 80, float or per-axis dict): the wc you intend to
    fly. The b0 table is b0_eff at this wc; see fit_plant_from_csv_indirect().

    verbose=False leaves the detail columns (VERBOSE_COLUMNS: yaw HF gain,
    fit band, data source, wc*(tau+d)) out of the displayed, saved and
    returned tables. verbose=True keeps them.

    adrc_flight=False drops the closed-loop 'adrc' column and any warning
    that depends on it; wc/wo/b0 can then be None (see
    fit_plant_from_csv_indirect()).

    suggest_wc=False skips the bandwidth / wc analysis entirely: no bandwidth
    table (bw_table is returned as None) and only the b0-fit warnings.

    save_outputs=False writes no files: no plot/metrics log from the fit and
    no summary tables .txt. Everything is still displayed in the notebook.

    motor_lag=True also displays the motor lag table (motor_lag_table) and
    writes it to the summary .txt; see fit_plant_from_csv_indirect().

    Returns (results, b0_table, bw_table).
    '''
    if results is None:
        results = fit_plant_from_csv_indirect(
            csv_path, wo=wo, b0=b0, wc=wc, order=order, nperseg=nperseg,
            f_lo=f_lo, f_hi=f_hi, coh_min=coh_min, cross_check=cross_check,
            use_rpm=use_rpm, include_dterm=include_dterm, out_dir=out_dir,
            adrc_flight=adrc_flight, suggest_wc=suggest_wc,
            save_outputs=save_outputs, motor_lag=motor_lag, lag_f_hi=lag_f_hi,
            lag_coh_min=lag_coh_min, lag_nperseg=lag_nperseg,
            yaw_lead=yaw_lead, yaw_f_hi=yaw_f_hi, wc_proposed=wc_proposed,
        )
    wc_p = _axis_dict(wc_proposed)
    wc_bw = _axis_dict(wc) if adrc_flight else wc_p
    wo, b0 = _axis_dict(wo), _axis_dict(b0)

    if suggest_wc and any(results[a].get('bandwidth') is None for a in AXIS_NAMES):
        raise ValueError("results were computed with suggest_wc=False; rerun "
                         "without passing `results` to get the bandwidth table")

    # Only needed for the extra 60 deg sweep in the bandwidth table
    if suggest_wc:
        df = load_blackbox_csv(csv_path)
        hdr = read_blackbox_header(csv_path)
    # b0 table from each of the three methods
    # Pulls straight from `results` (fit_plant_from_csv_indirect output) and
    # the wc/wo/b0 dicts passed in.

    def _style_table(df):
        # Zebra-striped, borderless table styling to match the reference tables.
        return (df.style
                .hide(axis='index')
                .set_table_styles([
                    {'selector': 'th',
                     'props': [('text-align', 'left'), ('font-weight', 'bold'),
                               ('border-bottom', '2px solid #444'),
                               ('padding', '6px 14px'), ('background-color', 'white')]},
                    {'selector': 'td',
                     'props': [('text-align', 'left'), ('padding', '6px 14px'),
                               ('border-bottom', '0px')]},
                    {'selector': 'tr:nth-child(even) td',
                     'props': [('background-color', '#f2f2f2')]},
                    {'selector': 'tr:nth-child(odd) td',
                     'props': [('background-color', 'white')]},
                    {'selector': 'table',
                     'props': [('border-collapse', 'collapse'), ('font-family', 'sans-serif'),
                               ('font-size', '13px')]},
                ]))

    b0_rows = []
    _b0_vals = []
    for axis in AXIS_NAMES:
        res = results[axis]

        adrc_v, adrc_sd = b0_uncertainty(res, wc_p[axis])

        cf_v = cf_sd = float('nan')
        if res.get('fit_cf') is not None:
            cf_v, cf_sd = b0_uncertainty(res['fit_cf'], wc_p[axis])

        rpm_v = rpm_sd = float('nan')
        if res.get('fit_rpm') is not None:
            rpm_v, rpm_sd = b0_uncertainty(res['fit_rpm'], wc_p[axis])

        # Gap between the two *independent* methods, matching the console
        # footer: adrc vs eRPM when flown with ADRC (ctrl-free shares data
        # with adrc so it's excluded), else ctrl-free vs eRPM.
        gap_str = 'n/a'
        if rpm_v == rpm_v:
            gap = abs(adrc_v - rpm_v) / max(0.5 * (adrc_v + rpm_v), 1e-9)
            gap_str = f'{gap*100:.0f}%'

        _b0_vals.append([adrc_v, cf_v, rpm_v])
        row = {'axis': axis}
        if adrc_flight:
            # `res` is the adrc fit; the ctrl-free run sits alongside it
            row['closed-loop / adrc'] = f'{adrc_v:.0f} +/- {adrc_sd:.0f}'
            row['ctrl-free check'] = (f'{cf_v:.0f} +/- {cf_sd:.0f}'
                                      if cf_v == cf_v else 'n/a')
        else:
            # `res` IS the ctrl-free fit; there is no adrc column to show
            row['ctrl-free'] = f'{adrc_v:.0f} +/- {adrc_sd:.0f}'
        row['eRPM'] = f'{rpm_v:.0f} +/- {rpm_sd:.0f}' if rpm_v == rpm_v else 'n/a'
        row['gap'] = gap_str
        if adrc_flight:
            # only meaningful when b0 is the value actually flown; with
            # adrc_flight=False it's a hypothetical the log can't confirm,
            # so there is nothing useful to compare it against here
            row['configured'] = f'{b0[axis]:.0f}'
        # Yaw only: the |G|*w plateau, which does not move with wc
        if any(results[a].get('zero') for a in AXIS_NAMES):
            if res.get('zero'):
                hf, hf_sd = hf_gain_uncertainty(res)
                extra = [f"{lbl} {hf_gain(fx):.0f}" for lbl, fx in
                         (('ctrl-free', res.get('fit_cf')), ('eRPM', res.get('fit_rpm')))
                         if fx is not None and fx.get('zero')]
                row['yaw HF gain'] = (f'{hf:.0f} +/- {hf_sd:.1f}'
                                      + (f" ({', '.join(extra)})" if extra else ''))
            else:
                row['yaw HF gain'] = '-'
        b0_rows.append(row)

    b0_table = pd.DataFrame(b0_rows)
    if not verbose:
        b0_table = drop_verbose_columns(b0_table)
    b0_table.attrs['wc_proposed'] = wc_label(wc_proposed)
    b0_by_axis = {r['axis']: v for r, v in zip(b0_rows, _b0_vals)}

    display(_style_table(b0_table).set_caption(
        f"b0 (deg/s per u) at proposed wc = {wc_label(wc_proposed)}"))

    lag_table = None
    if any(results[a].get('motor_lag') is not None for a in AXIS_NAMES):
        lag_table = motor_lag_table(results, wc=wc_p, wo=(wo if adrc_flight else None))
        if not verbose:
            lag_table = drop_verbose_columns(lag_table)
        display(_style_table(lag_table))

    # Bandwidth table
    # achieved -3dB basically tells the fastest response the system actually keeps up with
    # vs wc ratio basically says how fast the controller is responding, this should be around 2x
    # if vs wc ratio is too high that means something is off, likely b0

    bw_table = None
    bw60_by_axis = {a: None for a in AXIS_NAMES}
    if suggest_wc:
        bw_rows = []
        bw60_by_axis = {}
        for axis in AXIS_NAMES:
            res = results[axis]
            bw45 = res['bandwidth']

            t, r, y, u = get_axis_signals(df, axis)
            fs = 1.0 / np.median(np.diff(t))
            bw_fit = _sweep_fit(res, res.get('fit_rpm'))
            bw60 = suggest_bandwidth(r, y, fs, bw_fit, wc_bw[axis], wo[axis], b0[axis],
                                     hdr=hdr, pm_target=60.0, axis=axis,
                                     coh_min=coh_min)
            bw60_by_axis[axis] = bw60

            f3, f3sd = bw45['f_3db_hz'], bw45['f_3db_sd']
            wc_cfg, w3 = bw45['wc_cfg'], bw45['w_3db']

            # rad/s, to match the wc / max wc / PM columns either side of it.
            # The console footer still gives Hz as well.
            achieved = (f'{w3:.1f} +/- {f3sd * 2 * np.pi:.1f} rad/s'
                        if f3 == f3 else 'n/a')
            pk = bw45['cl_peak_db']
            peak_str = (f"{pk:+.1f} dB @ {bw45['cl_peak_hz']:.1f} Hz"
                        if pk == pk else 'n/a')
            vs_wc = f'{wc_cfg:.0f} ({wc_cfg/w3:.1f}x -3dB)' if (f3 == f3 and w3 > 0) else f'{wc_cfg:.0f}'

            def _max_wc(bw):
                return ('not bounded' if bw['wc_max_at_bound']
                        else f"{bw['wc_max']:.0f} +/- {bw['wc_max_sd']:.1f} rad/s")

            pm_at_wc = (f"{bw45['pm_cfg']:.1f} +/- {bw45['pm_cfg_sd']:.2f} deg"
                        if bw45['pm_cfg'] == bw45['pm_cfg'] else 'n/a')

            # CL peak is still computed in suggest_bandwidth and still drives the
            # peaking warning and the console footer; it is just not a column.
            brow = {'axis': axis, 'achieved -3dB': achieved}
            if adrc_flight:
                # the flown wc, with the ratio to the achieved -3dB when there is one
                brow['vs wc'] = vs_wc
            brow[f"max wc @{bw45['pm_target']:.0f} deg"] = _max_wc(bw45)
            brow['max wc @60 deg'] = _max_wc(bw60)
            brow['PM at wc' if adrc_flight else 'PM at proposed wc'] = pm_at_wc
            bw_rows.append(brow)

        bw_table = pd.DataFrame(bw_rows)

        display(_style_table(bw_table))

    # Same checks as the per-axis console footer, repeated here so the caveats
    # sit next to the numbers they apply to rather than scrolled off above.
    warn_lines = []
    for axis in AXIS_NAMES:
        res = results[axis]
        bw_fit = _sweep_fit(res, res.get('fit_rpm'))
        flags = axis_warnings(res, res['bandwidth'] if suggest_wc else None, bw_fit,
                              b0_values=b0_by_axis[axis], adrc_flight=adrc_flight,
                              lag=res.get('motor_lag'))
        # a ceiling that is band-limited at 45 deg is band-limited at 60 too,
        # but the 60 deg column can trip the bound flag on its own
        bw60 = bw60_by_axis[axis]
        if (bw60 is not None and bw60['wc_max_at_bound']
                and not res['bandwidth']['wc_max_at_bound']):
            flags.append("60 deg ceiling not bounded by this model")
        for fl in flags:
            warn_lines.append(f"[WARNING: {axis:>5}] {fl}")

    if warn_lines:
        print()
        for ln in warn_lines:
            print(ln)
    else:
        print("\nNo warnings.")

    if save_outputs:
        # --- export both tables as a readable, aligned .txt, alongside the
        # other outputs -----------------------------------------------------
        out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
        base = Path(csv_path).name.split('.')[0]
        out_tables_txt = out_dir / f'{base} Fit summary tables.txt'
        with open(out_tables_txt, 'w') as fh:
            title = f"{base} -- plant fit summary"
            fh.write(f"{title}\n{'=' * len(title)}\n\n")

            _t = f"b0 (deg/s per u) by method, at proposed wc = {wc_label(wc_proposed)}"
            fh.write(_t + "\n" + "-" * len(_t) + "\n")
            fh.write(_txt_table(b0_table))

            if lag_table is not None:
                fh.write("\n\nMotor lag\n")
                fh.write("---------\n")
                fh.write(_txt_table(lag_table))

            if bw_table is not None:
                fh.write("\n\nBandwidth\n")
                fh.write("---------\n")
                fh.write(_txt_table(bw_table))

            fh.write("\n\nWarnings\n")
            fh.write("--------\n")
            fh.write('\n'.join(warn_lines) if warn_lines else 'None.')
            fh.write("\n")
        print(f"Saved summary tables to {out_tables_txt}")

    return results, b0_table, bw_table
