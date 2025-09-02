#!/usr/bin/env python
# coding: utf-8

# In[1]:


# =============================================================================
# EKF + MH-ZNN Innovation Inversion (CPU-only, NumPy) — Full Suite
# Compares: STD, MOM, FT, VPFT, VAF, IE, DIE, HYBRID
# Optional extras: Newton–Schulz polish, 200 Hz stress, bursty noise
# =============================================================================
import numpy as np
import time
import os, csv  # NEW: for saving artifacts

# ----------------------------- Global toggles ---------------------------------
APPLY_POLISH       = True     # symmetric Newton–Schulz (2 steps) for all solvers
POLISH_ITERS       = 2

DO_STRESS_200HZ    = True     # set True to re-run at 200 Hz
DO_BURSTY_NOISE    = True     # set True to inject occasional 3σ measurement bursts
BURST_RATE         = 0.02     # 2% of steps are bursts if enabled
BURST_STD_MULT     = 3.0

# Gains (you can tweak per solver if needed)
DEFAULT_K_GAIN     = 100.0
DEFAULT_ALPHA      = 600.0
DT_INNER           = 5e-4     # inner ODE step (Euler everywhere)
STD_ITERS          = 230
MOM_ITERS          = 500
FT_ITERS           = 230
VPFT_ITERS         = 230
VAF_ITERS          = 230
IE_ITERS           = 230
DIE_ITERS          = 230
HYB_POLISH_ITERS   = 30       # stage-2 polish steps in hybrid
TAU_REL            = 1e-6     # hysteresis scale for hybrid switch
DWELL_MIN          = 1        # NEW: require ≥1 step between flip decisions

# ----------------------------- EKF Setup --------------------------------------
n_state = 6       # [px, py, pz, vx, vy, vz]
m_meas  = 3       # [px, py, pz]
dt_kf   = 0.01    # 100 Hz
tf_kf   = 10.0

F = np.eye(n_state)
F[:3, 3:] = np.eye(3) * dt_kf

q_pos, q_vel = 1e-2, 1e-1
Q = np.diag([q_pos]*3 + [q_vel]*3)
H = np.hstack((np.eye(3), np.zeros((3, 3))))
R = np.eye(m_meas) * 0.1

# Trajectory + measurements (seeded)
def build_data(dt, bursty=False, rng_seed=0):
    rng = np.random.default_rng(rng_seed)
    t_grid = np.arange(0, tf_kf, dt)
    true_pos = np.stack([np.cos(t_grid), np.sin(t_grid), 0.5 * t_grid], axis=1)
    true_vel = np.stack([-np.sin(t_grid), np.cos(t_grid), 0.5 * np.ones_like(t_grid)], axis=1)
    true_X   = np.hstack([true_pos, true_vel])
    meas = true_pos + rng.normal(0, np.sqrt(0.1), size=true_pos.shape)
    if bursty:
        mask = rng.random(len(t_grid)) < BURST_RATE
        noise_burst = rng.normal(0, BURST_STD_MULT*np.sqrt(0.1), size=true_pos.shape)
        meas[mask] += noise_burst[mask]
    return t_grid, true_pos, true_vel, true_X, meas

t_grid, true_pos, true_vel, true_X, meas = build_data(dt_kf, bursty=DO_BURSTY_NOISE, rng_seed=0)

# ----------------------------- ESBP & helpers ---------------------------------
p_vec = np.array([0.3, 0.4, 0.5])

def esbp(U, p_vec):
    a   = np.abs(U)
    out = np.zeros_like(U)
    for p in p_vec:
        out += a**p + a**(1.0/p)
    return np.sign(U) * out

def esbp_with_p(U, p_list):
    a   = np.abs(U)
    out = np.zeros_like(U)
    for p in p_list:
        out += a**p + a**(1.0/p)
    return np.sign(U) * out

# NEW: minimax φ entry surrogate + optional α* auto-tuner (not enabled by default)
def phi_min_entry(p_list):
    p = np.array(p_list, dtype=float)
    return float(p.sum() + (1.0/p).sum())

def autotune_alpha(k_gain, dt_inner, p_list=(0.3, 0.4, 0.5)):
    alpha_star = 4.0 * k_gain * phi_min_entry(p_list)
    alpha_cap  = 2.0 / dt_inner  # Euler stability cap for momentum update
    return min(alpha_star, alpha_cap), alpha_star, alpha_cap

def make_chol_solve(A):
    L = np.linalg.cholesky(A)
    def A_solve(B):
        Y = np.linalg.solve(L, B)
        return np.linalg.solve(L.T, Y)
    return A_solve

def make_rhs_std(A, k_gain=DEFAULT_K_GAIN):
    I = np.eye(A.shape[0])
    A_solve = make_chol_solve(A)
    def rhs(X):
        E   = A @ X - I
        Phi = esbp(E, p_vec)
        Y   = A_solve(Phi)
        return -k_gain * Y
    return rhs

def polish_newton_schulz(A, X, iters=POLISH_ITERS):
    I = np.eye(A.shape[0])
    for _ in range(iters):
        # symmetric NS to better preserve SPD structure
        X = 0.5*( X @ (2*I - A @ X) + (2*I - X @ A) @ X )
        X = 0.5*(X + X.T)
    return X

# ----------------------------- Solver variants --------------------------------
def inverse_std(A, X0=None, k_gain=DEFAULT_K_GAIN, iters=STD_ITERS):
    rhs = make_rhs_std(A, k_gain)
    X   = np.zeros_like(A) if X0 is None else X0.copy()
    for _ in range(iters):
        X += rhs(X) * DT_INNER
        X  = 0.5*(X + X.T)
    return X

def inverse_momentum_only(A, X0=None, V0=None, k_gain=DEFAULT_K_GAIN, alpha=DEFAULT_ALPHA, iters=MOM_ITERS):
    rhs = make_rhs_std(A, k_gain)
    X = np.zeros_like(A) if X0 is None else X0.copy()
    V = np.zeros_like(A) if V0 is None else V0.copy()
    # safety cap (no change for your defaults): alpha <= 2/DT_INNER
    alpha = min(alpha, 2.0/DT_INNER)
    for _ in range(iters):
        v_tgt = rhs(X)
        V += alpha * (v_tgt - V) * DT_INNER
        X += V * DT_INNER
        X  = 0.5*(X + X.T)
    return X, V

def inverse_ft(A, X0=None, k_gain=DEFAULT_K_GAIN, iters=FT_ITERS):
    I = np.eye(A.shape[0])
    A_solve = make_chol_solve(A)
    p_ft = [0.2, 0.25, 0.33]
    X = np.zeros_like(A) if X0 is None else X0.copy()
    for _ in range(iters):
        E   = A @ X - I
        Phi = esbp_with_p(E, p_ft)
        Y   = A_solve(Phi)
        X  += -k_gain * Y * DT_INNER
        X   = 0.5*(X + X.T)
    return X

def inverse_vpft(A, X0=None, k_gain=DEFAULT_K_GAIN, iters=VPFT_ITERS):
    I = np.eye(A.shape[0])
    A_solve = make_chol_solve(A)
    X = np.zeros_like(A) if X0 is None else X0.copy()
    for _ in range(iters):
        E = A @ X - I
        r = float(np.linalg.norm(E, ord='fro'))
        base = np.array([0.25, 0.33, 0.5])
        p_dyn = np.clip(base - 0.10*(r/(1.0+r)), 0.18, 0.50)
        Phi = esbp_with_p(E, p_dyn.tolist())
        Y   = A_solve(Phi)
        X  += -k_gain * Y * DT_INNER
        X   = 0.5*(X + X.T)
    return X

def inverse_vaf(A, X0=None, k_gain=DEFAULT_K_GAIN, iters=VAF_ITERS, beta=0.2):
    I = np.eye(A.shape[0])
    A_solve = make_chol_solve(A)
    X = np.zeros_like(A) if X0 is None else X0.copy()
    Ef = np.zeros_like(A)
    for _ in range(iters):
        E   = A @ X - I
        Ef  = (1.0 - beta)*Ef + beta*E
        Phi = esbp(Ef, p_vec)
        Y   = A_solve(Phi)
        X  += -k_gain * Y * DT_INNER
        X   = 0.5*(X + X.T)
    return X

def inverse_ie(A, X0=None, k_gain=DEFAULT_K_GAIN, ki=40.0, iters=IE_ITERS):
    I = np.eye(A.shape[0])
    A_solve = make_chol_solve(A)
    X = np.zeros_like(A) if X0 is None else X0.copy()
    Z = np.zeros_like(A)  # integral state
    for _ in range(iters):
        E   = A @ X - I
        Z  += E * DT_INNER
        Phi = esbp(E, p_vec)
        Y1  = A_solve(Phi)
        Y2  = A_solve(Z)
        X  += (-k_gain * Y1 - ki * Y2) * DT_INNER
        X   = 0.5*(X + X.T)
    return X

def inverse_die(A, X0=None, k_gain=DEFAULT_K_GAIN, ki1=60.0, ki2=15.0, iters=DIE_ITERS):
    I = np.eye(A.shape[0])
    A_solve = make_chol_solve(A)
    X = np.zeros_like(A) if X0 is None else X0.copy()
    Z1 = np.zeros_like(A); Z2 = np.zeros_like(A)
    for _ in range(iters):
        E   = A @ X - I
        Z1 += E * DT_INNER
        Z2 += Z1 * DT_INNER
        Phi = esbp(E, p_vec)
        Y1  = A_solve(Phi)
        YI1 = A_solve(Z1)
        YI2 = A_solve(Z2)
        X  += (-k_gain * Y1 - ki1 * YI1 - ki2 * YI2) * DT_INNER
        X   = 0.5*(X + X.T)
    return X

def inverse_hybrid(A, X0=None, V0=None, k_gain=DEFAULT_K_GAIN, alpha=DEFAULT_ALPHA,
                   max_iter_momentum=MOM_ITERS, polish_iters=HYB_POLISH_ITERS, tau_rel=TAU_REL):
    # Stage 1: momentum with slope-flip + hysteresis + dwell; Stage 2: std polish
    # Safe cap (paper-aligned): α ≤ 2/DT_INNER. Optional auto-tune if alpha=None.
    if alpha is None:
        alpha, alpha_star, alpha_cap = autotune_alpha(k_gain, DT_INNER, p_vec)
    else:
        _, _, alpha_cap = autotune_alpha(k_gain, DT_INNER, p_vec)
        alpha = min(alpha, alpha_cap)

    rhs = make_rhs_std(A, k_gain)
    I = np.eye(A.shape[0])
    X = np.zeros_like(A) if X0 is None else X0.copy()
    V = np.zeros_like(A) if V0 is None else V0.copy()

    def res(X_): return np.linalg.norm(A @ X_ - I, ord='fro')

    r_hist = [res(X)]
    last_decision_idx = -10  # NEW: dwell book-keeping
    for it in range(max_iter_momentum):  # was: for _ in range(...)
        v_tgt = rhs(X)
        V += alpha * (v_tgt - V) * DT_INNER
        X += V * DT_INNER
        X  = 0.5*(X + X.T)
        r_hist.append(res(X))
        if len(r_hist) >= 4:
            d1 = r_hist[-1] - r_hist[-2]
            d2 = r_hist[-2] - r_hist[-3]
            tau = tau_rel * max(1.0, r_hist[-2])
            dwell_ok = (it - last_decision_idx) >= DWELL_MIN  # NEW
            if (d2 < -tau) and (d1 > tau) and dwell_ok:
                last_decision_idx = it
                break

    for _ in range(polish_iters):
        X += rhs(X) * DT_INNER
        X  = 0.5*(X + X.T)

    return X, V

def chol_inverse(A):
    L = np.linalg.cholesky(A)
    Linv = np.linalg.inv(L)
    return Linv.T @ Linv

# ----------------------------- Stats & reporting -------------------------------
def wilson_hilferty_chi2_interval(k, conf=0.95):
    z = 1.959963984540054  # ~N^{-1}(0.975)
    mu = 1.0 - 2.0/(9.0*k)
    sig = (2.0/(9.0*k))**0.5
    low = k * (mu - z*sig)**3
    high = k * (mu + z*sig)**3
    return float(low), float(high)

def summarize_times(times, tick_s):
    t = np.array(times)
    return {
        "mean_ms": 1e3*t.mean(),
        "median_ms": 1e3*np.median(t),
        "p90_ms": 1e3*np.percentile(t, 90),
        "p99_ms": 1e3*np.percentile(t, 99),
        "max_ms":  1e3*t.max(),
        "pct_of_tick_mean": 100.0*(t.mean()/tick_s),
        "pct_of_tick_p99": 100.0*(np.percentile(t, 99)/tick_s),
        "deadline_miss_pct": 100.0*np.mean(t > tick_s)
    }

# One EKF pass with a chosen inverse "mode"
def run_kf_with_logging(mode="HYBRID", dt_tick=dt_kf, parity_stride=10):
    x_hat = np.zeros(n_state)
    P     = np.eye(n_state)
    I_m   = np.eye(m_meas)

    timing, nis_list, resI_list, parity_list = [], [], [], []
    pos_est_history = []

    prev_X = None; prev_V = None

    for step, z_k in enumerate(meas):
        # Predict
        x_pred = F @ x_hat
        P_pred = F @ P @ F.T + Q

        # Innovation
        y_tilde = z_k - H @ x_pred
        S = H @ P_pred @ H.T + R   # 3x3 SPD

        # Inversion (timed)
        t0 = time.perf_counter()
        if mode == "STD":
            S_inv = inverse_std(S, X0=prev_X, k_gain=DEFAULT_K_GAIN, iters=STD_ITERS)
            prev_X = S_inv
        elif mode == "MOM":
            S_inv, prev_V = inverse_momentum_only(S, X0=prev_X, V0=prev_V,
                                                  k_gain=DEFAULT_K_GAIN, alpha=DEFAULT_ALPHA, iters=MOM_ITERS)
            prev_X = S_inv
        elif mode == "FT":
            S_inv = inverse_ft(S, X0=prev_X, k_gain=DEFAULT_K_GAIN, iters=FT_ITERS)
            prev_X = S_inv
        elif mode == "VPFT":
            S_inv = inverse_vpft(S, X0=prev_X, k_gain=DEFAULT_K_GAIN, iters=VPFT_ITERS)
            prev_X = S_inv
        elif mode == "VAF":
            S_inv = inverse_vaf(S, X0=prev_X, k_gain=DEFAULT_K_GAIN, iters=VAF_ITERS)
            prev_X = S_inv
        elif mode == "IE":
            S_inv = inverse_ie(S, X0=prev_X, k_gain=DEFAULT_K_GAIN, ki=40.0, iters=IE_ITERS)
            prev_X = S_inv
        elif mode == "DIE":
            S_inv = inverse_die(S, X0=prev_X, k_gain=DEFAULT_K_GAIN, ki1=60.0, ki2=15.0, iters=DIE_ITERS)
            prev_X = S_inv
        elif mode == "HYBRID":
            # Keep your default alpha unless you pass alpha=None (auto-tune)
            S_inv, prev_V = inverse_hybrid(S, X0=prev_X, V0=prev_V,
                                           k_gain=DEFAULT_K_GAIN, alpha=None,
                                           max_iter_momentum=MOM_ITERS, polish_iters=HYB_POLISH_ITERS, tau_rel=TAU_REL)
            prev_X = S_inv
        else:
            raise ValueError("Unknown mode")

        if APPLY_POLISH:
            S_inv = polish_newton_schulz(S, S_inv, iters=POLISH_ITERS)

        timing.append(time.perf_counter() - t0)

        # Kalman update
        K = P_pred @ H.T @ S_inv
        x_hat = x_pred + K @ y_tilde
        P = (np.eye(n_state) - K @ H) @ P_pred
        P = 0.5*(P + P.T)

        # Logs (outside timed region where needed)
        nis_list.append(float(y_tilde.T @ S_inv @ y_tilde))
        resI_list.append(float(np.linalg.norm(S @ S_inv - I_m, ord='fro')))
        if (step % parity_stride) == 0:
            S_inv_chol = chol_inverse(S)
            parity_list.append(float(np.linalg.norm(S_inv - S_inv_chol, ord='fro')))

        pos_est_history.append(x_hat[:3])

    pos_est_history = np.array(pos_est_history)
    rmse = float(np.sqrt(np.mean(np.sum((true_pos - pos_est_history)**2, axis=1))))
    stats = summarize_times(timing, dt_tick)
    logs = {
        "timing": np.array(timing),
        "nis": np.array(nis_list),
        "resI": np.array(resI_list),
        "parity": np.array(parity_list),
        "rmse": rmse
    }
    return logs, stats

def report_suite(modes=("STD","MOM","FT","VPFT","VAF","IE","DIE","HYBRID"), tick_s=dt_kf, title_hz=100):
    band_lo, band_hi = wilson_hilferty_chi2_interval(m_meas, conf=0.95)
    os.makedirs("figures", exist_ok=True)  # NEW
    out_csv = f"figures/ekf_{title_hz}Hz_summary.csv"  # NEW
    rows = [["mode","mean_ms","median_ms","p90_ms","p99_ms","max_ms","miss_%","mean%tick","p99%tick",
             "rmse","nis_mean","nis_std","nis_inband_%","resI_mean","resI_p99"]]  # NEW

    print("\n========== EKF Innovation Inversion @ {} Hz ==========".format(title_hz))
    for mode in modes:
        logs, stats = run_kf_with_logging(mode, dt_tick=tick_s, parity_stride=10)
        nis = logs["nis"]
        frac_in = 100.0*np.mean((nis >= band_lo) & (nis <= band_hi))
        print(f"\n[{mode}]")
        print(f"  time (ms): mean {stats['mean_ms']:.3f} | median {stats['median_ms']:.3f} | p90 {stats['p90_ms']:.3f} | p99 {stats['p99_ms']:.3f} | max {stats['max_ms']:.3f}")
        print(f"  deadline misses (> {tick_s*1e3:.1f} ms): {stats['deadline_miss_pct']:.2f}% | tick budget: mean {stats['pct_of_tick_mean']:.1f}% | p99 {stats['pct_of_tick_p99']:.1f}%")
        print(f"  NIS: mean {nis.mean():.2f} ± {nis.std(ddof=1):.2f} | 95% band [{band_lo:.2f}, {band_hi:.2f}] | in-band {frac_in:.1f}%")
        print(f"  ||SS⁻¹−I||_F: mean {logs['resI'].mean():.2e} | p99 {np.percentile(logs['resI'],99):.2e}")
        if len(logs["parity"]):
            print(f"  parity vs Chol. (stride=10): median {np.median(logs['parity']):.2e}")
        print(f"  RMSE (pos): {logs['rmse']:.3f} m")

        # NEW: append a CSV row
        rows.append([
            mode, f"{stats['mean_ms']:.3f}", f"{stats['median_ms']:.3f}", f"{stats['p90_ms']:.3f}",
            f"{stats['p99_ms']:.3f}", f"{stats['max_ms']:.3f}", f"{stats['deadline_miss_pct']:.2f}",
            f"{stats['pct_of_tick_mean']:.1f}", f"{stats['pct_of_tick_p99']:.1f}",
            f"{logs['rmse']:.3f}", f"{nis.mean():.2f}", f"{nis.std(ddof=1):.2f}", f"{frac_in:.1f}",
            f"{logs['resI'].mean():.2e}", f"{np.percentile(logs['resI'],99):.2e}"
        ])

    # NEW: write CSV once per suite
    with open(out_csv, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    print(f"\n[OK] Wrote summary: {out_csv}")

# ----------------------------- Optional: stress / noise ------------------------
def rebuild_scenario(new_dt, bursty=False, rng_seed=0):
    global dt_kf, F, t_grid, true_pos, true_vel, true_X, meas
    dt_kf = new_dt
    F = np.eye(n_state)
    F[:3, 3:] = np.eye(3) * dt_kf
    t_grid, true_pos, true_vel, true_X, meas = build_data(dt_kf, bursty=bursty, rng_seed=rng_seed)

# ----------------------------- Run --------------------------------------------
if __name__ == "__main__":
    # Baseline 100 Hz
    report_suite(modes=("STD","MOM","FT","VPFT","VAF","IE","DIE","HYBRID"), tick_s=dt_kf, title_hz=int(1.0/dt_kf))

    if DO_STRESS_200HZ:
        rebuild_scenario(0.005, bursty=DO_BURSTY_NOISE, rng_seed=1)
        report_suite(modes=("STD","MOM","FT","VPFT","VAF","IE","DIE","HYBRID"), tick_s=dt_kf, title_hz=int(1.0/dt_kf))


# In[ ]:




