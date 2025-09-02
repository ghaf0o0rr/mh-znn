#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python3
# Benchmark 11 — RK45 (adaptive) on ill-conditioned SPD path
# Aligns with paper methodology:
# - Minimax α*: α* = 4 k φ_min with entry-scale surrogate φ_min ≈ Σp + Σ(1/p)
# - Stage-1 RK45 step cap: dt_max_stage1 = min(DT_MAX, 2 / α*)
# - Single, refined handoff (prev/mid/end using two RK4 half-steps)
# - True-slope flip test with absolute hysteresis + dwell
# - Baselines: Standard (ESBP), VAF-ZNN, FT-ZNN, VP-FTZNN (all RK45)

import numpy as np
import matplotlib.pyplot as plt
import csv, os

# ───────────────────────── Reproducibility ─────────────────────────
np.random.seed(7)

# ───────────────────────── Problem size & window ───────────────────
n          = 50
t0, tf     = 0.0, 0.06
I          = np.eye(n)

# ───────────────────────── Ill-conditioned SPD A(t) ────────────────
# A(t) = Q diag(s(t)) Q^T, with s(t) log-spaced (cond ≈ 1e8), smooth time drift
cond_target = 1e8
amp         = 0.15               # ±15% amplitude; keeps s(t)>0 (SPD)
omega       = 4.0                # temporal frequency

# Orthonormal Q (well-conditioned similarity)
Q, _  = np.linalg.qr(np.random.randn(n, n))
QT    = Q.T

# Log-spaced eigenvalues in [1/cond, 1]
log_s_base = np.linspace(np.log(1.0/cond_target), 0.0, n)
s_base     = np.exp(log_s_base)

def s_of_t(t):
    return s_base * (1.0 + amp*np.sin(omega*t))

def sdot_of_t(t):
    return s_base * (amp*omega*np.cos(omega*t))

# Fast operators using the eigenstructure
def apply_A(t, X):
    s = s_of_t(t)
    return (Q * s) @ (QT @ X)

def apply_Ad(t, X):
    sd = sdot_of_t(t)
    return (Q * sd) @ (QT @ X)

def apply_A_inv(t, B):
    s = s_of_t(t)
    return Q @ ((QT @ B) / s[:, None])

def A_func(t):
    s = s_of_t(t)
    return (Q * s) @ QT

# ─────────────────────── Hyper-parameters ────────────
# NOTE: kept exactly as provided by you (do not change).
p_vec5    = np.array([0.25, 0.50, 0.75, 0.75, 0.75, 0.8, 0.90], dtype=float)
m_choices = [5, 6, 7]

k_gain     = 10.0                # global gain (paper uses higher k for faster decay)

# VAF parameters (typical)
a1 = a2 = a3 = a4 = 1.0
h_exponent, w_exponent = 0.2, 5.0

# FT / VP-FT parameters (γ = 2/3, k0 = 8)
gamma_ft  = 2.0 / 3.0
k0_vpft   = 8.0

# ───────────────────────── RK45 controls (Dormand–Prince) ─────────
ATOL      = 1e-8
RTOL      = 1e-6
DT0       = 1e-5
DT_MIN    = 1e-8
DT_MAX    = 1e-2
SAFETY    = 0.9
MAX_STEPS = 200000

# Paper-aligned switch guards for adaptive steps:
TAU_ABS   = 1e-8   # absolute slope hysteresis (|de/dt|)
DWELL_MIN = 1      # require at least 1 accepted step between decisions

# ─────────────────────── Multi-exponent ESBP(u,m) ──────────────────
def esbp_multiexp(E, p_vec5, m):
    p_sub = p_vec5[:m]
    absE  = np.abs(E)
    term1 = (absE[..., None] ** p_sub).sum(-1)
    term2 = (absE[..., None] ** (1.0 / p_sub)).sum(-1)
    return np.sign(E) * (term1 + term2)

# α* self-tuning for Hybrid Stage 1: α* = 4 k φ_min(entry), φ_min ≈ Σ(p)+Σ(1/p)
def alpha_star_from_p(p_vec5, m, k):
    p_sub = p_vec5[:m]
    phi_min_entry = float(p_sub.sum() + (1.0/p_sub).sum())
    return 4.0 * k * phi_min_entry

# ─────────────────────── Additional activation functions ───────────
def vaf(E):
    absE, sgnE = np.abs(E), np.sign(E)
    return (a1 * absE**h_exponent * sgnE +
            a2 * absE**w_exponent * sgnE +
            a3 * E +
            a4 * sgnE)

def ft_term(E):
    return -(E + np.sign(E) * np.abs(E)**gamma_ft)

def vpft_term(E, t):
    k_t = k0_vpft / (t + 1.0)
    return -k_t * (E + np.sign(E) * np.abs(E)**gamma_ft)

# ─────────────────────────── RHS helpers (Ẋ) — fast solves ──────────
def znn_rhs_m(t, X, m):
    E     = apply_A(t, X) - I
    rhs   = -apply_Ad(t, X) - k_gain * esbp_multiexp(E, p_vec5, m)
    return apply_A_inv(t, rhs)

def momentum_rhs_m(t, X, V, alpha, m):
    E     = apply_A(t, X) - I
    v_tgt = apply_A_inv(t, -apply_Ad(t, X) - k_gain * esbp_multiexp(E, p_vec5, m))
    return V, alpha * (v_tgt - V)

# Single-activation RHS (not swept over m)
def vaf_rhs(t, X):
    E   = apply_A(t, X) - I
    rhs = -apply_Ad(t, X) - k_gain * vaf(E)
    return apply_A_inv(t, rhs)

def ft_rhs(t, X):
    E   = apply_A(t, X) - I
    rhs = -apply_Ad(t, X) + k_gain * ft_term(E)
    return apply_A_inv(t, rhs)

def vpft_rhs(t, X):
    E   = apply_A(t, X) - I
    rhs = -apply_Ad(t, X) + k_gain * vpft_term(E, t)
    return apply_A_inv(t, rhs)

# ───────────────────────── Utilities for RK45 (DP5(4)) ─────────────
def _fro(a): return np.linalg.norm(a, 'fro')

def _combine_norm(y):
    if isinstance(y, tuple):
        return np.sqrt(sum(_fro(yi)**2 for yi in y))
    return _fro(y)

def _lincomb(*terms):
    c0, y0 = terms[0]
    if isinstance(y0, tuple):
        out = []
        for j in range(len(y0)):
            acc = c0 * y0[j]
            for c, y in terms[1:]:
                acc = acc + c * y[j]
            out.append(acc)
        return tuple(out)
    acc = c0 * y0
    for c, y in terms[1:]:
        acc = acc + c * y
    return acc

# Dormand–Prince tableau
_c = np.array([0.0, 1/5, 3/10, 4/5, 8/9, 1.0, 1.0])
_a = [
    [],
    [1/5],
    [3/40,        9/40],
    [44/45,      -56/15,      32/9],
    [19372/6561, -25360/2187, 64448/6561,   -212/729],
    [9017/3168,  -355/33,     46732/5247,    49/176,   -5103/18656],
]
_b5 = np.array([35/384, 0.0, 500/1113, 125/192, -2187/6784, 11/84, 0.0])
_b4 = np.array([5179/57600, 0.0, 7571/16695, 393/640, -92097/339200, 187/2100, 1/40])

def _rk45_step(f, t, y, dt):
    # k1
    k1 = f(t + _c[0]*dt, y)
    # k2
    y2 = _lincomb((1.0, y), (dt*_a[1][0], k1))
    k2 = f(t + _c[1]*dt, y2)
    # k3
    y3 = _lincomb((1.0, y), (dt*_a[2][0], k1), (dt*_a[2][1], k2))
    k3 = f(t + _c[2]*dt, y3)
    # k4
    y4s = _lincomb((1.0, y),
                   (dt*_a[3][0], k1), (dt*_a[3][1], k2), (dt*_a[3][2], k3))
    k4 = f(t + _c[3]*dt, y4s)
    # k5
    y5s = _lincomb((1.0, y),
                   (dt*_a[4][0], k1), (dt*_a[4][1], k2),
                   (dt*_a[4][2], k3), (dt*_a[4][3], k4))
    k5 = f(t + _c[4]*dt, y5s)
    # k6 (uses the 'a[5]' row — not b5!)
    y6s = _lincomb((1.0, y),
                   (dt*_a[5][0], k1), (dt*_a[5][1], k2),
                   (dt*_a[5][2], k3), (dt*_a[5][3], k4), (dt*_a[5][4], k5))
    k6 = f(t + _c[5]*dt, y6s)

    # 5th order solution
    y5 = _lincomb((1.0, y),
                  (dt*_b5[0], k1), (dt*_b5[2], k3),
                  (dt*_b5[3], k4), (dt*_b5[4], k5), (dt*_b5[5], k6))

    # k7 for the embedded 4th-order estimate
    k7 = f(t + _c[6]*dt, y5)

    # 4th order solution
    y4 = _lincomb((1.0, y),
                  (dt*_b4[0], k1), (dt*_b4[2], k3), (dt*_b4[3], k4),
                  (dt*_b4[4], k5), (dt*_b4[5], k6), (dt*_b4[6], k7))

    err = _lincomb((1.0, y5), (-1.0, y4))
    return y5, err


def _err_ratio(y_new, err, atol=ATOL, rtol=RTOL):
    num = _combine_norm(err)
    denom = atol + rtol * max(_combine_norm(y_new), 1.0)
    return num / denom

def integrate_adaptive(f, y0, t0, tf, residual_fn,
                       atol=ATOL, rtol=RTOL,
                       dt0=DT0, dt_min=DT_MIN, dt_max=DT_MAX,
                       safety=SAFETY, max_steps=MAX_STEPS,
                       callback=None):
    t = t0
    y = y0
    dt = dt0
    ts = [t]
    errs = [residual_fn(t, y)]
    steps = 0
    p = 5.0
    while t < tf - 1e-15 and steps < max_steps:
        if t + dt > tf:
            dt = tf - t
        y_new, e = _rk45_step(f, t, y, dt)
        r = _err_ratio(y_new, e, atol, rtol)
        if r <= 1.0 or dt <= dt_min*1.01:
            # accept
            t += dt
            y = y_new
            ts.append(t)
            errs.append(residual_fn(t, y))
            # propose new dt
            factor = 2.0 if r == 0 else safety * (1.0 / r)**(1.0/(p+1.0))
            factor = min(5.0, max(0.2, factor))
            dt = min(dt_max, max(dt_min, dt * factor))
            steps += 1
            if callback is not None:
                action = callback(t, y, ts, errs)
                if action == "break":
                    break
        else:
            # reject
            factor = safety * (1.0 / r)**(1.0/(p+1.0))
            factor = min(5.0, max(0.2, factor))
            dt = max(dt_min, dt * factor)
            steps += 1
    return np.array(ts), np.array(errs), y

# ───────────────────────── Residual helpers ────────────────────────
def residual_single(t, X):
    return np.linalg.norm(apply_A(t, X) - I, 'fro')

def residual_momentum(t, state):
    X, V = state
    return np.linalg.norm(apply_A(t, X) - I, 'fro')

# ────────────────────────── RK45 variants (m-swept) ────────────────
def run_standard_m(m):
    f = lambda t, X: znn_rhs_m(t, X, m)
    X0 = np.zeros((n,n))
    return integrate_adaptive(f, X0, t0, tf, residual_single)

# Hybrid (RK45) with refined auto-switch (local minimum), for given m
def _rk4_step_mom_m(t, X, V, dt, alpha, m):
    def f(tt, XX, VV): return momentum_rhs_m(tt, XX, VV, alpha, m)
    k1X, k1V = f(t, X, V)
    k2X, k2V = f(t + 0.5*dt, X + 0.5*dt*k1X, V + 0.5*dt*k1V)
    k3X, k3V = f(t + 0.5*dt, X + 0.5*dt*k2X, V + 0.5*dt*k2V)
    k4X, k4V = f(t + dt, X + dt*k3X, V + dt*k3V)
    Xn = X + (dt/6.0)*(k1X + 2*k2X + 2*k3X + k4X)
    Vn = V + (dt/6.0)*(k1V + 2*k2V + 2*k3V + k4V)
    return Xn, Vn

def run_hybrid_autoswitch_refined_m(m, atol=ATOL, rtol=RTOL,
                                    dt0=2e-5, dt_min=DT_MIN, dt_max=DT_MAX,
                                    refine_depth=1):
    # α* and RK45 Stage-1 cap (paper)
    alpha_s1 = alpha_star_from_p(p_vec5, m, k_gain)
    dt_max_cap = 2.0 / alpha_s1
    dt_max_s1  = min(dt_max, dt_max_cap)

    state0 = (np.zeros((n,n)), np.zeros((n,n)))
    best = {"switch_t": None, "switch_state": None, "switch_err": None}

    f_mom = lambda t, state: momentum_rhs_m(t, state[0], state[1], alpha_s1, m)

    prev = {"t": t0, "state": state0, "err": residual_momentum(t0, state0), "dec_idx": 0}

    def cb(t, state, ts_loc, errs_loc):
        # true slope-based flip with tiny hysteresis + dwell
        if len(errs_loc) >= 3:
            dt1 = max(1e-15, (ts_loc[-2] - ts_loc[-3]))
            dt2 = max(1e-15, (ts_loc[-1] - ts_loc[-2]))
            g_prev = (errs_loc[-2] - errs_loc[-3]) / dt1
            g_curr = (errs_loc[-1] - errs_loc[-2]) / dt2
            dwell_ok = ((len(ts_loc) - 1) - prev.get("dec_idx", 0)) >= DWELL_MIN

            if (g_prev <= -TAU_ABS) and (g_curr >= +TAU_ABS) and dwell_ok:
                # refine within [prev.t, t] using two RK4 half-steps
                t_prev, (X_prev, V_prev), e_prev = prev["t"], prev["state"], prev["err"]
                dt_local = t - t_prev
                X_mid, V_mid = _rk4_step_mom_m(t_prev, X_prev, V_prev, 0.5*dt_local, alpha_s1, m)
                e_mid = residual_single(t_prev + 0.5*dt_local, X_mid)
                X_end, V_end = _rk4_step_mom_m(t_prev + 0.5*dt_local, X_mid, V_mid, 0.5*dt_local, alpha_s1, m)
                e_end = residual_single(t_prev + dt_local, X_end)

                candidates = [(t_prev, (X_prev, V_prev), e_prev),
                              (t_prev + 0.5*dt_local, (X_mid, V_mid), e_mid),
                              (t_prev + dt_local, (X_end, V_end), e_end)]
                t_best, state_best, e_best = min(candidates, key=lambda z: z[2])
                best["switch_t"], best["switch_state"], best["switch_err"] = t_best, state_best, e_best
                prev["dec_idx"] = len(ts_loc) - 1
                return "break"

        prev["t"] = ts_loc[-1]
        prev["state"] = state
        prev["err"] = errs_loc[-1]
        return None

    # Stage 1 (momentum, adaptive) up to first refined minimum
    ts1, errs1, state_at_break = integrate_adaptive(
        f=f_mom, y0=state0, t0=t0, tf=tf, residual_fn=residual_momentum,
        atol=atol, rtol=rtol, dt0=dt0, dt_min=dt_min, dt_max=dt_max_s1, callback=cb
    )

    # Choose handoff point (refined best if found; else last accepted)
    if best["switch_state"] is None:
        switch_t = ts1[-1]
        (Xs, Vs) = state_at_break
        ts_switch, errs_switch = ts1, errs1
    else:
        switch_t = best["switch_t"]
        (Xs, Vs) = best["switch_state"]
        ts_switch = ts1[:-1].copy()
        errs_switch = errs1[:-1].copy()
        if ts_switch[-1] < switch_t - 1e-18:
            ts_switch = np.append(ts_switch, switch_t)
            errs_switch = np.append(errs_switch, best["switch_err"])
        else:
            errs_switch[-1] = best["switch_err"]

    # Stage 2 (standard same-m ZNN, adaptive)
    f_std = lambda tt, X: znn_rhs_m(tt, X, m)
    ts2, errs2, Xf = integrate_adaptive(
        f_std, Xs, switch_t, tf, residual_single,
        atol=atol, rtol=rtol, dt0=DT0, dt_min=dt_min, dt_max=dt_max
    )

    ts_full   = np.concatenate([ts_switch, ts2[1:]])
    errs_full = np.concatenate([errs_switch, errs2[1:]])
    return ts_full, errs_full, Xf, (switch_t, errs_switch[-1] if best["switch_state"] is None else best["switch_err"])

# Generic single-activation RK45 runner (for VAF/FT/VP-FT)
def run_generic(rhs):
    f = lambda t, X: rhs(t, X)
    X0 = np.zeros((n,n))
    return integrate_adaptive(f, X0, t0, tf, residual_single)

# ───────────────────────────── Driver ──────────────────────────────
if __name__ == "__main__":
    runs, X_final, switches = {}, {}, {}

    # Conditioning report (w.r.t. spectrum at t0)
    s0 = s_of_t(t0)
    kappa = s0.max() / s0.min()
    print(f"cond(A(t)) ≈ {kappa:.2e}  (target {cond_target:.1e})")

    # Standard & Hybrid families for m ∈ {5,6,7}  (kept as provided)
    for m in m_choices:
        ts, errs, Xf_std = run_standard_m(m)
        runs[f"Standard (m={m})"] = (ts, errs)
        X_final[f"Standard (m={m})"] = Xf_std

        th, eh, Xf_h, (t_sw, e_sw) = run_hybrid_autoswitch_refined_m(
            m, atol=ATOL, rtol=RTOL, dt0=2e-5, dt_min=DT_MIN, dt_max=DT_MAX, refine_depth=1
        )
        runs[f"Hybrid (m={m})"] = (th, eh)
        X_final[f"Hybrid (m={m})"] = Xf_h
        switches[m] = (t_sw, e_sw)

    # Other variants (single curves)
    for name, rhs in [("VAF-ZNN (RK45)", vaf_rhs),
                      ("FT-ZNN (RK45)",  ft_rhs),
                      ("VP-FTZNN (RK45)", vpft_rhs)]:
        ts, errs, Xf = run_generic(rhs)
        runs[name] = (ts, errs)
        X_final[name] = Xf

    # Plot residuals (and save for artifacts)
    os.makedirs("figures", exist_ok=True)
    plt.figure(figsize=(9.2, 5.4))

    # Families (Standard solid, Hybrid dashed)
    for m in m_choices:
        plt.semilogy(*runs[f"Standard (m={m})"], linewidth=2, label=f"Standard (m={m})")
    for m in m_choices:
        th, eh = runs[f"Hybrid (m={m})"]
        plt.semilogy(th, eh, linewidth=2, linestyle="--", label=f"Hybrid (m={m})")
        t_sw, e_sw = switches[m]
        if t_sw is not None:
            plt.plot([t_sw], [e_sw], marker="x", ms=7)

    # Other variants
    for label in ["VAF-ZNN (RK45)", "FT-ZNN (RK45)", "VP-FTZNN (RK45)"]:
        plt.semilogy(*runs[label], linewidth=2, label=label)

    plt.xlabel("Time (s)")
    plt.ylabel(r"$\|A(t)X - I\|_F$  (log scale)")
    plt.title(f"ZNN Variants on 50×50 Ill-Conditioned SPD  (κ≈{kappa:.1e}, RK45)")
    plt.ylim(1e-9, 1e2)
    plt.xlim(t0, tf)
    plt.grid(True, which="both", ls=":")
    plt.legend(ncol=2)
    plt.tight_layout()
    png_path = os.path.join("figures", "bench11_rk45_spd50.png")
    plt.savefig(png_path, dpi=200)
    print(f"[OK] Plot saved: {png_path}")

    # Accuracy at tf against exact inverse A(tf)^{-1} (fast via structure)
    Ainv_tf = apply_A_inv(tf, I)

    # Convergence summary (grid-crossing times) + inv_err(tf)
    os.makedirs("figures", exist_ok=True)
    csv_path = os.path.join("figures", "bench11_rk45_summary.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["label", "t@1e-2", "t@1e-3", "t@1e-4", "inv_err(tf)"])
        for label, (ts, errs) in runs.items():
            hit_1e2 = np.where(errs <= 1e-2)[0]
            hit_1e3 = np.where(errs <= 1e-3)[0]
            hit_1e4 = np.where(errs <= 1e-4)[0]
            t_1e2   = f"{ts[hit_1e2[0]]:.6f}" if hit_1e2.size else "n/a"
            t_1e3   = f"{ts[hit_1e3[0]]:.6f}" if hit_1e3.size else "n/a"
            t_1e4   = f"{ts[hit_1e4[0]]:.6f}" if hit_1e4.size else "n/a"
            inv_err = np.linalg.norm(X_final[label] - Ainv_tf, 'fro')
            w.writerow([label, t_1e2, t_1e3, t_1e4, f"{inv_err:.2e}"])
            print(f"{label:23s} | t@1e-2: {t_1e2:>9}  t@1e-3: {t_1e3:>9}  t@1e-4: {t_1e4:>9}  inv_err(tf): {inv_err:.2e}")

    print("\nHybrid switch markers (refined):")
    for m in m_choices:
        t_sw, e_sw = switches[m]
        print(f"  m={m}: t_switch={t_sw}, residual={e_sw}")

    # Report Stage-1 RK45 cap for transparency
    for m in m_choices:
        alpha_s1 = alpha_star_from_p(p_vec5, m, k_gain)
        print(f"Stage-1 cap for m={m}: dt_max ≤ min(DT_MAX, 2/α*) = min({DT_MAX:g}, {2.0/alpha_s1:.3e})")


# In[ ]:




