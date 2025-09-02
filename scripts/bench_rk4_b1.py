#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python3
# ============================================================================
# MH-ZNN Benchmark 1 — paper-aligned (ESBP m=3/4/5, multi-rate RK4)
#
# What this script does (matches the methodology/bench protocol):
# - ESBP orders m ∈ {3,4,5} drawn from a length-5 p_vec
# - Runs Standard ZNN and Hybrid MH-ZNN (single switch) for each m
# - Fixed-step RK4:
#     • Baseline Std ZNN uses Δt
#     • Hybrid Stage-1 momentum uses Δt₁ = 2Δt   (multi-rate)
#     • Hybrid Stage-2 ZNN uses Δt₂ = Δt
# - Stage-1 momentum self-tunes α via α_base = 4 k φ_sur (entry-scale surrogate)
#   and then applies the discrete RK4 stability cap:
#       α ≤ min{ cR/h,  cI² / (k φ_max h²) }   with (cR,cI)=(2.5,2.5), h=Δt₁
#   Here φ_max uses the same entry-scale surrogate region as φ_min (paper practice).
# - Single guarded switch on residual slope flip with absolute hysteresis τ_abs
#   and 3-point refinement {prev, mid, end}; picks the min residual; one-way switch.
# - Reports first grid-crossing times for R(t)=||A(t)X−I||_F at thresholds (e.g., 1e-2)
# - No momentum-only baseline in outputs (momentum lives only inside Stage-1).
# ============================================================================

import numpy as np
import matplotlib.pyplot as plt
import time

# ───────────────  Hyper-parameters  ───────────────
k_gain = 1.0
t0, tf = 0.0, 1.0

# Base fixed RK4 step; Hybrid is multi-rate per the benchmark protocol
DT = 1e-5
dt_standard  = DT
dt_hybrid_s1 = 2 * DT   # Stage-1 momentum step (multi-rate)
dt_hybrid_s2 = DT       # Stage-2 ZNN step

# Switching guard (absolute hysteresis) and dwell in samples
tau_abs     = 1e-6
dwell_steps = 1

# Discrete RK4 caps (Prop. 14 / eqs. 11–12)
cR = 2.5
cI = 2.5

# Plot controls
DO_PLOT = True

# ───────────────  Problem definition (2×2 rotation path)  ───────────────
def A_func(t):
    return np.array([[ np.sin(4*t), -np.cos(4*t)],
                     [ np.cos(4*t),  np.sin(4*t)]], dtype=float)

def Adot_func(t):
    return 4*np.array([[ np.cos(4*t),  np.sin(4*t)],
                       [-np.sin(4*t),  np.cos(4*t)]], dtype=float)

n = 2
I = np.eye(n, dtype=float)

# ───────────────  ESBP exponents (length-5); m picks the first m entries ───────────────
# Middle entries must lie in (0,1)
p_vec5 = np.array([0.25, 0.50, 0.50, 0.75, 0.75], dtype=float)
M_LIST = [3, 4, 5]

# ───────────────  ESBP activation (entrywise)  ───────────────
def esbp(u, p_sub):
    s  = np.sign(u)
    au = np.abs(u)
    a = (au[..., None] ** p_sub).sum(-1)
    b = (au[..., None] ** (1.0/p_sub)).sum(-1)
    return s * (a + b)

# ───────────────  Practical φ surrogates at entry scale (E(0)=-I ⇒ |u|≈1) ───────────────
def phi_surrogate(p_sub):
    # H'(1) = sum(p) + sum(1/p)  (used as φ_min surrogate and φ_max for caps)
    return float(p_sub.sum() + (1.0/p_sub).sum())

def alpha_with_rk4_cap(alpha_base, k, phi_max, h, cR=2.5, cI=2.5):
    cap = min(cR / h, (cI**2) / (k * phi_max * h * h))
    return min(alpha_base, cap), cap

# ───────────────  Residual (Frobenius)  ───────────────
def err_norm(t, X):
    return np.linalg.norm(A_func(t) @ X - I, ord='fro')

# ───────────────  RHS factories (depend on p_sub) ───────────────
def make_znn_rhs(p_sub):
    def znn_rhs(t, X):
        At, Adt = A_func(t), Adot_func(t)
        Ax = At @ X
        rhs = -Adt @ X - k_gain * esbp(Ax - I, p_sub)
        return np.linalg.solve(At, rhs)  # no explicit inverse
    return znn_rhs

def make_momentum_rhs(p_sub, alpha):
    def momentum_rhs(t, X, V):
        At, Adt = A_func(t), Adot_func(t)
        Ax = At @ X
        v_tgt = np.linalg.solve(At, -Adt @ X - k_gain * esbp(Ax - I, p_sub))
        dX = V
        dV = alpha * (v_tgt - V)
        return dX, dV
    return momentum_rhs

# ───────────────  RK4 steppers  ───────────────
def rk4_step_znn(t, X, dt, znn_rhs):
    k1 = znn_rhs(t, X)
    k2 = znn_rhs(t + 0.5*dt, X + 0.5*dt*k1)
    k3 = znn_rhs(t + 0.5*dt, X + 0.5*dt*k2)
    k4 = znn_rhs(t + dt,     X + dt*k3)
    return X + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)

def rk4_step_mom(t, X, V, dt, mom_rhs):
    k1_X, k1_V = mom_rhs(t,               X,               V)
    k2_X, k2_V = mom_rhs(t + 0.5*dt, X + 0.5*dt*k1_X, V + 0.5*dt*k1_V)
    k3_X, k3_V = mom_rhs(t + 0.5*dt, X + 0.5*dt*k2_X, V + 0.5*dt*k2_V)
    k4_X, k4_V = mom_rhs(t + dt,     X + dt*k3_X,     V + dt*k3_V)
    Xn = X + (dt/6.0)*(k1_X + 2*k2_X + 2*k3_X + k4_X)
    Vn = V + (dt/6.0)*(k1_V + 2*k2_V + 2*k3_V + k4_V)
    return Xn, Vn

# ───────────────  Solvers for a given p_sub  ───────────────
def run_standard(p_sub):
    znn_rhs = make_znn_rhs(p_sub)
    t = t0
    X = np.zeros((n, n), dtype=float)
    T = [t]
    E = [err_norm(t, X)]
    while t < tf - 1e-15:
        X = rk4_step_znn(t, X, dt_standard, znn_rhs)
        t += dt_standard
        T.append(t); E.append(err_norm(t, X))
    return np.array(T), np.array(E)

def run_hybrid_autoswitch(p_sub):
    """
    Stage 1: momentum RK4 with α = min{ 4k φ_sur , RK4-cap } until first refined residual
             minimum, detected by slope flip with absolute hysteresis + dwell.
             Sub-step refinement picks best of {prev, mid, end}. Then single switch.
    Stage 2: standard ZNN RK4.
    """
    # α* with cap
    phi_sur = phi_surrogate(p_sub)
    alpha_base = 4.0 * k_gain * phi_sur
    alpha_star, alpha_cap = alpha_with_rk4_cap(alpha_base, k_gain, phi_sur, dt_hybrid_s1, cR=cR, cI=cI)
    mom_rhs = make_momentum_rhs(p_sub, alpha_star)
    znn_rhs = make_znn_rhs(p_sub)

    # -- Stage 1 --
    t = t0
    X = np.zeros((n, n), dtype=float)
    V = np.zeros((n, n), dtype=float)

    T = [t]
    E = [err_norm(t, X)]
    t_switch = None
    e_switch = None
    last_decision_idx = 0

    X_prev, V_prev, t_prev, e_prev = X.copy(), V.copy(), t, E[-1]

    while t < tf - 1e-15:
        # advance momentum step
        X_prev, V_prev, t_prev, e_prev = X, V, t, E[-1]
        X, V = rk4_step_mom(t, X, V, dt_hybrid_s1, mom_rhs)
        t   += dt_hybrid_s1
        e_cur = err_norm(t, X)
        T.append(t); E.append(e_cur)

        # slope flip with *absolute* hysteresis + dwell
        if len(E) >= 3:
            g_prev = (E[-2] - E[-3]) / dt_hybrid_s1
            g_curr = (E[-1] - E[-2]) / dt_hybrid_s1
            dwell_ok = (len(E) - last_decision_idx) >= dwell_steps

            if (g_prev <= -tau_abs) and (g_curr >= +tau_abs) and dwell_ok:
                # sub-step refinement on [t_prev, t_prev+dt]
                best_t, best_X, best_V, best_e = t_prev, X_prev.copy(), V_prev.copy(), e_prev
                dt_half = 0.5 * dt_hybrid_s1

                # midpoint
                X_mid, V_mid = rk4_step_mom(t_prev, X_prev, V_prev, dt_half, mom_rhs)
                e_mid = err_norm(t_prev + dt_half, X_mid)
                if e_mid < best_e:
                    best_t, best_X, best_V, best_e = t_prev + dt_half, X_mid, V_mid, e_mid

                # end
                X_end, V_end = rk4_step_mom(t_prev + dt_half, X_mid, V_mid, dt_half, mom_rhs)
                e_end = err_norm(t_prev + dt_hybrid_s1, X_end)
                if e_end < best_e:
                    best_t, best_X, best_V, best_e = t_prev + dt_hybrid_s1, X_end, V_end, e_end

                # land exactly at best_t (overwrite last sample)
                T.pop(); E.pop()
                if T[-1] < best_t - 1e-18:
                    T.append(best_t); E.append(best_e)
                else:
                    E[-1] = best_e

                X, V, t = best_X, best_V, best_t
                t_switch, e_switch = best_t, best_e
                last_decision_idx = len(E)
                break

    # -- Stage 2 (standard ZNN) --
    while t < tf - 1e-15:
        X  = rk4_step_znn(t, X, dt_hybrid_s2, znn_rhs)
        t += dt_hybrid_s2
        T.append(t); E.append(err_norm(t, X))

    meta = {
        "alpha_base": alpha_base,
        "alpha_cap":  alpha_cap,
        "alpha_used": alpha_star,
        "phi_sur":    phi_sur,
        "h_stage1":   dt_hybrid_s1,
        "t_switch":   t_switch,
        "e_switch":   e_switch
    }
    return np.array(T), np.array(E), meta

# ───────────────  Utilities  ───────────────
def grid_first_crossing_time(t, e, thresh=1e-2):
    idx = np.where(e <= thresh)[0]
    return float(t[idx[0]]) if idx.size else None

# ───────────────  Run modes for m=3/4/5  ───────────────
def main():
    results = {}   # dict[(method, m)] -> (T,E)
    walls   = {}   # dict[(method, m)] -> wall time
    switches = {}  # dict[m] -> (t_switch, e_switch)
    alphas   = {}  # dict[m] -> (alpha_base, alpha_cap, alpha_used, phi_sur, h_stage1)

    for m in M_LIST:
        p_sub = p_vec5[:m]

        # Standard
        t0_ = time.time()
        T_std, E_std = run_standard(p_sub)
        walls[("Std", m)] = time.time() - t0_
        results[("Std", m)] = (T_std, E_std)

        # Hybrid
        t0_ = time.time()
        T_hyb, E_hyb, meta = run_hybrid_autoswitch(p_sub)
        walls[("Hybrid", m)] = time.time() - t0_
        results[("Hybrid", m)] = (T_hyb, E_hyb)
        switches[m] = (meta["t_switch"], meta["e_switch"])
        alphas[m]   = meta

    # ───────── Plot ─────────
    if DO_PLOT:
        plt.figure(figsize=(10, 6))

        colors = {3: None, 4: None, 5: None}
        # draw Std (dashed) then Hybrid (solid) for each m
        for m in M_LIST:
            T_std, E_std = results[("Std", m)]
            line_std, = plt.semilogy(T_std, E_std, ls="--", lw=2, label=f"Std ESBP m={m}")
            colors[m] = line_std.get_color()  # reuse color for Hybrid

        for m in M_LIST:
            T_hyb, E_hyb = results[("Hybrid", m)]
            plt.semilogy(T_hyb, E_hyb, ls="-", lw=2, color=colors[m], label=f"Hybrid ESBP m={m}")
            t_sw, e_sw = switches[m]
            if t_sw is not None:
                plt.plot([t_sw], [e_sw], marker="o", ms=5, color=colors[m])

        plt.xlabel("Time (s)")
        plt.ylabel(r"Residual  $\|A(t)X-I\|_F$ (log scale)")
        plt.title("MH-ZNN Benchmark 1 — Fixed-step RK4 (ESBP m=3/4/5), multi-rate Hybrid")
        plt.ylim(1e-9, 1e2)
        plt.xlim(t0, tf)
        plt.grid(True, which="both", ls=":")
        plt.legend()
        plt.tight_layout()
        plt.show()

    # ───────── Summary ─────────
    print("\nGrid-based times to reach residual ≤ 1e-2:")
    for m in M_LIST:
        for method in ("Std", "Hybrid"):
            tarr, earr = results[(method, m)]
            t_to = grid_first_crossing_time(tarr, earr, 1e-2)
            print(f"{method:6s}  m={m}  t@1e-2 = {t_to}   wall = {walls[(method, m)]:.4f}s")
        t_sw, e_sw = switches[m]
        meta = alphas[m]
        print(f"  Hybrid m={m}: switch at t≈{t_sw}  (res≈{e_sw:.3e}),  "
              f"α_base={meta['alpha_base']:.6f}, cap={meta['alpha_cap']:.6f}, used={meta['alpha_used']:.6f}")

    print(f"\nStd step Δt = {DT:g}; Hybrid steps Δt₁ = {dt_hybrid_s1:g}, Δt₂ = {dt_hybrid_s2:g}")
    print(f"p_vec5 = {p_vec5.tolist()}  (ESBP uses the first m entries per run)")
    print(f"RK4 caps with (cR,cI)=({cR},{cI}). k = {k_gain}")

if __name__ == "__main__":
    main()


# In[ ]:




