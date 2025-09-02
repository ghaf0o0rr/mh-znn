# MH-ZNN: A Self-Tuning Heavy-Ball–Augmented Zeroing Neural Network

This repository contains **submission-ready artifacts** for our paper. It includes:
- **Benchmark 1 (RK4, multi-rate)** with ESBP \(m=3/4/5\)
- **Benchmark 11 (RK45, adaptive)** on ill-conditioned SPD \(50\times50,\ \kappa\approx10^8\) with ESBP \(m=5/6/7\)
- **EKF application** (innovation inversion, 3×3 SPD) comparing STD, MOM, FT, VPFT, VAF, IE, DIE, HYBRID

All scripts are **CPU-only NumPy**, save outputs in `figures/`, and print summary tables.

---

## Quick start

### Option A — pip (recommended)
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
