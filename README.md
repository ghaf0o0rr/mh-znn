# MH-ZNN: A Self-Tuning Heavy-Ball–Augmented Zeroing Neural Network

This repository contains **submission-ready artifacts** for the paper. It includes:

- **Benchmark 1 (RK4, multi-rate)** — ESBP \(m=3/4/5\)
- **Benchmark 11 (RK45, adaptive)** — ill-conditioned SPD \(50\times50,\ \kappa\approx10^8\) with ESBP \(m=5/6/7\)
- **EKF application** — innovation inversion (3×3 SPD) comparing: `STD, MOM, FT, VPFT, VAF, IE, DIE, HYBRID`

All scripts are **CPU-only (NumPy)**. Outputs (plots/CSVs) are written to `figures/` where applicable, and key summaries are printed to the console.

---

## TL;DR — Reproduce results

```bash
# create & activate a virtual environment (pip option)
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# install dependencies
pip install -r requirements.txt

# run artifacts
make bench1     # Benchmark 1: RK4 (multi-rate), ESBP m=3/4/5
make bench11    # Benchmark 11: RK45 (adaptive), SPD 50x50 (κ≈1e8), ESBP m=5/6/7
make ekf        # EKF application: 3x3 innovation inversion suite
