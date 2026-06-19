# Enhanced SecureDSC — B.Tech Project

**Base paper:** Shi et al., "Secure Transmission in Wireless Semantic Communications With Adversarial Training," *IEEE Communications Letters*, Vol. 29, No. 3, March 2025.

---

## Enhancements Over Base Paper

| # | What | Where in code | Replaces |
|---|------|---------------|----------|
| 1 | CSI-based dynamic key generation | `model.py → CSIKeyGenerator` | Random session key `k` |
| 2 | Adaptive λ scheduler | `model.py → AdaptiveLambdaScheduler` | Fixed hyperparameter `λ=6` |

---

## Project Structure

```
code/
├── model.py      ← Full SecureDSC model + both enhancements
├── train.py      ← Training loop (modified Algorithm 1)
├── evaluate.py   ← BLEU scoring + key agreement rate
└── README.md     ← This file
```

---

## Setup

```bash
# Optional: Create a virtual environment
python -m venv securedsc_env
# Windows: .\securedsc_env\Scripts\activate
# Linux/Mac: source securedsc_env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## Quick Start

### Train (small subset for testing)
```bash
python train.py --epochs 5 --snr 12 --batch_size 32 --dataset_size 500
```

### Train (full dataset)
```bash
python train.py --epochs 100 --snr 12 --batch_size 64 --dataset_size 0
```

### Evaluate
```bash
python evaluate.py --model_path securedsc_enhanced.pt --snr_range 0 3 6 9 12 15
```

---

## Key Design Decisions

### Enhancement 1 — CSI Key Generation

The original paper draws session keys randomly per batch, which requires a secure
out-of-band key exchange. We replace this with **physical layer key generation**:

1. Alice and Bob exchange pilot signals over the wireless channel
2. Both independently estimate the channel response h
3. A shared MLP `f_θ(h)` maps the estimate to a key vector
4. Channel reciprocity (h_AB ≈ h_BA) ensures Alice and Bob get the same key
5. An extra training loss `L_key = MSE(k_A, k_B)` enforces key agreement
6. Eve, at a different physical location, sees h_E ≠ h_AB and cannot reproduce the key

**New metric:** Key Agreement Rate (KAR) — fraction of trials where quantised
keys match. Target: KAR > 95% at 12 dB SNR.

### Enhancement 2 — Adaptive λ Scheduler

The original paper uses a fixed `λ=6` in the adversarial joint loss:

```
L_joint = L_Bob + |L_Eve − λ|
```

This is brittle — the right λ depends on the SNR regime and training phase.
We replace it with a sign-based control loop that runs **once per epoch**:

```
gap        = L_Eve − L_Bob           # current separation
direction  = +1 if gap < target_gap else −1
λ(t+1)    = clip(λ(t) + η·direction, λ_min, λ_max)
```

This removes the need for manual hyperparameter tuning and keeps training stable
across different SNR values without any retuning.

---

## Expected Results (from base paper, for comparison)

| Metric | Base SecureDSC | Enhanced SecureDSC (expected) |
|--------|----------------|-------------------------------|
| Bob BLEU-1 @ 15 dB | 0.97 | ≥ 0.97 |
| Eve BLEU-1 (all SNR) | < 0.20 | < 0.20 |
| Key source | Random | Physically derived (CSI) |
| λ tuning | Manual (λ=6) | Automatic |
| KAR @ 12 dB | N/A | > 95% (target) |

---

## References

1. Shi et al. (2025), IEEE Comm. Letters — base paper
2. Vaswani et al. (2017), "Attention Is All You Need" — Transformer backbone
3. Papineni et al. (2002), ACL — BLEU metric
4. Xie et al. (2021), IEEE Trans. Signal Process. — DeepSC baseline
