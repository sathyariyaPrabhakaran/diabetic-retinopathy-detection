# Diabetic Retinopathy Detection

## Cost-Aware Adaptive Medical AI

This repository implements a research-oriented ML system that investigates whether **selective deep inference** can reduce computational cost in diabetic-retinopathy screening while preserving sensitivity.

This is deliberately not a single-model image classifier. The engineering contribution is the **adaptive inference policy**: a lightweight model handles straightforward cases and a learned router escalates difficult cases to a stronger expert model.

### Architecture

```text
Fundus image
    |
    v
MobileNetV3-Small (low-cost model)
    |
    +--> class probabilities
    +--> confidence
    +--> entropy
    +--> probability margin
    |
    v
Learned routing model
    |
    +-----------------------------+
    |                             |
    v                             v
Easy / confident             Difficult / uncertain
    |                             |
    v                             v
Lightweight result          EfficientNet-B0 expert
                                  |
                                  v
                           Final screening result
```

### Why this is higher-level than a conventional classifier

The project evaluates a **cost-performance operating point**, rather than maximizing accuracy alone. The router is trained from validation/calibration outcomes to estimate when expert escalation is useful. Its operating threshold is selected under a minimum sensitivity constraint.

The evaluation compares four systems:

1. **Lightweight-only** — MobileNetV3-Small for every case.
2. **Expert-only** — EfficientNet-B0 for every case.
3. **Fixed routing baseline** — expert escalation from a fixed confidence rule.
4. **Learned adaptive routing** — our learned escalation policy.

Measured outputs include accuracy, balanced accuracy, macro F1, macro sensitivity, expert escalation rate, parameter count, measured forward time, and estimated adaptive inference time.

### Dataset

Use a public diabetic-retinopathy fundus-image dataset under its applicable license/terms. The loader expects class folders:

```text
data/retina/
  0_no_dr/
  1_mild/
  2_moderate/
  3_severe/
  4_proliferative/
```

Any number of class folders >= 2 is supported. Do not commit the dataset to GitHub.

### Local setup

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

Place the dataset under `data/retina/`, then run:

```bash
python src/train.py --data-dir data/retina --epochs 5
python src/report.py
```

Results are written locally to `results/evaluation.json` and `results/REPORT.md`; trained weights remain ignored by Git.

### Research protocol

Before claiming improvement, run and preserve:

- the same train/validation/test split for every model;
- lightweight-only and expert-only baselines;
- fixed-threshold routing baseline;
- learned-router ablation;
- multiple minimum-sensitivity constraints;
- runtime measurement on the same machine/device;
- parameter-count comparison;
- confusion matrices and per-class sensitivity;
- external/generalization testing when a compatible second public dataset is available.

A cost-saving result is only meaningful if sensitivity remains acceptable. **No accuracy or cost-saving percentage is fabricated in this repository.**

### Safety

This is a research prototype and is **not a clinical diagnostic device**. A real clinical deployment would require prospective validation, external validation, regulatory review, and clinical oversight.
