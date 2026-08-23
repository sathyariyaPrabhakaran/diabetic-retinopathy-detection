# Project Results — Cost-Aware Adaptive Diabetic Retinopathy Screening

## System

The system performs five-class diabetic-retinopathy classification using a two-stage inference policy:

1. MobileNetV3-Small lightweight model processes every image.
2. A learned router evaluates the lightweight prediction.
3. Only uncertain/risky cases are escalated to the EfficientNet-B0 expert model.

## Dataset split

| Split | Images |
|---|---:|
| Train | 2,562 |
| Validation | 549 |
| Test | 551 |

### Training distribution

| Class | Count |
|---|---:|
| No DR | 1,263 |
| Mild | 259 |
| Moderate | 699 |
| Severe | 135 |
| Proliferative | 206 |

## Model size

| Model | Parameters |
|---|---:|
| Lightweight | 1,522,981 |
| Expert | 4,013,953 |

## Adaptive routing

The final saved evaluation selected a routing threshold of approximately **0.21** and escalated **298 of 551 test images (54.08%)** to the expert model.

The adaptive policy therefore demonstrates selective inference rather than sending every test image through the more expensive model.

## Test performance

The saved evaluation reported:

- Lightweight-only accuracy: **77.50%**
- Expert-only accuracy: **75.50%**
- Fixed-confidence router accuracy: **76.23%**
- Learned adaptive router accuracy: **75.50%**
- Learned adaptive router macro F1: **0.6094**
- Learned adaptive router macro sensitivity: **0.6533**

The expert model's measured full-test forward time was approximately **30.90 seconds**, while the adaptive estimate was approximately **23.35 seconds** for the 551-image test set.

## Interpretation

The main contribution is the adaptive inference architecture and its cost/sensitivity evaluation, not a claim of clinical superiority. The current experiment does **not** demonstrate the requested 0.90 macro-sensitivity target, so the project should not claim that target was achieved.

The current result supports a research finding that selective routing can reduce estimated inference time relative to expert-only execution on the evaluated machine, while the accuracy/sensitivity trade-off remains an important limitation.

## Limitations

- The experiment uses an APTOS-derived labeled dataset organized into five classes.
- Evaluation is on a single held-out split.
- The router does not yet meet the 0.90 macro-sensitivity target.
- CPU runtime is hardware-dependent.
- No clinical deployment or diagnostic claim is made.
- External validation on an independent dataset is still required for stronger generalization evidence.

## Reproducibility

Run:

```powershell
git pull origin main
python src\train.py --data-dir data\retina
python src\report.py
streamlit run app.py
```

The trained model artifacts are intentionally kept out of source control where configured by `.gitignore`.
