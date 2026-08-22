# Diabetic Retinopathy Detection

## Cost-Aware Adaptive Medical AI

A research-oriented machine-learning system for diabetic-retinopathy screening that investigates whether expensive deep-model inference can be reduced through learned adaptive routing while preserving screening sensitivity.

### Research problem

A conventional pipeline sends every retinal image through the same deep model. This project instead studies **selective inference**: a lightweight model handles easy cases, while a learned router escalates uncertain or difficult cases to an expert model.

```text
Fundus image
     |
     v
Lightweight model
     |
     +--> routing features
     |      - confidence
     |      - entropy
     |      - probability margin
     |      - image-quality signal
     |
     v
Learned adaptive router
     |
     +--> easy ----------------> lightweight prediction
     |
     +--> difficult -----------> expert model
                                      |
                                      v
                               final prediction
```

### Models

- Lightweight model: MobileNetV3-Small
- Expert model: EfficientNet-B0
- Router: Logistic Regression over uncertainty/routing features
- Operating threshold: selected on a validation split under a minimum sensitivity constraint

### Evaluation

The implementation compares:

1. Lightweight-only inference
2. Expert-only inference
3. Fixed confidence-style routing baseline
4. Learned adaptive routing

Metrics include accuracy, balanced accuracy, macro F1, macro sensitivity, escalation rate, inference time, parameter count, and the computation/performance trade-off.

**No performance or cost-saving percentage is claimed before experiments are run.**

### Dataset

Use a public diabetic-retinopathy fundus-image dataset according to its license and terms. Place images in class folders under `data/retina/`.

```text
data/retina/
  0_no_dr/
  1_mild/
  2_moderate/
  3_severe/
  4_proliferative/
```

The loader also supports a different number of class folders.

### Status

Research implementation in progress. Dataset selection, preprocessing, training, baseline experiments, routing experiments, ablation studies, and external/generalization evaluation will be completed before drawing conclusions.

> This is a research prototype and is not a clinical diagnostic device.
