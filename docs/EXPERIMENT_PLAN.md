# Experiment Plan

## Research question
Can learned selective inference reduce expert-model execution while preserving a minimum screening sensitivity?

## Baselines
1. Lightweight-only MobileNetV3-Small
2. Expert-only EfficientNet-B0
3. Fixed confidence router
4. Learned adaptive router

## Ablations
- Remove entropy feature
- Remove probability margin
- Remove learned routing and use confidence only
- Different minimum sensitivity constraints: 0.85, 0.90, 0.95
- Different escalation thresholds

## Primary measurements
- Macro sensitivity (primary safety-oriented metric)
- Macro F1
- Balanced accuracy
- Escalation rate
- Parameters
- Measured inference time
- Estimated adaptive inference time

## Cost analysis
The project treats computational burden as the engineering cost target. A cost-saving claim is only made after measuring the same test set on the same machine/runtime for all compared systems.

## Safety boundary
This is a research prototype. It is not a medical device and must not be presented as a replacement for ophthalmologist diagnosis.
