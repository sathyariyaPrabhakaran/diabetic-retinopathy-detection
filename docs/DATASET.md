# Dataset Setup

This repository intentionally does not commit medical datasets or pretrained weights.

## Expected structure

```text
data/retina/
  class_a/
    image1.jpg
  class_b/
    image2.jpg
```

For a five-grade diabetic-retinopathy dataset, folders can represent:

- no DR
- mild
- moderate
- severe
- proliferative DR

The loader discovers class names automatically.

## Rules

- Use only a dataset whose license permits the intended research use.
- Keep patient-level duplicates out of different splits when patient identifiers are available.
- Do not use the test set for router threshold selection.
- Report class imbalance and split counts.
- Keep the external/generalization dataset completely separate from training and calibration.
