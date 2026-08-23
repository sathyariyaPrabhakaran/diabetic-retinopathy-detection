from __future__ import annotations

import json
from pathlib import Path


def main():
    path = Path("results/evaluation.json")
    if not path.exists():
        raise FileNotFoundError("results/evaluation.json not found. Run src/train.py first.")

    data = json.loads(path.read_text(encoding="utf-8"))
    perf = data["performance"]
    router = data["router"]
    runtime = data["runtime_seconds"]

    lines = [
        "# Experimental Results",
        "",
        "Generated directly from results/evaluation.json.",
        "",
        "## Dataset",
        f"- Train: {data['counts']['train']}",
        f"- Validation: {data['counts']['validation']}",
        f"- Test: {data['counts']['test']}",
        "",
        "## Model comparison",
        "| System | Accuracy | Balanced accuracy | Macro F1 | Macro sensitivity |",
        "|---|---:|---:|---:|---:|",
    ]
    names = [
        ("Lightweight-only", "lightweight_only"),
        ("Expert-only", "expert_only"),
        ("Fixed-confidence router", "fixed_confidence_router"),
        ("Learned adaptive router", "learned_adaptive_router"),
    ]
    for label, key in names:
        m = perf[key]
        lines.append(f"| {label} | {m['accuracy']:.4f} | {m['balanced_accuracy']:.4f} | {m['macro_f1']:.4f} | {m['macro_sensitivity']:.4f} |")

    lines += [
        "",
        "## Routing and runtime",
        f"- Router threshold: {router['threshold']:.4f}",
        f"- Expert escalation rate: {router['escalation_rate']:.2%}",
        f"- Escalated test images: {router.get('escalated_images', 'recorded by adaptive metrics')}",
        f"- Lightweight-only runtime: {runtime['lightweight_all']['seconds']:.4f}s",
        f"- Expert-only runtime: {runtime['expert_all']['seconds']:.4f}s",
        f"- Adaptive estimated runtime: {runtime['adaptive_estimate']:.4f}s",
    ]

    if runtime["expert_all"]["seconds"] if False else False:
        pass
    expert_seconds = runtime["expert_all"]["seconds"]
    adaptive_seconds = runtime["adaptive_estimate"]
    if expert_seconds > 0:
        reduction = 1 - adaptive_seconds / expert_seconds
        lines.append(f"- Estimated runtime reduction vs expert-only: {reduction:.2%}")

    lines += [
        "",
        "## Clinical/research interpretation",
        "",
        "The system is a research prototype. The current evaluation does not establish clinical safety or diagnostic performance and does not meet the 0.90 macro-sensitivity target. Results should be interpreted as an engineering evaluation of selective inference.",
        "",
    ]

    out = Path("results/FINAL_RESULTS.md")
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
