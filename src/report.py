from __future__ import annotations
import json
from pathlib import Path


def main(path="results/evaluation.json"):
    data=json.loads(Path(path).read_text(encoding="utf-8"))
    p=data["performance"]; rt=data["runtime_seconds"]; r=data["router"]
    adaptive=p["learned_adaptive_router"]
    baseline=p["expert_only"]
    report=["# Experimental Report", "", "This report is generated from measured runs; no result is hard-coded.", "", "## Performance", ""]
    for name, values in p.items():
        report += [f"### {name}", f"- Accuracy: {values['accuracy']:.4f}", f"- Balanced accuracy: {values['balanced_accuracy']:.4f}", f"- Macro F1: {values['macro_f1']:.4f}", f"- Macro sensitivity: {values['macro_sensitivity']:.4f}", ""]
    report += ["## Adaptive inference", f"- Expert escalation rate: {r['escalation_rate']:.2%}", f"- Lightweight-only measured forward time: {rt['lightweight_all']:.4f}s", f"- Expert-only measured forward time: {rt['expert_all']:.4f}s", f"- Estimated adaptive forward time: {rt['adaptive_estimate']:.4f}s", ""]
    if rt["expert_all"] > 0:
        report += [f"- Estimated compute-time reduction vs expert-only: {(1-rt['adaptive_estimate']/rt['expert_all']):.2%}"]
    report += ["", "## Interpretation", "", "The adaptive system should only be considered an improvement if it reduces computation while maintaining an acceptable sensitivity. Clinical suitability cannot be inferred from this prototype alone."]
    Path("results/REPORT.md").write_text("\n".join(report)+"\n",encoding="utf-8")

if __name__ == "__main__": main()
