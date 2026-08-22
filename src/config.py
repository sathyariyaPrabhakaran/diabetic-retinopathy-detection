from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
    image_size: int = 224
    batch_size: int = 32
    epochs: int = 5
    seed: int = 42
    lr: float = 2e-4
    weight_decay: float = 1e-4
    min_sensitivity: float = 0.90

    @staticmethod
    def paths(root: str = "."):
        base = Path(root)
        return {
            "data": base / "data" / "retina",
            "models": base / "models",
            "results": base / "results",
        }
