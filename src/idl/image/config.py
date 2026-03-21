from dataclasses import dataclass
from pathlib import Path
import yaml
import json

@dataclass
class ViTConfig:
    model_name: str = "google/vit-base-patch16-224-in21k"
    data_dir: Path = Path(__file__).resolve().parents[3] / "data" / "image"
    train_split: str = "train"
    val_split: str = "validation"
    per_device_batch_size: int = 32
    learning_rate: float = 5e-5
    num_labels: int = 10
    num_epochs: int = 1
    time_limit_seconds: int = 30 * 60
    mode: str = "train"
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    warmup_batches: int = 2

    def update_from_file(self, config: Path):
        with open(config, "r") as f:
            data = yaml.safe_load(f)
        if data is None:
            return
        for key, value in data.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def __str__(self):
        return json.dumps(vars(self), indent=4, default=str)