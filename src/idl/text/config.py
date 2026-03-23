from dataclasses import dataclass
from pathlib import Path
import yaml
import json


@dataclass
class MarianMTConfig:
    model_name: str = "Helsinki-NLP/opus-mt-en-es"
    dataset_name: str = "Iker/OpenHermes-2.5-English-Spanish"
    raw_data_dir: Path = Path(__file__).resolve().parents[3] / "data" / "text"
    src_lang_col: str = "en"
    tgt_lang_col: str = "es"
    max_seq_len: int = 256
    per_device_batch_size: int = 32
    learning_rate: float = 5e-5
    num_epochs: int = 1
    time_limit_seconds: int = 60 * 60
    mode: str = "train"
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    warmup_batches: int = 2
    warmup_steps: int = 500
    max_grad_norm: float = 1.0
    label_smoothing: float = 0.1
    checkpoint_dir: str = "models"
    bleu_eval_samples: int = 2000

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
