
from dataclasses import dataclass, field
from pathlib import Path
import os
import tempfile
import yaml
import json

from accelerate import ProfileKwargs, Accelerator

@dataclass
class ProfileConfig:
    profile: bool = False
    activities: list[str] = None
    profile_memory: bool = True
    record_shapes: bool = True
    schedule_option: dict = None
    trace_payloads: list[dict] = field(default_factory=list, init=False, repr=False)

    def update_from_file(self, config: Path):
        with open(config, "r") as f:
            data = yaml.safe_load(f)
        if data is None:
            return
        for key, value in data.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def __str__(self):
        return json.dumps(
            {k: v for k, v in vars(self).items() if k != "trace_payloads"},
            indent=4,
            default=str,
        )

    def trace_handler(self, p):
        with tempfile.NamedTemporaryFile(mode="w+", suffix=".json", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            p.export_chrome_trace(tmp_path)
            with open(tmp_path, "r") as f:
                trace_json = f.read()

            self.trace_payloads.append(
                {
                    "step_num": getattr(p, "step_num", None),
                    "chrome_trace_json": trace_json,
                }
            )
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def consume_trace_payloads(self) -> list[dict]:
        payloads = list(self.trace_payloads)
        self.trace_payloads = []
        return payloads

    def generate_profile_kwargs(self) -> ProfileKwargs | None:
        if not self.profile:
            return None
        return ProfileKwargs(
            activities=self.activities,
            profile_memory=self.profile_memory,
            record_shapes=self.record_shapes,
            schedule_option=self.schedule_option,
            on_trace_ready=self.trace_handler,
        )

    def accelerator(self) -> Accelerator:
        kwargs_handlers = []
        profile_kwargs = self.generate_profile_kwargs()
        if profile_kwargs is not None:
            kwargs_handlers.append(profile_kwargs)

        activities = self.activities or []
        normalized = {str(a).lower() for a in activities}
        force_cpu = "cpu" in normalized and "cuda" not in normalized

        if force_cpu:
            return Accelerator(kwargs_handlers=kwargs_handlers, cpu=True)
        return Accelerator(kwargs_handlers=kwargs_handlers)


def write_to_file(data: dict, path: Path | str, accelerator: Accelerator | None = None):
    """Dump stats and traces to output directory.
    
    When running multi-GPU, only the main process writes.
    """
    if accelerator is not None and not accelerator.is_main_process:
        return

    out_dir = Path(path)
    out_dir.mkdir(parents=True, exist_ok=True)

    stats = data.get("stats", {})
    traces = data.get("traces", {}).get("chrome", [])

    stats_file = out_dir / "stats.json"
    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=4, default=str)

    if traces:
        traces_dir = out_dir / "traces"
        traces_dir.mkdir(parents=True, exist_ok=True)
        for idx, trace in enumerate(traces):
            step = trace.get("step_num")
            if step is None:
                trace_file = traces_dir / f"trace_{idx}.json"
            else:
                trace_file = traces_dir / f"trace_step_{step}.json"

            with open(trace_file, "w") as f:
                f.write(trace.get("chrome_trace_json", ""))
