#!/usr/bin/env python

from idl.text.config import MarianMTConfig
from idl.text.marianmt import MarianMT
from idl.accelerate import ProfileConfig, write_to_file
import argparse
from datetime import datetime


def parse_args():
    """Parse command-line arguments for the MarianMT benchmark script."""
    parser = argparse.ArgumentParser(description="MarianMT inference/training benchmark with Accelerate")
    parser.add_argument("--model", type=str, help="Path to the model config YAML file", default=None)
    parser.add_argument("--trace", type=str, help="Path to the profile config YAML file", default=None)
    parser.add_argument("--output", type=str, help="Path to the output directory", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    model_config = MarianMTConfig()
    profile_config = ProfileConfig()

    if args.model:
        model_config.update_from_file(args.model)
    if args.trace:
        profile_config.update_from_file(args.trace)

    # Resolve output directory.
    output_dir = args.output
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"results/{timestamp}"

    mt = MarianMT(config=model_config, profile=profile_config)

    # Print resolved configs on main process.
    if mt.accelerator is None or mt.accelerator.is_main_process:
        print(f"Model config:\n{model_config}")
        print(f"Profile config:\n{profile_config}")
        print(f"Output directory: {output_dir}")

    stats = mt.run()
    write_to_file(stats, output_dir, accelerator=mt.accelerator)
