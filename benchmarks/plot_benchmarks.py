#!/usr/bin/env python3
"""
Plot TurboQuant benchmark results stored as llama-bench JSONL files.

Given a directory containing turbo3.jsonl and f16.jsonl, this script
plots prompt and generation throughput for both cache types.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_runs(jsonl_path: Path) -> Dict[str, Dict]:
    """Return a dict with 'prompt' and 'generation' entries from a JSONL file."""
    runs: Dict[str, Dict] = {}
    with jsonl_path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("n_prompt", 0) > 0:
                runs["prompt"] = record
            elif record.get("n_gen", 0) > 0:
                runs["generation"] = record
    return runs


def build_dataset(data_dir: Path) -> Tuple[List[str], Dict[str, List[float]]]:
    turbo3_runs = load_runs(data_dir / "turbo3.jsonl")
    f16_runs = load_runs(data_dir / "f16.jsonl")
    categories = ["prompt", "generation"]
    datasets = {
        "turbo3": [turbo3_runs[cat]["avg_ts"] for cat in categories],
        "f16": [f16_runs[cat]["avg_ts"] for cat in categories],
    }
    return categories, datasets


def plot(categories: List[str], datasets: Dict[str, List[float]], output: Path) -> None:
    width = 0.35
    x = range(len(categories))
    fig, ax = plt.subplots(figsize=(6, 4))

    ax.bar([i - width / 2 for i in x], datasets["turbo3"], width, label="turbo3 (tq3_0)")
    ax.bar([i + width / 2 for i in x], datasets["f16"], width, label="f16")

    ax.set_ylabel("Throughput (tokens/s)")
    ax.set_title("TurboQuant turbo3 vs f16 (tinyllama, ngl=1)")
    ax.set_xticks(list(x))
    ax.set_xticklabels(cat.capitalize() for cat in categories)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    print(f"[plot] wrote {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("benchmarks/2026-03-31-tinyllama-ngl1"),
        help="Directory containing turbo3.jsonl and f16.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/2026-03-31-tinyllama-ngl1/turbo3_vs_f16.png"),
        help="Where to write the PNG plot",
    )
    args = parser.parse_args()

    categories, datasets = build_dataset(args.data_dir)
    plot(categories, datasets, args.output)


if __name__ == "__main__":
    main()
