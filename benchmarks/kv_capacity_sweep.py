#!/usr/bin/env python3
"""
Simulate KV-cache memory usage for TurboQuant vs f16.

We assume a llama-style decoder-only transformer and estimate how much VRAM
is consumed by the K/V caches at different context lengths.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass
class ModelCfg:
    name: str
    n_layers: int
    d_model: int
    available_vram_gb: float
    reserve_gb: float = 1.0  # leave headroom for activations/other buffers

    @property
    def usable_vram_bytes(self) -> float:
        return (self.available_vram_gb - self.reserve_gb) * (1024**3)


BYTES_PER_VALUE = {
    "f16": 2.0,  # 16-bit float per value
    "tq3_0": 14.0 / 32.0,  # TurboQuant packs 32 values into 14 bytes
}


def kv_bytes(n_ctx: int, cfg: ModelCfg, bytes_per_value: float) -> float:
    """Bytes for both K and V caches at a given context length."""
    return n_ctx * cfg.d_model * cfg.n_layers * bytes_per_value * 2


def max_ctx(cfg: ModelCfg, bytes_per_value: float) -> int:
    """Largest n_ctx that fits into usable VRAM."""
    return int(cfg.usable_vram_bytes / (cfg.d_model * cfg.n_layers * bytes_per_value * 2))


def sweep(cfg: ModelCfg, ctx_values: List[int]) -> Dict[str, List[float]]:
    data: Dict[str, List[float]] = {"f16": [], "tq3_0": []}
    for dtype, bpf in BYTES_PER_VALUE.items():
        for ctx in ctx_values:
            data[dtype].append(kv_bytes(ctx, cfg, bpf) / (1024**3))
    return data


def plot(ctx_values: List[int], data: Dict[str, List[float]], cfg: ModelCfg, output: Path) -> None:
    plt.figure(figsize=(7, 4))
    for dtype, label in [("tq3_0", "TurboQuant (tq3_0)"), ("f16", "Baseline f16")]:
        plt.plot(ctx_values, data[dtype], marker="o", label=label)
    plt.axhline(cfg.usable_vram_bytes / (1024**3), color="red", linestyle="--", label="Usable VRAM")
    plt.title(f"KV cache VRAM vs context — {cfg.name}")
    plt.xlabel("Context length (tokens)")
    plt.ylabel("Memory (GiB)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output)
    print(f"[plot] wrote {output}")


def write_table(ctx_values: List[int], data: Dict[str, List[float]], cfg: ModelCfg, path: Path) -> None:
    payload = {
        "model": cfg.__dict__,
        "ctx_values": ctx_values,
        "memory_gib": data,
        "max_ctx": {dtype: max_ctx(cfg, bpf) for dtype, bpf in BYTES_PER_VALUE.items()},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
    print(f"[data] wrote {path}")


def main() -> None:
    cfg = ModelCfg(name="Llama 7B (d=4096, L=32)", n_layers=32, d_model=4096, available_vram_gb=8.0, reserve_gb=1.0)
    ctx_values = [512, 1024, 2048, 4096, 8192, 12288, 16384, 24576, 32768]
    data = sweep(cfg, ctx_values)
    out_dir = Path("benchmarks/2026-03-31-kv-capacity")
    plot(ctx_values, data, cfg, out_dir / "kv_vram_vs_context.png")
    write_table(ctx_values, data, cfg, out_dir / "kv_vram_vs_context.json")


if __name__ == "__main__":
    main()
