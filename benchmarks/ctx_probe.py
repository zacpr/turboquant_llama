#!/usr/bin/env python3
"""
Context-size probe tailored for TurboQuant KV caches.

We sweep increasingly large `-p/--n-prompt` sizes via `llama-bench` for each cache
format (turbo3 vs f16) to expose when the baseline cache runs out of VRAM while
TurboQuant continues to work. Each run's stdout/stderr is persisted alongside a
JSON summary that records the largest successful prompt length.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional


DEFAULT_PROMPTS = [4096, 8192, 12288, 14336, 16384, 24576, 32768, 65536]
DEFAULT_CACHE_TYPES = ["turbo3", "f16"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--llama-bench",
        type=Path,
        default=Path("llama.cpp/build-gcc13/bin/llama-bench"),
        help="Path to llama-bench binary (default: %(default)s).",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("llama.cpp/models/Qwen2.5-7B-Instruct-Q4_K_M.gguf"),
        help="Model to benchmark.",
    )
    parser.add_argument(
        "--cache-types",
        nargs="+",
        default=DEFAULT_CACHE_TYPES,
        choices=DEFAULT_CACHE_TYPES,
        help="Cache formats to compare (default: turbo3 f16).",
    )
    parser.add_argument(
        "--prompt-tokens",
        type=int,
        nargs="+",
        default=DEFAULT_PROMPTS,
        help="Prompt token counts to sweep (ascending).",
    )
    parser.add_argument(
        "--n-gen",
        type=int,
        default=0,
        help="Number of tokens to generate per run (default: 0, prefill-only).",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=512,
        help="Batch size (-b).",
    )
    parser.add_argument(
        "--ubatch",
        type=int,
        default=256,
        help="Micro-batch size (-ub).",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=16,
        help="CPU threads for llama-bench.",
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=1,
        help="Repetitions per benchmark (-r).",
    )
    parser.add_argument(
        "--ngl",
        type=int,
        default=1,
        help="Number of GPU layers (-ngl).",
    )
    flash_group = parser.add_mutually_exclusive_group()
    flash_group.add_argument(
        "--flash-attn",
        dest="flash_attn",
        action="store_true",
        default=True,
        help="Enable FlashAttention (-fa 1).",
    )
    flash_group.add_argument(
        "--no-flash-attn",
        dest="flash_attn",
        action="store_false",
        help="Disable FlashAttention.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=900,
        help="Per-run timeout in seconds.",
    )
    parser.add_argument(
        "--ld-library-path",
        type=str,
        default="/usr/local/cuda/lib64",
        help="Value to prepend to LD_LIBRARY_PATH.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmarks/2026-03-31-qwen2.5-ctx-probe"),
        help="Directory to store per-run logs and summary JSON.",
    )
    return parser.parse_args()


def extend_env(ld_path: Optional[str]) -> Dict[str, str]:
    env = os.environ.copy()
    if ld_path:
        existing = env.get("LD_LIBRARY_PATH")
        env["LD_LIBRARY_PATH"] = f"{ld_path}:{existing}" if existing else ld_path
    env.setdefault("LLAMA_LOG_COLORS", "0")
    return env


def build_cmd(
    llama_bench: Path,
    model: Path,
    cache_type: str,
    prompt_tokens: int,
    args: argparse.Namespace,
) -> List[str]:
    cmd = [
        str(llama_bench),
        "-m",
        str(model),
        "-ctk",
        cache_type,
        "-ctv",
        cache_type,
        "-p",
        str(prompt_tokens),
        "-n",
        str(args.n_gen),
        "-b",
        str(args.batch),
        "-ub",
        str(args.ubatch),
        "-t",
        str(args.threads),
        "-r",
        str(args.repetitions),
        "-ngl",
        str(args.ngl),
        "-o",
        "jsonl",
    ]
    if args.flash_attn:
        cmd.extend(["-fa", "1"])
    return cmd


def parse_avg_ts(stdout: str) -> Optional[float]:
    avg = None
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "avg_ts" in record:
            avg = record["avg_ts"]
    return avg


def run_probe(args: argparse.Namespace) -> Dict[str, List[Dict]]:
    llama_bench = args.llama_bench.resolve()
    model = args.model.resolve()
    env = extend_env(args.ld_library_path)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prompts = sorted(args.prompt_tokens)
    results: Dict[str, List[Dict]] = {}

    for cache_type in args.cache_types:
        cache_runs: List[Dict] = []
        print(f"[probe] cache={cache_type}")
        for prompt_tokens in prompts:
            cmd = build_cmd(llama_bench, model, cache_type, prompt_tokens, args)
            print(f"[probe]  prompt={prompt_tokens} running {' '.join(cmd)}")
            start = time.monotonic()
            try:
                proc = subprocess.run(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    env=env,
                    timeout=args.timeout,
                )
                duration = time.monotonic() - start
                timeout = False
                stdout_payload = proc.stdout
                stderr_payload = proc.stderr
            except subprocess.TimeoutExpired as exc:
                proc = exc  # type: ignore[assignment]
                duration = time.monotonic() - start
                timeout = True
                stdout_payload = exc.output if isinstance(exc.output, str) else ""
                stderr_payload = exc.stderr if isinstance(exc.stderr, str) else ""
            stdout_path = args.output_dir / f"{cache_type}-{prompt_tokens}.jsonl"
            stderr_path = args.output_dir / f"{cache_type}-{prompt_tokens}.log"
            stdout_path.write_text(stdout_payload)
            stderr_path.write_text(stderr_payload)

            avg_ts = parse_avg_ts(stdout_payload)
            returncode = proc.returncode if not timeout else None
            entry = {
                "cache_type": cache_type,
                "prompt_tokens": prompt_tokens,
                "returncode": returncode,
                "duration_s": duration,
                "stdout": stdout_path.name,
                "stderr": stderr_path.name,
                "timeout": timeout,
            }
            if avg_ts is not None:
                entry["avg_ts"] = avg_ts
            cache_runs.append(entry)
            if timeout:
                print(f"[probe]    timeout after {duration:.1f}s")
                break
            if returncode == 0:
                print(f"[probe]    success ({duration:.1f}s, avg_ts={avg_ts})")
            else:
                print(f"[probe]    failed rc={returncode} ({duration:.1f}s)")
                break
        results[cache_type] = cache_runs
    return results


def summarize(results: Dict[str, List[Dict]]) -> Dict[str, Optional[int]]:
    summary: Dict[str, Optional[int]] = {}
    for cache_type, runs in results.items():
        max_prompt = None
        for run in runs:
            if run["returncode"] == 0:
                max_prompt = run["prompt_tokens"]
            else:
                break
        summary[cache_type] = max_prompt
    return summary


def main() -> None:
    args = parse_args()
    results = run_probe(args)
    summary = summarize(results)
    payload = {
        "model": str(args.model.resolve()),
        "llama_bench": str(args.llama_bench.resolve()),
        "ngl": args.ngl,
        "prompt_tokens": sorted(args.prompt_tokens),
        "cache_types": args.cache_types,
        "results": results,
        "max_prompt_tokens": summary,
        "n_gen": args.n_gen,
        "batch": args.batch,
        "ubatch": args.ubatch,
        "threads": args.threads,
        "repetitions": args.repetitions,
        "flash_attn": args.flash_attn,
    }
    out_file = args.output_dir / "ctx_probe_results.json"
    out_file.write_text(json.dumps(payload, indent=2))
    print(f"[probe] wrote {out_file}")
    for cache_type, max_prompt in summary.items():
        print(f"[probe] cache={cache_type} max_prompt={max_prompt}")


if __name__ == "__main__":
    main()
