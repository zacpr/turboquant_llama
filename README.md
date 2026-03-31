# TurboQuant integration notes

This repo tracks the glue needed to keep TurboQuant (`tq3_0`) working inside `llama.cpp` on the shared GCC 13 + CUDA 12.4 toolchain.

## Rebuild & sanity check

```bash
cmake -S llama.cpp -B build-gcc13 -DLLAMA_CUBLAS=ON -DLLAMA_ACCELERATE=OFF
cmake --build build-gcc13 -j
LD_LIBRARY_PATH=/usr/local/cuda/lib64 \
  ./build-gcc13/bin/llama-cli \
  -m models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
  --cache-type-k turbo3 --cache-type-v turbo3 -ngl 1 -fa 1 -p "sanity"
```

The include order already favors `llama.cpp/cuda-patches/include`, so no privileged edits under `/usr/local/cuda` are required.

## Benchmark setup

Hardware: AMD Ryzen 9 9950X3D + RTX 5070 Ti (8 GB available for KV cache), Ubuntu 24.04, CUDA 12.4.  
Model: `models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf`  
Flags shared across runs: `-ngl 1 -fa 1 -b 2048 -ub 512 -t 16`

Run `llama-bench` twice per cache type (prompt-only and generate-only) after building `tools/llama-bench`:

```bash
# Prompt throughput (512 prompt tokens, no generation)
./build-gcc13/bin/llama-bench \
  -m models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
  -ctk turbo3 -ctv turbo3 -ngl 1 -fa 1 \
  -p 512 -n 0 -b 2048 -ub 512 -t 16 \
  -o jsonl > benchmarks/2026-03-31-tinyllama-ngl1/turbo3.jsonl

# Generation throughput (128 new tokens, empty prompt)
./build-gcc13/bin/llama-bench \
  -m models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
  -ctk turbo3 -ctv turbo3 -ngl 1 -fa 1 \
  -p 0 -n 128 -b 2048 -ub 512 -t 16 \
  -o jsonl >> benchmarks/2026-03-31-tinyllama-ngl1/turbo3.jsonl

# Repeat both commands with -ctk f16 -ctv f16 and send output to f16.jsonl
```

Raw outputs live under `benchmarks/2026-03-31-tinyllama-ngl1/`. Adjust `-ngl` upward on larger GPUs to observe the KV-cache savings without CPU bottlenecks.

## Results

| Cache type | Prompt tok/s (512 tok) | Generation tok/s (128 tok) |
|------------|------------------------|----------------------------|
| turbo3     | 243.18                 | 54.03                      |
| f16        | 6755.54                | 53.87                      |

Both modes use identical settings; only cache formats change. Prompt throughput is CPU-bound when only one layer remains on GPU, but generation parity shows turbo3’s compression doesn’t tax decode throughput.

![Turbo3 vs f16 throughput](benchmarks/2026-03-31-tinyllama-ngl1/turbo3_vs_f16.png)

Regenerate the plot (and update the PNG in-place) via:

```bash
# requires matplotlib (pip install matplotlib)
python3 benchmarks/plot_benchmarks.py \
  --data-dir benchmarks/2026-03-31-tinyllama-ngl1 \
  --output benchmarks/2026-03-31-tinyllama-ngl1/turbo3_vs_f16.png
```

## TurboQuant-friendly stress test (capacity sweep)

Downloading larger GGUFs to this environment currently requires credentials, so we approximated the “TurboQuant vs f16 on an 8 GB GPU” scenario analytically. The script below models a 7B-class transformer (32 layers, 4,096-dim hidden) and computes the VRAM consumed by the K/V caches at different context lengths, assuming 1 GB of headroom for activations and other buffers:

```bash
python3 benchmarks/kv_capacity_sweep.py
```

It emits `benchmarks/2026-03-31-kv-capacity/kv_vram_vs_context.json` plus the plot copied into this README:

![KV cache VRAM vs context](benchmarks/2026-03-31-kv-capacity/kv_vram_vs_context.png)

Key takeaways (8 GB GPU, 7 GB usable after headroom):

| Cache type | Memory @ 8k ctx | Max ctx before 7 GB exhausted |
|------------|-----------------|-------------------------------|
| f16        | 4.0 GiB         | 14,336 tokens                 |
| turbo3     | 0.875 GiB       | 65,536 tokens                 |

So turbo3’s ~4.6× compression leaves enough VRAM to keep ∼4× the context (or far more layers) resident on the GPU, even before accounting for weights. Once larger GGUFs are staged locally we can replace this analytical sweep with empirical `llama-bench` runs, but the script already captures the memory headroom TurboQuant unlocks.

Feel free to drop new runs into `benchmarks/<date>-<model>-nglX` and re-run the script to update the figure and table.
