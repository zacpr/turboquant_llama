# Copilot Instructions for TurboQuant Specs

## Build, Test, and Lint
- This repository is documentation-only; implement TurboQuant inside llama.cpp or the referenced forks (0xSero/turboquant, TheTom/turboquant_plus, RecursiveIntell/turbo-quant) and run their standard toolchain.
- Build llama.cpp with CUDA support before integrating: `cmake -S . -B build -DLLAMA_CUBLAS=ON && cmake --build build -j`.
- Validate a single KV-cache round trip by running `./build/bin/main -m <model> --cache-type-k turbo3 --cache-type-v turbo3 -ngl 1 -p "sanity"`; this exercises the rotation and quantization path without the full suite.
- Reuse upstream formatting/lint targets (for example `cmake --build build --target format`) so GGML/CUDA files stay consistent with the parent project.

## High-Level Architecture
- TurboQuant applies a two-stage pipeline to llama.cpp’s KV cache: a randomized FWHT rotation that forces each 32-value chunk into a predictable Beta distribution, followed by a 3-bit Lloyd–Max scalar quantizer that maps to precomputed centroids.
- Quantized data uses `block_tq3_0`, packing 32 elements into 14 bytes (16-bit scale plus 12-byte payload) to achieve ~5.3× compression without exceeding 16 GB VRAM.
- Integration work spans both CPU and CUDA: register `GGML_TYPE_TQ3_0` in `ggml.h`, wire its metadata in `ggml.c`, add `dequantize_block_tq3_0` kernels in `ggml-cuda.cu`, and expose `--cache-type-{k,v}=turbo3` via `common/arg.cpp`.

## Key Conventions and Constraints
- FWHT kernels must stay warp-local and use `__shfl_xor_sync` butterfly passes plus `1/sqrt(32)` normalization to avoid shared-memory stalls on Blackwell SM 10.x GPUs.
- Always load packed KV blocks with 128-bit transactions (`LDG.E.128`) to saturate the 5070 Ti’s GDDR7 bandwidth; falling back to 32-bit loads reintroduces the FWHT bottleneck.
- Keep the block shape immutable (32 values, 3.5 bits per weight including scale); any change requires updating the ggml type traits and all CUDA kernels together.
- Rotation randomness should match the reference implementations so the quantizer continues to see the expected Beta distribution and centroid table.
- CLI hooks must accept `turbo3` for both `--cache-type-k` and `--cache-type-v`; mismatched types leave stale cache entries and invalidate throughput numbers.

## Workflow Expectations
- Maintain a living todo list for every TurboQuant integration item (type registration, CUDA kernels, CLI plumbing, tests, etc.). Track it in this repository or an adjacent session database so Copilot agents can query progress quickly.
- Before starting a task, add a todo entry with enough context (file paths, dependencies). Mark it `in_progress` while working, then `done` when complete.
- When closing a todo, append a short journal entry (1–3 sentences) capturing what changed, any caveats, and insights that future tasks should reuse.
