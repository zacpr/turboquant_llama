#implementation.md

TurboQuant (TQ3) Integration Spec for llama.cpp
1. Core Architecture

TurboQuant on llama.cpp uses a two-stage pipeline to compress KV cache by ~5.3x.

    Stage 1 (Rotation): A randomized Fast Walsh-Hadamard Transform (FWHT) rotates the KV vectors to normalize their distribution into a predictable Beta distribution.

    Stage 2 (Quantization): A 3-bit Lloyd-Max scalar quantizer maps the rotated values to pre-calculated optimal centroids.

2. Target Repositories for Reference

If the CLI agent needs to "read" existing implementations to port them:

    Primary Reference (Python/Triton): 0xSero/turboquant (The most stable ICLR 2026 implementation).

    Llama.cpp Fork (Metal/Apple Silicon): TheTom/turboquant_plus (Already has the --cache-type-k turbo3 CLI logic; needs porting from Metal to CUDA).

    Standalone Rust: RecursiveIntell/turbo-quant (Best for the pure math logic of the rotation).

3. Data Structure (ggml.h)

To minimize VRAM on your 16GB card, we pack 32 elements into a 14-byte block.
C++

// 3.5 bits per weight including scale
typedef struct {
    ggml_fp16_t d;       // 16-bit Scale (2 bytes)
    uint8_t qs[12];      // 32 weights @ 3-bits packed (12 bytes)
} block_tq3_0; 

4. CUDA Implementation Details (ggml-cuda.cu)

For your 5070 Ti (GDDR7), the bottleneck is the FWHT rotation. We must use Warp Shuffles to avoid shared memory latency.
The FWHT "Butterfly" Kernel logic:
C++

__device__ __forceinline__ void fwht_32_warp(float &val, int lane_id) {
    #pragma unroll
    for (int i = 1; i < 32; i <<= 1) {
        float other = __shfl_xor_sync(0xffffffff, val, i);
        val = (lane_id & i) ? (other - val) : (other + val);
    }
    val *= 0.1767767f; // Normalization: 1/sqrt(32)
}

5. Integration Roadmap

    Register Type: Add GGML_TYPE_TQ3_0 to the enum in ggml.h.

    Add Metadata: Define type_traits in ggml.c (Block size: 32, Type size: 14).

    Implement Dequant: Create dequantize_block_tq3_0 in ggml-cuda.cu.

    CLI Hooks: Update common/arg.cpp to accept turbo3 as a valid --cache-type-k and --cache-type-v value.

Pro-Tip for the CLI Agent:

    "The 5070 Ti uses the Blackwell-based architecture (SM 10.x). Ensure the CUDA kernels use vectorized 128-bit loads (LDG.E.128) for the KV cache blocks to fully saturate the GDDR7 bus."
