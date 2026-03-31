# Building llama.cpp with CUDA using GCC 13

## Overview
CUDA 12.x requires GCC 13 or earlier. This system has both GCC 16 (default) and GCC 13 installed to allow CUDA compilation while maintaining the default system compiler.

## Compiler Installation Status
- **Default GCC**: 16.0.1 (/usr/bin/gcc, /usr/bin/g++)
- **CUDA-compatible GCC**: 13.3.0 (/opt/gcc-13/, symlinked to /usr/bin/gcc-13 and /usr/bin/g++-13)

## Building with CUDA Support

### Method 1: Using CMake with GCC 13
```bash
# Create build directory
mkdir -p build_cuda
cd build_cuda

# Configure with GCC 13 as CUDA host compiler
CC=gcc-13 CXX=g++-13 cmake .. \
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/gcc-13 \
  -DGGML_CUDA=ON

# Build
cmake --build . --config Release -j 12
```

### Method 2: Using Make
```bash
CC=gcc-13 CXX=g++-13 make -j 12
```

### Method 3: Setting Environment Variables
```bash
# Export for current session
export CC=gcc-13
export CXX=g++-13
export CMAKE_CUDA_HOST_COMPILER=/usr/bin/gcc-13

# Then run your normal build commands
cmake -B build -DGGML_CUDA=ON
cmake --build build -j 12
```

## Verification

### Check GCC 13 is accessible:
```bash
gcc-13 --version
g++-13 --version
```

### Test CUDA compilation:
```bash
# Build a simple CUDA test to verify the toolchain works
cd build_cuda
cmake --build . --target all 2>&1 | grep -i "cuda\|error" | head -20
```

## Troubleshooting

### If CUDA compilation fails with "unsupported compiler"
1. Verify GCC 13 is being used: `echo $CC $CXX`
2. Check CMake configuration: `grep CUDA build_cuda/CMakeCache.txt | head -10`
3. Ensure CUDA toolkit is installed: `nvcc --version`

### If only GCC 16 is available
GCC 13 was built from source and installed to `/opt/gcc-13/`. The build process:
1. Downloaded GCC 13.3.0 source
2. Disabled libsanitizer (Fedora 44 compatibility fix)
3. Installed to /opt/gcc-13 to avoid conflicts with system GCC 16
4. Created symlinks in /usr/bin for convenience

### Location Reference
- GCC 13 binary: `/opt/gcc-13/bin/gcc` and `/opt/gcc-13/bin/g++`
- Symlinks: `/usr/bin/gcc-13` and `/usr/bin/g++-13`
- Build source: `/tmp/gcc-13.3.0/` (can be deleted to free ~14GB)

## Environment Setup for Automation

For automated builds, source this in your CI/CD or build scripts:
```bash
export CC=gcc-13
export CXX=g++-13
export CMAKE_CUDA_HOST_COMPILER=/usr/bin/gcc-13
```

Or use shell aliases:
```bash
alias cmake-cuda="cmake -DCMAKE_C_COMPILER=gcc-13 -DCMAKE_CXX_COMPILER=g++-13 -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/gcc-13"
```
