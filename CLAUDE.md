# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

llama.cpp is a C/C++ library for efficient LLM inference with minimal dependencies, targeting diverse hardware platforms. The project enables state-of-the-art performance running language models locally on CPUs, GPUs (CUDA, Metal, Vulkan, HIP, etc.), and specialized accelerators.

- **Primary Languages**: C/C++ (200k+ lines), Python utilities for model conversion
- **Core Architecture**: `libllama` library + 40+ tools/examples + ggml tensor library
- **License**: MIT
- **Build System**: CMake 3.14+ (Makefile is deprecated)

## Build Commands

### Basic Build (CPU)
Always use CMake. The Makefile is deprecated and will error out.

```bash
# Configure and build (Release)
cmake -B build
cmake --build build --config Release -j $(nproc)

# Built binaries appear in: build/bin/
```

**Note**: ccache is auto-detected and highly recommended for faster builds.

### Debug Build
```bash
# Single-config generators (Unix Makefiles)
cmake -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build

# Multi-config generators (Visual Studio, Xcode)
cmake -B build -G "Xcode"
cmake --build build --config Debug
```

### Backend-Specific Builds
```bash
# CUDA (NVIDIA GPUs)
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release -j $(nproc)

# Metal (Apple Silicon - enabled by default on macOS)
cmake -B build -DGGML_METAL=ON
cmake --build build --config Release -j $(nproc)

# Vulkan
cmake -B build -DGGML_VULKAN=ON
cmake --build build --config Release -j $(nproc)

# HIP (AMD GPUs)
HIPCXX="$(hipconfig -l)/clang" HIP_PATH="$(hipconfig -R)" \
  cmake -B build -DGGML_HIP=ON -DGPU_TARGETS=gfx1030 -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j $(nproc)
```

**Important**: While multiple backends can be built if dependencies are installed, only the CPU backend can be tested without corresponding hardware. GPU backends require actual hardware to run.

## Testing

### Run Full Test Suite
```bash
# Run all tests (38 tests total, ~30 seconds)
ctest --test-dir build --output-on-failure -j $(nproc)

# Expected: 2-3 network-dependent tests may fail without internet
```

### Server-Specific Tests
```bash
# Build server first
cmake --build build --target llama-server

# Run server unit tests
cd tools/server/tests
source ../../../.venv/bin/activate
./tests.sh
```

### Manual Testing
```bash
# Test basic inference
./build/bin/llama-cli --version
./build/bin/llama-cli -m path/to/model.gguf -p "Hello" -n 10

# Benchmark performance
./build/bin/llama-bench -m model.gguf

# Measure perplexity
./build/bin/llama-perplexity -m model.gguf -f dataset.txt

# Test backend operations
./build/bin/test-backend-ops
```

### Running Local CI
```bash
# Full CI validation before submitting PRs
mkdir tmp
bash ./ci/run.sh ./tmp/results ./tmp/mnt

# Runtime: 30-60 minutes depending on backend
# Add "ggml-ci" to commit message to trigger heavy CI on custom infrastructure
```

## Code Formatting and Linting

### C++ Code
**Always format before committing:**
```bash
git clang-format
```

Configuration: `.clang-format`
- 4-space indentation
- 120 column limit
- Pointer/reference alignment: `void * ptr`, `int & ref` (middle)

### Python Code
**Always activate virtual environment:**
```bash
source .venv/bin/activate
```

Configuration files:
- `.flake8`: max-line-length=125
- `pyrightconfig.json`: type checking config

### Pre-commit Hooks
```bash
pre-commit run --all-files
```

## Architecture

### Core Components

**Libraries:**
- `src/` - Main llama library implementation (llama-*.cpp files)
- `ggml/` - Tensor computation library (core dependency)
- `common/` - Shared utilities across examples/tools

**Public API:**
- `include/llama.h` - Main C API (~2000 lines)
- `include/llama-cpp.h` - C++ wrapper

**Key Executables (in build/bin/):**
- `llama-cli` - Main inference tool (formerly "main")
- `llama-server` - OpenAI-compatible HTTP server
- `llama-quantize` - Model quantization
- `llama-perplexity` - Model evaluation
- `llama-bench` - Performance benchmarking
- `llama-run` - Comprehensive example (used by RamaLama)
- `llama-simple` - Minimal example for developers

### Directory Structure

- `src/` - llama library: modular implementation split across ~30 files (llama-model.cpp, llama-vocab.cpp, llama-context.cpp, llama-graph.cpp, etc.)
- `include/` - Public API headers
- `ggml/` - GGML tensor library with backend implementations
  - `ggml/src/ggml-cuda/` - CUDA backend
  - `ggml/src/ggml-metal/` - Metal backend
  - `ggml/src/ggml-vulkan/` - Vulkan backend
  - `ggml/src/ggml-cpu/` - CPU backend with SIMD optimizations
- `common/` - Shared utilities (arg parsing, chat templates, sampling, console I/O)
- `tools/` - Primary tools (cli, server, quantize, perplexity, bench)
- `examples/` - 30+ example applications
- `tests/` - Test suite with CTest integration
- `docs/` - Documentation (build guides, backend docs, model addition guide)
- `scripts/` - Utility scripts for CI and automation
- `gguf-py/` - Python library for GGUF format
- `convert_hf_to_gguf.py` - HuggingFace to GGUF conversion script

### Model Architecture Representation

The library is highly modular in how it represents model architectures:

1. **Architecture Definition** (`src/llama-arch.cpp`, `src/llama-arch.h`):
   - Each model architecture has an enum value in `llm_arch`
   - Tensor layout mappings in `LLM_TENSOR_NAMES`
   - Architecture names in `LLM_ARCH_NAMES`

2. **Model Loading** (`src/llama-model-loader.cpp`, `src/llama-model.cpp`):
   - GGUF file parsing and metadata extraction
   - Tensor loading with mmap support
   - Multi-file model support (splits)

3. **Computation Graph** (`src/llama-graph.cpp`):
   - GGML computation graph construction
   - Architecture-specific graph building (attention mechanisms, FFN, etc.)
   - Backend-agnostic graph representation

4. **Vocabulary/Tokenization** (`src/llama-vocab.cpp`):
   - Support for SPM, BPE, WPM, UGM, RWKV tokenizers
   - Chat template processing

5. **Context Management** (`src/llama-context.cpp`, `src/llama-kv-cache.cpp`):
   - KV cache management (standard, ring, infinite-state Wasserstein)
   - Batch processing
   - Memory management

### GGUF Format

GGUF is the file format for storing models. It contains:
- Metadata (architecture, hyperparameters, tokenizer config)
- Tensor data (weights, quantized or unquantized)
- Supports multiple quantization formats (Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, etc.)

Conversion: `convert_hf_to_gguf.py` converts HuggingFace models to GGUF.

### Backend System

GGML supports multiple compute backends via a unified interface:
- CPU with SIMD optimizations (AVX, AVX2, AVX512, NEON, RVV)
- CUDA (NVIDIA GPUs)
- Metal (Apple Silicon)
- Vulkan (cross-platform GPU)
- HIP/ROCm (AMD GPUs)
- SYCL (Intel GPUs)
- CANN (Ascend NPUs)
- MUSA (Moore Threads GPUs)
- OpenCL (Adreno GPUs)
- RPC (remote execution)

Backend selection at runtime: `--device <backend>`, `--list-devices`

## Common Development Workflows

### Adding a New Model Architecture

See [docs/development/HOWTO-add-model.md](docs/development/HOWTO-add-model.md) for detailed guide. Summary:

1. **Convert to GGUF** (Python):
   - Add model class to `convert_hf_to_gguf.py`
   - Define architecture enum in `gguf-py/gguf/constants.py`
   - Map tensor names in `gguf-py/gguf/tensor_mapping.py`

2. **Define Architecture** (C++):
   - Add enum to `src/llama-arch.h`
   - Add mappings to `src/llama-arch.cpp`
   - Add loading logic to `src/llama-model-loader.cpp`

3. **Build Computation Graph**:
   - Implement graph building in `src/llama-graph.cpp`
   - Test with CPU backend first, then add GPU support in follow-up PRs

4. **Validate**:
   - Test with llama-cli, llama-server, llama-quantize, llama-imatrix
   - Run on CPU, CUDA, Metal backends
   - Check perplexity against reference implementation

### Modifying Existing Code

**Before Changes:**
1. Read relevant source files
2. Check CONTRIBUTING.md for coding guidelines
3. Consider cross-platform compatibility

**After Changes:**
1. Format: `git clang-format`
2. Build: `cmake --build build --config Release`
3. Test: `ctest --test-dir build --output-on-failure`
4. Server tests (if applicable): `cd tools/server/tests && source ../../../.venv/bin/activate && ./tests.sh`
5. Manual validation with affected tools

### Performance Optimization

- Use `llama-bench` to measure throughput (tokens/second)
- Use `llama-perplexity` to validate quality
- Profile with backend-specific tools (nvprof, Instruments, etc.)
- Check `test-backend-ops` for operation correctness across backends

## Coding Guidelines

- **Dependencies**: Avoid adding third-party dependencies
- **Compatibility**: Always consider cross-platform (Linux, macOS, Windows, Android)
- **Style**: Simple C++17, avoid complex STL/templates, use basic `for` loops
- **Naming**: `snake_case` for functions/variables/types, optimize for longest common prefix
- **Types**: Use `int32_t`, `float`, etc. in public API
- **Enums**: Upper case values prefixed with enum name (e.g., `LLAMA_VOCAB_TYPE_BPE`)
- **Tensors**: Row-major order, dimension 0=columns, 1=rows, 2=matrices
- **API Changes**: Changes to `include/llama.h` require careful consideration

## Git Workflow

- Fork and create feature branches from `master`
- Create separate PRs for each feature/fix
- For new models: focus on CPU support first, GPU in follow-ups
- Allow maintainer write access to your branch for faster reviews
- Rebase on latest `master` if PR becomes stale
- Never commit build artifacts (`build/`, `*.o`, `*.gguf` models)
- Using AI for PRs is permitted but must be disclosed + manually reviewed

## Key Files Reference

- [include/llama.h](include/llama.h) - Main C API
- [src/llama.cpp](src/llama.cpp) - Main library entry point
- [src/llama-model.cpp](src/llama-model.cpp) - Model loading and architecture definitions (8000+ lines)
- [src/llama-vocab.cpp](src/llama-vocab.cpp) - Tokenization
- [src/llama-graph.cpp](src/llama-graph.cpp) - Computation graph building
- [src/llama-context.cpp](src/llama-context.cpp) - Context and inference
- [CMakeLists.txt](CMakeLists.txt) - Build configuration
- [CONTRIBUTING.md](CONTRIBUTING.md) - Contribution guidelines
- [convert_hf_to_gguf.py](convert_hf_to_gguf.py) - HF model conversion

## Dependencies

**Required:**
- CMake 3.14+
- C++17 compiler (GCC 13.3+, Clang, MSVC)

**Optional (for model downloading):**
- libcurl (enabled by default, disable with `-DLLAMA_CURL=OFF`)

**Backend-Specific:**
- CUDA Toolkit 11.2+ (NVIDIA)
- ROCm SDK (AMD)
- Vulkan SDK
- Intel oneAPI (SYCL)
- Metal framework (macOS, auto-detected)

**Bundled (header-only):**
- cpp-httplib (HTTP server)
- nlohmann/json (JSON parsing)
- minja (Jinja template parsing)

## Notes

- **Performance-Critical**: This is a high-performance inference library - consider performance impact
- **Python Environment**: Always activate `.venv` when working with Python code
- **Model Files**: Never commit model files (*.gguf) - they are large and belong in model repositories
- **Test Expectations**: 2-3 network tests may fail without internet access (expected behavior)
- **Backend Testing**: Only CPU backend can be tested without specific hardware
