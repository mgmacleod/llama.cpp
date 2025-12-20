# Context Management in llama.cpp

This document explains how to manage the llama_context object, configure it for different use cases, and handle memory and threading.

## Table of Contents

1. [Overview](#overview)
2. [Context Creation](#context-creation)
3. [Context Parameters](#context-parameters)
4. [Memory Management](#memory-management)
5. [Threading Configuration](#threading-configuration)
6. [Context State](#context-state)
7. [Best Practices](#best-practices)

## Overview

The `llama_context` is the central object for inference in llama.cpp. It encapsulates:

- **Memory system**: KV cache or recurrent state
- **Computation resources**: Backends, schedulers, graphs
- **Configuration**: Context size, batch size, threading
- **State**: Current sequence positions, cached computations

**Key concept**: One model can have multiple contexts with different configurations.

```cpp
llama_model * model = llama_load_model_from_file("model.gguf", mparams);

// Context 1: Large batch, many sequences
llama_context * ctx1 = llama_new_context_with_model(model, params1);

// Context 2: Small batch, single sequence
llama_context * ctx2 = llama_new_context_with_model(model, params2);

// Both share the same model weights but have independent state
```

## Context Creation

### Basic Creation

```cpp
// Load model first
llama_model_params mparams = llama_model_default_params();
llama_model * model = llama_load_model_from_file("model.gguf", mparams);

// Create context
llama_context_params cparams = llama_context_default_params();
cparams.n_ctx = 2048;      // Context size
cparams.n_batch = 512;     // Batch size
cparams.n_ubatch = 512;    // Microbatch size
cparams.n_seq_max = 1;     // Max parallel sequences

llama_context * ctx = llama_new_context_with_model(model, cparams);

if (!ctx) {
    fprintf(stderr, "Failed to create context\n");
    return 1;
}

// ... use context ...

// Cleanup
llama_free(ctx);
llama_free_model(model);
```

### Context from Model and Default Params

```cpp
llama_context * ctx = llama_new_context_with_model(
    model,
    llama_context_default_params()
);
```

## Context Parameters

### llama_context_params Structure

From [include/llama.h](../../include/llama.h):

```cpp
struct llama_context_params {
    uint32_t n_ctx;              // Context size (max tokens)
    uint32_t n_batch;            // Batch size (prompt processing)
    uint32_t n_ubatch;           // Physical batch size (hardware limit)
    uint32_t n_seq_max;          // Max parallel sequences
    uint32_t n_threads;          // Threads for generation
    uint32_t n_threads_batch;    // Threads for batch processing

    enum llama_rope_scaling_type rope_scaling_type; // RoPE scaling
    enum llama_pooling_type pooling_type;           // Pooling for embeddings
    enum llama_attention_type attention_type;       // Attention mechanism

    float rope_freq_base;        // RoPE base frequency
    float rope_freq_scale;       // RoPE frequency scale
    float yarn_ext_factor;       // YaRN extension factor
    float yarn_attn_factor;      // YaRN attention factor
    float yarn_beta_fast;        // YaRN beta fast
    float yarn_beta_slow;        // YaRN beta slow
    float defrag_thold;          // KV cache defrag threshold
    uint32_t yarn_orig_ctx;      // YaRN original context

    ggml_backend_sched_eval_callback cb_eval;     // Eval callback
    void * cb_eval_user_data;

    enum ggml_type type_k;       // KV cache K data type
    enum ggml_type type_v;       // KV cache V data type

    bool logits_all;             // Return logits for all tokens
    bool embeddings;             // Return embeddings (not logits)
    bool offload_kqv;            // Offload KQV to GPU
    bool flash_attn;             // Use flash attention
    bool no_perf;                // Don't collect performance metrics

    ggml_abort_callback abort_callback;  // Abort callback
    void * abort_callback_data;
};
```

### Key Parameters Explained

#### `n_ctx` - Context Size

Maximum number of tokens the context can hold:

```cpp
cparams.n_ctx = 2048;  // Can process up to 2048 tokens
```

**Considerations**:
- Larger = more memory (KV cache size proportional to n_ctx)
- Must be ≤ model's trained context size (usually)
- Some models support context extension (RoPE scaling)

**Memory impact**: For LLaMA-7B with n_ctx=2048, KV cache ≈ 512 MB

#### `n_batch` - Logical Batch Size

Maximum tokens to process in one call:

```cpp
cparams.n_batch = 512;  // Can decode up to 512 tokens at once
```

**Use cases**:
- Prompt processing: Large batches (512-2048) for fast prefill
- Token generation: Small batches (1-32) for individual tokens
- Parallel generation: Multiple sequences in one batch

**Constraint**: n_batch ≤ n_ctx

#### `n_ubatch` - Physical Batch Size

Maximum tokens processed by hardware in one go:

```cpp
cparams.n_ubatch = 256;  // Hardware processes 256 tokens per iteration
```

**Purpose**: Split large batches to fit GPU/CPU memory

**Relationship**: `n_batch` may be split into multiple `n_ubatch` chunks

**Example**:
```cpp
cparams.n_batch = 1024;   // Logical batch
cparams.n_ubatch = 256;   // Physical batch
// Batch will be processed as 4 ubatches of 256 tokens each
```

#### `n_seq_max` - Maximum Sequences

Maximum number of independent sequences:

```cpp
cparams.n_seq_max = 4;  // Can process 4 sequences in parallel
```

**Use cases**:
- **1**: Single conversation
- **4-8**: Multiple conversations or beam search
- **16+**: Batch inference for multiple users

**Memory impact**: Some overhead per sequence, but mainly shares KV cache

#### `n_threads` and `n_threads_batch`

CPU threading configuration:

```cpp
cparams.n_threads = 4;        // 4 threads for generation (1 token)
cparams.n_threads_batch = 8;  // 8 threads for batch processing (multiple tokens)
```

**Guideline**:
- `n_threads`: Physical cores (for single token latency)
- `n_threads_batch`: Physical cores × 2 (for batch throughput)

**Auto-detection**:
```cpp
cparams.n_threads = 0;        // Auto-detect
cparams.n_threads_batch = 0;  // Auto-detect
```

#### `type_k` and `type_v` - KV Cache Quantization

Data type for KV cache storage:

```cpp
cparams.type_k = GGML_TYPE_F16;   // 16-bit float (default)
cparams.type_v = GGML_TYPE_F16;

// Or quantize for memory savings
cparams.type_k = GGML_TYPE_Q8_0;  // 8-bit quantized
cparams.type_v = GGML_TYPE_Q8_0;
```

**Memory savings**:
- F16: Full precision, 2 bytes per element
- Q8_0: Quantized, 1 byte per element (50% savings)
- Q4_0: Quantized, 0.5 bytes per element (75% savings, some quality loss)

**Quality impact**: Q8_0 has minimal impact; Q4_0 may affect long contexts

#### `logits_all` - Return All Logits

```cpp
cparams.logits_all = true;  // Return logits for all tokens, not just last
```

**Use cases**:
- Perplexity measurement
- Token probability analysis
- Training/fine-tuning

**Performance**: Slower, more memory

#### `embeddings` - Embedding Mode

```cpp
cparams.embeddings = true;  // Return embeddings instead of logits
```

**Use case**: When using model for embeddings (semantic search, etc.)

**Note**: Requires appropriate pooling_type

#### `offload_kqv` - GPU KV Cache

```cpp
cparams.offload_kqv = true;  // Store KV cache on GPU
```

**Benefit**: Faster attention, less CPU↔GPU transfer

**Requirement**: GPU backend must be available

#### `flash_attn` - Flash Attention

```cpp
cparams.flash_attn = true;  // Use flash attention algorithm
```

**Benefit**: Faster attention for long contexts

**Availability**: Backend-dependent (CUDA, HIP)

## Memory Management

### Memory Initialization

Context memory is allocated during creation:

```cpp
llama_context * ctx = llama_new_context_with_model(model, cparams);
// Allocates:
// - KV cache (or recurrent state)
// - Computation graph buffers
// - Batch processing buffers
```

### Memory Breakdown

Query memory usage:

```cpp
// Get memory breakdown by buffer type
auto mem_map = llama_memory_breakdown(ctx);

for (const auto & [type, size] : mem_map) {
    printf("%s: %.2f MB\n",
        ggml_backend_buft_name(type),
        size / (1024.0 * 1024.0));
}
```

Output example:
```
CPU: 128.5 MB        (KV cache, computation buffers)
CUDA0: 512.0 MB      (Model layers on GPU)
```

### KV Cache Defragmentation

Over time, the KV cache can become fragmented:

```cpp
// Manual defragmentation
llama_kv_cache_defrag(ctx);

// Or configure automatic defragmentation
cparams.defrag_thold = 0.1;  // Defrag when 10% fragmented
```

**When to defrag**:
- After many sequence operations (remove, copy, etc.)
- When KV cache allocation fails despite available space
- Periodically in long-running applications

### Clearing Context

Reset context to initial state:

```cpp
llama_kv_cache_clear(ctx);  // Clear KV cache
// Context is now ready for new generation
```

## Threading Configuration

### CPU Threading

llama.cpp uses thread pools for CPU computation:

```cpp
// Set threads at creation
cparams.n_threads = 8;
cparams.n_threads_batch = 16;
llama_context * ctx = llama_new_context_with_model(model, cparams);

// Or change at runtime
llama_set_n_threads(ctx, 4, 8);
```

### Thread Count Guidelines

**For generation** (1 token at a time):
- Use ≤ physical cores
- Hyperthreading often doesn't help
- Example: 8-core CPU → n_threads = 8

**For batch processing** (many tokens):
- Can use more threads
- Example: 8-core CPU → n_threads_batch = 12-16

**Auto-detection**:
```cpp
cparams.n_threads = 0;  // Uses std::thread::hardware_concurrency()
```

### Backend Threading

GPU backends handle threading automatically:
- CUDA: Managed by CUDA runtime
- Metal: Managed by Metal framework
- Vulkan: Configured via device queues

## Context State

### Saving Context State

Save KV cache and context state:

```cpp
// Get size needed
size_t state_size = llama_state_get_size(ctx);

// Allocate buffer
std::vector<uint8_t> state_data(state_size);

// Save state
size_t written = llama_state_get_data(ctx, state_data.data(), state_size);

// Write to file
FILE * f = fopen("state.bin", "wb");
fwrite(state_data.data(), 1, written, f);
fclose(f);
```

### Loading Context State

```cpp
// Read from file
FILE * f = fopen("state.bin", "rb");
fseek(f, 0, SEEK_END);
size_t state_size = ftell(f);
fseek(f, 0, SEEK_SET);

std::vector<uint8_t> state_data(state_size);
fread(state_data.data(), 1, state_size, f);
fclose(f);

// Load state
llama_state_set_data(ctx, state_data.data(), state_size);

// Context now has restored KV cache
```

### Per-Sequence State

Save/load state for specific sequences:

```cpp
// Save sequence 0
size_t size = llama_state_seq_get_size(ctx, 0);
std::vector<uint8_t> data(size);
llama_state_seq_get_data(ctx, data.data(), size, 0);

// Load into sequence 1
llama_state_seq_set_data(ctx, data.data(), size, 1);
```

## Best Practices

### Context Size

1. **Match your use case**:
   - Chat: 2048-4096
   - Long documents: 8192-32768
   - Code completion: 4096-8192

2. **Consider memory**: Each doubling of n_ctx doubles KV cache size

3. **Test scaling**: Some models handle extended context better than others

### Batch Configuration

1. **Prompt processing**: Large batches (512-2048) for speed

2. **Generation**: Small batches (1-32) for low latency

3. **Parallel sequences**: Balance batch size and sequence count

### Threading

1. **Start with defaults**: Let auto-detection choose

2. **Profile**: Measure with different thread counts

3. **Don't over-thread**: More threads ≠ faster beyond a point

### Memory

1. **Monitor usage**: Use `llama_memory_breakdown()` to track memory

2. **Quantize KV cache**: Use Q8_0 for large contexts

3. **Defrag periodically**: Especially with dynamic sequence management

### Multi-Context Usage

```cpp
// Good: Share model, separate contexts
llama_model * model = /* load once */;
llama_context * ctx1 = llama_new_context_with_model(model, params1);
llama_context * ctx2 = llama_new_context_with_model(model, params2);

// Bad: Multiple models for same weights
llama_model * model1 = llama_load_model_from_file("model.gguf", params);
llama_model * model2 = llama_load_model_from_file("model.gguf", params);
// Wastes memory!
```

## Example: Adaptive Context Configuration

```cpp
// Configure based on task
llama_context_params get_params_for_task(TaskType task) {
    llama_context_params params = llama_context_default_params();

    switch (task) {
        case TASK_CHAT:
            params.n_ctx = 4096;
            params.n_batch = 512;
            params.n_seq_max = 1;
            break;

        case TASK_CODE_COMPLETION:
            params.n_ctx = 8192;
            params.n_batch = 1024;
            params.n_seq_max = 1;
            params.type_k = GGML_TYPE_Q8_0;  // Save memory
            params.type_v = GGML_TYPE_Q8_0;
            break;

        case TASK_BATCH_INFERENCE:
            params.n_ctx = 2048;
            params.n_batch = 512;
            params.n_seq_max = 16;  // Multiple sequences
            params.n_ubatch = 256;  // Smaller physical batches
            break;

        case TASK_EMBEDDINGS:
            params.n_ctx = 512;
            params.n_batch = 512;
            params.embeddings = true;
            params.pooling_type = LLAMA_POOLING_TYPE_MEAN;
            break;
    }

    return params;
}

// Usage
llama_context_params params = get_params_for_task(TASK_CHAT);
llama_context * ctx = llama_new_context_with_model(model, params);
```

## Performance Monitoring

```cpp
// Get timing information
struct llama_perf_context_data perf = llama_perf_context(ctx);

printf("Load time: %.2f ms\n", perf.t_load_ms);
printf("Sample time: %.2f ms\n", perf.t_sample_ms);
printf("Prompt eval time: %.2f ms\n", perf.t_p_eval_ms);
printf("Eval time: %.2f ms\n", perf.t_eval_ms);
printf("Total time: %.2f ms\n", perf.t_total_ms);

printf("Tokens evaluated: %d\n", perf.n_eval);
printf("Tokens per second: %.2f\n", 1000.0 * perf.n_eval / perf.t_eval_ms);
```

---

**Previous**: [Sampling and Decoding](05-sampling.md) | **Back to**: [Overview](01-overview.md)
