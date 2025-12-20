# Inference in llama.cpp - Overview

This document provides a comprehensive overview of how inference works in llama.cpp, covering the entire pipeline from input tokens to output generation.

## Table of Contents

1. [Overview](#overview)
2. [Key Components](#key-components)
3. [Inference Pipeline](#inference-pipeline)
4. [Related Documentation](#related-documentation)

## Overview

llama.cpp implements efficient Large Language Model (LLM) inference through a modular architecture that separates concerns between model loading, memory management, computation graph construction, and token generation. The inference system is designed for:

- **Performance**: Hardware-accelerated computation via GGML backends (CPU, CUDA, Metal, Vulkan, etc.)
- **Flexibility**: Support for multiple model architectures and various generation strategies
- **Efficiency**: Optimized memory usage through KV caching and batch processing
- **Simplicity**: Clean C API with minimal dependencies

## Key Components

### 1. **llama_context** - Inference State

The central object that holds all inference-related state:
- KV cache for attention computations
- Memory management system
- Computation graphs
- Batch processing allocator
- Thread configuration

**Location**: [src/llama-context.cpp](../../src/llama-context.cpp)

### 2. **llama_batch** - Input Specification

Represents a batch of tokens to process. Each token can have:
- `token`: The token ID or embedding vector
- `pos`: Position in the sequence
- `seq_id`: Which sequence(s) this token belongs to (for parallel processing)
- `logits`: Whether to compute output logits for this token

**Location**: [src/llama-batch.cpp](../../src/llama-batch.cpp)

### 3. **llama_memory_i** - KV Cache Management

Abstract interface for managing the key-value cache with multiple implementations:
- Standard KV cache (most common)
- Ring buffer cache (for continuous generation)
- Recurrent state cache (for Mamba-like models)
- Infinite-state Wasserstein cache

**Location**: [src/llama-kv-cache.cpp](../../src/llama-kv-cache.cpp)

### 4. **llm_graph** - Computation Graph

GGML computation graphs that define the forward pass through the model:
- Model-specific graph construction
- Graph reuse optimization
- Backend-agnostic representation

**Location**: [src/llama-graph.cpp](../../src/llama-graph.cpp)

### 5. **llama_sampler** - Token Selection

Chainable sampling operations for selecting the next token:
- Logit transformations (temperature, top-k, top-p, min-p)
- Repetition penalties
- Grammar constraints
- DRY sampler for reducing repetition

**Location**: [src/llama-sampling.cpp](../../src/llama-sampling.cpp)

### 6. **llama_batch_allocr** - Batch Processing

Automatic batch field generation and validation:
- Auto-generates missing positions from KV cache state
- Auto-generates sequence IDs
- Validates batch consistency
- Splits large batches into smaller ubatches

**Location**: [src/llama-batch.cpp](../../src/llama-batch.cpp)

## Inference Pipeline

The complete inference pipeline follows these steps:

```
┌─────────────────────────────────────────────────────────────┐
│ 1. User Input                                               │
│    - Tokenize text to token IDs                            │
│    - Create llama_batch with tokens                        │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. llama_decode(ctx, batch)                                 │
│    ┌─────────────────────────────────────────────────┐     │
│    │ a. Batch Validation & Auto-generation           │     │
│    │    - Validate token IDs, positions, seq_ids     │     │
│    │    - Auto-generate missing pos/seq_id fields    │     │
│    │    - Validate consistency with KV cache state   │     │
│    └─────────────────────────────────────────────────┘     │
│    ┌─────────────────────────────────────────────────┐     │
│    │ b. Memory Allocation                             │     │
│    │    - Reserve KV cache slots for new tokens      │     │
│    │    - Check if context has enough capacity       │     │
│    └─────────────────────────────────────────────────┘     │
│    ┌─────────────────────────────────────────────────┐     │
│    │ c. Graph Construction/Reuse                      │     │
│    │    - Build or reuse GGML computation graph      │     │
│    │    - Set tensor data from batch and KV cache    │     │
│    └─────────────────────────────────────────────────┘     │
│    ┌─────────────────────────────────────────────────┐     │
│    │ d. Backend Execution                             │     │
│    │    - Execute graph on CPU/GPU backends          │     │
│    │    - Compute attention using cached K/V         │     │
│    │    - Store new K/V pairs in cache               │     │
│    └─────────────────────────────────────────────────┘     │
│    ┌─────────────────────────────────────────────────┐     │
│    │ e. Synchronization                               │     │
│    │    - Wait for GPU computation to complete       │     │
│    │    - Copy results back to CPU if needed         │     │
│    └─────────────────────────────────────────────────┘     │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. llama_get_logits_ith(ctx, -1)                            │
│    - Retrieve output logits for the last token              │
│    - Shape: [n_vocab] - probability distribution            │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. llama_sampler_sample(sampler, ctx, -1)                   │
│    - Apply sampling chain (temp, top-k, top-p, etc.)        │
│    - Select next token based on probabilities               │
│    - Update sampler state (for grammar, repetition, etc.)   │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ 5. Loop Back                                                │
│    - Create new batch with selected token                   │
│    - Continue until stopping condition (EOS, max length)    │
└─────────────────────────────────────────────────────────────┘
```

### Minimal Example

From [tools/run/run.cpp](../../tools/run/run.cpp):

```cpp
// Create context and load model
llama_model * model = llama_load_model_from_file("model.gguf", params);
llama_context * ctx = llama_new_context_with_model(model, params);

// Tokenize prompt
std::vector<llama_token> tokens = llama_tokenize(model, prompt, true);

// Process initial prompt
llama_batch batch = llama_batch_get_one(tokens.data(), tokens.size());
llama_decode(ctx, batch);

// Generate tokens
while (!done) {
    // Sample next token
    llama_token next = llama_sampler_sample(sampler, ctx, -1);

    if (llama_vocab_is_eog(vocab, next)) {
        break;  // End of generation
    }

    // Decode next token
    batch = llama_batch_get_one(&next, 1);
    llama_decode(ctx, batch);
}
```

## Performance Considerations

### KV Cache Optimization

The KV cache is the primary memory consumer during inference. llama.cpp optimizes this by:
- **Reusing cache slots**: Old sequences can be removed to make room for new ones
- **Sequence copying**: Share cached data between similar sequences
- **Defragmentation**: Compact cache when it becomes fragmented

### Batch Processing

Processing multiple tokens in a single batch is more efficient than processing one at a time:
- **Prompt processing**: All prompt tokens processed together (prefill phase)
- **Parallel sequences**: Multiple independent sequences can be processed simultaneously
- **Automatic splitting**: Large batches automatically split to fit hardware constraints

### Graph Reuse

Computation graphs are expensive to build, so llama.cpp reuses them when:
- Same batch size and structure
- Same model and context configuration
- Graph is still valid (not invalidated by parameter changes)

## Related Documentation

- [Batch Processing Details](02-batch-processing.md)
- [KV Cache System](03-kv-cache.md)
- [Computation Graphs](04-computation-graphs.md)
- [Sampling and Generation](05-sampling.md)
- [Context Management](06-context-management.md)

## API Reference

Key functions from [include/llama.h](../../include/llama.h):

### Decoding
- `llama_decode()` - Process a batch of tokens (main inference function)
- `llama_encode()` - Process tokens with encoder (for encoder-decoder models)
- `llama_get_logits()` - Get all output logits
- `llama_get_logits_ith()` - Get logits for specific token

### Batching
- `llama_batch_get_one()` - Quick single-sequence batch
- `llama_batch_init()` - Allocate a new batch
- `llama_batch_free()` - Free batch memory

### Sampling
- `llama_sampler_sample()` - Sample next token
- `llama_sampler_chain_*()` - Manage sampler chain

### Memory Management
- `llama_kv_cache_clear()` - Clear entire cache
- `llama_kv_cache_seq_rm()` - Remove specific sequence
- `llama_kv_cache_seq_cp()` - Copy sequence data
- `llama_kv_cache_defrag()` - Defragment cache

---

**Next**: [Batch Processing](02-batch-processing.md)
