# Inference Documentation

This directory contains comprehensive documentation on how inference works in llama.cpp.

## Documentation Structure

### 1. [Overview](01-overview.md)
High-level overview of the inference pipeline, key components, and how they work together.

**Topics covered**:
- What is inference in llama.cpp
- Key components (context, batches, KV cache, graphs, samplers)
- Complete inference pipeline from input to output
- Minimal usage examples
- Performance considerations

**Start here** if you're new to llama.cpp inference.

### 2. [Batch Processing](02-batch-processing.md)
Deep dive into how batches work, including structure, validation, and splitting.

**Topics covered**:
- Batch structure and fields
- Creating and managing batches
- Auto-generation of batch fields
- Batch validation rules
- Batch splitting (ubatches)
- Multi-sequence batching
- Best practices

**Read this** when working with custom batching logic or multiple sequences.

### 3. [KV Cache System](03-kv-cache.md)
Comprehensive guide to the Key-Value cache, the core memory system for transformer inference.

**Topics covered**:
- What is KV cache and why it matters
- Cache structure and types
- Cache operations (clear, copy, remove, shift)
- Memory management and optimization
- Cache slots and cells
- Defragmentation
- Performance considerations

**Read this** when optimizing memory usage or working with long contexts.

### 4. [Computation Graphs](04-computation-graphs.md)
Explanation of GGML computation graphs and how they enable efficient inference.

**Topics covered**:
- What are computation graphs
- Graph building process
- Graph structure and components
- Graph reuse optimization
- Model-specific graph building
- Graph execution and scheduling
- Adding new architectures

**Read this** when adding new models or optimizing performance.

### 5. [Sampling and Token Generation](05-sampling.md)
Complete guide to sampling strategies and token selection.

**Topics covered**:
- From logits to tokens
- Sampling chain architecture
- All sampling strategies (greedy, temperature, top-k, top-p, etc.)
- Repetition penalties and constraints
- Grammar-constrained generation
- Common sampler high-level API
- Performance optimization

**Read this** when tuning generation quality or implementing custom sampling.

### 6. [Context Management](06-context-management.md)
Guide to managing llama_context, configuration, and resources.

**Topics covered**:
- Context creation and parameters
- Memory management
- Threading configuration
- State save/load
- Best practices for different use cases
- Performance monitoring

**Read this** when configuring contexts or managing resources.

## Quick Reference

### Basic Inference Flow

```cpp
// 1. Load model
llama_model * model = llama_load_model_from_file("model.gguf", mparams);

// 2. Create context
llama_context_params cparams = llama_context_default_params();
cparams.n_ctx = 2048;
llama_context * ctx = llama_new_context_with_model(model, cparams);

// 3. Tokenize input
std::vector<llama_token> tokens = llama_tokenize(model, "Hello", true);

// 4. Create batch
llama_batch batch = llama_batch_get_one(tokens.data(), tokens.size());

// 5. Decode (process tokens)
llama_decode(ctx, batch);

// 6. Sample next token
llama_token next = llama_sampler_sample(sampler, ctx, -1);

// 7. Repeat steps 4-6 for generation
```

### Key Functions

#### Model & Context
- `llama_load_model_from_file()` - Load model from GGUF file
- `llama_new_context_with_model()` - Create inference context
- `llama_free()` - Free context
- `llama_free_model()` - Free model

#### Batching
- `llama_batch_get_one()` - Quick single-sequence batch
- `llama_batch_init()` - Allocate custom batch
- `llama_batch_free()` - Free batch

#### Inference
- `llama_decode()` - Main inference function
- `llama_encode()` - Encoder inference (for encoder-decoder models)
- `llama_get_logits()` - Get output logits
- `llama_get_logits_ith()` - Get logits for specific token

#### KV Cache
- `llama_kv_cache_clear()` - Clear entire cache
- `llama_kv_cache_seq_rm()` - Remove sequence
- `llama_kv_cache_seq_cp()` - Copy sequence
- `llama_kv_cache_defrag()` - Defragment cache

#### Sampling
- `llama_sampler_chain_init()` - Create sampler chain
- `llama_sampler_chain_add()` - Add sampler to chain
- `llama_sampler_sample()` - Sample token
- `llama_sampler_free()` - Free sampler

#### High-Level Sampling (common)
- `common_sampler_init()` - Create sampler from parameters
- `common_sampler_sample()` - Sample with grammar support
- `common_sampler_accept()` - Accept token and update state
- `common_sampler_free()` - Free sampler

## Common Use Cases

### 1. Simple Text Generation

See: [Overview](01-overview.md) and [Sampling](05-sampling.md)

```cpp
// Process prompt
llama_batch batch = llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size());
llama_decode(ctx, batch);

// Generate tokens
while (!done) {
    llama_token token = llama_sampler_sample(sampler, ctx, -1);
    if (llama_vocab_is_eog(vocab, token)) break;

    batch = llama_batch_get_one(&token, 1);
    llama_decode(ctx, batch);
}
```

### 2. Parallel Sequence Generation

See: [Batch Processing](02-batch-processing.md)

```cpp
// Process shared prefix once
llama_batch batch = /* prefix tokens in sequence 0 */;
llama_decode(ctx, batch);

// Branch into multiple sequences
for (int i = 1; i < n_sequences; i++) {
    llama_kv_cache_seq_cp(ctx, 0, i, -1, -1);
}

// Generate from each sequence independently
// (using appropriate seq_id in batches)
```

### 3. Long Context with Sliding Window

See: [KV Cache System](03-kv-cache.md)

```cpp
if (n_past + n_tokens > n_ctx) {
    int n_discard = n_ctx / 2;

    // Remove first half
    llama_kv_cache_seq_rm(ctx, 0, 0, n_discard);

    // Shift remaining positions
    llama_kv_cache_seq_add(ctx, 0, n_discard, n_past, -n_discard);

    n_past -= n_discard;
}
```

### 4. Grammar-Constrained Generation

See: [Sampling](05-sampling.md)

```cpp
// Define grammar
const char * grammar = R"(
    root ::= object
    object ::= "{" pair ("," pair)* "}"
    pair ::= string ":" value
    string ::= "\"" [^"]* "\""
    value ::= string | number
)";

// Create sampler with grammar
common_params_sampling params;
params.grammar = grammar;
common_sampler * gsmpl = common_sampler_init(model, params);

// Sample (will respect grammar)
llama_token token = common_sampler_sample(gsmpl, ctx, -1);
```

### 5. State Save/Resume

See: [Context Management](06-context-management.md)

```cpp
// Save state
size_t state_size = llama_state_get_size(ctx);
std::vector<uint8_t> state(state_size);
llama_state_get_data(ctx, state.data(), state_size);

// ... later ...

// Restore state
llama_state_set_data(ctx, state.data(), state.size());
```

## Performance Tips

1. **Batch Size**: Use large batches (512-2048) for prompt processing, small batches (1-32) for generation

2. **KV Cache**: Quantize with Q8_0 for large contexts (50% memory savings, minimal quality loss)

3. **Threading**: Let auto-detection choose thread counts; don't over-thread

4. **Graph Reuse**: Keep batch structures consistent to enable graph reuse

5. **Defragmentation**: Defrag KV cache periodically when using dynamic sequences

6. **GPU Offloading**: Use `offload_kqv = true` for GPU-accelerated attention

## Debugging

### Enable Debug Output

```bash
# Batch processing debug
export LLAMA_BATCH_DEBUG=1

# KV cache debug
export LLAMA_KV_CACHE_DEBUG=1

# Graph debug
export LLAMA_GRAPH_RESULT_DEBUG=1
```

### Common Issues

**Issue**: "Could not find a KV slot for the batch"
- **Cause**: KV cache full
- **Solution**: Increase `n_ctx`, use sliding window, or defragment cache

**Issue**: Slow first inference, fast subsequent inferences
- **Cause**: Graph building on first run
- **Solution**: This is normal; graph is reused for subsequent runs

**Issue**: Out of memory
- **Cause**: Large context, many sequences, or unoptimized parameters
- **Solution**: Reduce `n_ctx`, quantize KV cache, reduce `n_seq_max`

**Issue**: Repetitive output
- **Cause**: Insufficient sampling diversity or repetition penalty
- **Solution**: Adjust temperature, add repetition penalty, use DRY sampler

## Contributing

When updating inference documentation:

1. Keep examples concise and practical
2. Reference specific files and line numbers when possible
3. Include both high-level explanations and code examples
4. Update this README if adding new documents

## Additional Resources

- [Main README](../../README.md) - Project overview
- [Build Instructions](../../CLAUDE.md) - How to build llama.cpp
- [Model Addition Guide](../development/HOWTO-add-model.md) - Adding new architectures
- [Examples](../../examples/) - Complete working examples
- [include/llama.h](../../include/llama.h) - Public API documentation

## Questions?

- Check the [examples](../../examples/) for working code
- See [common/](../../common/) for high-level helper utilities
- Ask on GitHub Discussions or Issues

---

*Last updated: 2025-12-16*
