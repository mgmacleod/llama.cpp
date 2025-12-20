# KV Cache System in llama.cpp

This document explains the Key-Value (KV) cache system, which is crucial for efficient transformer inference by avoiding recomputation of attention keys and values.

## Table of Contents

1. [What is KV Cache?](#what-is-kv-cache)
2. [Cache Structure](#cache-structure)
3. [Cache Types](#cache-types)
4. [Cache Operations](#cache-operations)
5. [Memory Management](#memory-management)
6. [Cache Slots and Cells](#cache-slots-and-cells)
7. [Performance Considerations](#performance-considerations)

## What is KV Cache?

In transformer models, the attention mechanism computes queries (Q), keys (K), and values (V) for each token. During autoregressive generation:

- **Without cache**: For each new token, recompute K and V for **all previous tokens** (expensive!)
- **With cache**: Store computed K and V values, only compute K and V for the **new token** (fast!)

### Memory Savings Example

Generating 1000 tokens without cache:
- Token 1: Compute K,V for 1 token
- Token 2: Compute K,V for 2 tokens
- Token 3: Compute K,V for 3 tokens
- ...
- Token 1000: Compute K,V for 1000 tokens
- **Total**: ~500,000 K,V computations

Generating 1000 tokens with cache:
- Token 1: Compute K,V for 1 token, store
- Token 2: Compute K,V for 1 token, store (reuse previous)
- ...
- Token 1000: Compute K,V for 1 token, store (reuse all previous)
- **Total**: 1,000 K,V computations (500x faster!)

## Cache Structure

### llama_kv_cache Class

From [src/llama-kv-cache.h:20](../../src/llama-kv-cache.h#L20):

```cpp
class llama_kv_cache : public llama_memory_i {
    // Configuration
    uint32_t n_seq_max;   // Maximum number of sequences
    uint32_t n_stream;    // Number of cache streams
    uint32_t n_pad;       // Required padding
    uint32_t n_swa;       // Sliding window attention size

    // Storage
    std::vector<kv_layer> layers;  // One per model layer
    std::vector<llama_kv_cells> v_cells;  // Cell allocation tracking

    // Per-layer storage
    struct kv_layer {
        uint32_t il;  // Layer index
        ggml_tensor * k;  // Key cache   [n_embd_k_gqa, n_kv_heads, kv_size]
        ggml_tensor * v;  // Value cache [n_embd_v_gqa, n_kv_heads, kv_size]

        // For multi-stream caches
        std::vector<ggml_tensor *> k_stream;
        std::vector<ggml_tensor *> v_stream;
    };
};
```

### Cache Dimensions

For each layer `l`, the cache stores:

**Keys**: `[n_embd_k_gqa, n_kv_heads, kv_size]`
- `n_embd_k_gqa`: Key embedding dimension (after GQA grouping)
- `n_kv_heads`: Number of key-value heads
- `kv_size`: Maximum number of cached positions

**Values**: `[n_embd_v_gqa, n_kv_heads, kv_size]` or transposed
- Typically transposed for better memory access patterns
- `v_trans = true`: `[kv_size, n_embd_v_gqa, n_kv_heads]`

### Memory Usage

For a model with:
- `n_layers = 32`
- `n_kv_heads = 8`
- `n_embd_head = 128`
- `kv_size = 2048`
- Quantization: `Q8_0` (1 byte per element)

**Memory per layer**:
- Keys: `128 × 8 × 2048 = 2,097,152 bytes ≈ 2 MB`
- Values: `128 × 8 × 2048 = 2,097,152 bytes ≈ 2 MB`

**Total cache**: `32 layers × 4 MB = 128 MB`

With `kv_size = 32768` (32K context): **2 GB** cache!

## Cache Types

llama.cpp supports multiple cache implementations via the `llama_memory_i` interface:

### 1. Standard KV Cache

**Class**: `llama_kv_cache` in [src/llama-kv-cache.cpp](../../src/llama-kv-cache.cpp)

The default cache used for most models:
- Stores K and V for all layers
- Supports multiple sequences
- Dynamic allocation of cache cells
- Sequence operations (copy, remove, shift)

**Use case**: Standard transformer models (GPT, LLaMA, Mistral, etc.)

### 2. Ring Buffer Cache

Part of the standard cache, activated with sliding window attention:

```cpp
llama_context_params params = llama_context_default_params();
params.n_ctx = 4096;  // Total context
// Model's sliding window size determines ring buffer behavior
```

**Use case**: Models with sliding window attention (Mistral, etc.)

### 3. Recurrent State Cache

For models without traditional attention (Mamba, RWKV):
- Stores recurrent state instead of K/V
- Much smaller memory footprint
- Different memory interface

**Use case**: Recurrent/SSM models

### 4. Infinite-State Wasserstein (ISWA) Cache

**Class**: `llama_iswa_cache` in [src/llama-kv-cache-iswa.h](../../src/llama-kv-cache-iswa.h)

Advanced cache that maintains a fixed-size cache by:
- Keeping a sliding window of recent tokens
- Keeping a compressed summary of older tokens
- Using Wasserstein distance for selection

**Use case**: Very long context generation with fixed memory budget

## Cache Operations

### Initialization

Cache is created with context:

```cpp
llama_context_params params = llama_context_default_params();
params.n_ctx = 2048;        // Context size
params.n_batch = 512;       // Batch size
params.n_seq_max = 4;       // Max parallel sequences

llama_context * ctx = llama_new_context_with_model(model, params);
// Cache is automatically created and initialized
```

### Sequence Operations

#### Clear All
```cpp
llama_kv_cache_clear(ctx);
```
Removes all cached data, resets all sequences.

#### Remove Sequence
```cpp
llama_kv_cache_seq_rm(ctx, seq_id, p0, p1);
```
- `seq_id`: Which sequence to remove
- `p0, p1`: Position range (`-1, -1` for all positions)

Example:
```cpp
// Remove sequence 2 entirely
llama_kv_cache_seq_rm(ctx, 2, -1, -1);

// Remove positions 100-200 from sequence 0
llama_kv_cache_seq_rm(ctx, 0, 100, 200);
```

#### Copy Sequence
```cpp
llama_kv_cache_seq_cp(ctx, src_seq, dst_seq, p0, p1);
```

Copies cached K/V from one sequence to another.

Example - Branching:
```cpp
// Process shared prompt in sequence 0
llama_decode(ctx, prompt_batch);  // All tokens in seq 0

// Branch into two generation paths
llama_kv_cache_seq_cp(ctx, 0, 1, -1, -1);  // Copy seq 0 -> seq 1

// Now seq 0 and seq 1 have identical cache
// Continue generation in both sequences independently
```

#### Keep Only Sequence
```cpp
llama_kv_cache_seq_keep(ctx, seq_id);
```

Removes all sequences except `seq_id`.

#### Shift Positions
```cpp
llama_kv_cache_seq_add(ctx, seq_id, p0, p1, shift);
```

Shifts position indices in the cache (for sliding window).

Example - Sliding window:
```cpp
if (cache_is_full) {
    // Remove first half of context
    llama_kv_cache_seq_rm(ctx, 0, 0, n_ctx / 2);

    // Shift positions of second half
    llama_kv_cache_seq_add(ctx, 0, n_ctx / 2, -1, -(n_ctx / 2));
}
```

#### Divide Positions
```cpp
llama_kv_cache_seq_div(ctx, seq_id, p0, p1, d);
```

Divides position indices (for specific model requirements).

### Defragmentation

Over time, the cache can become fragmented with gaps from removed sequences:

```cpp
llama_kv_cache_defrag(ctx);
```

Compacts the cache, removing gaps and improving memory locality.

**When to use**:
- After many sequence removal operations
- When cache utilization is low but allocation fails
- Periodically in long-running applications

## Memory Management

### Cache Cells

The cache is divided into "cells", each storing K/V for one position:

From [src/llama-kv-cache.h:34](../../src/llama-kv-cache.h#L34):

```cpp
struct slot_info {
    uint32_t s0, s1;  // Stream range

    std::vector<llama_seq_id> strm;  // Stream IDs
    std::vector<idx_vec_t> idxs;     // Cell indices for each stream
};
```

### Finding Slots

When processing a batch, the cache must find available cells:

```cpp
slot_info find_slot(const llama_ubatch & ubatch, bool cont) const;
```

- `cont = true`: Requires contiguous slots (for some optimizations)
- `cont = false`: Accepts non-contiguous slots

**Ring buffer strategy**: Start searching from last allocation point, wrap around.

### Cache Streams

For certain models (with multiple attention streams), the cache can be split into separate streams:

```cpp
llama_kv_cache(
    // ...
    n_seq_max,   // Maximum sequences
    n_stream,    // Number of streams
    // ...
);
```

Each stream maintains independent cache cells.

### Applying Batch to Cache

After finding slots, the batch is applied to the cache:

```cpp
void apply_ubatch(const slot_info & sinfo, const llama_ubatch & ubatch);
```

This updates the cell metadata:
- Marks cells as occupied
- Associates cells with sequences
- Updates position tracking

The actual K/V data is written during graph execution.

## Cache Slots and Cells

### Cell Structure

Each cell tracks:
- Which sequences use it
- Position in each sequence
- Whether it's currently occupied

From [src/llama-kv-cells.h](../../src/llama-kv-cells.h):

```cpp
struct llama_kv_cell {
    llama_pos pos;        // Position in sequence
    uint32_t  seq_ids;    // Bitset of sequence IDs
    bool      has_seq_id(llama_seq_id id) const;
};
```

### Slot Allocation Example

Processing batch with 3 tokens for sequence 0:

```
Before:
Cache: [used][used][FREE][FREE][FREE][used]...
         pos0  pos1

Processing batch: [tok1, tok2, tok3]

After:
Cache: [used][used][NEW][NEW][NEW][used]...
         pos0  pos1  pos2 pos3 pos4
                     tok1 tok2 tok3
```

### Multi-Sequence Example

Two sequences sharing a prefix:

```
Sequence 0: [A][B][C][D][E]
Sequence 1: [A][B][C][F][G]
                     ↑
                  Divergence point

Cache cells:
[A]: seq {0, 1}  pos 0  (shared)
[B]: seq {0, 1}  pos 1  (shared)
[C]: seq {0, 1}  pos 2  (shared)
[D]: seq {0}     pos 3  (unique to seq 0)
[E]: seq {0}     pos 4  (unique to seq 0)
[F]: seq {1}     pos 3  (unique to seq 1)
[G]: seq {1}     pos 4  (unique to seq 1)
```

## Performance Considerations

### Cache Size vs Context Size

`kv_size` (cache size) should be ≥ maximum context you'll use:

```cpp
params.n_ctx = 2048;  // Can process up to 2048 tokens
// Cache sized for 2048 positions
```

**Trade-off**:
- Larger cache: More memory, longer context
- Smaller cache: Less memory, must use sliding window

### Quantization

Cache can be quantized to save memory:

```cpp
params.type_k = GGML_TYPE_Q8_0;  // 8-bit keys
params.type_v = GGML_TYPE_Q8_0;  // 8-bit values
```

**Quality impact**: Minimal for Q8_0, small for Q4_0

### Offloading

Cache can be offloaded to GPU:

```cpp
params.offload_kqv = true;  // Offload KV cache to GPU
```

**Benefit**: Faster attention computation, reduced CPU-GPU transfers

### Defragmentation Cost

Defragmentation copies memory and rebuilds metadata:
- Time: O(cache_size)
- Best done when performance is not critical
- Not needed after simple sequential generation

### Cache Reuse

For multiple similar queries, cache can be reused:

```cpp
// Process shared prefix once
llama_decode(ctx, shared_prefix);

// Branch into multiple completions
for (int i = 0; i < n_branches; i++) {
    llama_kv_cache_seq_cp(ctx, 0, i + 1, -1, -1);
    // Generate from branch i+1
}
```

### Memory Monitoring

Check cache usage:

```cpp
// Get current memory usage
std::map<ggml_backend_buffer_type_t, size_t> mem =
    llama_memory_breakdown(ctx);

for (const auto & [type, size] : mem) {
    printf("%s: %.2f MB\n", ggml_backend_buft_name(type), size / 1024.0 / 1024.0);
}
```

## Best Practices

1. **Size appropriately**: Set `n_ctx` to your maximum expected context length

2. **Use quantization**: Q8_0 cache saves 4x memory with minimal quality loss

3. **Defragment periodically**: If doing many sequence operations, defragment occasionally

4. **Reuse cache**: For multiple similar queries, process shared prefix once and branch

5. **Monitor memory**: Watch cache memory usage, especially for long contexts

6. **Clear when done**: Call `llama_kv_cache_clear()` between unrelated generations

7. **Use sequences wisely**: Don't create more sequences than needed; clean up unused ones

## Debugging

Enable KV cache debugging:

```bash
export LLAMA_KV_CACHE_DEBUG=1  # Enable debug output
```

Query cache state:

```cpp
// Get position range for a sequence
llama_pos min_pos = llama_memory_seq_pos_min(ctx, seq_id);
llama_pos max_pos = llama_memory_seq_pos_max(ctx, seq_id);

printf("Sequence %d: positions [%d, %d]\n", seq_id, min_pos, max_pos);
```

---

**Previous**: [Batch Processing](02-batch-processing.md) | **Next**: [Computation Graphs](04-computation-graphs.md)
