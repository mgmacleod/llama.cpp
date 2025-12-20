# Computation Graphs in llama.cpp

This document explains how llama.cpp builds and executes computation graphs for model inference using the GGML backend.

## Table of Contents

1. [What are Computation Graphs?](#what-are-computation-graphs)
2. [Graph Building Process](#graph-building-process)
3. [Graph Structure](#graph-structure)
4. [Graph Reuse Optimization](#graph-reuse-optimization)
5. [Model-Specific Graph Building](#model-specific-graph-building)
6. [Graph Execution](#graph-execution)
7. [Performance Considerations](#performance-considerations)

## What are Computation Graphs?

A **computation graph** is a directed acyclic graph (DAG) that represents the mathematical operations needed to perform a forward pass through the neural network. In llama.cpp, graphs are built using GGML (Generic Graph Machine Learning), a tensor library that provides:

- **Backend abstraction**: Same graph runs on CPU, CUDA, Metal, Vulkan, etc.
- **Automatic differentiation**: Though primarily used for inference
- **Memory planning**: Efficient tensor memory allocation
- **Operation fusion**: Combines operations for better performance

### Graph vs Direct Computation

**Without graphs** (immediate execution):
```cpp
tensor_a = matmul(weights, input);
tensor_b = add(tensor_a, bias);
tensor_c = relu(tensor_b);
// Each operation executes immediately
```

**With graphs** (deferred execution):
```cpp
// Build phase: Define operations
tensor_a = ggml_matmul(ctx, weights, input);
tensor_b = ggml_add(ctx, tensor_a, bias);
tensor_c = ggml_relu(ctx, tensor_b);

ggml_build_forward_expand(gf, tensor_c);

// Execute phase: Run entire graph
ggml_backend_sched_graph_compute(sched, gf);
```

**Benefits**:
- Backend can optimize the entire graph
- Better memory management
- Can reuse graphs for similar inputs

## Graph Building Process

From [src/llama-context.cpp:975](../../src/llama-context.cpp#L975) - `llama_context::decode()`:

### Step 1: Determine if Graph Can Be Reused

```cpp
bool can_reuse_graph = gres && gres->can_reuse(params);

if (can_reuse_graph) {
    // Just update input tensors and execute
    gres->set_inputs(&ubatch);
} else {
    // Build new graph
    gres = build_graph(params);
}
```

### Step 2: Build Graph (if needed)

```cpp
llm_graph_result * build_graph(const llm_graph_params & params) {
    // Create graph result object
    auto * gres = new llm_graph_result(max_nodes);

    // Build model-specific graph
    switch (model_arch) {
        case LLM_ARCH_LLAMA:
            llm_build_llama(params);
            break;
        case LLM_ARCH_FALCON:
            llm_build_falcon(params);
            break;
        // ... other architectures
    }

    return gres;
}
```

### Step 3: Set Input Tensors

```cpp
gres->set_inputs(&ubatch);
// This copies batch data (tokens, positions, etc.) into graph input tensors
```

### Step 4: Execute Graph

```cpp
ggml_backend_sched_graph_compute(sched, gres->get_gf());
```

### Step 5: Synchronize and Extract Results

```cpp
ctx->synchronize();
float * logits = llama_get_logits_ith(ctx, -1);
```

## Graph Structure

### llm_graph_result

From [src/llama-graph.h:470](../../src/llama-graph.h#L470):

```cpp
class llm_graph_result {
    // GGML context for graph tensors
    ggml_context_ptr ctx_compute;

    // The actual computation graph
    ggml_cgraph * gf;

    // Important output tensors
    ggml_tensor * t_tokens;      // Input tokens
    ggml_tensor * t_logits;      // Output logits
    ggml_tensor * t_embd;        // Output embeddings
    ggml_tensor * t_embd_pooled; // Pooled embeddings

    // Input tensors (managed by llm_graph_input_i objects)
    std::vector<llm_graph_input_ptr> inputs;

    // Previous graph parameters (for reuse check)
    llm_graph_params params;
};
```

### llm_graph_context

From [src/llama-graph.h:537](../../src/llama-graph.h#L537):

Helper object used during graph construction:

```cpp
struct llm_graph_context {
    // Model configuration
    const llm_arch arch;
    const llama_hparams & hparams;
    const llama_cparams & cparams;
    const llama_ubatch  & ubatch;

    // Dimensions extracted from hparams
    const int64_t n_embd, n_layer, n_head, n_head_kv;
    // ... many more

    // GGML context for building
    ggml_context * ctx0;
    ggml_cgraph  * gf;

    // Backend scheduler
    ggml_backend_sched_t sched;

    // Memory context (KV cache, recurrent state, etc.)
    const llama_memory_context_i * mctx;

    // Helper methods for building common patterns
    ggml_tensor * build_norm(...);
    ggml_tensor * build_ffn(...);
    ggml_tensor * build_attn(...);
    // ... many more
};
```

## Graph Reuse Optimization

Building a graph is expensive (involves allocating tensors, setting up operations, memory planning). llama.cpp aggressively reuses graphs when possible.

### When Can a Graph Be Reused?

From [src/llama-graph.h:427](../../src/llama-graph.h#L427) - `llm_graph_params::allow_reuse()`:

A graph can be reused if the new batch has:

1. **Same batch structure**:
   - Same `n_tokens`
   - Same `n_seqs`
   - Same `n_seq_tokens`
   - Same `equal_seqs` flag
   - Same token vs embedding mode

2. **Same context configuration**:
   - Same `embeddings` mode
   - Same `causal_attn` setting

3. **Same model state**:
   - Same architecture
   - Same graph type (encoder/decoder)
   - Same adapters (LoRA, control vectors)

4. **Same input structure**:
   - All graph inputs can be reused (checked individually)

### Example: Graph Reuse During Generation

```cpp
// First decode: Build graph
batch = llama_batch_get_one(prompt_tokens, prompt_len);
llama_decode(ctx, batch);  // Builds new graph

// Second decode: Reuse graph
batch = llama_batch_get_one(&next_token, 1);
llama_decode(ctx, batch);  // Reuses if single-token structure is same

// Third decode: Reuse again
batch = llama_batch_get_one(&next_token2, 1);
llama_decode(ctx, batch);  // Reuses again
```

**Performance impact**: Building graphs can take 10-100ms; reusing takes <1ms

### Graph Input Objects

Each type of input (tokens, positions, KV cache indices, etc.) has a corresponding `llm_graph_input_i` object that knows:
- How to allocate its tensor
- How to populate its tensor from a batch
- Whether it can be reused for a new batch

From [src/llama-graph.h:79](../../src/llama-graph.h#L79):

```cpp
class llm_graph_input_i {
    virtual void set_input(const llama_ubatch * ubatch) = 0;
    virtual bool can_reuse(const llm_graph_params & params) { return false; }
};
```

Examples:
- `llm_graph_input_embd`: Tokens/embeddings
- `llm_graph_input_pos`: Position indices
- `llm_graph_input_attn_kv`: KV cache indices and attention mask
- `llm_graph_input_out_ids`: Output token indices

## Model-Specific Graph Building

Each model architecture has its own graph building function. These are defined in [src/models/*.cpp](../../src/models/).

### Example: LLaMA Graph Building

Simplified excerpt from [src/models/llama.cpp](../../src/models/llama.cpp):

```cpp
struct llm_build_context llm_build_llama(llm_graph_context & ctx) {
    const auto & model = ctx.model;
    const auto & hparams = ctx.hparams;

    // Build input embeddings
    struct ggml_tensor * cur = ctx.build_inp_embd(model.tok_embd);
    struct ggml_tensor * inpL = cur;

    // Build layers
    for (int il = 0; il < n_layer; ++il) {
        // Attention
        struct ggml_tensor * attn_norm_output =
            ctx.build_norm(cur, model.layers[il].attn_norm, ...);

        cur = ctx.build_attn(
            inp_kv,                          // KV cache input
            model.layers[il].wo,             // Output projection
            nullptr,                         // No bias
            q_cur, k_cur, v_cur,            // Q, K, V projections
            nullptr,                         // No bias
            nullptr,                         // No sinks
            nullptr,                         // No MLA
            kq_scale,                        // Attention scale
            il                               // Layer index
        );

        cur = ggml_add(ctx.ctx0, cur, inpL);  // Residual connection

        // Feed-forward
        struct ggml_tensor * ffn_inp = cur;
        cur = ctx.build_norm(cur, model.layers[il].ffn_norm, ...);

        cur = ctx.build_ffn(
            cur,
            model.layers[il].ffn_up,
            nullptr, nullptr,                // No up bias/scale
            model.layers[il].ffn_gate,
            nullptr, nullptr,                // No gate bias/scale
            model.layers[il].ffn_down,
            nullptr, nullptr,                // No down bias/scale
            nullptr,                         // No act scales
            LLM_FFN_SILU,                    // Activation: SiLU
            LLM_FFN_PAR,                     // Gate type: parallel
            il
        );

        cur = ggml_add(ctx.ctx0, cur, ffn_inp);  // Residual connection
        inpL = cur;
    }

    // Final norm and output
    cur = ctx.build_norm(cur, model.output_norm, ...);
    cur = ggml_mul_mat(ctx.ctx0, model.output, cur);

    // Set output tensor
    ctx.res->t_logits = cur;

    return ctx;
}
```

### Common Building Blocks

From [src/llama-graph.h:598-829](../../src/llama-graph.h#L598-L829):

#### Normalization
```cpp
ggml_tensor * build_norm(
    ggml_tensor * cur,     // Input tensor
    ggml_tensor * mw,      // Weight
    ggml_tensor * mb,      // Bias
    llm_norm_type type,    // LLM_NORM, LLM_NORM_RMS, etc.
    int il                 // Layer index
);
```

#### Feed-Forward Network
```cpp
ggml_tensor * build_ffn(
    ggml_tensor * cur,
    ggml_tensor * up, gate, down,  // FFN weights
    llm_ffn_op_type type_op,       // SILU, GELU, SWIGLU, etc.
    llm_ffn_gate_type type_gate,   // Sequential or parallel
    int il
);
```

#### Attention
```cpp
ggml_tensor * build_attn(
    llm_graph_input_attn_kv * inp,  // KV cache input
    ggml_tensor * wo,                // Output projection
    ggml_tensor * q_cur, k_cur, v_cur,  // Q, K, V
    float kq_scale,                  // Attention scale
    int il
);
```

#### Mixture of Experts (MoE)
```cpp
ggml_tensor * build_moe_ffn(
    ggml_tensor * cur,
    ggml_tensor * gate_inp,      // Expert gate
    ggml_tensor * up_exps,       // Expert up projections
    ggml_tensor * gate_exps,     // Expert gates
    ggml_tensor * down_exps,     // Expert down projections
    int64_t n_expert,
    int64_t n_expert_used,
    llm_ffn_op_type type_op,
    int il
);
```

## Graph Execution

### Backend Scheduling

llama.cpp uses `ggml_backend_sched` to:
1. **Allocate memory** for intermediate tensors
2. **Assign operations** to backends (CPU, CUDA, Metal, etc.)
3. **Schedule execution** order
4. **Handle data transfers** between backends

```cpp
ggml_backend_sched_graph_compute(sched, gf);
```

The scheduler:
- Analyzes graph dependencies
- Allocates temporary buffers
- Splits operations across available backends
- Executes in topological order
- Handles CPU↔GPU transfers when needed

### Execution Flow

```
1. ggml_backend_sched_graph_compute()
   ↓
2. Memory allocation for intermediate tensors
   ↓
3. For each operation in topological order:
   a. Select backend (CPU/GPU)
   b. Transfer inputs if needed
   c. Execute operation
   d. Transfer outputs if needed
   ↓
4. Return (results in output tensors)
```

### Synchronization

After scheduling execution, you must wait for completion:

```cpp
// Non-blocking: schedule execution
ggml_backend_sched_graph_compute(sched, gf);

// Blocking: wait for completion
llama_synchronize(ctx);

// Now safe to read results
float * logits = llama_get_logits(ctx);
```

## Performance Considerations

### Graph Building Cost

| Operation | Time | When |
|-----------|------|------|
| Build new graph | 10-100ms | First batch, or when batch structure changes |
| Reuse graph | <1ms | Same batch structure |
| Set inputs | <0.1ms | Every batch |

**Recommendation**: Keep batch structures consistent when possible.

### Memory Usage

Graphs allocate memory for:
- **All intermediate tensors**: Every layer's activation
- **Attention KV storage**: Already in KV cache
- **Temporary buffers**: Backend-specific

**Memory usage**: ~100MB-1GB depending on:
- Batch size
- Sequence length
- Model size
- Number of layers

### Backend Selection

The scheduler automatically selects backends, but you can influence it:

```cpp
// Offload specific operations to GPU
ggml_backend_sched_set_tensor_backend(sched, tensor, backend_gpu);

// Set number of layers on GPU
params.n_gpu_layers = 32;  // First 32 layers on GPU
```

### Graph Optimization

GGML performs several optimizations:
- **Operation fusion**: Combine multiple ops (e.g., matmul + bias + activation)
- **Memory reuse**: Reuse buffers for tensors with non-overlapping lifetimes
- **Layout optimization**: Arrange tensors for better cache locality

### Debugging Graphs

Enable graph visualization:

```cpp
// Set environment variable
export LLAMA_GRAPH_RESULT_DEBUG=1

// Graphs will be printed/saved for inspection
```

Dump graph to file:

```cpp
ggml_graph_dump_dot(gf, nullptr, "graph.dot");
// Convert: dot -Tpng graph.dot -o graph.png
```

## Best Practices

1. **Consistent batch sizes**: Try to keep batch sizes consistent to enable graph reuse

2. **Warm up**: Run a dummy batch first to build the graph, then reuse for actual inference

3. **Profile**: Use backend-specific profilers (nvprof, Instruments, etc.) to identify bottlenecks

4. **Offload wisely**: Not all operations benefit from GPU; scheduler usually makes good choices

5. **Monitor memory**: Watch for graph memory usage with large batches

6. **Test architectures**: Each model architecture has unique graph structure; test thoroughly

## Advanced: Adding a New Architecture

When adding support for a new model architecture:

1. **Create graph builder**: Add `llm_build_<arch>()` function in [src/models/](../../src/models/)

2. **Register architecture**: Add to architecture enum in [src/llama-arch.h](../../src/llama-arch.h)

3. **Implement forward pass**: Use `llm_graph_context` helper methods

4. **Test graph**: Verify outputs match reference implementation

5. **Optimize**: Add model-specific optimizations

See [docs/development/HOWTO-add-model.md](../../docs/development/HOWTO-add-model.md) for details.

---

**Previous**: [KV Cache System](03-kv-cache.md) | **Next**: [Sampling and Decoding](05-sampling.md)
