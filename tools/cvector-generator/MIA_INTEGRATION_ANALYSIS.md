# Llama.MIA Integration Analysis

Analysis of Llama.MIA's interpretability features and integration plan for modern llama.cpp.

---

## Executive Summary

**Llama.MIA Status:**
- 10 commits ahead, 5744 commits behind llama.cpp master
- Last updated ~2 years ago (circa late 2023)
- Uses old callback system that differs from modern `ggml_backend_sched_eval_callback`

**Key Finding:**
MIA's callback mechanism is **simpler but less flexible** than modern llama.cpp's system. However, **all MIA features can be ported** to use the modern callback API.

---

## MIA's Callback Architecture

###

 Old System (MIA's Approach)

**Type Definitions** (`mia_mod/ggml.h`):
```cpp
// MIA_DEV
typedef void (*ggml_compute_callback)(struct ggml_tensor * tensor);
typedef void (*ggml_compute_init_callback)(struct ggml_cgraph * cgraph);
```

**Storage** (`mia_mod/llama.cpp:1523-1524`):
```cpp
struct llama_context {
    // ... other members ...
    ggml_compute_callback ggml_cb = NULL;
    ggml_compute_init_callback ggml_init_cb = NULL;
};
```

**Registration** (`mia_mod/llama.cpp:1536-1542`):
```cpp
void add_ggml_callback(struct llama_context *ctx, ggml_compute_callback cb) {
    ctx->ggml_cb = cb;
}

void add_ggml_init_callback(struct llama_context *ctx, ggml_compute_init_callback cb) {
    ctx->ggml_init_cb = cb;
}
```

**Usage in mia.cpp** (lines 289-290):
```cpp
add_ggml_callback(ctx, tensor_process_callback);
add_ggml_init_callback(ctx, init_callback);
```

**Assignment to cgraph** (`mia_mod/llama.cpp:5941-5944`):
```cpp
// In llama_build_graph_impl, after building graph:
result->cb = lctx.ggml_cb;  // Assign tensor callback to cgraph

if (lctx.ggml_init_cb) {
    lctx.ggml_init_cb(result);  // Call init callback once
}
```

**Invocation in ggml** (`mia_mod/ggml.c:16343-16344, 16380-16381`):
```cpp
// During graph computation, for each node:
if (cgraph->cb) {
    cgraph->cb(node);  // Call callback on each tensor
}
```

**Key Characteristics:**
- ✅ Simple: One callback per tensor after computation
- ✅ Init callback: Called once when graph is built
- ❌ No ask/deliver protocol
- ❌ No filtering by callback
- ❌ Always called for ALL tensors
- ❌ Stored in cgraph struct (requires ggml.h modification)

---

## Modern llama.cpp Callback System

### New System (Current llama.cpp)

**Type Definition** (`include/ggml-backend.h:303`):
```cpp
typedef bool (*ggml_backend_sched_eval_callback)(
    struct ggml_tensor * t,
    bool ask,
    void * user_data
);
```

**Storage** (`common/common.h` - common_params):
```cpp
struct common_params {
    // ... other params ...
    ggml_backend_sched_eval_callback cb_eval = nullptr;
    void * cb_eval_user_data = nullptr;
};
```

**Integration** (in llama_init_from_params):
```cpp
// Passed to llama_context_params
llama_context_params cparams = ...;
cparams.cb_eval = params.cb_eval;
cparams.cb_eval_user_data = params.cb_eval_user_data;
```

**Two-Phase Protocol:**
1. **Ask phase** (`ask=true`): Callback returns true if it wants this tensor
2. **Deliver phase** (`ask=false`): Tensor data is provided for inspection

**Key Characteristics:**
- ✅ Flexible filtering (callback chooses which tensors)
- ✅ User data pointer (no globals needed)
- ✅ No ggml.h modifications needed
- ✅ Works with backend scheduler
- ❌ More complex protocol
- ❌ No dedicated init callback

---

## Comparison: MIA vs Modern

| Feature | MIA System | Modern System |
|---------|------------|---------------|
| **Callback signature** | `void callback(tensor)` | `bool callback(tensor, ask, user_data)` |
| **Filtering** | No - all tensors | Yes - ask/deliver protocol |
| **User data** | Global variables | Passed as parameter |
| **Init callback** | Yes - separate function | No - do in first ask |
| **ggml.h changes** | Required | Not required |
| **Complexity** | Simple | Moderate |
| **Flexibility** | Limited | High |
| **Used by** | MIA fork only | cvector-generator, eval-callback |

---

## MIA Features Analysis

### 1. Logit Lens

**Implementation** (`mia.cpp:170-177`):
```cpp
// logit lens
if ((strstr(t->name, mia.ll_layer.c_str()) && !mia.ll_layer.empty()) || (mia.ll_layer == "all")) {
    printf("\nunembed LN %d %s:\n", layer_num, t->name);
    for (int y = 0; y < ny; y++) {
        unembed(t, y);
    }
    printf("\n");
}
```

**How it works:**
1. Filter tensors by name (e.g., "l_out", "kqv_out", "ffn_out")
2. For each token position, apply unembedding transformation
3. Print top-K predicted tokens

**`unembed()` function**: Applies `output_norm` → `output` to get logits

**Parameters:**
- `--ll <layer_name>` - which tensor to unembed (e.g., "l_out", "kqv_out")
- `--ll all` - unembed all matching tensors
- `--ll-topk <K>` - show top K tokens (default: 10)

**Port to modern llama.cpp:**
- Easy - just filter by tensor name in callback
- Need to extract `output_norm` and `output` weights (can do in init)
- Modern equivalent: Similar to what cvector-generator does

---

### 2. Attention Visualization

**Implementation** (`mia.cpp:180-226`):
```cpp
// draw attention
if (ggml_n_dims(t) == 3) {  // 3D tensor = attention
    for (int z = 0; z < nz; z++) {  // For each head
        if (mia.draw && strstr(t->name, "kq_soft_max")) {
            // Draw attention pattern
            // Create heatmap visualization
            // Save as PNG
        }
    }
}
```

**How it works:**
1. Filter for `kq_soft_max` tensors (attention scores after softmax)
2. Extract 3D tensor: [n_tokens_q, n_tokens_k, n_heads]
3. For each head, create heatmap
4. Use OpenCV to generate PNG visualization

**Parameters:**
- `--draw` - enable visualization
- `--draw-path <path>` - output PNG filename

**Dependencies:**
- OpenCV (cv::Mat, cvCreateMat, etc.)

**Port to modern llama.cpp:**
- **Challenge:** `kq_soft_max` tensor may not exist in modern llama.cpp
  - Flash attention fuses operations
  - Attention scores not materialized
- **Solution:** Would need to modify graph building to explicitly save attention scores
- **Alternative:** Extract Q, K, V and manually compute attention

---

### 3. Attention Head Ablation

**Implementation** (`mia.cpp:194-217`):
```cpp
int head_i = (layer_num * 32 + z);  // Global head index
bool do_ablate = false;

// Check if this head should be ablated
for (int i = 0; i < mia.ablate_array.size(); i++) {
    if (mia.ablate_array[i] == head_i) {
        do_ablate = true;
    }
}

// Zero out attention scores
for (int y = 0; y < ny; y++) {
    for (int x = 0; x < ny; x++) {
        float *vp = (float *) ((char *) t->data + z*t->nb[2] + y*t->nb[1] + x*t->nb[0]);
        if (do_ablate) {
            *vp = 0.0f;  // Zero ablation
        }
    }
}
```

**How it works:**
1. User specifies head indices to ablate: `-a 0,1,2,65`
2. During inference, zero out attention scores for those heads
3. Measure effect on output

**Head Selection** (`-s <layer> <index>`):
```cpp
if (mia.select_layer >= 0 && mia.select_index >= 0) {
    if (z != mia.select_index && layer_num == mia.select_layer) {
        do_ablate = true;  // Ablate all EXCEPT this one head
    }
}
```

**Parameters:**
- `-a <indices>` - comma-separated list of head indices to ablate
- `-s <layer> <index>` - isolate one head (ablate all others in layer)

**Port to modern llama.cpp:**
- Same challenge as attention viz (need access to attention scores)
- Could also ablate at Q, K, or V level (easier to access)

---

### 4. Activation Patching

**Implementation** (`mia.cpp:122-168`):
```cpp
// patch a tensor with values read from disk
if (!mia.patch_layer_name.empty() && strstr(t->name, mia.patch_layer_name.c_str())) {
    // Load tensor from file
    FILE *fin = fopen(mia.patch_layer_filename1.c_str(), "rb");
    // ... read data ...

    // Patch specific token position
    char *src = (buf + mia.patch_from*t->nb[1]);
    char *dst = ((char *)t->data + mia.patch_to*t->nb[1]);
    memcpy(dst, src, patch_size);

    // Optional: average with second file
    if (!mia.patch_layer_filename2.empty()) {
        // ... average two tensors ...
    }
}
```

**How it works:**
1. Save activations from "clean" run: `--save <layer_name> <file>`
2. Run "corrupted" input, patch in clean activations: `--patch <layer_name> <file1> <from_idx> <to_idx>`
3. Optional averaging: `--patch-avg <file2>`

**Use case:**
Activation patching for causal analysis:
```bash
# Clean run
./mia -m model.gguf -p "The Eiffel Tower is in Paris" --save l_out-15 clean.bin

# Corrupted run with patch
./mia -m model.gguf -p "The Eiffel Tower is in London" --patch l_out-15 clean.bin 20 20
```

**Parameters:**
- `--save <layer> <file>` - save tensor to file
- `--patch <layer> <file> <from_idx> <to_idx>` - patch token position
- `--patch-avg <file2>` - average with second tensor

**Port to modern llama.cpp:**
- ✅ Easy - can do with modern callback
- Just need tensor I/O utilities
- Modern equivalent: Could extend cvector-generator

---

### 5. Tensor Save/Load

**Implementation** (`mia.cpp:113-120`):
```cpp
// save a tensor to disk
if (!mia.save_layer_name.empty() && strstr(t->name, mia.save_layer_name.c_str())) {
    FILE *fout = fopen(mia.save_layer_filename.c_str(), "wb");
    const size_t size = ggml_nbytes(t);
    int r = fwrite(t->data, sizeof(char), size, fout);
    printf("\nsave tensor %s to %s size %d\n", ...);
    fclose(fout);
}
```

**Port to modern llama.cpp:**
- ✅ Trivial - just file I/O in callback

---

### 6. Computation Graph Inspection

**Implementation** (`mia.cpp:253-255`):
```cpp
if (mia.print_cgraph) {
    ggml_graph_export(cgraph, 0);  // Export graph structure
}
```

**What it does:**
Prints all tensors, operations, dimensions, and connections in the computation graph.

**Parameters:**
- `--print-cgraph` - dump graph structure

**Port to modern llama.cpp:**
- ✅ Easy - can access graph structure in modern callbacks too

---

## Integration Strategy

### Option 1: Port MIA Features to Modern Callback System

**Approach:**
Reimplement MIA features using `ggml_backend_sched_eval_callback`.

**Pros:**
- ✅ No ggml.h modifications
- ✅ Works with current llama.cpp
- ✅ Clean integration
- ✅ Can merge upstream

**Cons:**
- ❌ Need to rewrite features
- ❌ No init callback (simulate with state machine)
- ❌ Attention features require graph modifications

**Implementation Plan:**

1. **Create new tool:** `llama-interp` (or extend cvector-generator)

2. **Implement logit lens:**
```cpp
static bool interp_callback(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * data = (interp_data *) user_data;

    if (ask) {
        // Filter for layer outputs, attention outputs, etc.
        return strstr(t->name, "l_out") || strstr(t->name, "kqv_out");
    }

    // Deliver phase: apply logit lens
    if (!data->output_weights_loaded) {
        // First time: extract output weights from model
        data->load_output_weights(model);
        data->output_weights_loaded = true;
    }

    logit_lens(t, data->output_norm, data->output);
    return true;
}
```

3. **Implement activation patching:**
```cpp
// Similar structure, but modify tensor data
if (matches_patch_layer(t->name)) {
    load_and_patch(t, patch_file, from_idx, to_idx);
}
```

4. **Implement tensor save:**
```cpp
if (matches_save_layer(t->name)) {
    save_tensor_to_file(t, save_file);
}
```

5. **Attention visualization:**
- Requires modifying graph building to save attention scores
- OR: Extract Q, K, V and manually compute

**Estimated Effort:**
- Logit lens: 1-2 days
- Activation patching: 1-2 days
- Tensor save/load: 0.5 days
- Graph inspection: 0.5 days
- Attention visualization: 3-5 days (requires graph mods)
- **Total: 6-10 days**

---

### Option 2: Port MIA Callback System to Modern llama.cpp

**Approach:**
Add MIA-style simple callbacks alongside modern callbacks.

**Pros:**
- ✅ Minimal changes to MIA code
- ✅ Faster porting
- ✅ Keep init callback

**Cons:**
- ❌ Requires ggml.h modifications
- ❌ Won't merge upstream
- ❌ Maintenance burden
- ❌ Two callback systems to maintain

**Not recommended** - better to port features properly.

---

### Option 3: Hybrid Approach

**Approach:**
1. Port easy features (logit lens, patching, save/load) to modern callbacks
2. Create separate attention analysis tool that modifies graph
3. Maintain MIA fork for bleeding-edge experiments

**Pros:**
- ✅ Get most features quickly
- ✅ Clean modern implementation
- ✅ Can upstream non-attention features
- ✅ Keep MIA fork for research

**Cons:**
- ⚠️ Attention features require separate workflow

**Recommended** for practical use.

---

## Detailed Feature Port Plans

### Logit Lens (High Priority, Easy)

**Modern Implementation:**
```cpp
struct logit_lens_data {
    bool initialized = false;
    ggml_tensor * output_norm = nullptr;
    ggml_tensor * output = nullptr;
    ggml_context * ctx_weights = nullptr;
    int top_k = 10;
    std::string target_layer = "l_out";
};

static bool logit_lens_callback(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * data = (logit_lens_data *) user_data;

    if (ask) {
        return strstr(t->name, data->target_layer.c_str()) != nullptr;
    }

    // First time: load weights
    if (!data->initialized) {
        // Extract output_norm and output tensors from model
        // (Need to access model somehow - pass in user_data)
        data->initialized = true;
    }

    // Copy tensor from GPU if needed
    size_t n_bytes = ggml_nbytes(t);
    std::vector<float> activations(n_bytes / sizeof(float));
    ggml_backend_tensor_get(t, activations.data(), 0, n_bytes);

    // Apply: norm(activations) -> output projection -> logits
    // Print top-k tokens

    return true;
}
```

**Usage:**
```bash
./llama-interp -m model.gguf -p "Test" --logit-lens l_out --logit-lens-topk 10
```

**Estimated time:** 1-2 days

---

### Activation Patching (High Priority, Medium)

**Modern Implementation:**
Similar to logit lens, but modify tensor data instead of read-only analysis.

**Key challenge:** Need to modify tensor data **before** it's used.
- Modern callback happens after computation
- Need to patch **during** graph building or **before** computation

**Solution:** Use state serialization
```bash
# Clean run - save full state
./llama-cli -m model.gguf -p "Clean prompt" --save-state clean.state

# Corrupted run - load and continue
./llama-cli -m model.gguf -p "Corrupted" --load-state clean.state
```

**Alternative:** Modify graph building to inject patching operations.

**Estimated time:** 2-3 days

---

### Tensor Save/Load (Low Priority, Easy)

**Trivial implementation** - just file I/O in callback.

**Estimated time:** 0.5 days

---

### Attention Visualization (Low Priority, Hard)

**Requires graph modifications** to expose attention scores.

**Two approaches:**

**Approach A: Modify graph building**
```cpp
// In build_attn(), add option to save attention scores:
if (params.save_attention_scores) {
    ggml_tensor * scores = ggml_mul_mat(ctx0, q, k);
    scores = ggml_scale(ctx0, scores, 1.0/sqrt(d));
    scores = ggml_soft_max(ctx0, scores);
    cb(scores, "attn_scores", il);  // Now accessible!
    cur = ggml_mul_mat(ctx0, scores, v);
} else {
    cur = ggml_flash_attn(...);  // Fast path
}
```

**Approach B: Extract Q, K, V and compute manually**
```cpp
// Capture Q, K, V in callback
// Offline: compute attention = softmax(Q·K^T / √d)
// Visualize
```

**Estimated time:** 3-5 days (graph mods) or 2-3 days (manual computation)

---

## Recommended Integration Plan

### Phase 1: Quick Wins (Week 1)
1. ✅ Create `llama-interp` tool skeleton
2. ✅ Implement logit lens
3. ✅ Implement tensor save/load
4. ✅ Test with Llama models

**Deliverable:** Working logit lens tool

### Phase 2: Activation Analysis (Week 2)
1. ✅ Implement activation patching via state save/load
2. ✅ Add activation statistics (like imatrix)
3. ✅ Add computation graph inspection

**Deliverable:** Activation patching for circuit discovery

### Phase 3: Attention (Week 3-4)
1. ⚠️ Decide: Graph mods vs manual computation
2. ✅ Implement Q/K/V extraction
3. ✅ Implement attention visualization (offline or online)
4. ✅ Add attention head ablation

**Deliverable:** Complete interpretability toolkit

### Phase 4: Documentation & Release (Week 4-5)
1. ✅ Write comprehensive documentation
2. ✅ Create examples and tutorials
3. ✅ Add to main llama.cpp (PR for non-attention features)
4. ✅ Maintain separate branch for attention features

---

## Code Modification Checklist

### No Modifications Needed:
- ✅ ggml.h
- ✅ ggml.c
- ✅ llama.h (API)
- ✅ llama.cpp (core)

### New Code Only:
- ✅ `examples/llama-interp/` or `tools/llama-interp/`
- ✅ Uses existing callback system
- ✅ Self-contained interpretability features

### Optional Graph Modifications (for attention):
- ⚠️ `src/llama-graph.cpp` - add attention score saving
- ⚠️ Only if want online attention visualization
- ⚠️ Could be optional flag: `--save-attention-scores`

---

## Comparison: Modern Port vs MIA Fork

| Feature | MIA Fork | Modern Port |
|---------|----------|-------------|
| **Up to date** | ❌ 5744 commits behind | ✅ Current |
| **GPU support** | ❌ CPU only | ✅ Full GPU |
| **Logit lens** | ✅ Working | ✅ Easy to port |
| **Attn viz** | ✅ Working | ⚠️ Requires work |
| **Ablation** | ✅ Working | ⚠️ Requires work |
| **Patching** | ✅ Working | ✅ Via state save/load |
| **Maintenance** | ❌ Unmaintained | ✅ Active |
| **Merge upstream** | ❌ No | ✅ Possible |
| **Complexity** | ✅ Simple | ⚠️ Moderate |

**Recommendation:** Port features to modern llama.cpp. Keep MIA fork for reference but don't try to merge it.

---

## Next Steps

### Immediate (This Week):
1. ✅ Create `examples/llama-interp/` or extend cvector-generator
2. ✅ Implement logit lens using modern callbacks
3. ✅ Test on Llama models

### Short-term (Next 2 Weeks):
4. ✅ Add activation patching
5. ✅ Add tensor save/load
6. ✅ Document usage

### Medium-term (Next Month):
7. ⚠️ Decide on attention feature approach
8. ✅ Implement Q/K/V extraction
9. ✅ Create visualization tools

### Long-term:
10. ✅ Propose upstream merge for non-attention features
11. ✅ Maintain attention features as optional extension
12. ✅ Build interpretability toolkit ecosystem

---

## Conclusion

**MIA's callback system is obsolete** but its **features are valuable**.

**Best path forward:**
- ✅ Port MIA features to modern callback system
- ✅ Start with logit lens (easy, high value)
- ✅ Add activation patching via state management
- ⚠️ Attention features need graph modifications or manual computation
- ✅ Create standalone interpretability tool
- ✅ Can merge most features upstream

**Estimated total effort:** 3-4 weeks for complete port.

**Quick win:** Logit lens working in 1-2 days.

**Would you like me to start implementing the logit lens feature using modern callbacks?**
