# Mechanistic Interpretability in llama.cpp

A comprehensive guide to what's possible for DIY mechanistic interpretability research with locally-run models in llama.cpp.

---

## Table of Contents

1. [Overview](#overview)
2. [What Exists](#what-exists)
3. [What's Missing](#whats-missing)
4. [Core Hook Systems](#core-hook-systems)
5. [Existing Interpretability Tools](#existing-interpretability-tools)
6. [Building Custom Tools](#building-custom-tools)
7. [Research Ideas](#research-ideas)
8. [Limitations](#limitations)

---

## Overview

**llama.cpp is well-suited for mechanistic interpretability** because:

- ✅ Full control over inference pipeline
- ✅ Access to all intermediate computations via callbacks
- ✅ CPU/GPU support for large-scale analysis
- ✅ Tensor inspection APIs for detailed analysis
- ✅ State serialization for reproducibility
- ✅ Active codebase with interpretability tools already implemented

**Key Advantage:** Unlike cloud APIs, you have complete access to model internals - every activation, every attention weight, every intermediate computation can be intercepted and analyzed.

---

## What Exists

### 1. Layer Activation Extraction

**Status:** ✅ **Fully Supported**

**How:**
- Use `ggml_backend_sched_eval_callback` to intercept layer outputs
- Filter tensors by name (e.g., `l_out` for layer outputs)
- Copy tensor data from GPU/CPU to host memory
- Process and save activations

**Example Implementation:** [cvector-generator](cvector-generator.cpp)

**What You Can Do:**
- Extract activations for any layer
- Compare activations across different prompts
- Compute activation statistics (mean, variance, sparsity)
- Track activation changes during generation
- Identify "steering" directions (control vectors)

---

### 2. Embedding Analysis

**Status:** ✅ **Fully Supported**

**APIs:**
```cpp
// Get embeddings for all tokens
float* llama_get_embeddings(ctx)

// Get embeddings for specific token
float* llama_get_embeddings_ith(ctx, token_idx)

// Get embeddings for specific sequence
float* llama_get_embeddings_seq(ctx, seq_id)
```

**Tools:**
- [embedding example](../../examples/embedding/) - Extract and normalize embeddings
- [retrieval example](../../examples/retrieval/) - Cosine similarity search

**What You Can Do:**
- Extract token/sequence embeddings
- Compute semantic similarity
- Build retrieval systems
- Analyze embedding geometry
- Track embedding evolution during generation

---

### 3. Logit Analysis

**Status:** ✅ **Fully Supported**

**APIs:**
```cpp
// Get logits for all tokens
float* llama_get_logits(ctx)

// Get logits for specific token
float* llama_get_logits_ith(ctx, token_idx)
```

**Tools:**
- [perplexity tool](../../tools/perplexity/) - Compute perplexity and log-softmax statistics

**What You Can Do:**
- Analyze token probability distributions
- Compute perplexity metrics
- Study logit lens (interpret intermediate representations as predictions)
- Track confidence/uncertainty

---

### 4. Activation Importance Analysis

**Status:** ✅ **Fully Supported**

**Tool:** [imatrix](../../tools/imatrix/)

**Capabilities:**
- Collects squared activation statistics per tensor
- Computes importance metrics:
  - Sum of squared activations (Σ Act²)
  - Min/Max values
  - Mean (μ) and standard deviation (σ)
  - % Active (sparsity)
  - Entropy and normalized entropy
  - Z-score distribution
  - Cosine similarity between layers

**Use Cases:**
- Identify important neurons/dimensions
- Find sparse/dense layers
- Quantization optimization
- Circuit discovery (which components activate together)

---

### 5. State Serialization

**Status:** ✅ **Fully Supported**

**APIs:**
```cpp
// Get size needed for state
size_t llama_state_get_size(ctx)

// Save full state (logits, embeddings, KV cache)
size_t llama_state_get_data(ctx, dst, size)

// Restore state
size_t llama_state_set_data(ctx, src, size)

// Per-sequence KV cache manipulation
size_t llama_state_seq_get_size(ctx, seq_id)
size_t llama_state_seq_get_data(ctx, dst, size, seq_id)
size_t llama_state_seq_set_data(ctx, src, size, seq_id)
```

**Tool:** [save-load-state example](../../examples/save-load-state/)

**What You Can Do:**
- Save exact model state for reproducibility
- Analyze KV cache contents
- Study how context affects generation
- Implement "rewind" for counterfactual analysis
- Extract and inspect cached attention keys/values

---

### 6. Control Vector Generation & Application

**Status:** ✅ **Fully Supported**

**Tools:**
- [cvector-generator](cvector-generator.cpp) - Extract control vectors
- `llama_apply_adapter_cvec()` - Apply during inference

**Command Line:**
```bash
# Generate vector
./cvector-generator -m model.gguf --method mean -o vector.gguf

# Apply vector
./llama-cli --control-vector-scaled vector.gguf 1.0 --control-vector-layer-range 10 31
```

**What You Can Do:**
- Extract directional features (sentiment, formality, expertise, etc.)
- Steer model behavior along specific dimensions
- Test causal hypotheses (does this direction control X?)
- Compose multiple control vectors
- Study representation geometry

**Research Applications:**
- Representation engineering
- Behavioral steering
- Feature discovery
- Causal intervention testing

---

## What's Missing

### 1. Attention Pattern Visualization

**Status:** ❌ **Not Available**

**What's Missing:**
- No built-in attention score export
- No attention pattern visualization tools
- No per-head attention analysis utilities

**Why It Matters:**
- Attention patterns reveal what the model "looks at"
- Critical for understanding information routing
- Needed for circuit discovery

**Workaround:**
Attention is computed internally via optimized kernels (flash attention). You would need to:
1. Hook into attention computation graph
2. Intercept attention scores before softmax
3. Copy scores from GPU (expensive!)
4. Custom visualization

**Technical Challenge:**
- Attention computed as: `softmax(Q·K^T / √d) · V`
- Most implementations use fused kernels (flash attention) that don't expose intermediate scores
- Named tensors like `kqv_out` give final weighted values, not attention weights
- Would need to modify graph building to explicitly compute and save attention scores

---

### 2. Gradient-Based Analysis

**Status:** ❌ **Not Available**

**What's Missing:**
- No backpropagation through model
- No gradient computation APIs
- No attribution methods (integrated gradients, etc.)

**Why It Matters:**
- Gradients show which inputs most affect outputs
- Needed for saliency maps
- Important for attribution analysis

**Why Not Implemented:**
- llama.cpp is inference-only
- No autograd system
- Designed for speed, not training

**Workaround:**
Use numerical gradients (finite differences):
```
∂f/∂x ≈ (f(x + ε) - f(x - ε)) / (2ε)
```
But very slow for large models.

---

### 3. Automated Circuit Discovery

**Status:** ❌ **Not Available**

**What's Missing:**
- No tools for finding computational circuits
- No automatic subgraph identification
- No path tracing through network

**Why It Matters:**
- Circuits are the "programs" the model runs
- Understanding circuits = understanding how models work
- Key to mechanistic interpretability research

**What You'd Need to Build:**
- Activation patching framework
- Path integration tools
- Automated importance scoring
- Visualization of computational graphs

---

### 4. Per-Head Attention Analysis

**Status:** ❌ **Not Available**

**What's Missing:**
- No direct access to individual attention heads
- No head importance scoring
- No head ablation tools

**Why It Matters:**
- Different heads specialize in different tasks
- Induction heads, copying heads, etc.
- Circuit-level understanding requires head analysis

**Partial Workaround:**
- Can extract Q, K, V tensors via callbacks
- Manually reconstruct per-head attention
- But requires custom implementation

---

## Core Hook Systems

### Primary Callback: `ggml_backend_sched_eval_callback`

**The main mechanism for intercepting computations.**

**Location:** `ggml/include/ggml-backend.h:303`

**Signature:**
```cpp
typedef bool (*ggml_backend_sched_eval_callback)(
    struct ggml_tensor * t,  // The tensor being computed
    bool ask,                // Phase: ask=true (register interest), ask=false (deliver data)
    void * user_data         // Your custom data
);
```

**Two-Phase Protocol:**

**Phase 1: Ask** (`ask=true`)
- Scheduler asks: "Do you want to inspect this tensor?"
- Return `true` if you want it, `false` to skip
- Filter by tensor name: `strncmp(t->name, "l_out", ...) == 0`

**Phase 2: Deliver** (`ask=false`)
- Scheduler provides tensor after computation
- Data may be on GPU - must copy to host
- Do your analysis, logging, etc.
- Return `true` to continue, `false` to abort

**Integration:**
```cpp
// Set callback in common_params
params.cb_eval = my_callback;
params.cb_eval_user_data = &my_data;

// Callback gets passed to llama_context
// Executed for every node in computation graph
```

**Example:** See [eval-callback](../../examples/eval-callback/)

---

### Tensor Naming Conventions

**Filter tensors by name to get specific computations:**

**Layer Outputs:**
- `l_out` - Final output of each layer (what cvector-generator captures)
- Format: `l_out` with layer index accessible via callback context

**Attention Components:**
- `Qcur`, `Kcur`, `Vcur` - Current query, key, value tensors
- `kqv_out` - Attention output (weighted sum of values)
- `attn_norm` - Pre-attention normalization
- `attn_out` - Post-attention projection

**FFN Components:**
- `ffn_norm` - Pre-FFN normalization
- `ffn_out` - FFN output
- `ffn_moe_out` - MoE FFN output (if applicable)

**Input/Output:**
- `inp_embd` - Input embeddings
- `result_norm` - Final layer normalization
- `result_output` - Final logits
- `result_embd` - Final embeddings (for embedding models)

**Architecture-Specific:**
- Names vary by model architecture
- Use eval-callback example to discover tensor names for your model
- Consistent pattern: `{component}_{type}-{layer}`

---

### Tensor Access APIs

**Copy tensor data from device to host:**

```cpp
// Copy full tensor
ggml_backend_tensor_get(tensor, dst, offset, size)

// Check if tensor is host-accessible
bool is_host = ggml_backend_buffer_is_host(tensor->buffer)

// Get tensor size
size_t size = ggml_nbytes(tensor)
```

**Tensor Properties:**
```cpp
t->name        // Tensor name (char*)
t->type        // Data type (F32, F16, BF16, Q4_0, etc.)
t->ne[4]       // Dimensions [dim0, dim1, dim2, dim3]
t->nb[4]       // Strides in bytes
t->src[2]      // Source tensors (for operations)
t->op          // GGML operation type
t->data        // Data pointer (if host-accessible)
```

**Common Patterns:**

```cpp
// Extract 2D tensor (e.g., layer output: [n_embd, n_tokens])
int n_embd = t->ne[0];
int n_tokens = t->ne[1];
size_t n_bytes = ggml_nbytes(t);

std::vector<float> data(n_embd * n_tokens);
ggml_backend_tensor_get(t, data.data(), 0, n_bytes);

// Access element [i, j]
float value = data[j * n_embd + i];
```

---

## Existing Interpretability Tools

### 1. cvector-generator

**Purpose:** Extract directional control vectors from activation differences

**Location:** `tools/cvector-generator/`

**How It Works:**
1. Run paired positive/negative prompts through model
2. Capture layer activations for both using eval callbacks
3. Compute difference: `positive_activations - negative_activations`
4. Apply PCA or mean to extract primary direction
5. Export as GGUF control vector

**Key Code:**
```cpp
// Callback filters for layer outputs
static bool cb_eval(struct ggml_tensor * t, bool ask, void * user_data) {
    const bool is_l_out = strncmp(t->name, "l_out", strlen("l_out")) == 0;
    if (ask) {
        return is_l_out;  // Only interested in layer outputs
    }
    // Copy tensor from GPU
    cb_data->save_tensor_for_layer(t);
    return true;
}
```

**Research Extensions:**
- Modify to extract other statistics (not just mean/PCA)
- Add attention-based filtering
- Implement different dimensionality reduction methods
- Study multi-dimensional subspaces (not just 1D directions)

See [ARCHITECTURE.md](ARCHITECTURE.md) and [ISSUES.md](ISSUES.md) for detailed analysis.

---

### 2. imatrix

**Purpose:** Compute importance matrices for quantization and activation analysis

**Location:** `tools/imatrix/`

**How It Works:**
1. Run calibration data through model
2. Collect squared activation statistics: `Σ(activation²)`
3. Compute per-tensor metrics (entropy, sparsity, etc.)
4. Export importance data

**Key Statistics:**
```cpp
// For each tensor:
- sum_sqr: Σ(act²) - total squared activation
- min/max: activation range
- mean/std: distribution parameters
- pct_active: % of non-zero activations (sparsity)
- entropy: information entropy
- norm_entropy: normalized by max possible
- zd_score: z-score distribution metric
- cos_sim: cosine similarity with other layers
```

**Research Uses:**
- Identify sparse vs dense layers
- Find "superposition" (many features in fewer dimensions)
- Locate important neurons
- Study activation distributions across layers
- Compare activation patterns between different inputs

**Extension Ideas:**
- Add correlation analysis between neurons
- Implement activation clustering
- Track activation changes during generation
- Compute feature importance scores

---

### 3. eval-callback Example

**Purpose:** Demonstrate tensor inspection during inference

**Location:** `examples/eval-callback/`

**What It Does:**
- Prints all operations and tensor data
- Shows tensor shapes, types, names
- Copies data from GPU when needed
- Demonstrates the callback protocol

**Use As:**
- Template for custom analysis tools
- Debugging aid for understanding model graph
- Reference for tensor naming conventions
- Starting point for custom interpretability tools

---

### 4. Embedding & Retrieval Tools

**Purpose:** Extract and analyze semantic embeddings

**Locations:**
- `examples/embedding/` - Embedding extraction
- `examples/retrieval/` - Similarity search

**Capabilities:**
- Extract token or sequence embeddings
- Support different pooling strategies (MEAN, CLS, LAST, RANK)
- Normalize embeddings
- Compute cosine similarity
- Top-k retrieval

**Research Uses:**
- Study embedding geometry
- Analyze semantic clusters
- Track embedding evolution during generation
- Compare embeddings across layers (via callbacks)
- Probe representations for specific concepts

---

### 5. Perplexity Tool

**Purpose:** Compute language modeling metrics

**Location:** `tools/perplexity/`

**What It Measures:**
- Perplexity: `exp(average negative log-likelihood)`
- Log-probabilities of tokens
- Model confidence on held-out data

**Research Uses:**
- Evaluate representation quality
- Test interventions (does changing activations hurt perplexity?)
- Study learning curves
- Analyze model uncertainty

---

## Building Custom Tools

### Recipe 1: Extract All Layer Activations

**Goal:** Save activations from every layer for later analysis

**Code Template:**
```cpp
struct activation_logger {
    std::vector<std::vector<float>> layer_outputs;  // [layer][data]
    int n_layers;
    int current_layer = 0;

    static bool callback(struct ggml_tensor * t, bool ask, void * user_data) {
        auto * logger = (activation_logger *) user_data;

        if (ask) {
            // Only capture layer outputs
            return strncmp(t->name, "l_out", 5) == 0;
        }

        // Copy tensor data
        size_t n_bytes = ggml_nbytes(t);
        std::vector<float> data(n_bytes / sizeof(float));
        ggml_backend_tensor_get(t, data.data(), 0, n_bytes);

        logger->layer_outputs.push_back(std::move(data));
        logger->current_layer++;

        return true;
    }
};

// Usage:
activation_logger logger;
logger.n_layers = llama_model_n_layer(model);
params.cb_eval = activation_logger::callback;
params.cb_eval_user_data = &logger;

// Run inference...
// Now logger.layer_outputs contains all activations
```

**What to do with activations:**
- Compute statistics (mean, variance, sparsity)
- Find neurons that activate on specific inputs
- Cluster similar activation patterns
- Compare activations across different prompts
- Implement activation patching

---

### Recipe 2: Track Attention Components

**Goal:** Extract Q, K, V tensors to analyze attention patterns

**Code Template:**
```cpp
struct attention_tracker {
    struct attention_data {
        std::vector<float> Q, K, V;
        int n_embd_head, n_heads, n_tokens;
    };

    std::vector<attention_data> layers;

    static bool callback(struct ggml_tensor * t, bool ask, void * user_data) {
        auto * tracker = (attention_tracker *) user_data;

        if (ask) {
            // Capture Q, K, V tensors
            return strncmp(t->name, "Qcur", 4) == 0 ||
                   strncmp(t->name, "Kcur", 4) == 0 ||
                   strncmp(t->name, "Vcur", 4) == 0;
        }

        // Determine which component
        char type = t->name[0];  // Q, K, or V
        size_t n_bytes = ggml_nbytes(t);
        std::vector<float> data(n_bytes / sizeof(float));
        ggml_backend_tensor_get(t, data.data(), 0, n_bytes);

        // Store appropriately
        // ... (extract layer index, save to correct location)

        return true;
    }
};
```

**What to do with Q, K, V:**
- Manually compute attention scores: `softmax(Q·K^T / √d)`
- Analyze per-head patterns
- Find copying heads, induction heads, etc.
- Implement attention ablation
- Study attention geometry

**Note:** Flash attention optimization may make this more complex. See limitations section.

---

### Recipe 3: Activation Patching Framework

**Goal:** Replace activations at specific layers to test causal hypotheses

**Approach:**
1. Run model on "clean" input, save activations
2. Run model on "corrupted" input
3. For each layer, patch in activations from clean run
4. Measure effect on output

**Pseudo-code:**
```cpp
// Run 1: Clean input
run_with_callback(clean_prompt, save_activations_callback);
std::vector<tensor> clean_activations = saved_activations;

// Run 2: Corrupted input, patch layer L
struct patch_data {
    int layer_to_patch;
    tensor patch_tensor;
};

bool patching_callback(ggml_tensor * t, bool ask, void * user_data) {
    auto * pd = (patch_data *) user_data;

    if (ask) {
        return check_if_layer_output(t, pd->layer_to_patch);
    }

    // Replace tensor data with patch
    ggml_backend_tensor_set(t, pd->patch_tensor.data, 0, size);
    return true;
}

// Test each layer
for (int L = 0; L < n_layers; L++) {
    patch_data pd = {L, clean_activations[L]};
    float output = run_with_callback(corrupted_prompt, patching_callback, &pd);
    // Record output, compute effect
}
```

**Research Applications:**
- Localization: Which layers are important for task X?
- Circuit discovery: Which paths are critical?
- Causal attribution: Which components cause behavior Y?

**Challenge:** `ggml_backend_tensor_set()` may not work on all backends. May need to modify graph directly.

---

### Recipe 4: Logit Lens Implementation

**Goal:** Interpret intermediate representations as predictions

**Concept:**
- At any layer, activations are in embedding space
- Project through final layer norm + LM head
- See what tokens the model "thinks" at each layer

**Code:**
```cpp
// Extract layer L activation
ggml_tensor * layer_L_output = extract_activation(L);

// Apply final layer norm
ggml_tensor * normed = ggml_rms_norm(ctx, layer_L_output, eps);

// Project through LM head
ggml_tensor * logits = ggml_mul_mat(ctx, model.output, normed);

// Get top-k predictions at layer L
std::vector<token> top_k = get_top_k_tokens(logits, k=10);
```

**What It Reveals:**
- How representations evolve across layers
- When model "decides" on output
- Emergence of correct answer
- Layer specialization

---

### Recipe 5: Neuron Activation Analysis

**Goal:** Find neurons that activate for specific concepts

**Approach:**
1. Collect activations for dataset with labels
2. For each neuron, compute correlation with labels
3. Find maximally activating inputs
4. Visualize neuron function

**Code Sketch:**
```cpp
// Collect activations for many examples
std::map<std::string, std::vector<float>> activations_by_label;

for (auto & example : dataset) {
    run_model(example.text);
    auto acts = extract_layer_activations(layer=15);
    activations_by_label[example.label].push_back(acts);
}

// Find neuron specialization
for (int neuron = 0; neuron < n_embd; neuron++) {
    for (auto & [label, acts] : activations_by_label) {
        float mean_act = compute_mean_activation(acts, neuron);
        // High mean_act = neuron activates for this label
    }
}
```

**Extensions:**
- Automated neuron description (run maximally activating examples through model)
- Causal testing (activate neuron, measure effect)
- Polysemanticity analysis (does neuron have multiple functions?)

---

## Research Ideas

### 1. DIY Sparse Autoencoder (SAE) Training

**Goal:** Find interpretable features via sparse decomposition

**Why Autoencoders:**
- Models use "superposition" - pack many features into fewer dimensions
- SAE decomposes activations into interpretable sparse features
- Each SAE neuron ideally represents one concept

**How with llama.cpp:**

```
1. Extract activations from layer L using callbacks
2. Train autoencoder offline (PyTorch/JAX):
   - Encoder: x → f (sparse features)
   - Decoder: f → x̂ (reconstruction)
   - Loss: ||x - x̂||² + λ||f||₁ (sparsity penalty)
3. Export encoder/decoder as GGUF
4. Use llama.cpp to extract SAE features during inference
5. Analyze feature activation patterns
```

**What You Learn:**
- Interpretable feature dictionary
- Feature polysemanticity (how many concepts per feature)
- Feature composition (which features co-occur)
- Causal features (intervene on features, measure effect)

**Implementation:**
- Use cvector-generator as template for activation extraction
- Train SAE offline
- Build custom callback to apply SAE encoder during inference
- Save feature activations for analysis

---

### 2. Representation Reading & Writing

**Goal:** Decode information from activations and inject new information

**Reading (Linear Probes):**
```
1. Extract activations at layer L
2. Train linear classifier: activation → property
3. Test: Can you decode specific information?
   - Sentiment, topic, named entities, syntax, etc.
4. Compare across layers: When does information emerge?
```

**Writing (Activation Steering):**
```
1. Compute difference vector for property X
2. Add vector to activations at layer L
3. Measure effect on output
4. Iterate: Find optimal layer and magnitude
```

**Combines:**
- Reading: What information is present?
- Writing: Does that information cause behavior?

**llama.cpp Implementation:**
- Reading: Extract activations with callbacks, train probes offline
- Writing: Use control vector system or custom patching callback

---

### 3. Circuit Discovery via Activation Patching

**Goal:** Find minimal computational subgraphs that implement specific behaviors

**Method:**
1. **Identify behavior:** E.g., "model answers factual questions correctly"
2. **Ablation study:** Patch each layer with corrupted activations
3. **Find critical path:** Which layers are necessary?
4. **Zoom in:** Within critical layers, patch attention heads, MLP neurons
5. **Reconstruct circuit:** Map out full computational path

**llama.cpp Advantages:**
- Full control over activations
- Can patch at any granularity (layer, head, neuron)
- Fast iteration with local model

**Example: Factual Recall Circuit**
```
1. Clean: "The Eiffel Tower is in" → "Paris"
2. Corrupted: "The Golden Gate is in" → ???
3. Patch layer L from clean run
4. Does it recover "Paris"? → Layer L is important
5. Repeat for all layers → Find minimal circuit
```

---

### 4. Attention Head Analysis

**Goal:** Understand what different attention heads do

**Method:**
1. Extract Q, K, V from all heads via callbacks
2. Compute attention patterns manually
3. Analyze head behavior:
   - **Copying heads:** Attend to previous token, copy it
   - **Induction heads:** Attend to pattern AB, predict B after A
   - **Previous token heads:** Always attend to immediate predecessor
   - **Syntax heads:** Track grammatical structure

**llama.cpp Implementation:**
```cpp
// Callback captures Qcur, Kcur, Vcur (post-RoPE)
// Tensors are shaped: [n_embd_head, n_heads, n_tokens]

// Extract per-head Q, K, V
for (int head = 0; head < n_heads; head++) {
    auto Q_h = extract_head(Q_tensor, head);
    auto K_h = extract_head(K_tensor, head);

    // Compute attention: scores = softmax(Q·K^T / √d)
    auto scores = softmax(matmul(Q_h, transpose(K_h)) / sqrt(d));

    // Analyze pattern
    classify_head_behavior(scores, head);
}
```

**Discoveries:**
- Head specialization (which heads do what)
- Layer specialization (early = syntax, late = semantics)
- Compositional circuits (how heads chain together)

---

### 5. Feature Visualization via Optimization

**Goal:** Generate inputs that maximally activate specific neurons

**Method:**
1. Select target neuron at layer L
2. Optimize input to maximize activation
3. Interpret resulting text

**Challenge:** llama.cpp is inference-only, no gradients

**Workaround: Discrete Optimization**
```
1. Start with random token sequence
2. For each position:
   a. Try all possible tokens
   b. Measure target neuron activation
   c. Keep token that maximizes activation
3. Iterate until convergence
```

**Alternative: Dataset Search**
```
1. Run model on large corpus
2. Record neuron activations
3. Find examples with highest activation
4. Manually inspect to interpret neuron
```

**llama.cpp Advantage:**
- Fast inference enables trying many examples
- Callback system makes activation tracking easy

---

### 6. Probe Training Across Layers

**Goal:** Track information flow through the network

**Method:**
1. Extract activations from all layers for labeled dataset
2. Train linear probes for each layer:
   - `probe_L: activation_L → property`
3. Plot probe accuracy vs layer
4. Discover:
   - **When** information emerges
   - **Where** information is processed
   - **Which layers** are critical

**Example Properties:**
- Part-of-speech tags
- Named entity types
- Sentiment
- Topic
- Factual knowledge
- Syntax trees

**llama.cpp Workflow:**
```bash
# 1. Extract activations
./my_activation_extractor -m model.gguf -f dataset.txt -o activations/

# 2. Train probes (offline Python)
python train_probes.py --acts activations/ --labels labels.json

# 3. Analyze
python analyze_probe_accuracy.py --results probe_results/
```

---

### 7. Counterfactual Analysis via State Manipulation

**Goal:** Answer "what if" questions by editing model state

**Method:**
1. Run model to checkpoint, save state
2. Edit internal state (activations, KV cache)
3. Continue generation
4. Compare with unedited baseline

**Example:**
```
Baseline: "The Eiffel Tower is in Paris. It is very"
          → "tall"

Edit: Replace "Paris" representation with "London" in KV cache
Continue: "The Eiffel Tower is in [London]. It is very"
          → ??? (if it says "famous" instead of "tall", location matters)
```

**llama.cpp Implementation:**
```cpp
// Save state
size_t state_size = llama_state_get_size(ctx);
std::vector<uint8_t> state(state_size);
llama_state_get_data(ctx, state.data(), state_size);

// Edit state (e.g., modify KV cache)
modify_state(state, edit_spec);

// Restore and continue
llama_state_set_data(ctx, state.data(), state_size);
llama_decode(...);  // Continue generation
```

**Research Questions:**
- Which representations are causal?
- How does context influence generation?
- Can we perform "conceptual surgery"?

---

## Limitations

### 1. No Direct Attention Access

**Problem:**
- Attention scores not directly accessible
- Flash attention fuses QK^T computation with softmax
- Optimized kernels don't expose intermediate values

**Impact:**
- Can't easily visualize attention patterns
- Harder to analyze attention head behavior
- Circuit discovery more difficult

**Workarounds:**
- Extract Q, K, V via callbacks
- Manually recompute attention (slow)
- Modify graph to explicitly save scores (requires code changes)

**Why It's Hard:**
Most models use `build_attn()` which calls `build_attn_mha()`:
```cpp
// Simplified attention computation
cur = build_attn_mha(q, k, v, kq_mask, kq_scale, il);
cb(cur, "kqv_out", il);  // Only final output exposed
```

Attention scores are never materialized as a separate tensor - they're computed inside fused kernel.

**Possible Fix:**
Modify `build_attn_mha()` to optionally save attention scores:
```cpp
if (save_attention_scores) {
    ggml_tensor * scores = ggml_mul_mat(ctx0, q, k);  // QK^T
    scores = ggml_scale(ctx0, scores, 1.0/sqrt(d));
    scores = ggml_soft_max(ctx0, scores);
    cb(scores, "attn_scores", il);  // ← Now accessible!
    cur = ggml_mul_mat(ctx0, scores, v);
} else {
    cur = ggml_flash_attn(...);  // Fast path
}
```

---

### 2. No Gradient Computation

**Problem:**
- llama.cpp is inference-only
- No autograd/backpropagation
- Can't compute gradients efficiently

**Impact:**
- No gradient-based attribution methods
- No saliency maps
- No integrated gradients
- Harder to train probes/SAEs
- Can't do gradient-based optimization

**Workarounds:**
- Numerical gradients (very slow)
- Train components offline (PyTorch/JAX), use for inference
- Discrete optimization instead of gradient descent

**Why Not Added:**
- Inference focus: Speed over training capability
- Complexity: Would require full autograd system
- Use case: Research typically done in Python frameworks

---

### 3. Memory Constraints for Large-Scale Analysis

**Problem:**
- Large models + full activations = lots of memory
- Example: Llama-70B, layer size ~8192
- Saving all layers × all tokens × all examples → TB of data

**Impact:**
- Can't easily save all activations for large datasets
- Need to process in batches
- Trade-off between analysis depth and scale

**Workarounds:**
- Stream processing: Analyze on-the-fly, save statistics only
- Selective extraction: Only save specific layers/components
- Compression: Quantize saved activations
- Chunked datasets: Process in manageable batches

---

### 4. Flash Attention Optimization

**Problem:**
- Flash attention is great for speed
- Bad for interpretability (fused operations)

**Impact:**
- Can't intercept intermediate attention computations
- QK^T scores never materialized
- Harder to analyze attention patterns

**Trade-off:**
- Disable flash attention → slower but interpretable
- Keep flash attention → fast but opaque

**Current Status:**
No easy way to disable flash attention and get explicit scores.

---

### 5. Limited Per-Component Ablation

**Problem:**
- Can patch full layers easily
- Harder to patch individual heads or neurons
- Graph structure is optimized, not granular

**Impact:**
- Circuit discovery requires custom graph modifications
- Can't easily ablate single attention heads
- Neuron-level intervention needs workarounds

**Possible Solutions:**
- Modify graph building to separate components
- Implement custom patching at GGML op level
- Build parallel analysis graph with explicit components

---

## Best Practices for DIY Interpretability

### 1. Start Simple

**Begin with:**
- Layer output extraction (cvector-generator pattern)
- Embedding analysis
- Logit analysis

**Then progress to:**
- Attention component extraction
- Activation patching
- Circuit discovery

### 2. Validate Your Tools

**Always:**
- Test callbacks on small examples first
- Verify tensor shapes match expectations
- Check that data is copied correctly (GPU→CPU)
- Validate results against known baselines

**Common Pitfalls:**
- Forgetting to copy from GPU
- Wrong tensor shapes (row-major vs column-major)
- Incorrect layer indexing
- Float16 vs Float32 precision issues

### 3. Manage Memory

**For large-scale analysis:**
- Process in batches
- Save statistics, not raw activations
- Use streaming where possible
- Compress saved data

**Example:**
```cpp
// Instead of:
std::vector<float> all_activations;  // OOM for large models!

// Do:
struct activation_stats {
    float mean, variance, max, min;
    std::vector<float> top_k_values;
};
std::vector<activation_stats> layer_stats;  // Much smaller
```

### 4. Use Existing Tools as Templates

**cvector-generator is your friend:**
- Study its callback implementation
- Copy its tensor handling patterns
- Adapt its structure for your analysis

**eval-callback shows you:**
- Tensor naming conventions
- GPU↔CPU data transfer
- Callback protocol details

### 5. Document and Share

**Good interpretability research:**
- Documents methods clearly
- Shares code and tools
- Validates findings across models
- Builds on prior work

**Contribute back:**
- Share useful interpretability tools
- Document interesting findings
- Submit PRs for general-purpose utilities
- Help build the interpretability toolkit

---

## Comparison to Other Frameworks

### vs. TransformerLens (Python)

**llama.cpp Advantages:**
- ✅ Much faster inference
- ✅ Runs locally on consumer hardware
- ✅ Lower memory usage (quantization)
- ✅ Full control over C++ internals

**llama.cpp Disadvantages:**
- ❌ No autograd/gradients
- ❌ Less convenient API
- ❌ Fewer built-in interpretability tools
- ❌ Requires C++ knowledge

**Best Use:**
- llama.cpp: Large-scale activation analysis, fast iteration, production deployments
- TransformerLens: Gradient-based methods, rapid prototyping, research

### vs. Neuroscope/Anthropic Tools

**llama.cpp Advantages:**
- ✅ Open source, fully customizable
- ✅ Works with any GGUF model
- ✅ No API costs, full privacy

**llama.cpp Disadvantages:**
- ❌ Fewer pre-built visualizations
- ❌ No web interface
- ❌ Need to build tools yourself

**Best Use:**
- llama.cpp: Privacy-sensitive research, custom analyses, open models
- Anthropic: Claude-specific features, polished UI

### vs. PyTorch Hooks

**llama.cpp Advantages:**
- ✅ Faster inference (optimized C++)
- ✅ Better quantization support
- ✅ Lower memory footprint

**llama.cpp Disadvantages:**
- ❌ More complex setup
- ❌ Fewer examples/tutorials
- ❌ No gradients

**Best Use:**
- llama.cpp: Large models, many examples, speed-critical
- PyTorch: Small models, gradient methods, easy prototyping

---

## Getting Started Checklist

Ready to do mechanistic interpretability with llama.cpp?

- [ ] Compile llama.cpp with desired backend (CUDA/Metal/CPU)
- [ ] Test basic inference with your model
- [ ] Run eval-callback example to see tensor names
- [ ] Start with cvector-generator to understand callbacks
- [ ] Extract layer activations for a small dataset
- [ ] Build a simple analysis tool (activation statistics)
- [ ] Validate results manually on known examples
- [ ] Scale up to larger analyses
- [ ] Document and share your findings!

---

## Further Resources

- [ARCHITECTURE.md](ARCHITECTURE.md) - cvector-generator technical details
- [ISSUES.md](ISSUES.md) - Known issues affecting interpretability
- [BEST_PRACTICES.md](BEST_PRACTICES.md) - Practical usage guide
- [llama.cpp main repo](https://github.com/ggerganov/llama.cpp)
- [examples/](../../examples/) - Reference implementations
- [ggml documentation](https://github.com/ggerganov/ggml)

---

## Contributing

Have you built cool interpretability tools with llama.cpp? We'd love to see:
- New analysis utilities
- Visualization tools
- Circuit discovery methods
- Attention pattern extraction
- SAE implementations
- Probe training frameworks

Share your work in the llama.cpp repository or start your own interpretability toolkit!

**Let's build the open-source mechanistic interpretability stack together.** 🔬🧠
