# Control Vector Generator - Architecture Overview

## Purpose

The `cvector-generator` tool extracts directional feature vectors from language models for representation engineering. These control vectors can steer model behavior along specific dimensions (e.g., sentiment, formality, expertise level).

## How It Works

### High-Level Process

1. **Input**: Pairs of positive/negative text prompts demonstrating contrasting behaviors
2. **Extraction**: Run both prompts through the model, capture hidden layer activations
3. **Difference**: Compute `positive_activations - negative_activations` for each layer
4. **Reduction**: Apply PCA or mean to extract the primary direction of difference
5. **Output**: GGUF file containing one control vector per layer

### Example

```
Positive: "Act extremely happy" → [activations_happy]
Negative: "Act extremely sad"   → [activations_sad]
Difference: activations_happy - activations_sad = [happiness_direction]
```

The resulting vector can be added to model activations during inference to increase "happiness" or subtracted to increase "sadness".

---

## Code Architecture

### File Structure

```
tools/cvector-generator/
├── cvector-generator.cpp  # Main program, orchestration
├── pca.hpp                # PCA via power iteration
├── mean.hpp               # Simple averaging method
├── positive.txt           # Example positive prompts
├── negative.txt           # Example negative prompts
└── completions.txt        # (Currently unused)
```

### Core Components

#### 1. `callback_data` (cvector-generator.cpp:55-173)

Callback handler for capturing layer activations during model inference.

**Key Members:**
- `v_pos` - Vector of tensors storing positive prompt activations (one tensor per layer)
- `v_neg` - Vector of tensors storing negative prompt activations
- `v_diff_filtered` - Difference tensors with zero rows removed
- `ctx_ggml` - GGML context for tensor storage

**Key Methods:**
- `save_tensor_for_layer(t)` - Copies tensor from GPU/CPU to host memory
- `calc_diff()` - Computes `v_pos - v_neg` element-wise
- `filter_nonzero_rows(t)` - Removes rows where all values are near zero
- `reset()` - Clears tensors for next prompt pair

**Lifecycle:**
1. Created once at startup
2. Reused for each prompt pair
3. `is_eval_pos` flag switches between positive/negative capture
4. Reset after each pair is processed

---

#### 2. `train_context` (cvector-generator.cpp:179-268)

Manages the accumulation and processing of differences across multiple prompt pairs.

**Key Members:**
- `positive_entries` / `negative_entries` - Loaded prompt strings
- `v_diff_tmp` - Temporary byte buffers accumulating difference tensors
- `v_diff` - Final concatenated/transposed difference matrices (one per layer)
- `v_final` - Output control vectors (one 1D vector per layer)
- `ctx_ggml` - GGML context for tensors

**Key Methods:**
- `concat_diff_tmp(diff_filtered)` - Appends new differences to temporary buffers
- `build_v_diff(transpose)` - Converts byte buffers to tensors, optionally transposes
- Destructor frees all allocated tensor data

**Data Flow:**
```
[Prompt pair 1] → diff_filtered → concat → v_diff_tmp
[Prompt pair 2] → diff_filtered → concat → v_diff_tmp
[Prompt pair N] → diff_filtered → concat → v_diff_tmp
                                            ↓
                                      build_v_diff()
                                            ↓
                                         v_diff (ready for PCA/mean)
                                            ↓
                                      PCA or mean()
                                            ↓
                                         v_final (output vectors)
```

---

#### 3. `tokenized_prompt` (cvector-generator.cpp:270-294)

Handles tokenization and padding for prompt pairs.

**Process:**
1. Tokenize positive and negative prompts independently
2. Find `max_seq_len = max(len_positive, len_negative)`
3. Pad shorter sequence with space character tokens

**Important:** Both sequences must have identical length for batched processing.

---

#### 4. PCA Implementation (pca.hpp)

Implements Principal Component Analysis via **power iteration** to find the dominant eigenvector.

**Algorithm:**
```
Given: X (difference matrix, shape [n_samples, n_embd])
Goal: Find v such that X^T X v = λv (largest eigenvalue)

1. Compute square matrix: S = X^T X
2. Initialize random unit vector: v₀
3. Iterate:
   v_{k+1} = S v_k
   v_{k+1} = v_{k+1} / ||v_{k+1}||  (normalize)
   distance = ||v_{k+1} - v_k||
   if distance < tolerance: break
4. Return v_final
```

**Key Components:**

##### `pca_model` (pca.hpp:52-132)
- Manages GPU/CPU backend and tensors
- `dev_input` - Input difference matrix (on device)
- `dev_square` - Cached S = X^T X matrix
- `dev_eigenvector` - Current eigenvector estimate
- Initialized with random normalized vector

##### `build_graph_piter()` (pca.hpp:134-192)
- Constructs computation graph for batched iterations
- Batching reduces graph overhead
- Computes square matrix only on first iteration
- Each batch runs `n_batch` normalization steps

##### `compute_piter()` (pca.hpp:194-243)
- Executes the computation graph
- Extracts eigenvectors and convergence distances
- Returns results for inspection

##### `power_iteration()` (pca.hpp:245-294)
- Main entry point
- Runs batched iterations until convergence
- Copies final eigenvector to output tensor

**Backend Support:**
- CUDA: Fully supported
- CPU: Fallback
- Metal: Disabled (waiting for GGML_OP_SQRT support)

**Parameters:**
- `n_threads` - CPU thread count
- `n_batch` - Iterations per graph execution (default: 20)
- `n_iterations` - Total iterations (default: 1000)
- `tolerance` - Convergence threshold (1e-7)

---

#### 5. Mean Implementation (mean.hpp)

Simpler alternative to PCA - just averages all difference vectors.

**Algorithm:**
```
Given: X (difference matrix, shape [n_embd, n_samples])
1. For each embedding dimension i:
   mean[i] = (1/n_samples) * Σ X[i, j]
2. Normalize: mean = mean / ||mean||
```

**When to use:**
- Faster computation
- More stable (no sign ambiguity)
- Good for initial exploration
- Use PCA when you have many prompt pairs and want the "strongest" direction

---

### Main Execution Flow (cvector-generator.cpp:394-508)

```
1. Parse command-line arguments
2. Initialize LLAMA backend and load model
3. Load positive.txt and negative.txt
4. Create train_context(n_embd, n_layers)

5. FOR EACH prompt pair:
   a. Tokenize and pad both prompts
   b. Evaluate positive prompt → capture activations in cb_data.v_pos
   c. Evaluate negative prompt → capture activations in cb_data.v_neg
   d. Calculate difference: cb_data.calc_diff()
   e. Filter zero rows: cb_data.filter_nonzero_rows()
   f. Append to train_context: ctx_train.concat_diff_tmp()
   g. Reset callback data: cb_data.reset()

6. Build final difference matrices:
   ctx_train.build_v_diff(transpose = use_pca)

7. Apply dimensionality reduction:
   IF PCA:
     PCA::run_pca(params, ctx_train.v_diff, ctx_train.v_final)
   ELSE:
     mean::run(ctx_train.v_diff, ctx_train.v_final)

8. Export to GGUF:
   export_gguf(ctx_train.v_final, output_file, model_hint)

9. Cleanup and exit
```

---

## Data Shapes

Understanding tensor dimensions is crucial:

### During Activation Capture
```
tokens_pos: [max_seq_len]               # Token IDs
tokens_neg: [max_seq_len]               # Token IDs

Per layer:
l_out (from model): [n_embd, n_tokens]  # Raw layer output
v_pos[layer]:       [n_embd, n_tokens]  # Saved positive activations
v_neg[layer]:       [n_embd, n_tokens]  # Saved negative activations
```

### After Difference Calculation
```
v_diff_filtered[layer]: [n_embd, n_nonzero_rows]
# n_nonzero_rows ≤ n_tokens (some rows filtered out)
```

### After Concatenation (all prompt pairs)
```
v_diff_tmp[layer]: byte buffer of size (n_embd * total_nonzero_rows * sizeof(float))
```

### After build_v_diff()
```
If transpose=true (PCA):
  v_diff[layer]: [total_nonzero_rows, n_embd]  # Each row is a sample, each col is a feature

If transpose=false (mean):
  v_diff[layer]: [n_embd, total_nonzero_rows]  # Each row is a feature, each col is a sample
```

### Final Output
```
v_final[layer]: [n_embd]  # One direction vector per layer (excluding last layer)
# Total layers in output: n_layers - 1
```

---

## GGUF Export Format

The output file uses GGUF format with metadata:

```
Metadata:
  general.architecture = "controlvector"
  controlvector.model_hint = <model architecture name>
  controlvector.layer_count = n_layers - 1

Tensors:
  direction.1: [n_embd] (layer 0)
  direction.2: [n_embd] (layer 1)
  ...
  direction.N: [n_embd] (layer N-2, since last layer is excluded)
```

The model hint (e.g., "llama", "mistral") helps ensure compatibility when applying vectors.

---

## Memory Management

**Current Approach:**
- Manual `malloc()` for tensor data
- Explicit `free()` in destructors and reset methods
- GGML contexts use no_alloc mode (manual memory management)

**Potential Issues:**
- No RAII (Resource Acquisition Is Initialization)
- Risk of leaks if exceptions occur (though C++ exceptions rarely used in this codebase)
- TODO comments indicate authors want to improve this

---

## Command-Line Interface

**Basic Usage:**
```bash
# CPU only
./cvector-generator -m model.gguf

# GPU accelerated
./cvector-generator -m model.gguf -ngl 99

# Custom files
./cvector-generator -m model.gguf \
  --positive-file my_positive.txt \
  --negative-file my_negative.txt \
  -o my_vector.gguf

# Use mean instead of PCA
./cvector-generator -m model.gguf --method mean

# Advanced PCA tuning
./cvector-generator -m model.gguf \
  --pca-iter 2000 \
  --pca-batch 100 \
  -ngl 99
```

**Applying Control Vectors:**
```bash
./llama-cli -m model.gguf \
  -p "Your prompt here" \
  --control-vector-scaled vector.gguf 0.8 \
  --control-vector-layer-range 10 31
```

Scale factor (0.8) controls strength; layer range specifies which layers to affect.

---

## Design Rationale

### Why difference vectors?
Subtracting activations isolates the "direction" of the behavioral difference while canceling out common features.

### Why filter zero rows?
Tokens where positive and negative activations are identical contribute no directional information and would dilute the signal.

### Why PCA?
With multiple prompt pairs, you get many difference vectors. PCA finds the single direction that captures the most variance (i.e., the most consistent difference across examples).

### Why mean?
Simpler, faster, and more interpretable. If all your prompt pairs are consistent, mean and PCA should give similar results.

### Why exclude last layer?
The final layer's activations are very close to token predictions - too specific for general behavioral steering. Middle layers capture more abstract semantic features.

### Why normalize vectors?
Allows scaling factor to have consistent meaning across different control vectors and models.
