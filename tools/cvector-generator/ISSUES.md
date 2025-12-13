# Control Vector Generator - Known Issues

This document catalogs issues that affect the quality, reliability, and reproducibility of control vectors for representation engineering.

---

## Critical Issues (Affect Correctness)

### 🔴 CRITICAL: PCA Sign Instability

**Location:** [pca.hpp:292-293](pca.hpp#L292-L293)

**Severity:** CRITICAL - Can produce opposite behavioral effects

**Description:**

Eigenvectors are mathematically defined only up to sign. The power iteration algorithm can converge to either `v` or `-v`, both of which are valid solutions. This means:

- Running the same data twice can produce opposite vectors
- A "happiness" vector might become a "sadness" vector randomly
- No reproducibility between runs
- **Applying the wrong sign steers behavior in the opposite direction**

**Code Evidence:**
```cpp
// TODO @ngxson : The output vector is randomly inverted
// Solution: https://github.com/ggerganov/llama.cpp/pull/8069#issuecomment-2185328171
```

**Impact on Representation Engineering:**
- Complete lack of reproducibility
- Testing/validation is impossible
- Could accidentally make model behaviors worse instead of better
- User has no way to know if sign is correct without manual testing

**Reproduction:**
```bash
# Run twice with identical inputs
./cvector-generator -m model.gguf --method pca -o vec1.gguf
./cvector-generator -m model.gguf --method pca -o vec2.gguf
# Compare vectors - signs may differ
```

**Recommended Fix:**

From the referenced PR discussion, standardize sign by:
1. Checking the sign of the first non-zero component
2. Flipping vector if it's negative
3. OR: Maintain a reference vector and ensure positive correlation

**Workaround:**

Use `--method mean` instead of PCA for now - mean doesn't have sign ambiguity.

---

### 🔴 CRITICAL: Zero-Row Filtering Sign Asymmetry

**Location:** [cvector-generator.cpp:117-126](cvector-generator.cpp#L117-L126)

**Severity:** CRITICAL - Introduces systematic bias

**Description:**

The zero-row filter only checks if values are greater than epsilon, not absolute value:

```cpp
auto is_row_all_zeros = [](struct ggml_tensor * t, int row, float eps) -> bool {
    int n_cols = t->ne[0];
    for (int col = 0; col < n_cols; ++col) {
        if (ggml_get_f32_nd(t, col, row, 0, 0) > eps) {  // ← BUG: Should be abs() > eps
            return false;
        }
    }
    return true;
};
```

**Consequences:**

1. **Rows with all negative values are filtered out**
   - Example: `[-0.1, -0.2, -0.3, ...]` considered "all zeros" and removed
   - Only positive-biased rows survive

2. **Asymmetric treatment of positive vs negative differences**
   - If negative prompt produces stronger activations, entire row filtered
   - Systematically biases vectors toward positive activation differences

3. **Loss of meaningful signal**
   - Negative activations are just as informative as positive ones
   - Filtering them corrupts the directional information

**Impact on Representation Engineering:**
- Systematic bias in extracted directions
- Lost information from inhibitory effects
- Unpredictable behavior depending on which prompt produces stronger activations
- May explain why some control vectors don't work as expected

**Example:**

```
Token position 5:
  Positive prompt activations: [0.1, 0.2, 0.1, ...]
  Negative prompt activations: [0.5, 0.6, 0.4, ...]
  Difference: [-0.4, -0.4, -0.3, ...]  ← All negative, FILTERED OUT!

This row contained valid signal (negative is suppressing this feature) but is lost.
```

**Recommended Fix:**

```cpp
if (std::abs(ggml_get_f32_nd(t, col, row, 0, 0)) > eps) {
    return false;
}
```

**Priority:** Should be fixed immediately before generating production control vectors.

---

## High-Impact Issues (Affect Quality)

### 🟡 HIGH: Hardcoded Zero-Row Epsilon

**Location:** [cvector-generator.cpp:129](cvector-generator.cpp#L129)

**Severity:** HIGH - Model and use-case dependent

**Description:**

The threshold for filtering "zero" rows is hardcoded to `1e-6`:

```cpp
if (!is_row_all_zeros(a, i_row, 1e-6)) {
    rows_to_copy.push_back(i_row);
}
```

**Problems:**

1. **Model-dependent activation scales**
   - Different architectures have different activation ranges
   - `1e-6` might be too aggressive for models with small activations
   - `1e-6` might be too lenient for models with large activations

2. **Context-dependent sensitivity**
   - Subtle behavioral differences produce small activation differences
   - Aggressive filtering loses weak but meaningful signals
   - Important for fine-grained representation engineering

3. **No way to tune for specific use cases**
   - Users can't experiment with different thresholds
   - Can't adapt to different prompt contrast strengths

**Impact on Representation Engineering:**
- May filter out weak but important directional signals
- Reduces sensitivity for subtle behavioral adjustments
- Can't be tuned for specific models or use cases

**Example Scenarios:**

```
Scenario 1: Subtle emotion control (happy → slightly happy)
  Small activation differences → many filtered as "zero" → weak control vector

Scenario 2: Extreme behavior control (formal → street slang)
  Large activation differences → epsilon too lenient → keeps noise rows
```

**Recommended Fix:**

Add command-line parameter:
```bash
./cvector-generator -m model.gguf --zero-threshold 1e-7
```

Alternatively, use adaptive threshold based on activation statistics (e.g., percentile-based).

---

### 🟡 HIGH: Padding Token Contamination

**Location:** [cvector-generator.cpp:286-293](cvector-generator.cpp#L286-L293)

**Severity:** HIGH - Contaminates training signal

**Description:**

Prompts of different lengths are padded with space character tokens:

```cpp
void padding_seq(llama_context * ctx, std::vector<llama_token> & tokens, size_t len) {
    // TODO: customize padding token
    std::vector<llama_token> pad_tokens = common_tokenize(ctx, " ", false);
    llama_token pad_tok = pad_tokens.back();
    while (tokens.size() < len) {
        tokens.push_back(pad_tok);
    }
}
```

**Problems:**

1. **Space has semantic meaning**
   - Not a true padding token (not `<pad>`, `<eos>`, or null)
   - Model processes it as real input
   - Generates meaningful activations that contaminate differences

2. **Unequal padding creates asymmetric noise**
   ```
   Positive prompt: 20 tokens → 0 padding tokens
   Negative prompt: 12 tokens → 8 padding tokens

   Difference includes:
   - 12 positions: real signal (pos - neg)
   - 8 positions: noise (pos - space_activations)
   ```

3. **No attention masking**
   - Unlike training/inference, padding tokens aren't masked
   - Their activations fully participate in difference calculation
   - Signal-to-noise ratio degrades proportional to padding amount

4. **Model-specific padding conventions ignored**
   - Different model families use different padding strategies
   - Some models expect specific padding tokens
   - Universal space padding may behave unexpectedly

**Impact on Representation Engineering:**
- Weakened control vectors when prompt lengths differ significantly
- Spurious directions related to "presence of text" vs semantic content
- Unpredictable behavior across different models
- Reduced effectiveness with varied prompt lengths

**Example:**

```
Positive: "Act extremely happy" (4 tokens)
Negative: "Act extremely sad" (4 tokens)
→ No padding, clean signal ✓

Positive: "Act like a person who is extremely happy and enthusiastic" (12 tokens)
Negative: "Act sad" (3 tokens)
→ 9 padding tokens in negative, contaminated signal ✗
```

**Recommended Fix:**

1. **Use model-native padding token:**
   ```cpp
   llama_token pad_tok = llama_token_eos(model);  // or model-specific pad token
   ```

2. **Implement attention masking:**
   - Track which positions are padding
   - Zero out their contributions to differences
   - Or exclude them entirely from calculations

3. **Warn users about length mismatches:**
   ```
   Warning: Prompt length mismatch (pos=45, neg=12).
   Consider using similar lengths for better quality.
   ```

**Workaround:**

Manually ensure all prompt pairs have similar lengths (within 10% recommended).

---

### 🟡 MEDIUM: Per-Layer Normalization Loses Magnitude Information

**Location:** [mean.hpp:32-42](mean.hpp#L32-L42), [pca.hpp:169-174](pca.hpp#L169-L174)

**Severity:** MEDIUM - Affects interpretability and scaling

**Description:**

Each layer's control vector is independently normalized to unit length:

```cpp
// mean.hpp
float norm = 0.0;
for (int i = 0; i < ggml_nelements(ctrl_out); i++) {
    float f = ggml_get_f32_1d(ctrl_out, i);
    norm += f*f;
}
norm = sqrt(norm);
for (int i = 0; i < ggml_nelements(ctrl_out); i++) {
    float f = ggml_get_f32_1d(ctrl_out, i);
    ggml_set_f32_1d(ctrl_out, i, f / norm);
}
```

**Problems:**

1. **Original magnitude information is lost**
   ```
   Before normalization:
     Layer 10: magnitude = 0.05 (weak effect)
     Layer 20: magnitude = 3.2  (strong effect)

   After normalization:
     Layer 10: magnitude = 1.0
     Layer 20: magnitude = 1.0

   Information about which layers naturally have stronger effects is gone.
   ```

2. **Uniform scaling affects all layers equally**
   ```bash
   ./llama-cli --control-vector-scaled vector.gguf 0.8
   # Applies 0.8 to ALL layers, but maybe:
   # Layer 10 should be 0.01 and Layer 20 should be 2.5
   ```

3. **Can't distinguish natural vs artificial strength**
   - Is this layer strong because the difference was large?
   - Or because the original activations were small?
   - Normalization removes this information

**Impact on Representation Engineering:**
- Need more experimentation to find correct scaling factors
- May over-amplify weak layers or under-amplify strong layers
- Harder to interpret which layers are most important
- Per-layer scaling requires manual tuning

**Current Behavior:**

When applying vectors, you use a single scale for all layers:
```bash
# This applies 0.8× to layer 10 AND layer 20
--control-vector-scaled vector.gguf 0.8
```

You can use layer ranges to partially address this:
```bash
# Apply 1.0 to layers 10-20, 0.5 to layers 21-30
--control-vector-scaled vector.gguf 1.0 --control-vector-layer-range 10 20
--control-vector-scaled vector.gguf 0.5 --control-vector-layer-range 21 30
```

But you need to experiment to find the right values.

**Alternative Approaches:**

1. **Don't normalize - preserve magnitudes:**
   - Pro: Natural strength information preserved
   - Con: Harder to compare across different models/vectors

2. **Global normalization:**
   - Normalize across all layers combined
   - Preserves relative layer strengths

3. **Store magnitude metadata in GGUF:**
   ```
   direction.1: [n_embd]
   direction.1.original_magnitude: 0.05
   direction.2: [n_embd]
   direction.2.original_magnitude: 3.2
   ```

**Recommended Action:**

Document expected magnitude ranges for reference, or add option to skip normalization.

---

## Medium-Impact Issues (Affect Usability)

### 🟢 MEDIUM: No PCA Centering

**Location:** [pca.hpp:155](pca.hpp#L155)

**Severity:** MEDIUM - May not be a bug, but differs from standard PCA

**Description:**

Standard PCA centers data before computing covariance:
```python
X_centered = X - X.mean(axis=0)
cov = X_centered.T @ X_centered
```

This implementation doesn't center:
```cpp
tmp_square = ggml_mul_mat(ctx0, model.dev_input, model.dev_input);  // Just X^T @ X
```

**Implications:**

1. **First PC may capture mean instead of variance**
   - Standard PCA: PC1 = direction of maximum variance
   - Non-centered: PC1 = direction of maximum "presence"

2. **May be intentional for this use case**
   - We're already working with differences (pos - neg)
   - The mean of differences might be the signal we want
   - Centering could remove important information

3. **Differs from standard practice**
   - Users familiar with PCA may be surprised
   - Harder to compare with other PCA-based methods

**Impact on Representation Engineering:**
- Unclear if this is beneficial or detrimental
- May capture different information than expected
- Hard to predict without empirical testing

**Investigation Needed:**

Compare centered vs non-centered PCA empirically:
1. Test both on same data
2. Evaluate control vector quality
3. Document which works better and why

---

### 🟢 MEDIUM: completions.txt File Unused

**Location:** File exists but not referenced in code

**Severity:** LOW - Indicates incomplete feature or technical debt

**Description:**

The file `completions.txt` exists in the directory but is never loaded or used by the code.

**Possible Explanations:**

1. **Deprecated feature** - Previously used, now obsolete
2. **Incomplete implementation** - Feature partially implemented
3. **Documentation artifact** - Example file for future use

**Impact:**
- Minimal, but suggests the tool may have unfinished features
- Could confuse users about expected input format

**Recommended Action:**

Either remove the file or implement the feature it was intended for. Add documentation explaining what it's for if keeping it.

---

## Low-Impact Issues (Technical Debt)

### 🟢 LOW: Manual Memory Management

**Location:** Multiple files - search for `malloc` and `free`

**Severity:** LOW - Works but fragile

**Description:**

The code uses manual `malloc()` and `free()` instead of RAII patterns:

```cpp
t_layer->data = malloc(n_bytes); // TODO @ngxson : get rid of this malloc somehow
```

**Problems:**

1. **Risk of memory leaks**
   - If exception occurs between malloc and free
   - If early return paths are added
   - If error handling forgets to free

2. **No automatic cleanup**
   - RAII (smart pointers, containers) would handle automatically
   - Manual tracking required

3. **Code smell**
   - TODO comments indicate developers are aware
   - Suggests they want to fix but haven't yet

**Impact:**
- Low in practice (code seems to free correctly)
- Makes maintenance harder
- Could leak in edge cases

**Recommended Fix:**

Use GGML's buffer allocation system or smart pointers:
```cpp
std::unique_ptr<void, decltype(&free)> data(malloc(n_bytes), free);
```

Or GGML's context-based allocation which frees on context destruction.

---

### 🟢 LOW: Limited Error Messages

**Location:** Various error paths

**Severity:** LOW - Annoying but not breaking

**Examples:**

```cpp
fprintf(stderr, "error: unable to open file: %s\n", path.c_str());
// Doesn't say which file (positive? negative?) or why
```

```cpp
fprintf(stderr, "%s : failed to eval\n", __func__);
// Which prompt failed? What was the error?
```

**Impact:**
- Harder to debug when things go wrong
- Users can't self-diagnose issues
- May require code inspection to understand failures

**Recommended Fix:**

Add context to error messages:
```cpp
fprintf(stderr, "error: unable to open positive prompt file '%s': %s\n",
        path.c_str(), strerror(errno));
```

---

## Summary Table

| Issue | Severity | Impact | Fix Priority | Workaround Available? |
|-------|----------|--------|--------------|----------------------|
| PCA sign instability | CRITICAL | Wrong direction | IMMEDIATE | Use `--method mean` |
| Zero-row sign asymmetry | CRITICAL | Systematic bias | IMMEDIATE | None - must fix |
| Hardcoded epsilon | HIGH | Loss of signal | HIGH | Use strong contrasts |
| Padding contamination | HIGH | Noise in signal | HIGH | Match prompt lengths |
| Normalization loses magnitude | MEDIUM | Scaling difficulty | MEDIUM | Manual layer tuning |
| No PCA centering | MEDIUM | Unclear | LOW | Use mean method |
| Unused completions.txt | LOW | Confusion | LOW | Ignore file |
| Manual malloc | LOW | Maintenance | LOW | None needed |
| Poor error messages | LOW | Debug difficulty | LOW | Read code |

---

## Recommendations by Use Case

### For Production Representation Engineering

**Must Fix:**
1. PCA sign instability (CRITICAL)
2. Zero-row sign asymmetry (CRITICAL)

**Should Fix:**
3. Hardcoded epsilon (HIGH)
4. Padding contamination (HIGH)

### For Experimental/Research Use

**Workarounds Sufficient:**
- Use `--method mean` to avoid PCA issues
- Keep prompt pairs similar length (±10%)
- Use strong behavioral contrasts (reduces epsilon sensitivity)
- Manually test vectors to verify direction is correct

### For DIY Hobbyist Use

**Safe Practices:**
1. Always use `--method mean` for now
2. Create 10+ prompt pairs for robust signal
3. Test control vectors with both positive and negative scaling
4. Keep positive/negative prompts within 5 tokens of each other
5. Use very clear behavioral contrasts

---

## Testing Recommendations

### Validate Your Control Vectors

```bash
# 1. Check sign is correct
./llama-cli -m model.gguf -p "Neutral prompt" --control-vector-scaled vec.gguf 1.0
# Should increase target behavior

./llama-cli -m model.gguf -p "Neutral prompt" --control-vector-scaled vec.gguf -1.0
# Should decrease target behavior (opposite effect)

# 2. Check reproducibility (if using PCA)
./cvector-generator -m model.gguf --method pca -o vec1.gguf
./cvector-generator -m model.gguf --method pca -o vec2.gguf
# Compare vec1 and vec2 - should be identical or exact negatives

# 3. Check prompt length sensitivity
# Create two sets: one with matched lengths, one with mismatched
# Compare quality of resulting vectors
```

### Detect Issues

```bash
# Check for sign asymmetry bug:
# Look for control vectors that don't work at all
# If vector has no effect in either direction, filtering bug may be culprit

# Check epsilon sensitivity:
# Generate vectors with subtle vs extreme prompt contrasts
# Subtle contrasts that produce weak vectors may be over-filtered
```
