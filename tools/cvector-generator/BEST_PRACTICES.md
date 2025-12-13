# Control Vector Generator - Best Practices Guide

A practical guide for creating high-quality control vectors for representation engineering with locally-run models.

---

## Quick Start: Safe Defaults

If you just want to get started without reading all the issues, use these safe settings:

```bash
# 1. Use mean method (more stable than PCA for now)
./cvector-generator -m your_model.gguf \
  --positive-file positive.txt \
  --negative-file negative.txt \
  --method mean \
  -o control_vector.gguf

# 2. Test the vector works in both directions
./llama-cli -m your_model.gguf \
  -p "Write a story about a cat" \
  --control-vector-scaled control_vector.gguf 1.0

./llama-cli -m your_model.gguf \
  -p "Write a story about a cat" \
  --control-vector-scaled control_vector.gguf -1.0

# If +1.0 and -1.0 produce opposite behaviors, your vector works!
```

---

## Prompt Design Guidelines

### Rule 1: Match Prompt Lengths

**Why:** Unequal lengths require padding, which contaminates the signal with noise.

**Bad:**
```
positive.txt:
Act like an extremely happy and enthusiastic person who loves life

negative.txt:
Act sad
```

**Good:**
```
positive.txt:
Act like an extremely happy person

negative.txt:
Act like an extremely sad person
```

**Target:** Keep positive/negative pairs within ±10% length (in tokens).

**Pro Tip:** Use the same sentence structure, just swap the key words:
```
Positive: You are a highly [expert/formal/creative] assistant
Negative: You are a highly [novice/casual/boring] assistant
```

---

### Rule 2: Create Strong Behavioral Contrasts

**Why:** Weak contrasts produce small activation differences that may be filtered out.

**Weak Contrast:**
```
Positive: You are somewhat happy
Negative: You are neutral
```
→ Small activation difference, sensitive to noise

**Strong Contrast:**
```
Positive: You are the happiest person alive, filled with joy and excitement!
Negative: You are completely miserable, drowning in sadness and despair
```
→ Large activation difference, robust signal

**Rule of Thumb:** Use extreme adjectives and clear emotional/behavioral markers.

---

### Rule 3: Use Multiple Diverse Examples

**Why:** Multiple examples average out noise and capture the concept more robustly.

**Minimum:** 4 prompt pairs
**Recommended:** 10-20 prompt pairs
**Ideal:** 50+ prompt pairs for production use

**Example: Formality Control**

```
positive.txt:
<|start_header_id|>system<|end_header_id|>\n\nYou are a highly formal and professional assistant<|eot_id|>
<|start_header_id|>system<|end_header_id|>\n\nSpeak with utmost professionalism and courtesy<|eot_id|>
<|start_header_id|>system<|end_header_id|>\n\nUse formal language and proper grammar at all times<|eot_id|>
... (7 more examples)

negative.txt:
<|start_header_id|>system<|end_header_id|>\n\nYou are a casual and relaxed assistant, dude<|eot_id|>
<|start_header_id|>system<|end_header_id|>\n\nSpeak casually like you're talking to a friend<|eot_id|>
<|start_header_id|>system<|end_header_id|>\n\nUse slang and informal language, no need to be proper<|eot_id|>
... (7 more examples)
```

**Variation Strategies:**
- Different phrasings of the same concept
- Different contexts (system prompt, user interaction, different tasks)
- Different intensities (very formal → extremely formal)

---

### Rule 4: Use Proper Model-Specific Formatting

**Why:** Models are trained on specific prompt formats; using the wrong format produces poor activations.

**Llama 3 Format:**
```
<|start_header_id|>system<|end_header_id|>\n\nYour instruction here<|eot_id|>
```

**ChatML Format (Mistral, etc.):**
```
<|im_start|>system\nYour instruction here<|im_end|>
```

**Check your model's documentation** for the correct format!

**Pro Tip:** Include some completion text to activate response patterns:
```
<|start_header_id|>system<|end_header_id|>\n\nAct extremely happy<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nHello<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nHi! I'm having the best day ever!
```

---

### Rule 5: Validate Concept Isolation

**Why:** Make sure you're capturing ONLY the concept you want, not confounding factors.

**Bad - Confounded:**
```
Positive: You are a brilliant expert scientist with a PhD
Negative: You are uneducated and don't know anything
```
→ Captures expertise AND education level AND confidence AND verbosity

**Good - Isolated:**
```
Positive: You are an expert in this field
Negative: You are a novice in this field
```
→ Captures expertise level specifically

**Test:** Can you describe the concept in one word/phrase? That's what your prompts should isolate.

---

## Method Selection: PCA vs Mean

### Use **Mean** When:
- ✅ You have consistent prompt pairs (all examples demonstrate same concept)
- ✅ You want maximum stability and reproducibility
- ✅ You're experimenting and need fast iterations
- ✅ You have fewer than 10 prompt pairs
- ✅ You want to avoid sign ambiguity issues

**Command:**
```bash
./cvector-generator -m model.gguf --method mean
```

### Use **PCA** When:
- ✅ You have many prompt pairs (20+) with natural variation
- ✅ You want to extract the "strongest" common direction
- ✅ You're willing to verify sign consistency
- ✅ You have diverse examples and want to find the main pattern

**Command:**
```bash
./cvector-generator -m model.gguf --method pca --pca-iter 2000 --pca-batch 100
```

**⚠️ Warning:** PCA currently has sign instability - see [ISSUES.md](ISSUES.md#-critical-pca-sign-instability). Validate output carefully.

### Validation: Compare Both Methods

```bash
./cvector-generator -m model.gguf --method mean -o mean_vec.gguf
./cvector-generator -m model.gguf --method pca -o pca_vec.gguf

# Test both
./llama-cli -m model.gguf -p "Test prompt" --control-vector-scaled mean_vec.gguf 1.0
./llama-cli -m model.gguf -p "Test prompt" --control-vector-scaled pca_vec.gguf 1.0

# Use whichever produces better behavioral control
```

---

## GPU Acceleration

### When to Use GPU

- ✅ Large models (>7B parameters)
- ✅ Many prompt pairs (>10)
- ✅ Using PCA with high iteration counts

### Command:
```bash
./cvector-generator -m model.gguf -ngl 99 --method pca
```

`-ngl 99` offloads 99 layers to GPU (adjust based on your VRAM).

### Memory Considerations

GPU memory needed ≈ model size + (n_embd × n_tokens × n_layers × 2)

Example for Llama-3-8B:
- Model: ~8GB
- Activations: ~0.5GB per prompt pair
- Total: ~8.5GB minimum

If you run out of VRAM, reduce `-ngl` or process fewer prompt pairs at once.

---

## Testing and Validation

### Step 1: Verify Direction Correctness

```bash
# Positive scaling should INCREASE target behavior
./llama-cli -m model.gguf \
  -p "Write a short story" \
  --control-vector-scaled vector.gguf 1.0 \
  --control-vector-layer-range 10 31

# Negative scaling should DECREASE target behavior
./llama-cli -m model.gguf \
  -p "Write a short story" \
  --control-vector-scaled vector.gguf -1.0 \
  --control-vector-layer-range 10 31
```

If both produce similar output, your vector may be broken.

---

### Step 2: Find Optimal Scaling Factor

```bash
# Try different scales
for scale in 0.2 0.5 0.8 1.0 1.5 2.0; do
  echo "Scale: $scale"
  ./llama-cli -m model.gguf -p "Test prompt" \
    --control-vector-scaled vector.gguf $scale \
    --control-vector-layer-range 10 31 -n 50
done
```

**Typical ranges:**
- Subtle effects: 0.1 - 0.5
- Moderate effects: 0.5 - 1.0
- Strong effects: 1.0 - 2.0
- Very strong effects: 2.0+

**Warning:** Too high scaling can cause incoherence or repetition.

---

### Step 3: Find Optimal Layer Range

Different layers control different aspects:

- **Early layers (1-10):** Low-level features (syntax, basic patterns)
- **Middle layers (10-20):** Semantic concepts, moderate abstraction
- **Late layers (20-30):** High-level concepts, complex behaviors
- **Final layers (30+):** Token prediction (usually not recommended)

**Experimentation:**
```bash
# Only early layers
./llama-cli ... --control-vector-layer-range 5 15

# Only middle layers (recommended starting point)
./llama-cli ... --control-vector-layer-range 10 25

# Only late layers
./llama-cli ... --control-vector-layer-range 20 31

# Full range except final layer
./llama-cli ... --control-vector-layer-range 1 31
```

**Pro Tip:** Start with layers 10-31, then narrow down based on results.

---

### Step 4: Test on Diverse Prompts

Don't just test on prompts similar to your training examples!

**Good Test Set:**
- Similar to training prompts (should work strongly)
- Different phrasing of same concept (should work moderately)
- Different tasks entirely (reveals generalization)
- Edge cases (very short prompts, unusual formats)

**Example for happiness vector:**
```bash
# Similar to training
./llama-cli ... -p "How are you feeling today?"

# Different phrasing
./llama-cli ... -p "Describe your current emotional state"

# Different task
./llama-cli ... -p "Write a poem about nature"

# Edge case
./llama-cli ... -p "Hi"
```

---

## Common Issues and Debugging

### Issue: Vector has no effect

**Possible Causes:**
1. Scale too low → Try higher values (1.0, 2.0, 5.0)
2. Wrong layer range → Try 10-31
3. Prompts too similar → Recreate with stronger contrast
4. Sign flipped (PCA) → Try negative scaling

**Debug:**
```bash
# Extreme test - should definitely do something
./llama-cli -m model.gguf -p "Test" \
  --control-vector-scaled vector.gguf 5.0 \
  --control-vector-layer-range 1 31
```

---

### Issue: Vector produces incoherent output

**Possible Causes:**
1. Scale too high → Reduce to 0.5 or lower
2. Layer range includes final layers → Exclude layers > 30
3. Prompts poorly designed → Review prompt quality

**Fix:**
```bash
# Start conservative
./llama-cli ... --control-vector-scaled vector.gguf 0.3 --control-vector-layer-range 10 25
```

---

### Issue: Vector works inconsistently

**Possible Causes:**
1. Prompt length variation in training data
2. Weak behavioral contrast
3. Too few examples
4. PCA sign instability

**Solutions:**
- Recreate with matched lengths
- Use stronger contrasts
- Add more prompt pairs (aim for 20+)
- Switch to mean method

---

### Issue: Vector works in reverse (wrong direction)

**Cause:** PCA sign instability (see [ISSUES.md](ISSUES.md#-critical-pca-sign-instability))

**Fix:**
```bash
# Just negate the scale factor
./llama-cli ... --control-vector-scaled vector.gguf -1.0  # Instead of 1.0
```

Or regenerate using mean method:
```bash
./cvector-generator -m model.gguf --method mean -o fixed_vector.gguf
```

---

## Advanced Techniques

### Multi-Concept Vectors

Apply multiple control vectors simultaneously:

```bash
./llama-cli -m model.gguf -p "Your prompt" \
  --control-vector-scaled formality.gguf 1.0 \
  --control-vector-scaled expertise.gguf 0.8 \
  --control-vector-scaled creativity.gguf -0.5
```

This creates: formal + expert + less creative responses.

**Limitation:** Vectors may interfere if they're not orthogonal. Test carefully.

---

### Layer-Specific Scaling

Apply different strengths to different layer ranges:

```bash
# Strong in middle layers, weak in late layers
./llama-cli -m model.gguf -p "Test" \
  --control-vector-scaled vector.gguf 1.5 --control-vector-layer-range 10 20 \
  --control-vector-scaled vector.gguf 0.5 --control-vector-layer-range 21 31
```

Useful when early/late layers have different sensitivities.

---

### Interpolation Between Behaviors

Create smooth transitions:

```bash
# More negative (behavior A)
./llama-cli ... --control-vector-scaled vector.gguf -1.0

# Neutral
./llama-cli ... --control-vector-scaled vector.gguf 0.0

# More positive (behavior B)
./llama-cli ... --control-vector-scaled vector.gguf 1.0

# Very positive (extreme behavior B)
./llama-cli ... --control-vector-scaled vector.gguf 2.0
```

Think of it as a slider from behavior A (negative) to behavior B (positive).

---

## Recommended Workflow

### Phase 1: Preparation
1. Choose your target concept (one clear behavior/trait)
2. Design 10-20 prompt pairs with strong contrast
3. Match prompt lengths (±10%)
4. Use model-appropriate formatting

### Phase 2: Generation
```bash
# Start with mean method (safer)
./cvector-generator -m model.gguf \
  --positive-file positive.txt \
  --negative-file negative.txt \
  --method mean \
  -o vector_mean.gguf \
  -ngl 99  # If you have GPU
```

### Phase 3: Validation
```bash
# Test both directions
./llama-cli -m model.gguf -p "Neutral test prompt" \
  --control-vector-scaled vector_mean.gguf 1.0 \
  --control-vector-layer-range 10 31

./llama-cli -m model.gguf -p "Neutral test prompt" \
  --control-vector-scaled vector_mean.gguf -1.0 \
  --control-vector-layer-range 10 31
```

### Phase 4: Tuning
```bash
# Find best scale
for scale in 0.5 0.8 1.0 1.5 2.0; do
  ./llama-cli -m model.gguf -p "Test prompt" \
    --control-vector-scaled vector_mean.gguf $scale \
    --control-vector-layer-range 10 31 -n 100 > output_$scale.txt
done

# Compare outputs manually
```

### Phase 5: Production Use
```bash
# Use optimized settings from tuning
./llama-cli -m model.gguf -p "Your production prompt" \
  --control-vector-scaled vector_mean.gguf 0.8 \
  --control-vector-layer-range 12 28
```

---

## Example: Creating a "Technical Expertise" Vector

### Step 1: Design Prompts

**positive.txt:**
```
<|start_header_id|>system<|end_header_id|>\n\nYou are a highly technical expert with deep knowledge of computer science<|eot_id|>
<|start_header_id|>system<|end_header_id|>\n\nYou are an experienced software architect with expertise in system design<|eot_id|>
<|start_header_id|>system<|end_header_id|>\n\nYou are a senior engineer with advanced technical knowledge<|eot_id|>
<|start_header_id|>system<|end_header_id|>\n\nYou are a technical specialist with expert-level understanding<|eot_id|>
<|start_header_id|>system<|end_header_id|>\n\nYou are a professional engineer with deep technical expertise<|eot_id|>
```

**negative.txt:**
```
<|start_header_id|>system<|end_header_id|>\n\nYou are a beginner learning the basics of computer science<|eot_id|>
<|start_header_id|>system<|end_header_id|>\n\nYou are new to software development with limited technical knowledge<|eot_id|>
<|start_header_id|>system<|end_header_id|>\n\nYou are a junior developer still learning fundamental concepts<|eot_id|>
<|start_header_id|>system<|end_header_id|>\n\nYou are a novice with basic understanding of technical topics<|eot_id|>
<|start_header_id|>system<|end_header_id|>\n\nYou are an entry-level programmer learning the fundamentals<|eot_id|>
```

### Step 2: Generate Vector

```bash
./cvector-generator -m llama-3-8b.Q4_K_M.gguf \
  --positive-file technical_expert_pos.txt \
  --negative-file technical_expert_neg.txt \
  --method mean \
  -o technical_expertise.gguf \
  -ngl 99
```

### Step 3: Test

```bash
# Baseline (no vector)
./llama-cli -m llama-3-8b.Q4_K_M.gguf \
  -p "Explain what a mutex is" -n 100 > baseline.txt

# Expert mode (positive)
./llama-cli -m llama-3-8b.Q4_K_M.gguf \
  -p "Explain what a mutex is" \
  --control-vector-scaled technical_expertise.gguf 1.0 \
  --control-vector-layer-range 10 31 -n 100 > expert.txt

# Beginner mode (negative)
./llama-cli -m llama-3-8b.Q4_K_M.gguf \
  -p "Explain what a mutex is" \
  --control-vector-scaled technical_expertise.gguf -1.0 \
  --control-vector-layer-range 10 31 -n 100 > beginner.txt
```

### Step 4: Evaluate

Compare outputs:
- **baseline.txt**: Standard explanation
- **expert.txt**: Should use technical jargon, mention implementation details, discuss edge cases
- **beginner.txt**: Should use simple language, analogies, avoid jargon

---

## Quality Checklist

Before considering your control vector production-ready:

- [ ] Tested with positive scaling (increases target behavior)
- [ ] Tested with negative scaling (decreases target behavior)
- [ ] Tested on diverse prompts (not just training examples)
- [ ] Found optimal scaling factor (not too weak, not incoherent)
- [ ] Found optimal layer range (best trade-off)
- [ ] Prompt pairs have matched lengths (±10%)
- [ ] Used strong behavioral contrasts
- [ ] Used 10+ diverse examples
- [ ] Verified effect is specific to intended concept
- [ ] Documented optimal settings for future use

---

## Resources

- [ARCHITECTURE.md](ARCHITECTURE.md) - Technical details of how the tool works
- [ISSUES.md](ISSUES.md) - Known issues and their impact
- [llama.cpp Documentation](https://github.com/ggerganov/llama.cpp) - Main project docs
- [Original PR #5970](https://github.com/ggml-org/llama.cpp/pull/5970) - Control vector support
- [Example PR #7514](https://github.com/ggml-org/llama.cpp/pull/7514) - This tool's implementation

---

## Community Examples

Share your successful control vectors! Document:
1. Target concept
2. Model used
3. Number of prompt pairs
4. Method (PCA/mean)
5. Optimal scaling factor and layer range
6. Example outputs

This helps build a knowledge base of what works well.
