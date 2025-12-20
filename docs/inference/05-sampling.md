# Sampling and Token Generation in llama.cpp

This document explains how llama.cpp samples tokens from model outputs to generate text, including the sampling chain, various sampling strategies, and grammar constraints.

## Table of Contents

1. [Overview](#overview)
2. [From Logits to Tokens](#from-logits-to-tokens)
3. [Sampling Chain](#sampling-chain)
4. [Sampling Strategies](#sampling-strategies)
5. [Grammar Constraints](#grammar-constraints)
6. [Common Sampler](#common-sampler)
7. [Performance Optimization](#performance-optimization)

## Overview

After the model computes logits for the next token, we need to **sample** (select) one token from the vocabulary. This is not as simple as picking the highest probability token - different sampling strategies can produce more creative, more focused, or more diverse outputs.

### The Sampling Pipeline

```
Model Output (Logits)
    ↓
Convert to Probabilities (softmax)
    ↓
Apply Sampling Chain
    - Temperature scaling
    - Top-K filtering
    - Top-P (nucleus) sampling
    - Min-P filtering
    - Repetition penalty
    - Frequency/presence penalty
    - DRY (Don't Repeat Yourself)
    - Grammar constraints
    ↓
Select Token
    ↓
Return Selected Token ID
```

## From Logits to Tokens

### Step 1: Get Logits

After `llama_decode()`, retrieve the output logits:

```cpp
float * logits = llama_get_logits_ith(ctx, -1);  // Get last token's logits
// logits is array of size n_vocab with raw scores
```

### Step 2: Create Token Candidates

```cpp
// Convert logits to token candidates
llama_token_data_array candidates;
candidates.data = /* array of llama_token_data */;
candidates.size = n_vocab;
candidates.selected = -1;
candidates.sorted = false;

for (int i = 0; i < n_vocab; i++) {
    candidates.data[i].id = i;
    candidates.data[i].logit = logits[i];
    candidates.data[i].p = 0.0f;  // Will be set by softmax
}
```

### Step 3: Apply Sampler Chain

```cpp
llama_token token = llama_sampler_sample(sampler, ctx, -1);
```

This applies all samplers in the chain and returns the selected token.

### Step 4: Accept Token

```cpp
// Update sampler state (for grammar, repetition penalty, etc.)
llama_sampler_accept(sampler, token);
```

## Sampling Chain

The **sampler chain** is a sequence of samplers that each modify the token probability distribution. Samplers are applied in order.

### Creating a Sampler Chain

From [include/llama.h](../../include/llama.h):

```cpp
// Create empty chain
llama_sampler * chain = llama_sampler_chain_init(params);

// Add samplers in desired order
llama_sampler_chain_add(chain, llama_sampler_init_top_k(40));
llama_sampler_chain_add(chain, llama_sampler_init_top_p(0.95, 1));
llama_sampler_chain_add(chain, llama_sampler_init_temp(0.8));
llama_sampler_chain_add(chain, llama_sampler_init_dist(seed));

// Use the chain
llama_token token = llama_sampler_sample(chain, ctx, -1);

// Clean up
llama_sampler_free(chain);
```

### Sampler Chain Order Matters!

Different orders produce different results:

**Typical order** (most to least restrictive):
1. **Logit modifiers**: XTC, Penalties (repetition, frequency, presence)
2. **Probability filters**: Top-K, Top-P, Min-P
3. **Temperature scaling**
4. **Final selection**: Greedy or distribution sampling

**Example**:
```cpp
// Good: Filter then scale
chain → top_k(40) → temperature(0.8) → sample
// Keeps top 40, then makes distribution smoother

// Less effective: Scale then filter
chain → temperature(0.8) → top_k(40)
// Smooths first, then cuts off - may lose good candidates
```

## Sampling Strategies

### Greedy Sampling

Pick the token with highest probability. Deterministic.

```cpp
llama_sampler * sampler = llama_sampler_chain_init(params);
llama_sampler_chain_add(sampler, llama_sampler_init_greedy());
```

**Use case**: When you want the most likely output, don't need creativity

**Example output**: "The capital of France is Paris." (always the same)

### Temperature Sampling

Scale logits before softmax to control randomness:

```cpp
llama_sampler_chain_add(sampler, llama_sampler_init_temp(temperature));
```

- **temperature < 1.0**: More focused, less random (e.g., 0.7)
- **temperature = 1.0**: No change
- **temperature > 1.0**: More random, more creative (e.g., 1.2)
- **temperature → 0**: Approaches greedy

**Formula**: `logit' = logit / temperature`

**Example** (temperature effect):
- **T=0.1**: "The capital of France is Paris." (very focused)
- **T=1.0**: "The capital of France is Paris, a beautiful city."
- **T=2.0**: "The capital of France is Paris, which some people debate about..."

### Top-K Sampling

Keep only the K most probable tokens, discard the rest:

```cpp
llama_sampler_chain_add(sampler, llama_sampler_init_top_k(k));
```

- **k=1**: Greedy
- **k=40**: Common default
- **k=100**: More diversity

**Use case**: Limit to most reasonable options

### Top-P (Nucleus) Sampling

Keep the smallest set of tokens whose cumulative probability ≥ P:

```cpp
llama_sampler_chain_add(sampler, llama_sampler_init_top_p(p, min_keep));
```

- **p=1.0**: Keep all tokens
- **p=0.95**: Common default (keep top 95% probability mass)
- **p=0.9**: More focused

**Dynamic**: Number of kept tokens varies based on distribution shape

**Example**:
- Peaked distribution: Might keep only 10 tokens for p=0.95
- Flat distribution: Might keep 100 tokens for p=0.95

### Min-P Sampling

Keep tokens with probability ≥ (P × max_probability):

```cpp
llama_sampler_chain_add(sampler, llama_sampler_init_min_p(p, min_keep));
```

- **p=0.05**: Keep tokens with P ≥ 5% of max
- **p=0.1**: More restrictive

**Use case**: Alternative to top-p that scales with confidence

### Repetition Penalty

Penalize tokens that have appeared recently:

```cpp
llama_sampler_chain_add(sampler,
    llama_sampler_init_penalties(
        n_vocab,
        special_eos_id,
        linefeed_id,
        penalty_last_n,     // Look back N tokens
        penalty_repeat,     // Penalty for repetition (1.1)
        penalty_freq,       // Penalty based on frequency (0.0)
        penalty_present,    // Penalty for presence (0.0)
        penalize_nl,        // Penalize newlines?
        ignore_eos          // Ignore EOS when penalizing?
    ));
```

**Types**:
- **Repeat penalty**: Same penalty for all repeated tokens
- **Frequency penalty**: Stronger penalty for more frequent tokens
- **Presence penalty**: Binary - penalize if present at all

**Formula**: `logit' = logit - penalty * (repeat_count or presence)`

**Use case**: Reduce repetitive text

### DRY (Don't Repeat Yourself) Sampler

More sophisticated repetition avoidance:

```cpp
llama_sampler_chain_add(sampler,
    llama_sampler_init_dry(
        vocab,
        context_size,       // Max context to scan
        dry_multiplier,     // Penalty multiplier
        dry_base,           // Base penalty
        dry_allowed_length, // Allowed repetition length
        dry_penalty_last_n, // Look back N tokens
        seq_breakers        // Sequences that break patterns
    ));
```

**Advantage over basic penalty**: Detects longer repeated patterns

**Use case**: Prevent paragraph/sentence repetition

### XTC (Exclude Top Choices) Sampler

Excludes the most probable token with some probability to avoid "obvious" completions:

```cpp
llama_sampler_chain_add(sampler,
    llama_sampler_init_xtc(
        probability,  // Probability of excluding top token (0.1)
        threshold,    // Min probability to consider (0.1)
        min_keep,     // Min tokens to keep
        seed
    ));
```

**Use case**: More creative outputs by occasionally skipping the obvious choice

### Mirostat Sampling

Adaptive sampling that targets a specific perplexity:

```cpp
llama_sampler_chain_add(sampler,
    llama_sampler_init_mirostat(
        n_vocab,
        seed,
        tau,   // Target perplexity (5.0)
        eta,   // Learning rate (0.1)
        m      // Number of candidates (100)
    ));
```

**Advantage**: Automatically adjusts sampling to maintain consistent quality

### TFS (Tail Free Sampling)

Removes tokens from the "tail" of the distribution:

```cpp
llama_sampler_chain_add(sampler,
    llama_sampler_init_tail_free(z, min_keep));
```

**Use case**: Alternative to top-p that focuses on derivative

### Typical Sampling

Samples based on "typicality" rather than raw probability:

```cpp
llama_sampler_chain_add(sampler,
    llama_sampler_init_typical(p, min_keep));
```

**Use case**: Balance between creativity and coherence

### Final Sampler (Required)

The chain must end with a sampler that actually selects a token:

```cpp
// Option 1: Sample from distribution
llama_sampler_chain_add(sampler, llama_sampler_init_dist(seed));

// Option 2: Greedy (highest probability)
llama_sampler_chain_add(sampler, llama_sampler_init_greedy());
```

## Grammar Constraints

Grammar constraints ensure the output matches a specific format (JSON, programming language, etc.).

### Using Grammar

From [common/sampling.h](../../common/sampling.h):

```cpp
// Define grammar (GBNF format)
const char * grammar = R"(
    root ::= object
    object ::= "{" pair ("," pair)* "}"
    pair ::= string ":" value
    string ::= "\"" [^"]* "\""
    value ::= string | number | object
    number ::= [0-9]+
)";

// Initialize grammar sampler
llama_sampler * grammar_sampler = llama_sampler_init_grammar(
    vocab,
    grammar,
    "root"  // Start rule
);

// Add to chain (typically last, before final sampler)
llama_sampler_chain_add(chain, grammar_sampler);
```

### Grammar Performance

Applying grammar to all tokens can be slow. The `common_sampler` optimizes this:

```cpp
// Fast path: Sample first, check grammar
llama_token token = common_sampler_sample(gsmpl, ctx, idx, false);
// If token doesn't fit grammar, resample with grammar applied to all tokens

// Slow path: Apply grammar first (ensures all candidates fit)
llama_token token = common_sampler_sample(gsmpl, ctx, idx, true);
```

### Grammar Examples

**JSON output**:
```gbnf
root ::= object
object ::= "{" ws members? "}" ws
members ::= pair ("," ws pair)*
pair ::= string ":" ws value
value ::= string | number | "true" | "false" | "null" | object | array
string ::= "\"" ([^"\\] | "\\" .)* "\""
number ::= "-"? [0-9]+ ("." [0-9]+)?
array ::= "[" ws (value ("," ws value)*)? "]" ws
ws ::= [ \n\t]*
```

**Yes/No response**:
```gbnf
root ::= ("yes" | "no")
```

## Common Sampler

The `common_sampler` ([common/sampling.cpp](../../common/sampling.cpp)) is a high-level wrapper that:
- Manages sampler chain creation based on parameters
- Handles grammar efficiently
- Tracks sampling history
- Collects performance metrics

### Using Common Sampler

```cpp
#include "common.h"
#include "sampling.h"

// Configure sampling parameters
common_params_sampling params;
params.temp = 0.8;
params.top_k = 40;
params.top_p = 0.95;
params.penalty_repeat = 1.1;
params.penalty_last_n = 64;

// Create sampler
common_sampler * gsmpl = common_sampler_init(model, params);

// Sample
llama_token token = common_sampler_sample(gsmpl, ctx, -1);

// Accept (updates history and grammar state)
common_sampler_accept(gsmpl, token, true);

// Get last N tokens for context
std::string context = common_sampler_prev_str(gsmpl, ctx, 10);

// Clean up
common_sampler_free(gsmpl);
```

### Sampling Parameters

From [common/common.h](../../common/common.h):

```cpp
struct common_params_sampling {
    int32_t seed = -1;              // RNG seed
    int32_t n_prev = 64;            // Tokens to consider for penalties
    int32_t n_probs = 0;            // Show top N token probabilities
    int32_t min_keep = 0;           // Min tokens to keep in filtering
    int32_t top_k = 40;             // Top-K sampling
    float   top_p = 0.95f;          // Top-P sampling
    float   min_p = 0.05f;          // Min-P sampling
    float   tfs_z = 1.00f;          // Tail free sampling
    float   typ_p = 1.00f;          // Typical sampling
    float   temp = 0.80f;           // Temperature
    float   dynatemp_range = 0.00f; // Dynamic temperature range
    float   dynatemp_exponent = 1.00f;
    int32_t penalty_last_n = 64;    // Last N for penalties
    float   penalty_repeat = 1.00f; // Repetition penalty
    float   penalty_freq = 0.00f;   // Frequency penalty
    float   penalty_present = 0.00f;// Presence penalty
    int32_t mirostat = 0;           // 0=disabled, 1=v1, 2=v2
    float   mirostat_tau = 5.00f;   // Target perplexity
    float   mirostat_eta = 0.10f;   // Learning rate
    bool    penalize_nl = false;    // Penalize newlines
    bool    ignore_eos = false;     // Ignore EOS for penalties

    std::vector<common_sampler_type> samplers = {
        common_sampler_type::TOP_K,
        common_sampler_type::TFS_Z,
        common_sampler_type::TYP_P,
        common_sampler_type::TOP_P,
        common_sampler_type::MIN_P,
        common_sampler_type::TEMPERATURE
    };

    std::string grammar;            // Grammar string (GBNF)
    // ... more options
};
```

## Performance Optimization

### Sampling Performance Tips

1. **Use greedy when possible**: For non-creative tasks, greedy is fastest

2. **Limit top-k**: Smaller k = faster (but less diverse)

3. **Grammar optimization**: Let `common_sampler` handle grammar efficiently

4. **Batch sampling**: If generating multiple outputs, can batch some operations

5. **Profile samplers**: Use timing info to identify slow samplers

### Timing Information

```cpp
// Get timing from common_sampler
common_perf_print(ctx, gsmpl);
```

Output:
```
sampling:
    sampling time = 1.23 ms / 100 tokens (0.012 ms per token, 81300 tokens per second)
    total time = 123.45 ms
```

### Memory Considerations

Samplers maintain state:
- **Penalty samplers**: Store last N tokens
- **Grammar samplers**: Store grammar state
- **Mirostat**: Store adaptive parameters

**Tip**: Reset samplers between unrelated generations:
```cpp
common_sampler_reset(gsmpl);
```

## Best Practices

1. **Start with defaults**: temp=0.8, top_k=40, top_p=0.95

2. **Tune for task**:
   - Code generation: Lower temperature (0.2-0.5)
   - Creative writing: Higher temperature (0.8-1.2)
   - Factual Q&A: Lower temperature + top-k

3. **Use penalties wisely**: Repetition penalty helps but can make output unnatural if too high

4. **Test grammar**: Validate your grammar with simple examples first

5. **Monitor quality**: Watch for repetition, incoherence, or overly random output

6. **Profile**: Use timing info to optimize your sampling chain

## Example: Complete Generation Loop

```cpp
// Setup
common_params_sampling params;
params.temp = 0.8;
params.top_k = 40;
params.top_p = 0.95;
common_sampler * gsmpl = common_sampler_init(model, params);

// Tokenize prompt
std::vector<llama_token> tokens = llama_tokenize(model, prompt, true);

// Process prompt
llama_batch batch = llama_batch_get_one(tokens.data(), tokens.size());
llama_decode(ctx, batch);

// Generate
int n_generated = 0;
while (n_generated < max_tokens) {
    // Sample
    llama_token token = common_sampler_sample(gsmpl, ctx, -1);

    // Check for end
    if (llama_vocab_is_eog(vocab, token)) break;

    // Accept
    common_sampler_accept(gsmpl, token, true);

    // Decode
    batch = llama_batch_get_one(&token, 1);
    llama_decode(ctx, batch);

    // Print
    printf("%s", llama_token_to_piece(ctx, token).c_str());
    fflush(stdout);

    n_generated++;
}

// Cleanup
common_sampler_free(gsmpl);
```

---

**Previous**: [Computation Graphs](04-computation-graphs.md) | **Next**: [Context Management](06-context-management.md)
