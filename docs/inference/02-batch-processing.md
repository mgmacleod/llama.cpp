# Batch Processing in llama.cpp

This document explains how llama.cpp processes batches of tokens, including batch structure, validation, auto-generation of fields, and batch splitting.

## Table of Contents

1. [What is a Batch?](#what-is-a-batch)
2. [Batch Structure](#batch-structure)
3. [Creating Batches](#creating-batches)
4. [Batch Allocator](#batch-allocator)
5. [Batch Validation and Auto-Generation](#batch-validation-and-auto-generation)
6. [Batch Splitting (ubatch)](#batch-splitting-ubatch)
7. [Sequence Management](#sequence-management)

## What is a Batch?

A **batch** (`llama_batch`) is the fundamental unit of input to the inference engine. It represents one or more tokens that should be processed together. Batches enable:

- **Efficient prompt processing**: Process all prompt tokens in one pass
- **Parallel sequence processing**: Handle multiple independent sequences simultaneously
- **Flexible attention**: Control which tokens attend to which via positions and sequence IDs

## Batch Structure

Defined in [include/llama.h](../../include/llama.h):

```cpp
struct llama_batch {
    int32_t n_tokens;  // Number of tokens in this batch

    // Either tokens OR embeddings (not both)
    llama_token *  token;    // Token IDs [n_tokens]
    float *        embd;     // Token embeddings [n_tokens * n_embd]

    // Position and sequence information
    llama_pos *    pos;      // Position of each token [n_tokens]
    int32_t *      n_seq_id; // Number of sequences for each token [n_tokens]
    llama_seq_id **seq_id;   // Sequence IDs for each token [n_tokens][n_seq_id[i]]

    // Output control
    int8_t *       logits;   // Whether to compute logits for this token [n_tokens]
};
```

### Field Descriptions

#### `n_tokens`
The total number of tokens in the batch.

#### `token` / `embd`
- **`token`**: Array of token IDs (most common usage)
- **`embd`**: Array of token embeddings (for models that accept embedding inputs)
- Only one should be set; the other should be `nullptr`

#### `pos`
Position of each token in its sequence. Used for:
- Positional encodings (RoPE, ALiBi, etc.)
- Determining attention masks
- Can be `nullptr` - will be auto-generated from KV cache state

#### `n_seq_id` / `seq_id`
Each token can belong to one or more sequences:
- `n_seq_id[i]`: How many sequences token `i` belongs to
- `seq_id[i][j]`: The j-th sequence ID for token `i`
- Can be `nullptr` - will default to sequence 0

**Use cases**:
- **Single sequence**: All tokens have `n_seq_id[i] = 1` and `seq_id[i][0] = 0`
- **Multiple sequences**: Different tokens belong to different sequences (for parallel processing)
- **Shared context**: Some tokens can belong to multiple sequences (for shared prompt prefixes)

#### `logits`
Controls which tokens should produce output logits:
- `logits[i] != 0`: Compute logits for token `i`
- `logits[i] == 0`: Skip logits computation for token `i`
- Can be `nullptr` - will default to computing logits only for the last token

**Performance tip**: Only request logits for tokens where you need them (usually just the last token during generation).

## Creating Batches

### Method 1: Quick Single-Sequence Batch

From [src/llama-batch.cpp:861](../../src/llama-batch.cpp#L861):

```cpp
llama_batch batch = llama_batch_get_one(tokens, n_tokens);
```

This creates a simple batch where:
- All tokens belong to sequence 0
- Positions are auto-generated
- Logits are computed only for the last token

**Use case**: Simple single-sequence generation.

### Method 2: Full Custom Batch

From [src/llama-batch.cpp:875](../../src/llama-batch.cpp#L875):

```cpp
llama_batch batch = llama_batch_init(n_tokens_alloc, embd, n_seq_max);

// Fill in the batch fields manually
batch.n_tokens = actual_n_tokens;
for (int i = 0; i < batch.n_tokens; i++) {
    batch.token[i] = my_tokens[i];
    batch.pos[i] = my_positions[i];
    batch.n_seq_id[i] = 1;
    batch.seq_id[i][0] = my_seq_id;
    batch.logits[i] = (i == batch.n_tokens - 1);
}

// ... use batch ...

llama_batch_free(batch);  // Don't forget to free!
```

**Use case**: Complex scenarios with multiple sequences, custom positions, or shared contexts.

### Method 3: Let the Allocator Handle It

The most flexible approach - provide only the minimum information and let `llama_batch_allocr` fill in the rest:

```cpp
// Minimal batch - just tokens
llama_batch batch = {
    .n_tokens = n,
    .token = my_tokens,
    .embd = nullptr,
    .pos = nullptr,        // Will be auto-generated
    .n_seq_id = nullptr,   // Will be auto-generated
    .seq_id = nullptr,     // Will be auto-generated
    .logits = nullptr,     // Will be auto-generated
};

llama_decode(ctx, batch);  // Allocator handles the rest
```

## Batch Allocator

The `llama_batch_allocr` class ([src/llama-batch.cpp](../../src/llama-batch.cpp)) is responsible for:
1. Validating batch inputs
2. Auto-generating missing fields
3. Computing batch statistics
4. Splitting large batches into smaller "ubatches"

### Auto-Generation Rules

From [src/llama-batch.cpp:25](../../src/llama-batch.cpp#L25) `llama_batch_allocr::init()`:

#### Auto-generate `n_seq_id` and `seq_id`
If not provided:
- All tokens default to sequence 0
- `n_seq_id[i] = 1` for all tokens
- `seq_id[i][0] = 0` for all tokens

#### Auto-generate `pos`
If not provided:
- Start from the end of the KV cache for each sequence
- Increment position for each subsequent token
- Example: If KV cache has positions 0-9 for sequence 0, next tokens get positions 10, 11, 12, ...

```cpp
if (!batch.pos) {
    // Initialize starting position for each sequence
    for (uint32_t s = 0; s < n_seq_max; ++s) {
        if (!memory) {
            p0[s] = 0;  // No memory -> start from 0
        } else {
            p0[s] = memory->seq_pos_max(s) + 1;  // Continue from cache
        }
    }

    // Assign positions
    for (int32_t i = 0; i < batch.n_tokens; i++) {
        const llama_seq_id seq_id = batch.seq_id[i][0];
        pos[i] = p0[seq_id]++;  // Assign and increment
    }

    batch.pos = pos.data();
}
```

#### Auto-generate `logits`
If not provided:
- With `embeddings == true`: All tokens produce output
- Otherwise: Only the last token produces output

```cpp
if (!batch.logits) {
    if (output_all) {
        output.resize(batch.n_tokens, true);   // All true
    } else {
        output.resize(batch.n_tokens, false);
        output[output.size() - 1] = true;      // Only last
    }
    batch.logits = output.data();
}
```

## Batch Validation and Auto-Generation

### Validation Checks

From [src/llama-batch.cpp:40-384](../../src/llama-batch.cpp#L40-L384):

1. **Token ID validation**: All token IDs must be in range `[0, n_vocab)`

2. **Sequence ID validation**: All sequence IDs must be in range `[0, n_seq_max)`

3. **Position consistency**: Positions must be consistent with KV cache
   - For standard RoPE: Positions must be continuous (pos[i] = last_pos + 1)
   - For M-RoPE: Positions can "jump" forward but not backward

4. **No position gaps**: For standard models, no gaps in positions allowed

5. **No coupled sequences with diverged history**: If two sequences share tokens in the batch, they must have identical KV cache history

6. **No sequence subset changes**: Can't suddenly drop sequences from a token's sequence set

7. **No decreasing positions**: Within a sequence, positions must be non-decreasing

### Example Validation Error

```cpp
// Invalid batch - positions not continuous
llama_batch batch = {
    .n_tokens = 3,
    .token = (llama_token[]){1, 2, 3},
    .pos = (llama_pos[]){10, 15, 20},  // Gap between positions!
    // ...
};

llama_decode(ctx, batch);  // ERROR: positions are not continuous
```

## Batch Splitting (ubatch)

Large batches may not fit in hardware constraints (memory, attention size, etc.). The batch allocator can split a large batch into smaller "ubatches" that are processed sequentially.

### ubatch Structure

```cpp
struct llama_ubatch {
    bool     b_equal_seqs;   // Are all sequences equal length?
    uint32_t n_tokens;       // Total tokens
    uint32_t n_seq_tokens;   // Tokens per sequence
    uint32_t n_seqs;         // Number of sequences
    uint32_t n_seqs_unq;     // Number of unique sequences
    uint32_t n_pos;          // Positions per embedding (for M-RoPE)

    // Same fields as llama_batch
    llama_token  * token;
    float        * embd;
    llama_pos    * pos;
    int32_t      * n_seq_id;
    llama_seq_id **seq_id;
    llama_seq_id * seq_id_unq;  // Unique sequence IDs
    int32_t      * seq_idx;      // Sequence index mapping
    int8_t       * output;

    std::shared_ptr<data_t> data;  // Owns the memory
};
```

### Splitting Strategies

From [src/llama-batch.cpp:472-651](../../src/llama-batch.cpp#L472-L651):

#### 1. Simple Split (`split_simple`)
Takes tokens sequentially until reaching `n_ubatch` size.

```cpp
llama_ubatch ubatch = allocr.split_simple(n_ubatch);
```

Use case: Single sequence or when order doesn't matter.

#### 2. Equal Split (`split_equal`)
Takes equal number of tokens from each sequence to create balanced ubatches.

```cpp
llama_ubatch ubatch = allocr.split_equal(n_ubatch, sequential);
```

- `sequential = true`: Requires sequences to be in order (0, 1, 2, ...)
- `sequential = false`: Can handle any sequence ordering

Use case: Parallel sequence processing with balanced load.

#### 3. Sequence Split (`split_seq`)
Groups tokens by sequence, keeping sequences together.

```cpp
llama_ubatch ubatch = allocr.split_seq(n_ubatch);
```

Use case: When sequences shouldn't be interleaved.

### Example: Processing with Splits

```cpp
llama_batch_allocr allocr;
allocr.init(batch, vocab, memory, n_embd, n_seq_max, false);

// Split and process
while (true) {
    llama_ubatch ubatch = allocr.split_equal(512, false);
    if (ubatch.n_tokens == 0) break;  // Done

    // Process this ubatch
    process_ubatch(ctx, ubatch);
}
```

## Sequence Management

### Sequence Use Cases

1. **Single sequence generation**
   ```cpp
   // All tokens belong to sequence 0
   llama_batch batch = llama_batch_get_one(tokens, n);
   ```

2. **Parallel generation** (e.g., beam search, multiple prompts)
   ```cpp
   // Token 0,2,4 -> seq 0; Token 1,3,5 -> seq 1
   batch.seq_id[0][0] = 0;
   batch.seq_id[1][0] = 1;
   batch.seq_id[2][0] = 0;
   batch.seq_id[3][0] = 1;
   // ...
   ```

3. **Shared context** (multiple generations from same prefix)
   ```cpp
   // Prompt tokens belong to both sequences
   for (int i = 0; i < prefix_len; i++) {
       batch.n_seq_id[i] = 2;
       batch.seq_id[i][0] = 0;
       batch.seq_id[i][1] = 1;  // Shared between seq 0 and 1
   }

   // Generation tokens belong to individual sequences
   for (int i = prefix_len; i < n; i++) {
       batch.n_seq_id[i] = 1;
       batch.seq_id[i][0] = (i - prefix_len) % 2;  // Alternate
   }
   ```

### Sequence Operations

Sequences in the KV cache can be manipulated:

```cpp
// Remove a sequence from the cache
llama_kv_cache_seq_rm(ctx, seq_id, -1, -1);

// Copy sequence (e.g., for branching)
llama_kv_cache_seq_cp(ctx, src_seq_id, dst_seq_id, -1, -1);

// Keep only one sequence, remove all others
llama_kv_cache_seq_keep(ctx, seq_id);

// Shift positions (for sliding window)
llama_kv_cache_seq_add(ctx, seq_id, p0, p1, shift);
```

## Best Practices

1. **Use `llama_batch_get_one()` for simple cases**: Let the library handle the complexity

2. **Don't over-specify**: Provide only the fields you need to control; let auto-generation handle the rest

3. **Request logits only where needed**: Set `logits[i] = 0` for tokens where you don't need output

4. **Validate your batch**: The allocator provides detailed error messages if your batch is invalid

5. **Use sequences wisely**:
   - Single sequence for simple generation
   - Multiple sequences for parallel processing
   - Shared sequences for common prefixes

6. **Monitor memory**: Large batches consume more memory; consider splitting if needed

## Debugging

Set environment variable to enable batch debugging:

```bash
export LLAMA_BATCH_DEBUG=1  # Basic info
export LLAMA_BATCH_DEBUG=2  # Detailed token info
```

This will print detailed batch information during processing.

---

**Previous**: [Overview](01-overview.md) | **Next**: [KV Cache System](03-kv-cache.md)
