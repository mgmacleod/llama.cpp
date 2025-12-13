# Existing Mechanistic Interpretability Projects

A survey of existing tools and projects for mechanistic interpretability with Llama models and llama.cpp.

---

## llama.cpp-Based Projects

### 1. Llama.MIA (Mechanistic Interpretability Application)

**Status:** ✅ Active fork of llama.cpp
**Repository:** https://github.com/coolvision/llama.mia (mia branch)
**Author:** coolvision
**Blog Post:** https://grgv.xyz/blog/llama.mia/
**Delta with current llama.cpp**: tools/cvector-generator/MIA_INTEGRATION_ANALYSIS.md

**Description:**
A fork of llama.cpp specifically designed for mechanistic interpretability research. Currently supports CPU-only inference and has been tested with Llama2 models.

**Key Features:**

1. **Attention Visualization** (`--draw`)
   - Generates PNG visualizations of attention maps
   - Shows how attention heads interact across layers
   - Useful for studying attention patterns

2. **Computation Graph Analysis** (`--print-cgraph`)
   - Outputs detailed tensor information
   - Displays computation graph nodes
   - Shows tensor types, operations, dimensions, memory addresses
   - Complete model structure inspection

3. **Logit Lens** (`--ll` or `--logit-lens`)
   - Applies unembedding to intermediate layer representations
   - Reveals "what tokens the model is 'predicting' at each layer"
   - Works on:
     - Residual stream (`l_out`)
     - Attention outputs (`kqv_out`)
     - MLP outputs (`ffn_out`)
     - Other embedding-dimension tensors

4. **Attention Head Ablation**
   - **Zero ablation** (`-a`): Ablate specific heads (comma-separated indices)
   - **Head isolation** (`-s`): Retain only one head per layer
   - Useful for testing head importance and function

5. **Activation Patching** (`--patch`, `--patch-avg`)
   - Replace tensor values at specified token indices
   - Load patches from files
   - Enables causal analysis of component interactions
   - Test hypotheses about information flow

6. **Tensor I/O** (`--save`, `--load`)
   - Save tensors to files for offline analysis
   - Load and patch tensors from files
   - Enable experimental manipulation

**Technical Implementation:**
- Refactored to use hooks and callbacks (not global variables)
- Improved maintainability and thread safety
- Built directly into llama.cpp inference loop

**Limitations:**
- CPU only (no GPU support yet)
- Tested primarily with Llama2
- May not support latest llama.cpp features

**Use Cases:**
- Attention pattern analysis
- Circuit discovery via ablation
- Logit lens exploration
- Activation patching experiments
- Studying layer-wise representations

**How to Use:**
```bash
# Clone the fork
git clone https://github.com/coolvision/llama.mia.git -b mia
cd llama.mia

# Build
make

# Run with logit lens
./llama-cli -m model.gguf -p "Your prompt" --logit-lens

# Ablate attention heads 0,1,2
./llama-cli -m model.gguf -p "Your prompt" -a 0,1,2

# Generate attention visualization
./llama-cli -m model.gguf -p "Your prompt" --draw

# Activation patching
./llama-cli -m model.gguf -p "Your prompt" --patch tensor_file.bin --patch-avg
```

**Research Potential:**
- Study what each layer "knows"
- Find important attention heads
- Test causal hypotheses via patching
- Discover computational circuits
- Visualize attention patterns

---

## PyTorch/HuggingFace-Based Projects

### 2. llama3_interpretability_sae

**Status:** ✅ Active, complete pipeline
**Repository:** https://github.com/PaulPauls/llama3_interpretability_sae
**Author:** Paul Pauls
**Year:** 2024

**Description:**
A complete end-to-end pipeline for LLM interpretability using Sparse Autoencoders (SAEs) with Llama 3.2, written in pure PyTorch and fully reproducible.

**Key Features:**

1. **Full Pipeline:**
   - Capture training data from model activations
   - Train Sparse Autoencoders
   - Analyze learned features
   - Verify results experimentally

2. **Model:**
   - Currently provides SAE for Llama 3.2-3B
   - All code, data, and trained models available
   - Fully reproducible research

3. **Sparse Autoencoder Approach:**
   - Projects activations into large, sparse latent space
   - Untangles superimposed representations
   - Each SAE neuron represents one interpretable concept
   - Addresses superposition hypothesis

**Use Cases:**
- Feature discovery (what concepts does model represent?)
- Monosemanticity analysis (one feature = one concept?)
- Feature composition (which features co-occur?)
- Representation geometry

**Workflow:**
1. Extract activations from Llama 3.2 (using HuggingFace/PyTorch)
2. Train SAE on activations
3. Analyze which features activate for which inputs
4. Interpret and verify features

**Not llama.cpp compatible** - Uses PyTorch/HuggingFace directly.

---

### 3. Llama Scope

**Status:** ✅ Major research project
**Paper:** https://arxiv.org/abs/2410.20526
**Models:** https://huggingface.co/fnlp/Llama-Scope
**Code:** https://github.com/OpenMOSS/Language-Model-SAEs
**Authors:** Multiple institutions
**Year:** 2024

**Description:**
Large-scale SAE training on Llama-3.1-8B-Base, extracting millions of interpretable features across all layers.

**Key Features:**

1. **Scale:**
   - 256 SAEs total (one per layer and sublayer)
   - 32K and 128K feature variants
   - Trained on Llama-3.1-8B-Base

2. **Evaluation Metrics:**
   - Sparsity-fidelity trade-off
   - Latent firing frequency
   - Feature interpretability scores
   - Comprehensive benchmarking

3. **Infrastructure:**
   - Training tools
   - Interpretation utilities
   - Visualization framework
   - All available at OpenMOSS/Language-Model-SAEs

**Use Cases:**
- Large-scale feature extraction
- Cross-layer feature analysis
- Comprehensive model understanding
- Benchmark for SAE methods

**Not llama.cpp compatible** - Research project using standard PyTorch infrastructure.

---

### 4. Goodfire SAEs

**Status:** ✅ Open source, production-ready
**Website:** https://www.goodfire.ai/research/understanding-and-steering-llama-3
**Announcement:** https://www.goodfire.ai/blog/sae-open-source-announcement
**Organization:** Goodfire AI
**Year:** 2024

**Description:**
Production-quality Sparse Autoencoders for Llama 3.3 70B and Llama 3.1 8B, with steering capabilities.

**Key Features:**

1. **Models:**
   - Llama 3.3 70B SAEs
   - Llama 3.1 8B SAEs
   - Optimized for both understanding and steering

2. **Applications:**
   - Feature interpretation
   - Behavioral steering via feature activation
   - Research preview with live demo

3. **Production Focus:**
   - High-quality trained models
   - Tested for practical use
   - Accelerate interpretability research

**Use Cases:**
- Steering model behavior via features
- Understanding model representations
- Production deployments with interpretability

**Not llama.cpp compatible** - Designed for HuggingFace Transformers.

---

### 5. Language-Model-SAEs (OpenMOSS)

**Status:** ✅ Active framework
**Repository:** https://github.com/OpenMOSS/Language-Model-SAEs
**Organization:** OpenMOSS
**Year:** 2024

**Description:**
Performant framework for training, analyzing, and visualizing Sparse Autoencoders and their variants.

**Key Features:**

1. **Training:**
   - Scalable SAE training
   - Multiple SAE variants supported
   - Optimized for performance

2. **Analysis:**
   - Feature interpretation tools
   - Activation pattern analysis
   - Circuit discovery capabilities

3. **Visualization:**
   - Feature visualizations
   - Activation heatmaps
   - Interactive exploration

4. **Circuit Discovery:**
   - Dictionary learning improves patch-free circuit discovery
   - Case studies (e.g., Othello-GPT)
   - Automated circuit analysis

**Use Cases:**
- Train custom SAEs
- Visualize learned features
- Discover computational circuits
- Compare SAE architectures

**Not llama.cpp compatible** - Standalone framework.

---

## Related Tools & Frameworks

### 6. TransformerLens

**Status:** ✅ Mature library
**Repository:** https://github.com/TransformerLensOrg/TransformerLens
**Description:** Library for mechanistic interpretability of GPT-style language models

**Key Features:**
- Converts HuggingFace models to HookedTransformer
- Easy access to intermediate activations
- Extensive hooks for intervention
- Gradient-based analysis
- Activation patching
- Attention pattern visualization

**Comparison to llama.cpp:**
- More features, easier API
- Gradient support
- Slower inference
- Higher memory usage
- Python-only

**Not llama.cpp compatible** - Separate Python framework.

---

### 7. EleutherAI SAE Tools

**Status:** ✅ Active development
**Repository:** https://github.com/EleutherAI/sae (training)
**Repository:** https://github.com/EleutherAI/sae-auto-interp (interpretation)
**Organization:** EleutherAI
**Year:** 2024

**Description:**
Tools for training k-sparse autoencoders and automatically interpreting learned features.

**Key Features:**

1. **Training Library:**
   - Trains k-sparse autoencoders on HuggingFace activations
   - Follows "Scaling and evaluating sparse autoencoders" (Gao et al. 2024)
   - Optimized implementation

2. **Auto-Interpretation:**
   - Automated interpretation of SAE features
   - Generate feature descriptions
   - Validate interpretations
   - Open source library

**Use Cases:**
- Train SAEs on custom models
- Automatically interpret features
- Scale interpretability research

**Not llama.cpp compatible** - HuggingFace-focused.

---

## Comparison Matrix

| Project | llama.cpp Compatible | Activation Access | Attention Analysis | SAE Support | Circuit Discovery | Gradients | Status |
|---------|---------------------|-------------------|-------------------|-------------|-------------------|-----------|--------|
| **Llama.MIA** | ✅ Yes (fork) | ✅ Yes | ✅ Yes (visualize + ablate) | ❌ No | ⚠️ Partial (patching) | ❌ No | Active |
| **llama3_interp_sae** | ❌ No | ✅ Yes (PyTorch) | ❌ Limited | ✅ Yes | ❌ No | ✅ Yes | Active |
| **Llama Scope** | ❌ No | ✅ Yes (PyTorch) | ❌ No | ✅ Yes (256 SAEs) | ❌ No | ✅ Yes | Active |
| **Goodfire SAEs** | ❌ No | ✅ Yes (HF) | ❌ No | ✅ Yes | ❌ No | ✅ Yes | Active |
| **OpenMOSS SAE** | ❌ No | ✅ Yes (PyTorch) | ❌ No | ✅ Yes | ✅ Yes | ✅ Yes | Active |
| **TransformerLens** | ❌ No | ✅ Yes (hooks) | ✅ Yes | ⚠️ Via plugins | ✅ Yes | ✅ Yes | Mature |
| **EleutherAI SAE** | ❌ No | ✅ Yes (HF) | ❌ No | ✅ Yes | ❌ No | ✅ Yes | Active |
| **cvector-generator** | ✅ Yes (native) | ✅ Yes (callbacks) | ⚠️ Limited | ❌ No | ❌ No | ❌ No | Native |

---

## Key Insights

### What Exists for llama.cpp:

1. **Llama.MIA** is the **ONLY** llama.cpp-based interpretability tool
   - Provides attention visualization, logit lens, ablation, patching
   - CPU-only, tested with Llama2
   - Most directly relevant to your use case

2. **cvector-generator** (your current codebase)
   - Native llama.cpp tool
   - Activation extraction working
   - Could be extended with SAE or other methods

### What's Missing in llama.cpp Ecosystem:

1. **Sparse Autoencoders:**
   - All SAE work happens in PyTorch/HuggingFace
   - No llama.cpp-native SAE training or inference
   - Could build: Extract activations with llama.cpp → train SAE offline → apply SAE encoder during llama.cpp inference

2. **Gradient-Based Methods:**
   - All gradient work in PyTorch (llama.cpp is inference-only)
   - No integrated gradients, saliency maps, etc.
   - Workaround: numerical gradients (very slow)

3. **Attention Pattern Access:**
   - Llama.MIA has attention visualization
   - No direct access to attention scores in main llama.cpp
   - Flash attention makes this harder

### Opportunities:

1. **Extend cvector-generator:**
   - Add SAE encoder inference (train offline, run in llama.cpp)
   - Add more reduction methods beyond PCA/mean
   - Add activation statistics (like imatrix)
   - Add logit lens capability

2. **Port Llama.MIA features:**
   - Attention visualization
   - Logit lens
   - Ablation tools
   - Merge into main llama.cpp or keep as fork

3. **Hybrid Approach:**
   - Use llama.cpp for fast activation extraction
   - Use PyTorch for SAE training, gradient methods
   - Use llama.cpp for SAE inference (convert SAE to GGUF)
   - Best of both worlds: speed + features

4. **Novel Contributions:**
   - llama.cpp-native SAE inference
   - Efficient circuit discovery with activation caching
   - Large-scale activation analysis (leverage llama.cpp speed)
   - Real-time interpretability during inference

---

## Recommendations

### For Your DIY Mechanistic Interpretability Goals:

**Option 1: Use Llama.MIA**
- Fork/clone Llama.MIA
- Immediately get attention viz, logit lens, ablation
- Build on existing interpretability-focused codebase
- Contribute back improvements

**Pros:**
- Ready to use today
- Proven interpretability features
- llama.cpp-based (fast inference)

**Cons:**
- CPU only (GPU would require porting)
- Llama2 focus (may need updates for Llama3)
- Fork maintenance burden

**Option 2: Extend cvector-generator**
- You already understand this codebase
- Add features incrementally:
  1. Logit lens (easy - just apply output projection)
  2. Activation statistics (easy - like imatrix)
  3. Attention extraction (medium - hook Q,K,V)
  4. SAE inference (medium - train offline, run in llama.cpp)
  5. Circuit discovery (hard - activation patching)

**Pros:**
- Build on your existing work
- Control over features
- Can upstream to main llama.cpp
- GPU support already there

**Cons:**
- More development work
- Start from scratch for some features

**Option 3: Hybrid Approach**
- Extract activations with cvector-generator (fast, GPU)
- Train SAEs with llama3_interpretability_sae (full pipeline)
- Analyze with OpenMOSS/Language-Model-SAEs (visualizations)
- Use llama.cpp for production/inference

**Pros:**
- Best of all worlds
- Use mature tools for each step
- Leverage community work

**Cons:**
- More complex workflow
- Multiple codebases to manage

**Option 4: Contribute to Llama.MIA**
- Port GPU support to Llama.MIA
- Update for Llama3
- Add missing features (SAE, more ablations)
- Collaborate with coolvision

**Pros:**
- Community benefit
- Shared maintenance
- Build on proven foundation

**Cons:**
- Requires coordination
- May not match your exact needs

---

## Immediate Next Steps

**To explore existing tools:**

1. **Try Llama.MIA:**
```bash
git clone https://github.com/coolvision/llama.mia.git -b mia
cd llama.mia
make
./llama-cli -m your-llama2-model.gguf -p "Test prompt" --logit-lens
./llama-cli -m your-llama2-model.gguf -p "Test prompt" --draw
```

2. **Experiment with llama3_interpretability_sae:**
```bash
git clone https://github.com/PaulPauls/llama3_interpretability_sae.git
cd llama3_interpretability_sae
# Follow their pipeline to train SAE on Llama 3.2-3B
# Extract activations → Train SAE → Interpret features
```

3. **Compare approaches:**
- Which gives you the insights you need?
- Which fits your workflow?
- What features are missing?

**To build your own:**

1. **Start with cvector-generator extensions:**
   - Add logit lens (1-2 days)
   - Add activation statistics (1-2 days)
   - Add attention extraction (3-5 days)

2. **Research hybrid SAE approach:**
   - Extract activations with cvector-generator
   - Train SAE with PyTorch offline
   - Convert SAE to GGUF
   - Run SAE encoder in llama.cpp callback

3. **Document and share:**
   - Whatever you build, document it
   - Share with community
   - Upstream useful features

---

## Further Resources

### Papers:
- "Scaling Monosemanticity" (Anthropic, 2024) - SAE methodology
- "Sparse Autoencoders Find Highly Interpretable Features" (ICLR 2024)
- "Llama Scope: Extracting Millions of Features" (2024)

### Awesome Lists:
- https://github.com/cooperleong00/Awesome-LLM-Interpretability
- https://github.com/JShollaj/awesome-llm-interpretability
- https://github.com/Dakingrai/awesome-mechanistic-interpretability-lm-papers

### Communities:
- llama.cpp GitHub discussions
- Alignment Forum / LessWrong (mech interp posts)
- EleutherAI Discord (interpretability channel)

---

## Conclusion

**The good news:** Mechanistic interpretability for Llama models is a very active research area with multiple mature projects.

**The challenge:** Most tools are PyTorch/HuggingFace-based, not llama.cpp-native.

**The opportunity:** llama.cpp's speed and efficiency make it ideal for large-scale interpretability work, but the ecosystem needs more tools.

**Your options:**
1. Use Llama.MIA (ready now, llama.cpp-based)
2. Extend cvector-generator (build what you need)
3. Hybrid approach (llama.cpp + PyTorch tools)
4. Contribute to existing projects

**Best path forward:** Try Llama.MIA first to see if it meets your needs. If not, extend cvector-generator incrementally, starting with logit lens and activation statistics. Consider a hybrid approach for SAE work.

The foundation you've built with cvector-generator is solid - it's already doing the hard part (activation extraction via callbacks). Adding interpretability features on top is very doable.
