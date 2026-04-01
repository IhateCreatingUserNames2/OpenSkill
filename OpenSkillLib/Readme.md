### CURRENT STATUS:
<img width="1024" height="933" alt="image" src="https://github.com/user-attachments/assets/bcc48563-ec29-4144-84ba-d89f1afb623b" />


### ANALYSIS:

# OpenSkill vs. Reference Papers: Implementation Gaps Analysis

## TurboQuant
**OpenSkill captured the essence correctly:** orthogonal rotation via QR, 4-bit quantization, and QJL residual for inner product without bias.

**The gap:** The reference paper uses an optimal codebook by solving a continuous k-means based on the Beta distribution of rotated coordinates. OpenSkill instead uses a min-max uniform approach. This results in suboptimal distortion, though the two-stage structure (MSE quantizer + QJL residual) is implemented.

---

## S-PATH RAG
**The gap is more serious.**

The paper specifies:
- A differentiable path scorer trained with **InfoNCE + BCE** (the *sθ* scorer and *vη* verifier)
- **Gumbel-Softmax** for smooth selection

**OpenSkill's approach:**
- Uses a manual heuristic scoring system (tuned over several hours):
  - `0.3 * structural`
  - `0.5 * semantic`
  - `0.2 * diversity`
- While this works as a proxy, it does not match the paper's proposed method.

**Cross-attention injection:** Implemented via `SkillProjector`, but **not trained**. This was the "static noise" that Qwen 0.8B initially failed to process.

---

## MemCollab
**Most faithful implementation.**

- The **weak vs. strong contrast** for extracting invariants and violation patterns is implemented directly.
- **Scale difference:** Paper uses Qwen 7B vs. 32B; OpenSkill uses GPT-4o-mini vs. Claude 3.5 Sonnet.
- Despite the difference in model scale, the approach remains true to the underlying principle.

---

## Trace2Skill
- The central idea (*parallel fleet + patches + consolidation*) exists in the `evolve` component.
- **Missing piece:** The paper is very specific about **hierarchical consolidation** with programmatic detection of conflicts between patches—OpenSkill does not yet implement this.

---

## Overall Assessment

**OpenSkill is essentially a very solid proof of concept**—it implements the correct skeleton of all four systems.

### Most impactful gaps:
1. **Suboptimal codebook** in TurboQuant
2. **Untrained path scorer** (*sθ*, *vη*) in S-PATH
3. **Untrained GNN**

### Next steps:
- For a research/MVP project, OpenSkill is impressively well-founded.
- For production, the natural next step is to train the *sθ* and *vη* components of S-PATH with at least a few hundred examples of positive/negative query-path pairs.
