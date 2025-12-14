# OktoSemantic Training: Theoretical Foundation

**Author:** Ademir P. de Oliveira  
**Organization:** OktoSeek AI  
**Date:** December 2025

---

## 1. Problem Statement

### 1.1 Computational Bottleneck in Large-Vocabulary Models

Modern language models face a critical computational bottleneck when scaling vocabulary size:

**Traditional Cross-Entropy Training:**
- Compute logits: $\text{logits}_t = W_{\text{out}} h_t \in \mathbb{R}^{V}$
- Apply softmax: $p_t = \text{softmax}(\text{logits}_t) \in \mathbb{R}^{V}$
- Compute loss: $\ell_{\text{CE}} = -\log p_{t,y_t}$

**Computational Complexity**: $O(V)$ per token, where $V$ is the vocabulary size.

For vocabularies above 100K tokens, a significant fraction of training time is spent in the output layer:
- Computing $V$ logits
- Normalizing via softmax over $V$ classes
- Evaluating cross-entropy loss

Even though most semantic learning occurs in the backbone (GRU/Transformer), the softmax cost limits scalability.

### 1.2 Motivation

The key insight is that **semantic representation learning** and **discrete token resolution** can be decoupled:

1. **Semantic Learning**: The model learns contextual representations in a low-dimensional embedding space ($D_{\text{sem}} \ll V$)
2. **Token Resolution**: Alignment with discrete tokens is performed later through a lightweight fine-tuning phase

This decoupling allows the training cost to scale with a constant $K$ (number of negatives) and $D_{\text{sem}}$ (semantic dimension), rather than with $V$.

---

## 2. Method: Semantic Contrastive Token Approximation

### 2.1 Architecture Overview

OktoSemantic Training consists of three stages:

1. **Stage 1**: Semantic Contrastive Pre-training
2. **Stage 2**: Hybrid Alignment with Cross-Entropy
3. **Stage 3**: Standard Inference

---

### 2.2 Stage 1: Semantic Contrastive Pre-training

During this phase, the model predicts a **normalized semantic vector** instead of a token distribution.

#### 2.2.1 Semantic Projection

Given a hidden state $h_t \in \mathbb{R}^{d}$ from the backbone (GRU/Transformer), we project to a semantic space:

\[
z_t = \text{normalize}(W_{\text{sem}} h_t) \in \mathbb{R}^{D_{\text{sem}}}
\]

where:
- $W_{\text{sem}} \in \mathbb{R}^{D_{\text{sem}} \times d}$ is a learnable projection matrix
- $D_{\text{sem}} \ll V$ (typically $D_{\text{sem}} = 64$ for $V \geq 50K$)
- $\text{normalize}(x) = \frac{x}{\|x\|_2}$ ensures unit norm

#### 2.2.2 Semantic Embedding Table

We maintain a table of semantic embeddings, one per vocabulary token:

\[
E^{\text{sem}} \in \mathbb{R}^{V \times D_{\text{sem}}}
\]

Each embedding is normalized:

\[
e_i = \text{normalize}(E^{\text{sem}}_i) \in \mathbb{R}^{D_{\text{sem}}}
\]

#### 2.2.3 Contrastive Objective

For each target token $y_t$, we:
1. Use $e_{y_t}$ as the positive sample
2. Sample $K$ negative tokens $\{n_1, n_2, \ldots, n_K\}$ from the vocabulary
3. Compute the contrastive loss:

\[
\ell_{\text{sem}} = -\log \frac{\exp( z_t \cdot e_{y_t} / \tau)}
{\exp( z_t \cdot e_{y_t} / \tau) + \sum_{j=1}^{K} \exp( z_t \cdot e_{n_j} / \tau)}
\]

where:
- $\tau > 0$ is a temperature parameter (typically $\tau = 0.07$)
- $K$ is the number of negative samples (typically $K = 32$)
- $z_t \cdot e_i$ denotes the dot product (cosine similarity, since both are normalized)

#### 2.2.4 Computational Complexity

**Traditional CE**: $O(V)$ per token
- Compute $V$ logits: $O(V \cdot d)$
- Softmax over $V$: $O(V)$
- Cross-entropy: $O(1)$

**OktoSemantic**: $O(K \cdot D_{\text{sem}})$ per token
- Project to semantic space: $O(d \cdot D_{\text{sem}})$
- Sample $K$ negatives: $O(K)$ (amortized)
- Compute $K+1$ dot products: $O((K+1) \cdot D_{\text{sem}})$
- Contrastive loss: $O(K)$

**Key Advantage**: The cost scales with $K$ (constant) and $D_{\text{sem}}$ (constant), **not** with $V$.

For $V = 1M$, $K = 32$, $D_{\text{sem}} = 64$:
- CE: $O(1M)$
- OktoSemantic: $O(32 \cdot 64) = O(2048)$

This represents a **~500× reduction** in the output layer computation.

---

### 2.3 Stage 2: Hybrid Alignment with Cross-Entropy

After semantic pre-training, we introduce a standard token head for alignment.

#### 2.3.1 Token Head

\[
\text{logits}_t = W_{\text{tok}} h_t \in \mathbb{R}^{V}
\]

where $W_{\text{tok}} \in \mathbb{R}^{V \times d}$ is a learnable weight matrix.

#### 2.3.2 Training Strategy

- **Option A**: Train only the token head (freeze backbone and embeddings)
- **Option B**: Fine-tune the entire model with a reduced learning rate
- **Epochs**: Typically 1 epoch is sufficient, as the semantic backbone is already well-structured

#### 2.3.3 Cross-Entropy Loss

\[
\ell_{\text{CE}} = -\log \frac{\exp(\text{logits}_{t,y_t})}{\sum_{i=1}^{V} \exp(\text{logits}_{t,i})}
\]

This stage aligns the semantic backbone with discrete token identities, leveraging the already-structured semantic space.

---

### 2.4 Stage 3: Standard Inference

Once trained, the model behaves as a standard autoregressive language model, producing next-token distributions using the token head:

\[
p(\text{next token} = i | \text{context}) = \frac{\exp(\text{logits}_{t,i})}{\sum_{j=1}^{V} \exp(\text{logits}_{t,j})}
\]

---

## 3. Why It Works

### 3.1 Task Decoupling

The method treats semantic learning and token resolution as separate stages:
- **Stage 1**: Learn rich semantic representations in a low-dimensional space
- **Stage 2**: Map these representations to discrete tokens

This is analogous to:
- **Word2Vec**: Learn embeddings first, then use them for downstream tasks
- **Contrastive Learning**: Learn representations by contrasting positive and negative pairs

### 3.2 Computational Efficiency

The contrastive objective requires computing similarities only for:
- 1 positive sample
- $K$ negative samples

Total: $K+1$ comparisons, independent of $V$.

### 3.3 Gradient Flow

The contrastive loss provides gradients for:
- The semantic projection $W_{\text{sem}}$
- The semantic embeddings $E^{\text{sem}}$
- The backbone (via backpropagation through $h_t$)

These gradients are sufficient to learn meaningful semantic representations, as demonstrated by the experimental results.

---

## 4. Theoretical Analysis

### 4.1 Relationship to Negative Sampling

OktoSemantic can be viewed as a generalization of negative sampling (Mikolov et al., 2013):

- **Word2Vec Negative Sampling**: Approximates softmax over vocabulary
- **OktoSemantic**: Uses contrastive learning in a learned semantic space

The key difference is that OktoSemantic:
1. Learns a semantic projection (not just embeddings)
2. Operates in a reduced-dimensional space
3. Is followed by a fine-tuning stage for token alignment

### 4.2 Relationship to Contrastive Learning

The method is inspired by contrastive learning in computer vision (Chen et al., 2020):

- **SimCLR**: Contrastive learning for visual representations
- **OktoSemantic**: Contrastive learning for language model training

Both methods:
- Use normalized embeddings
- Sample negatives
- Apply temperature scaling

### 4.3 Approximation Quality

The contrastive objective approximates the full softmax:

\[
\ell_{\text{CE}} \approx \ell_{\text{sem}} + \text{regularization term}
\]

The approximation quality depends on:
- Number of negatives $K$ (more negatives → better approximation)
- Semantic dimension $D_{\text{sem}}$ (higher dimension → richer representations)
- Temperature $\tau$ (lower temperature → sharper distributions)

Experimental results show that $K=32$ and $D_{\text{sem}}=64$ are sufficient for vocabularies up to 1M tokens.

---

## 5. Scalability Analysis

### 5.1 Time Complexity

**Per Training Step:**

| Operation | CE | OktoSemantic |
|:----------|:---|:-------------|
| Forward pass (backbone) | $O(d^2)$ | $O(d^2)$ |
| Output projection | $O(V \cdot d)$ | $O(D_{\text{sem}} \cdot d)$ |
| Softmax/Loss | $O(V)$ | $O(K \cdot D_{\text{sem}})$ |
| **Total** | **$O(V \cdot d)$** | **$O(d^2 + K \cdot D_{\text{sem}})$** |

For large $V$, the output layer dominates in CE, but not in OktoSemantic.

### 5.2 Memory Complexity

**Memory per Batch:**

| Component | CE | OktoSemantic |
|:----------|:---|:-------------|
| Logits | $B \cdot T \cdot V$ | $B \cdot T \cdot (K+1)$ |
| Gradients | $O(V \cdot d)$ | $O(D_{\text{sem}} \cdot d + V \cdot D_{\text{sem}})$ |

For $B=16$, $T=64$, $V=1M$, $K=32$:
- CE logits: $16 \cdot 64 \cdot 1M = 1.024B$ floats
- OktoSemantic logits: $16 \cdot 64 \cdot 33 = 33,792$ floats

**~30,000× reduction** in logit memory.

---

## 6. Limitations and Future Work

### 6.1 Current Limitations

1. **Model Size**: Experiments conducted with GRU; validation needed for Transformers
2. **Dataset Scale**: Limited to 20K examples; larger-scale validation required
3. **Quality Metrics**: Focus on throughput; downstream task evaluation needed
4. **Hyperparameters**: Optimal $K$, $D_{\text{sem}}$, $\tau$ may vary with vocabulary size

### 6.2 Future Directions

1. **Transformer Validation**: Apply to Transformer architectures
2. **Extreme Vocabularies**: Explore 2M, 5M, 10M token vocabularies
3. **Quality Evaluation**: Measure perplexity, Hit@k, downstream task performance
4. **Hybrid Training**: Optimize the balance between semantic and token stages

---

## 7. References

- Mikolov, T., et al. (2013). Distributed Representations of Words and Phrases and their Compositionality. *NIPS*.
- Chen, T., et al. (2020). A Simple Framework for Contrastive Learning of Visual Representations. *ICML*.

---

## Scientific Archives

**Primary scientific record:**  
Open Science Framework (OSF): [10.17605/OSF.IO/MRKPU](https://doi.org/10.17605/OSF.IO/MRKPU)

**Repository archive:**  
Zenodo: [10.5281/zenodo.17931292](https://doi.org/10.5281/zenodo.17931292)

---

**Last Updated**: December 2025  
**Version**: 1.0
