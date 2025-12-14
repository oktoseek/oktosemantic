# OktoSemantic Training (SCTA)

**Semantic Contrastive Token Approximation for Large-Vocabulary Language Models**

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17931292.svg)](https://doi.org/10.5281/zenodo.17931292) [![OSF DOI](https://img.shields.io/badge/DOI-10.17605%2FOSF.IO%2FMRKPU-blue)](https://doi.org/10.17605/OSF.IO/MRKPU)

---

## Abstract

OktoSemantic Training, also known as **Semantic Contrastive Token Approximation (SCTA)**, is a novel training methodology for large-vocabulary language models that addresses the computational bottleneck caused by full softmax normalization over large vocabularies. The method decouples semantic representation learning from discrete token resolution by replacing full vocabulary normalization with a contrastive objective operating in a reduced embedding space. Experimental validation demonstrates up to **4.43× speedup** on vocabularies of 1M tokens, with superior scalability compared to traditional cross-entropy training.

---

## Key Experimental Results

### Performance Highlights

| GPU | Vocabulary | Method | Time (s) | **Tokens/sec** | **Examples/sec** | Speedup |
|:----|:-----------|:------|:---------|:---------------:|:----------------:|:-------:|
| **RTX A6000** | 1M | CE | 535.09 | 4,710 | 74.8 | 1.00× |
| **RTX A6000** | 1M | **SEMANTIC** | **120.81** | **20,860** | **331.1** | **4.43×** |
| **RTX A6000** | 200K | CE | 111.56 | 22,589 | 358.5 | 1.00× |
| **RTX A6000** | 200K | **SEMANTIC** | **37.56** | **67,097** | **1,065.0** | **2.97×** |
| **RTX 4070** | 1M | CE | ❌ Failed | - | - | - |
| **RTX 4070** | 1M | **SEMANTIC** | **312.22** | **8,071** | **128.1** | **∞** |

**Key Achievement**: SEMANTIC enables **1M-token vocabulary training** on consumer hardware (RTX 4070) where traditional CE fails completely.

### Speedup Scaling (RTX A6000)

![Speedup vs Vocabulary Size - RTX A6000](./figures/fig_speedup_vs_vocab_rtx_a6000.png)

*Speedup increases dramatically with vocabulary size: 1.61× (50K) → 2.97× (200K) → 4.43× (1M tokens)*

### Throughput Comparison (RTX A6000)

![Throughput vs Vocabulary Size - RTX A6000](./figures/fig_throughput_vs_vocab_rtx_a6000.png)

*SEMANTIC maintains significantly higher throughput (tokens/second) across all vocabulary sizes, with the advantage increasing for larger vocabularies.*

---

## Authorship

**Author:** Ademir P. de Oliveira  
**Organization:** OktoSeek AI  
**Date:** December 2025

This method was conceived, designed, implemented, and evaluated by the author using a private training engine. All reported experimental results were obtained from real training runs executed on actual hardware (NVIDIA RTX 4070 Laptop GPU and NVIDIA RTX A6000).

---

## Repository Purpose

This repository serves as a **scientific disclosure and prior-art record** for the SCTA training method. It is intended for:

- **Research communication** and authorship establishment
- **Formal documentation** of the mathematical formulation
- **Experimental evidence** of empirical validity
- **Prior art** documentation for scientific and patent purposes

The authoritative scientific record of this work is maintained on the Open Science Framework (OSF):  
https://doi.org/10.17605/OSF.IO/MRKPU

**This repository is NOT intended to provide a reference implementation.** Source code, training scripts, and optimized kernels are not included. The absence of implementation details is intentional and does not imply the absence of a working system.

---

## Contents

| Document | Description |
|:---------|:------------|
| **[THEORY.md](./THEORY.md)** | Mathematical formulation, architecture, and theoretical foundations |
| **[EXPERIMENTAL_RESULTS.md](./EXPERIMENTAL_RESULTS.md)** | Complete experimental results, hardware specifications, and scalability analysis |
| **[NOTICE.md](./NOTICE.md)** | Scientific disclosure statement and repository purpose |

---

## Complete Results Summary

### Full Performance Table

| GPU | Vocabulary | Method | Time (s) | Tokens/sec | Examples/sec | Speedup | Viable |
|:----|:-----------|:------|:---------|:-----------:|:------------:|:-------:|:------:|
| **RTX 4070** | 50K | CE | 65.78 | 38,308 | 608.1 | 1.00× | ✅ |
| **RTX 4070** | 50K | SEMANTIC | 29.58 | **85,206** | **1,352.5** | **2.22×** | ✅ |
| **RTX 4070** | 200K | CE | 255.31 | 9,870 | 156.7 | 1.00× | ✅ |
| **RTX 4070** | 200K | SEMANTIC | 75.19 | **33,513** | **532.0** | **3.40×** | ✅ |
| **RTX 4070** | 1M | CE | ❌ | - | - | - | ❌ |
| **RTX 4070** | 1M | SEMANTIC | 312.22 | **8,071** | **128.1** | **∞** | ✅ |
| **RTX A6000** | 50K | CE | 34.83 | 72,358 | 1,148.5 | 1.00× | ✅ |
| **RTX A6000** | 50K | SEMANTIC | 21.68 | **116,253** | **1,845.3** | **1.61×** | ✅ |
| **RTX A6000** | 200K | CE | 111.56 | 22,589 | 358.5 | 1.00× | ✅ |
| **RTX A6000** | 200K | SEMANTIC | 37.56 | **67,097** | **1,065.0** | **2.97×** | ✅ |
| **RTX A6000** | 1M | CE | 535.09 | 4,710 | 74.8 | 1.00× | ✅ |
| **RTX A6000** | 1M | SEMANTIC | 120.81 | **20,860** | **331.1** | **4.43×** | ✅ |

### Key Findings

1. **Superior Scalability**: Speedup increases with vocabulary size (1.61× → 2.97× → 4.43× on RTX A6000)
2. **Consumer Hardware Viability**: Enables training on 1M-token vocabularies where traditional CE fails
3. **Energy Efficiency**: 4.43× reduction in training time implies proportional energy savings
4. **Faster Convergence**: Lower and more stable loss values compared to CE baseline

---

## Experimental Validation

All experiments were conducted using:
- **Dataset**: 20,000 examples from ShareGPT
- **Model**: GRU-based language model (1 layer, hidden size 256)
- **Semantic Dimension**: 64
- **Sequence Length**: 64
- **Batch Size**: 16
- **Epochs**: 2 complete epochs
- **Hardware**: NVIDIA RTX 4070 Laptop GPU, NVIDIA RTX A6000

Complete experimental details, metrics, and analysis are available in [EXPERIMENTAL_RESULTS.md](./EXPERIMENTAL_RESULTS.md).

---

## Mathematical Formulation

The core innovation is the decoupling of semantic learning from token resolution:

**Semantic Contrastive Loss:**
\[
\ell_{\text{sem}} = -\log \frac{\exp( z_t \cdot e_{y_t} / \tau)}
{\exp( z_t \cdot e_{y_t} / \tau) + \sum_{j=1}^{K} \exp( z_t \cdot e_{n_j} / \tau)}
\]

where:
- $z_t \in \mathbb{R}^{D_{\text{sem}}}$ is the normalized semantic projection
- $e_i \in \mathbb{R}^{D_{\text{sem}}}$ are normalized semantic embeddings
- $K$ is the number of negative samples (constant, independent of vocabulary size $V$)
- $\tau$ is the temperature parameter

**Computational Complexity**: $O(K \cdot D_{\text{sem}})$ instead of $O(V)$

Complete mathematical formulation and theoretical analysis are available in [THEORY.md](./THEORY.md).

---

## Figures

Experimental results include high-resolution figures demonstrating:
- Throughput vs vocabulary size
- Speedup scaling
- Training time comparison
- Cross-GPU comparison

### RTX 4070 Laptop GPU

#### Throughput vs Vocabulary Size

![Throughput vs Vocabulary Size - RTX 4070](./figures/fig_throughput_vs_vocab_rtx_4070.png)

#### Speedup Scaling

![Speedup vs Vocabulary Size - RTX 4070](./figures/fig_speedup_vs_vocab_rtx_4070.png)

#### Training Time Comparison

![Training Time vs Vocabulary Size - RTX 4070](./figures/fig_time_vs_vocab_rtx_4070.png)

### RTX A6000

#### Throughput vs Vocabulary Size

![Throughput vs Vocabulary Size - RTX A6000](./figures/fig_throughput_vs_vocab_rtx_a6000.png)

#### Speedup Scaling

![Speedup vs Vocabulary Size - RTX A6000](./figures/fig_speedup_vs_vocab_rtx_a6000.png)

#### Training Time Comparison

![Training Time vs Vocabulary Size - RTX A6000](./figures/fig_time_vs_vocab_rtx_a6000.png)

### Cross-GPU Comparison

![Cross-GPU Comparison](./figures/fig_comparison_all_gpus.png)

**Note**: All figures are available in both PNG (300 DPI) and PDF formats in the [`figures/`](./figures/) directory. For detailed figure descriptions, see the [Figures section](./EXPERIMENTAL_RESULTS.md#figures) in EXPERIMENTAL_RESULTS.md.

---

## Citation

If you use or reference this work, please cite the authoritative research record:

**Oliveira, A. P. (2025).**  
**OktoSemantic: Scalable Training Methods for Large-Vocabulary Language Models.**  
Open Science Framework (OSF).  
https://doi.org/10.17605/OSF.IO/MRKPU

For software and repository archival purposes, see the Zenodo record:  
https://doi.org/10.5281/zenodo.17931292

### BibTeX — OSF (Primary Record)

```bibtex
@misc{oliveira2025oktosemantic,
  title        = {OktoSemantic: Scalable Training Methods for Large-Vocabulary Language Models},
  author       = {Oliveira, Ademir P. de},
  year         = {2025},
  howpublished = {\url{https://doi.org/10.17605/OSF.IO/MRKPU}},
  note         = {Open Science Framework},
  doi          = {10.17605/OSF.IO/MRKPU}
}
```

### BibTeX — Zenodo (Repository Archive)

```bibtex
@software{oliveira2025oktosemantic_zenodo,
  author  = {Oliveira, Ademir P. de},
  title   = {OktoSemantic Training (SCTA) — Repository Archive},
  year    = {2025},
  doi     = {10.5281/zenodo.17931292},
  url     = {https://doi.org/10.5281/zenodo.17931292}
}
```

---

## Scientific Archives

### Open Science Framework (OSF) — Primary Record

The authoritative scientific record of this work is maintained on the Open Science Framework:

**OSF DOI**: [10.17605/OSF.IO/MRKPU](https://doi.org/10.17605/OSF.IO/MRKPU)

[![OSF DOI](https://img.shields.io/badge/DOI-10.17605%2FOSF.IO%2FMRKPU-blue)](https://doi.org/10.17605/OSF.IO/MRKPU)

### Zenodo Archive — Repository Archive

This repository is also archived on Zenodo for software and repository archival purposes:

**Zenodo DOI**: [10.5281/zenodo.17931292](https://doi.org/10.5281/zenodo.17931292)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17931292.svg)](https://doi.org/10.5281/zenodo.17931292)

---

## Related Work

- **Manuscript prepared for arXiv submission**:  
  *SCTA: Semantic Contrastive Token Approximation*

- **OktoEngine**: Private training engine used for experimental validation

- **OktoSeek AI**: https://oktoseek.com

---

## License

**Proprietary Research License**

This repository contains research documentation and experimental results. The method, mathematical formulations, and experimental data are proprietary to OktoSeek AI.

See [LICENSE](./LICENSE) for complete terms.

---

## Contact

**OktoSeek AI**  
Website: [oktoseek.com](https://www.oktoseek.com)  
GitHub: [github.com/oktoseek](https://github.com/oktoseek)

---

**Last Updated**: December 2025  
**Version**: 1.0
