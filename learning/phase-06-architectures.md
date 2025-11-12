# Phase 6: Novel Architectures & Theory

[← Back to Learning Path](../learning-path.md) | [← Previous: Reasoning & Agents](phase-05-reasoning.md) | [📖 Glossary](glossary.md)

**Phase Overview**: [Transformers](glossary.md#transformer) dominate modern AI, but are they the final answer? This phase explores alternative architectures that challenge the transformer's supremacy: [state-space models](glossary.md#state-space-model) like [Mamba](glossary.md#mamba) that achieve linear-time inference, [retention networks](glossary.md#retnet-retentive-network) that blend RNN and transformer properties, [RWKV](glossary.md#rwkv)'s parallelizable RNN approach, and hybrid architectures that combine different mechanisms. Each offers different trade-offs between performance, efficiency, and scaling properties. Understanding these alternatives gives you insight into the fundamental principles that make architectures work—and hints at what might come next.

## 6.1 Alternative Architectures
**Goal**: Explore beyond standard transformers

1. [RWKV: Reinventing RNNs for the Transformer Era](https://arxiv.org/abs/2305.13048) (Peng et al., 2023)
   - *Why*: **Foundational non-transformer alternative** - combines efficient parallelizable training with O(1) inference; scaled to 14B parameters

2. [Nested Learning: The Illusion of Deep Learning Architectures](https://abehrouz.github.io/files/NL.pdf) (Behrouz et al., 2025) - NeurIPS 2025
   - *Why*: **New paradigm unifying architecture and optimization** - views models as nested optimization problems at multiple time scales; introduces Hope, a self-modifying architecture with continuum memory systems that achieves superior continual learning and mitigates catastrophic forgetting

3. [Kolmogorov–Arnold Networks (KAN)](https://arxiv.org/pdf/2404.19756) (2024)
   - *Why*: Novel learnable activation functions replacing fixed activations

4. [U-Nets as Belief Propagation: Efficient Classification, Denoising, and Diffusion in Generative Hierarchical Models](https://arxiv.org/pdf/2404.18444) (2024)
   - *Why*: Connecting neural networks to probabilistic inference

5. [Mamba or RWKV: Exploring High-Quality and High-Efficiency Segment Anything Model](https://arxiv.org/pdf/2409.15254) (2024)
   - *Why*: Comparison of alternative sequence modeling approaches

## 6.2 Theoretical Foundations
**Goal**: Understand the mathematical foundations

1. [Neural Tangent Kernel: Convergence and Generalization in Neural Networks](https://arxiv.org/abs/1806.07572) (Jacot et al., 2018)
   - *Why*: **Bridges theory and deep learning** - shows ANNs are equivalent to kernel methods in infinite-width limit; explains generalization

2. [Token embeddings violate the manifold hypothesis](https://arxiv.org/abs/2504.01002) (2025)
   - *Why*: Understanding embedding space geometry

3. [How much do language models memorize?](https://arxiv.org/pdf/2505.24832) (2025)
   - *Why*: Memorization vs. generalization theory

4. [Accelerating Training With Neuron Interaction And Nowcasting Networks](https://arxiv.org/pdf/2409.04434) (2024)
   - *Why*: Novel training acceleration theory

---

**Next**: [Phase 7: Model Interpretability & Evaluation →](phase-07-interpretability.md)
