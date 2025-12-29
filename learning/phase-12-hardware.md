# Phase 12: Hardware & Systems

[← Back to Learning Path](../learning-path.md) | [← Previous: Phase 11](phase-11-vision.md) | [📖 Glossary](glossary.md)

**Phase Overview**: AI doesn't run on theory alone—hardware and systems design fundamentally shape what's possible. This phase examines how memory bandwidth bottlenecks limit model performance, how specialized hardware accelerators exploit AI workload characteristics, and how [distributed training](glossary.md#distributed-training) systems coordinate thousands of [GPUs](glossary.md#gpu-graphics-processing-unit) to train massive models. Understanding the hardware-software co-design is crucial for making informed decisions about model architecture, for optimizing deployment costs, and for anticipating future directions as specialized AI chips become more prevalent.

## 12.1 Hardware Considerations
**Goal**: Understand hardware-algorithm co-design

1. [Making Deep Learning Go Brrrr From First Principles](https://horace.io/brrr_intro.html) (He, 2022)
   - *Why*: **Essential practitioner guide** - explains GPU memory hierarchy, compute vs memory bound operations, and why FlashAttention works; practical understanding for optimizing ML workloads

2. [Efficiently Scaling Transformer Inference](https://arxiv.org/abs/2211.05102) (Pope et al., 2022)
   - *Why*: **Google's inference optimization guide** - analyzes memory bandwidth vs compute tradeoffs for transformer inference; introduces key concepts like arithmetic intensity and roofline analysis

3. 🔒 [High-dimensional on-chip dataflow sensing and routing using spatial photonic networks](https://www.nature.com/articles/s41566-023-01272-3.pdf) (2023)
   - *Why*: Photonic computing for AI; next-generation hardware
   - *Note*: Paywalled - Nature Photonics journal

4. [A Log-Domain Implementation of the Diffusion Network in Very Large Scale Integration](https://papers.nips.cc/paper_files/paper/2010/file/7bcdf75ad237b8e02e301f4091fb6bc8-Paper.pdf) (2010)

---

**Next**: [Phase 13: Policy & Governance →](phase-13-policy.md)
