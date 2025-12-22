# Phase 2: Large Language Models - Core Concepts

[← Back to Learning Path](../learning-path.md) | [← Previous: Foundations](phase-01-foundations.md) | [📖 Glossary](glossary.md)

**Phase Overview**: This phase traces the explosive evolution of language models from [BERT](glossary.md#bert-bidirectional-encoder-representations-from-transformers)'s bidirectional pretraining breakthrough to [GPT](glossary.md#gpt-generative-pre-trained-transformer)-3's massive scale demonstration. You'll learn how the field discovered that [pre-training](glossary.md#pre-training) on vast amounts of text data creates models with remarkable [few-shot learning](glossary.md#few-shot-learning) abilities, and how different pretraining objectives (masked language modeling vs. autoregressive) lead to different capabilities. This progression from BERT to GPT-3 to instruction-tuned models forms the backbone of modern NLP and sets the stage for understanding today's ChatGPT-style systems.

## 2.1 LLM Foundations
**Goal**: Understand transformer architecture and pre-training

1. [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) (Bahdanau et al., 2014)
   - *Why*: **Introduces attention mechanism** - the foundational paper that introduced attention for sequence-to-sequence models; paved the way for transformers

2. [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017)
   - *Why*: **THE foundational transformer paper** - introduces self-attention, multi-head attention, and the transformer architecture that powers all modern LLMs

3. [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805) (Devlin et al., 2019)
   - *Why*: Introduces masked language modeling and bidirectional pre-training; revolutionized NLP fine-tuning

3. [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361) (Kaplan et al., 2020)
   - *Why*: **Empirical scaling laws** - establishes predictable relationships between model size, dataset size, compute, and performance; foundational for understanding how to scale LLMs effectively

4. [Language Models are Few-Shot Learners (GPT-3)](https://arxiv.org/abs/2005.14165) (Brown et al., 2020)
   - *Why*: Demonstrates emergent abilities at scale; introduces in-context learning and few-shot prompting

4. [ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators](https://arxiv.org/abs/2003.10555) (2020)
   - *Why*: Efficient alternative to masked language modeling; achieves BERT-level performance with less compute

5. [The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783) (2024)
   - *Why*: State-of-the-art open-source LLMs; demonstrates continued scaling benefits and instruction-tuning techniques

6. [OpenELM: An Efficient Language Model Family with Open-source Training and Inference Framework](https://arxiv.org/pdf/2404.14619) (2024)
   - *Why*: Efficient, open-source LLM architecture

7. [EuroLLM: Multilingual Language Models for Europe](https://arxiv.org/pdf/2409.11741) (2024)
   - *Why*: Multilingual capabilities and cross-lingual transfer

## 2.2 Training at Scale
**Goal**: Learn how to train massive models efficiently

1. [GPipe: Easy Scaling with Micro-Batch Pipeline Parallelism](https://arxiv.org/abs/1811.06965) (Huang et al., 2019)
   - *Why*: **Pipeline parallelism foundation** - enables training of very large models by splitting across devices with micro-batching; essential for scaling beyond single-device memory limits

2. [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053) (2019)
   - *Why*: Foundation of distributed LLM training

2. [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054) (2019)
   - *Why*: Memory-efficient training techniques

3. [Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM](https://arxiv.org/abs/2104.04473) (2021)
   - *Why*: Combining techniques for practical large-scale training

4. [The Potential of Second-Order Optimization for LLMs: A Study with Full Gauss-Newton](https://arxiv.org/pdf/2510.09378) (2025)
   - *Why*: Advanced optimization techniques for faster convergence

## 2.3 Memory & Efficiency Optimizations
**Goal**: Make models faster and more memory-efficient

1. [Cut Your Losses in Large-Vocabulary Language Models](https://arxiv.org/abs/2411.09009) (2024)
   - *Why*: Reducing memory footprint during training

2. [Scalable MatMul-free Language Modeling](https://arxiv.org/pdf/2406.02528) (2024)
   - *Why*: Eliminating expensive matrix multiplications

3. [The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits](https://arxiv.org/pdf/2402.17764) (2024)
   - *Why*: Extreme quantization techniques

---

**Next**: [Phase 3: Attention Mechanisms & Context →](phase-03-attention.md)
