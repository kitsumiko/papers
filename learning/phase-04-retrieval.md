# Phase 4: Retrieval & Knowledge Systems

[← Back to Learning Path](../learning-path.md) | [← Previous: Attention](phase-03-attention.md)

**Phase Overview**: Language models have limited memory and can hallucinate facts, but what if they could look up information before answering? This phase introduces retrieval-augmented generation (RAG), where models query external knowledge bases to ground their responses in factual sources. You'll learn about dense retrieval systems that find relevant documents, nearest-neighbor approaches that augment model capabilities, and the architectural patterns that make RAG practical. These techniques are essential for building reliable, factual AI systems and are widely used in production applications today.

## 4.1 Retrieval-Augmented Generation (RAG)
**Goal**: Combine retrieval with generation for better knowledge access

1. [REALM: Retrieval-Augmented Language Model Pre-Training](https://arxiv.org/abs/2002.08909) (Guu et al., 2020)
   - *Why*: **First major retrieval-augmented pretraining approach** - shows how to integrate retrieval into pre-training with backpropagation through millions of documents

2. [Generalization through Memorization: Nearest Neighbor Language Models (kNN-LM)](https://arxiv.org/abs/1911.00172) (Khandelwal et al., 2020)
   - *Why*: Memory-augmented decoding foundation; shows that similarity search is easier than next-word prediction

3. [REFRAG: Rethinking RAG based Decoding](https://arxiv.org/pdf/2509.01092) (2025)
   - *Why*: Optimizing RAG decoding strategies

4. [Semantic IDs for Joint Generative Search and Recommendation](https://arxiv.org/abs/2508.10478) (2025)
   - *Why*: Unified approach to search and recommendation

5. [Is Table Retrieval a Solved Problem? Exploring Join-Aware Multi-Table Retrieval](https://arxiv.org/pdf/2404.09889) (2024)
   - *Why*: Structured data retrieval challenges

## 4.2 Federated & Distributed Learning
**Goal**: Train models across distributed data

1. [Federated Learning with Ad-hoc Adapter Insertions: The Case of Soft-Embeddings for Training Classifier-as-Retriever](https://arxiv.org/pdf/2509.16508) (2025)
   - *Why*: Privacy-preserving distributed training

---

**Next**: [Phase 5: AI Reasoning & Agents →](phase-05-reasoning.md)
