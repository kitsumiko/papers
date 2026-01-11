# Phase 4: Retrieval & Knowledge Systems

[← Back to Learning Path](../learning-path.md) | [← Previous: Attention](phase-03-attention.md) | [📖 Glossary](glossary.md)

**Phase Overview**: Language models have limited memory and can [hallucinate](glossary.md#hallucination) facts, but what if they could look up information before answering? This phase introduces [retrieval-augmented generation (RAG)](glossary.md#rag-retrieval-augmented-generation), where models query external knowledge bases to ground their responses in factual sources. You'll learn about dense retrieval systems that find relevant documents, nearest-neighbor approaches that augment model capabilities, and the architectural patterns that make RAG practical. These techniques are essential for building reliable, factual AI systems and are widely used in production applications today.

## 4.1 Retrieval-Augmented Generation (RAG)
**Goal**: Combine retrieval with generation for better knowledge access

1. [Dense Passage Retrieval for Open-Domain Question Answering](https://arxiv.org/abs/2004.04906) (Karpukhin et al., 2020)
   - *Why*: **Foundational dense retrieval paper** - introduces [dual-encoder architecture](glossary.md#dual-encoder) for semantic search; enables efficient retrieval without explicit term matching

2. [REALM: Retrieval-Augmented Language Model Pre-Training](https://arxiv.org/abs/2002.08909) (Guu et al., 2020)
   - *Why*: **First major retrieval-augmented pretraining approach** - shows how to integrate retrieval into pre-training with backpropagation through millions of documents

3. [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) (Lewis et al., 2020)
   - *Why*: **THE RAG paper** - defines the retrieval-augmented generation paradigm; combines pre-trained retriever with seq2seq generator; foundational architecture for modern RAG systems

4. [Generalization through Memorization: Nearest Neighbor Language Models (kNN-LM)](https://arxiv.org/abs/1911.00172) (Khandelwal et al., 2020)
   - *Why*: Memory-augmented decoding foundation; shows that similarity search is easier than next-word prediction

5. [A Survey of Context Engineering for Large Language Models](https://arxiv.org/abs/2507.13334) (Mei et al., 2025)
   - *Why*: **Comprehensive survey (1400+ papers)** - defines context engineering as a discipline; covers context retrieval, processing, management, RAG architectures, memory systems, and multi-agent integration

6. [REFRAG: Rethinking RAG based Decoding](https://arxiv.org/pdf/2509.01092) (2025)
   - *Why*: Optimizing RAG decoding strategies

7. [Semantic IDs for Joint Generative Search and Recommendation](https://arxiv.org/abs/2508.10478) (2025)
   - *Why*: Unified approach to search and recommendation

8. [Is Table Retrieval a Solved Problem? Exploring Join-Aware Multi-Table Retrieval](https://arxiv.org/pdf/2404.09889) (2024)
   - *Why*: Structured data retrieval challenges

9. [Lost in the Middle: How Language Models Use Long Contexts](https://arxiv.org/abs/2307.03172) (Liu et al., 2023)
   - *Why*: **Critical analysis of long-context usage** - reveals U-shaped performance curve where models struggle with information in the middle of long contexts; important for RAG system design

## 4.2 Federated & Distributed Learning
**Goal**: Train models across distributed data

1. [Federated Learning with Ad-hoc Adapter Insertions: The Case of Soft-Embeddings for Training Classifier-as-Retriever](https://arxiv.org/pdf/2509.16508) (2025)
   - *Why*: Privacy-preserving distributed training

---

**Next**: [Phase 5: AI Reasoning & Agents →](phase-05-reasoning.md)
