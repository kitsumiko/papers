# AI/ML Research Papers Collection

> A curated, pedagogically-organized collection of essential research papers spanning the landscape of modern artificial intelligence and machine learning.

[![Papers](https://img.shields.io/badge/papers-100+-blue.svg)](by-date.md)
[![Learning Path](https://img.shields.io/badge/learning-13_phases-green.svg)](learning-path.md)
[![Last Updated](https://img.shields.io/badge/updated-October_2025-orange.svg)](by-date.md)

---

## 🎯 What's Inside

This repository contains **100+ carefully selected research papers** organized in two complementary ways:

1. **📚 [Structured Learning Path](learning-path.md)** - A 13-phase curriculum designed to take you from foundations to cutting-edge research
2. **📅 [Chronological Timeline](by-date.md)** - Papers organized by publication date (1997-2025)

### Why This Collection?

- **Curated for Learning**: Papers selected for their pedagogical value and impact
- **Pedagogically Organized**: Follow a structured path from basics to advanced topics
- **Modern & Comprehensive**: Covers transformers, LLMs, agents, vision, security, and more
- **Open Access Focused**: Most papers freely available; paywalled papers marked 🔒
- **Actively Maintained**: Updated with latest research from 2025

---

## 🚀 Quick Start

### For Beginners
Start with the **[Learning Path](learning-path.md)** and follow **Phase 1: Foundations**. Read papers in sequence, focusing on the "Why" explanations.

### For Practitioners
Jump to relevant phases:
- **LLMs & Training**: [Phase 2](learning/phase-02-llms.md)
- **Efficient Models**: [Phase 3](learning/phase-03-attention.md)
- **Production AI**: [Phase 4](learning/phase-04-retrieval.md) (RAG), [Phase 8](learning/phase-08-security.md) (Security)

### For Researchers
Browse the **[Chronological View](by-date.md)** to see latest 2025 research, or deep-dive into:
- [Phase 6: Alternative Architectures](learning/phase-06-architectures.md)
- [Phase 7: Interpretability](learning/phase-07-interpretability.md)
- [Phase 9: Advanced Topics](learning/phase-09-advanced.md)

---

## 📖 Learning Path Overview

The learning path is organized into **13 progressive phases**, each building on previous knowledge:

| Phase | Topic | Papers | Focus |
|-------|-------|--------|-------|
| **[1](learning/phase-01-foundations.md)** | 🏗️ **Foundations** | 15 | Deep learning basics, embeddings, CNNs, RNNs, GANs, tokenization |
| **[2](learning/phase-02-llms.md)** | 🤖 **Large Language Models** | 10 | Transformers, BERT, GPT, training at scale |
| **[3](learning/phase-03-attention.md)** | ⚡ **Attention Innovations** | 7 | FlashAttention, efficient attention, long context |
| **[4](learning/phase-04-retrieval.md)** | 🔍 **Retrieval & RAG** | 6 | RAG systems, kNN-LM, semantic search |
| **[5](learning/phase-05-reasoning.md)** | 🧠 **Reasoning & Agents** | 12 | RLHF, chain-of-thought, agentic systems |
| **[6](learning/phase-06-architectures.md)** | 🏛️ **Alternative Architectures** | 7 | RWKV, Mamba, state-space models, theory |
| **[7](learning/phase-07-interpretability.md)** | 🔬 **Interpretability** | 10 | LIME, integrated gradients, evaluation methods |
| **[8](learning/phase-08-security.md)** | 🛡️ **Security & Robustness** | 6 | Alignment, jailbreaking, adversarial ML |
| **[9](learning/phase-09-advanced.md)** | 🎯 **Advanced Applications** | 7 | Multimodal, scientific AI, test-time compute |
| **[10](learning/phase-10-probabilistic.md)** | 🎲 **Probabilistic Models** | 4 | Diffusion, probabilistic programming |
| **[11](learning/phase-11-vision.md)** | 👁️ **Vision & Multimodal** | 8 | ViT, CLIP, SAM, vision-language models |
| **[12](learning/phase-12-hardware.md)** | ⚙️ **Hardware & Systems** | 2 | Photonic computing, VLSI implementations |
| **[13](learning/phase-13-policy.md)** | 📜 **Policy & Governance** | 1 | AI safety, policy, societal impact |

**Total**: 95+ core papers across 13 phases

---

## 🗂️ Repository Structure

```
papers/
├── README.md                    # This file - repository overview
├── learning-path.md             # Main learning path navigation
├── by-date.md                   # Chronological paper listing
└── learning/                    # Phase-by-phase curriculum
    ├── glossary.md              # Comprehensive glossary of terms & concepts
    ├── phase-01-foundations.md
    ├── phase-02-llms.md
    ├── phase-03-attention.md
    ├── phase-04-retrieval.md
    ├── phase-05-reasoning.md
    ├── phase-06-architectures.md
    ├── phase-07-interpretability.md
    ├── phase-08-security.md
    ├── phase-09-advanced.md
    ├── phase-10-probabilistic.md
    ├── phase-11-vision.md
    ├── phase-12-hardware.md
    └── phase-13-policy.md
```

---

## 📊 Coverage by Topic

This collection spans the full spectrum of modern AI/ML research, organized by theme:

### 🏗️ Deep Learning Foundations
- Classic architectures: LeNet, AlexNet, GoogLeNet, ResNet
- Training techniques: Xavier initialization, Batch Normalization
- **Word Embeddings**: Word2Vec, GloVe, encoder-decoder architectures
- Sequence models: LSTM, time-series foundations (MOMENT)
- Generative models: GANs, diffusion models
- **Tokenization**: BPE, SentencePiece, tokenization bias and best practices

### 🤖 Large Language Models & Transformers
- **Foundational**: [Attention Is All You Need](https://arxiv.org/abs/1706.03762), [BERT](https://arxiv.org/abs/1810.04805), [GPT-3](https://arxiv.org/abs/2005.14165)
- **Modern LLMs**: Llama 3, OpenELM, EuroLLM
- **Training at Scale**: Megatron-LM, ZeRO, model parallelism, second-order optimization
- **Efficiency**: MatMul-free models, 1-bit LLMs, vocabulary optimization

### ⚡ Efficient Attention & Long Context
- FlashAttention: IO-aware attention algorithms
- Alternative mechanisms: RetNet, attention sinks, Infini-attention
- Hardware-aligned: Native sparse attention
- Context compression: DeepSeek-OCR, hierarchical speculative decoding

### 🔍 Retrieval & Knowledge Systems
- RAG fundamentals: REALM, kNN-LM, REFRAG
- Advanced retrieval: Semantic IDs, table retrieval, multi-table systems
- Federated approaches: Classifier-as-retriever with adapters

### � Reasoning, Alignment & Agents
- **Alignment**: RLHF, PPO, InstructGPT, Constitutional AI
- **Reasoning**: Hierarchical reasoning, recursive reasoning, reinforcement pre-training
- **Agents**: ReAct, Reflexion, Gato, DynaSaur
- **Multi-agent**: MAS-ZERO (zero-supervision design)
- **Efficiency**: Small language models for agents

### 🏛️ Novel Architectures & Theory
- Alternatives to transformers: RWKV, Mamba, state-space models
- Novel approaches: Kolmogorov-Arnold Networks (KAN), U-Nets as belief propagation
- Theoretical foundations: Neural Tangent Kernel, manifold hypothesis
- Training acceleration: Neuron interaction networks

### 🔬 Interpretability & Analysis
- Attribution methods: LIME, Integrated Gradients, SmoothGrad
- Frameworks: Rigorous interpretability science, interpretation surveys
- Evaluation: TruthfulQA, MLE-bench, model calibration
- Critical analysis: "The Illusion of Thinking", LLM evaluation challenges

### 🛡️ Security, Safety & Robustness
- **Alignment & Safety**: InstructGPT, Constitutional AI
- **Security**: Prompt sanitization, MCP security, agentic AI self-defense
- **Threats**: Cyber threat intelligence, reliability concerns

### 👁️ Computer Vision & Multimodal AI
- Vision transformers: ViT, CLIP, SAM
- Interpretability: DeconvNet, saliency maps, visualization techniques
- Segmentation: All-CNN, adversarial segmentation
- Multimodal understanding: Vision-language models

### 🎯 Advanced Applications
- **Scientific AI**: AI scientist, scientific software generation, architecture discovery
- **Specialized**: Audio generation (Stable Audio), molecular dynamics
- **Tabular**: TabPFN for small classification problems
- **Test-time compute**: o3 system, consciousness in AI

### 🎲 Probabilistic & Generative Models
- Probabilistic programming: Data analysis, scene perception (Picture)
- MCMC methods: Hamiltonian dynamics
- Bayesian approaches: Intuitive dynamics modeling

### ⚙️ Hardware & Systems
- Photonic computing for AI
- VLSI implementations: Log-domain diffusion networks
- Hardware-algorithm co-design

### 📜 Policy & Governance
- AI policy frameworks: Dual-use models, open weights
- Governance considerations for foundation models

---

## 🎓 How to Use This Repository

### Reading Strategies

**🌱 The Beginner Path** (3-6 months)
1. Start with [Phase 1: Foundations](learning/phase-01-foundations.md)
2. Read key papers: Attention Is All You Need → BERT → GPT-3
3. Focus on "Why" explanations before diving deep
4. Take notes on connections between papers

**⚡ The Practitioner Sprint** (1-2 months)
1. Read Phase 1 summaries for context
2. Deep-dive: [Phase 2](learning/phase-02-llms.md) + [Phase 3](learning/phase-03-attention.md) + [Phase 4](learning/phase-04-retrieval.md)
3. Skim related work sections to understand landscape
4. Implement key techniques from papers

**🔬 The Researcher Deep-Dive** (Ongoing)
1. Use [chronological view](by-date.md) for latest research
2. Focus on specific phases relevant to your research
3. Read citations and follow paper connections
4. Compare approaches across different papers

**🛠️ The Engineer Focus** (2-4 weeks)
1. Priority: [Phase 3](learning/phase-03-attention.md) (Efficiency), [Phase 8](learning/phase-08-security.md) (Security), [Phase 12](learning/phase-12-hardware.md) (Hardware)
2. Focus on implementation details and benchmarks
3. Note production considerations and trade-offs

**� The Security Specialist** (1-2 weeks)
1. Core: [Phase 8](learning/phase-08-security.md)
2. Context: [Phase 2](learning/phase-02-llms.md) (LLM basics), [Phase 5](learning/phase-05-reasoning.md) (Alignment)
3. Focus on threat models and defense mechanisms

### Paper Reading Tips

1. **Start with abstracts** - Understand the core contribution
2. **Read "Why" annotations** - Context before content
3. **Check the [📖 Glossary](learning/glossary.md)** - Look up unfamiliar terms
4. **Follow the narrative** - Papers build on each other
5. **Take notes** - Document connections and insights
6. **Implement key ideas** - Hands-on learning reinforces concepts

---

## 🆕 What's New

### Recent Additions (October 2025)
- ✨ **Structured Learning Path**: 13-phase curriculum with navigation
- 📝 **Phase Overviews**: Contextual introductions for each learning phase
- 🔓 **Open Access Focus**: Updated links to freely accessible versions
- 🔒 **Paywall Markers**: Clear indicators for paywalled papers
- 🧭 **Better Navigation**: Breadcrumb links between all phases

### Latest Papers (2025)
- [Small Language Models are the Future of Agentic AI](https://arxiv.org/abs/2506.02153) (June 2025)
- [The Illusion of Thinking](https://ml-site.cdn-apple.com/papers/the-illusion-of-thinking.pdf) (February 2025)
- [Native Sparse Attention](https://arxiv.org/abs/2502.11089) (February 2025)

See [by-date.md](by-date.md) for complete chronological listing.

---

## 🤝 Contributing

Have a paper that should be included? Found a broken link? Want to improve explanations?

**To suggest a paper:**
1. Check if it's already in [by-date.md](by-date.md)
2. Consider: Is it influential? Does it fit the learning path?
3. Open an issue with: Title, arXiv/URL, why it's important, suggested phase

**To fix issues:**
1. Broken links: Open an issue or PR with updated URL
2. Typos/improvements: PRs welcome!
3. Better explanations: Suggest edits to "Why" annotations

---

## 📚 Additional Resources

### Related Collections
- [Papers We Love](https://github.com/papers-we-love/papers-we-love) - Classic CS papers
- [Awesome Deep Learning Papers](https://github.com/terryum/awesome-deep-learning-papers) - DL fundamentals
- [ML Papers of The Week](https://github.com/dair-ai/ML-Papers-of-the-Week) - Weekly updates

### Tools & Platforms
- [arXiv](https://arxiv.org/) - Preprint repository
- [Papers With Code](https://paperswithcode.com/) - Papers + implementations
- [Semantic Scholar](https://www.semanticscholar.org/) - AI-powered paper search
- [Connected Papers](https://www.connectedpapers.com/) - Visual paper exploration

### Conference Deadlines
- 🎓 **[AI Deadlines](https://aideadlin.es/?sub=ML,CV,CG,NLP,RO,SP,DM,AP,KR,HCI)** - Track ML/AI conference submissions

---

## 📊 Repository Statistics

- **Total Papers**: 100+
- **Date Range**: 1997-2025
- **Coverage**: 13 major AI/ML topics
- **Open Access**: ~95% freely available
- **Structure**: 2 navigation modes (pedagogical + chronological)
- **Last Updated**: October 2025

---

## 📄 License

This repository contains links to research papers. All papers remain under their original licenses and copyrights held by authors and publishers.

The curation, organization, and annotations in this repository are provided for educational purposes.

---

## 🙏 Acknowledgments

Papers compiled from:
- Major AI/ML conferences (NeurIPS, ICML, ICLR, CVPR, ACL, etc.)
- Leading research institutions and labs
- arXiv preprint server
- Open access initiatives

Special thanks to the researchers, authors, and institutions making their work freely available.

---

## 📬 Contact & Feedback

Found this helpful? Have suggestions? Want to discuss a paper?

- **Issues**: [Open an issue](../../issues) for bugs, suggestions, or paper recommendations
- **Discussions**: [Start a discussion](../../discussions) for paper analysis or learning questions

---

**Happy Reading! 📖🚀**

*Building knowledge, one paper at a time.*