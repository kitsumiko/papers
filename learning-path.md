# Learning Path: AI/ML Research Curriculum

A structured curriculum organizing papers from this collection, designed to build knowledge progressively from foundational concepts to cutting-edge research.

---

## Phase 1: Foundations (Start Here)

### 1.1 Deep Learning Basics
**Goal**: Understand the fundamental building blocks of modern deep learning

1. [ImageNet Classification with Deep Convolutional Neural Networks (AlexNet)](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf) (2012)
   - *Why*: The paper that sparked the deep learning revolution
   
2. [Gradient-Based Learning Applied to Document Recognition (LeNet)](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf) (1997)
   - *Why*: Historical foundation of CNNs

3. [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) (2015)
   - *Why*: ResNets and skip connections - essential architecture innovation

4. [Going Deeper with Convolutions (GoogLeNet)](https://arxiv.org/abs/1409.4842) (2014)
   - *Why*: Inception modules and efficient architecture design

### 1.2 Sequence Modeling & Recurrent Networks
**Goal**: Learn time-series and sequential data processing

1. [Long Short-Term Memory (LSTM)](https://www.researchgate.net/publication/13853244_Long_Short-term_Memory) (1997)
   - *Why*: Foundation for sequence modeling

2. [MOMENT: A Family of Open Time-series Foundation Models](https://arxiv.org/pdf/2402.03885) (2024)
   - *Why*: Modern approach to time-series with foundation models

### 1.3 Generative Models
**Goal**: Understand how to generate new data

1. [Generative Adversarial Networks](https://arxiv.org/abs/1406.2661) (2014)
   - *Why*: Revolutionary generative modeling approach

2. [Dualscale Diffusion: Adaptive Feature Balancing for Low-Dimensional Generative Models](https://sakana.ai/assets/ai-scientist/adaptive_dual_scale_denoising.pdf) (2024)
   - *Why*: Modern diffusion models for generative tasks

---

## Phase 2: Large Language Models - Core Concepts

### 2.1 LLM Foundations
**Goal**: Understand transformer architecture and pre-training

1. [ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators](https://arxiv.org/abs/2003.10555) (2020)
   - *Why*: Efficient pre-training alternative to masked language modeling

2. [The Llama 3 Herd of Models](https://ai.meta.com/research/publications/the-llama-3-herd-of-models/) (2024)
   - *Why*: State-of-the-art open LLMs and their design principles

3. [OpenELM: An Efficient Language Model Family with Open-source Training and Inference Framework](https://arxiv.org/pdf/2404.14619) (2024)
   - *Why*: Efficient, open-source LLM architecture

4. [EuroLLM: Multilingual Language Models for Europe](https://arxiv.org/pdf/2409.11741) (2024)
   - *Why*: Multilingual capabilities and cross-lingual transfer

### 2.2 Training at Scale
**Goal**: Learn how to train massive models efficiently

1. [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053) (2019)
   - *Why*: Foundation of distributed LLM training

2. [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054) (2019)
   - *Why*: Memory-efficient training techniques

3. [Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM](https://arxiv.org/abs/2104.04473) (2021)
   - *Why*: Combining techniques for practical large-scale training

4. [The Potential of Second-Order Optimization for LLMs: A Study with Full Gauss-Newton](https://arxiv.org/pdf/2510.09378) (2025)
   - *Why*: Advanced optimization techniques for faster convergence

### 2.3 Memory & Efficiency Optimizations
**Goal**: Make models faster and more memory-efficient

1. [Cut Your Losses in Large-Vocabulary Language Models](https://arxiv.org/abs/2411.09009) (2024)
   - *Why*: Reducing memory footprint during training

2. [Scalable MatMul-free Language Modeling](https://arxiv.org/pdf/2406.02528) (2024)
   - *Why*: Eliminating expensive matrix multiplications

3. [The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits](https://arxiv.org/pdf/2402.17764) (2024)
   - *Why*: Extreme quantization techniques

---

## Phase 3: Attention Mechanisms & Context

### 3.1 Efficient Attention
**Goal**: Understand and optimize the core attention mechanism

1. [Efficient streaming language models with attention sinks](https://arxiv.org/pdf/2309.17453.pdf) (2024)
   - *Why*: Handling streaming/infinite sequences

2. [Leave No Context Behind: Efficient Infinite Context Transformers with Infini-attention](https://arxiv.org/pdf/2404.07143v1.pdf) (2024)
   - *Why*: Infinite context windows

3. [Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention](https://arxiv.org/abs/2502.11089) (2025)
   - *Why*: Hardware-efficient sparse attention

### 3.2 Long Context & Compression
**Goal**: Handle longer sequences efficiently

1. [TriForce: Lossless Acceleration of Long Sequence Generation with Hierarchical Speculative Decoding](https://arxiv.org/pdf/2404.11912v1.pdf) (2024)
   - *Why*: Speeding up long sequence generation

2. [DeepSeek-OCR: Contexts Optical Compression](https://arxiv.org/pdf/2510.18234) (2025)
   - *Why*: Novel compression via 2D optical mapping

3. [Agentic Context Engineering: Evolving Contexts for Self-Improving Language Models](https://arxiv.org/abs/2510.04618) (2025)
   - *Why*: Dynamic context optimization

---

## Phase 4: Retrieval & Knowledge Systems

### 4.1 Retrieval-Augmented Generation (RAG)
**Goal**: Combine retrieval with generation for better knowledge access

1. [REFRAG: Rethinking RAG based Decoding](https://arxiv.org/pdf/2509.01092) (2025)
   - *Why*: Optimizing RAG decoding strategies

2. [Semantic IDs for Joint Generative Search and Recommendation](https://arxiv.org/abs/2508.10478) (2025)
   - *Why*: Unified approach to search and recommendation

3. [Is Table Retrieval a Solved Problem? Exploring Join-Aware Multi-Table Retrieval](https://arxiv.org/pdf/2404.09889) (2024)
   - *Why*: Structured data retrieval challenges

### 4.2 Federated & Distributed Learning
**Goal**: Train models across distributed data

1. [Federated Learning with Ad-hoc Adapter Insertions: The Case of Soft-Embeddings for Training Classifier-as-Retriever](https://arxiv.org/pdf/2509.16508) (2025)
   - *Why*: Privacy-preserving distributed training

---

## Phase 5: AI Reasoning & Agents

### 5.1 Reasoning Architectures
**Goal**: Build models that can reason and solve complex problems

1. [Hierarchical Reasoning Model](https://arxiv.org/pdf/2506.21734) (2025)
   - *Why*: Brain-inspired hierarchical reasoning

2. [Less is More: Recursive Reasoning with Tiny Networks](https://arxiv.org/pdf/2510.04871) (2025)
   - *Why*: Efficient reasoning with small models

3. [Reinforcement Pre-Training](https://arxiv.org/abs/2506.08007) (2025)
   - *Why*: Using RL during pre-training phase

### 5.2 Agentic Systems
**Goal**: Create autonomous AI agents

1. [A Generalist Agent](https://arxiv.org/pdf/2205.06175) (2022)
   - *Why*: Multi-task agent foundations (Gato)

2. [Executable Code Actions Elicit Better LLM Agents](https://arxiv.org/pdf/2402.01030) (2024)
   - *Why*: Code execution for agents

3. [DynaSaur: Large Language Agents Beyond Predefined Actions](https://arxiv.org/abs/2411.01747) (2024)
   - *Why*: Dynamic action generation

4. [MAS-ZERO: Designing Multi-Agent Systems with Zero Supervision](https://arxiv.org/pdf/2505.14996) (2025)
   - *Why*: Automated multi-agent system design

---

## Phase 6: Novel Architectures & Theory

### 6.1 Alternative Architectures
**Goal**: Explore beyond standard transformers

1. [Kolmogorov–Arnold Networks (KAN)](https://arxiv.org/pdf/2404.19756) (2024)
   - *Why*: Novel learnable activation functions

2. [U-Nets as Belief Propagation: Efficient Classification, Denoising, and Diffusion in Generative Hierarchical Models](https://arxiv.org/pdf/2404.18444) (2024)
   - *Why*: Connecting neural networks to probabilistic inference

3. [Mamba or RWKV: Exploring High-Quality and High-Efficiency Segment Anything Model](https://arxiv.org/pdf/2409.15254) (2024)
   - *Why*: Alternative sequence modeling approaches

### 6.2 Theoretical Foundations
**Goal**: Understand the mathematical foundations

1. [Token embeddings violate the manifold hypothesis](https://arxiv.org/abs/2504.01002) (2025)
   - *Why*: Understanding embedding space geometry

2. [How much do language models memorize?](https://arxiv.org/pdf/2505.24832) (2025)
   - *Why*: Memorization vs. generalization theory

3. [Accelerating Training With Neuron Interaction And Nowcasting Networks](https://arxiv.org/pdf/2409.04434) (2024)
   - *Why*: Novel training acceleration theory

---

## Phase 7: Model Interpretability & Evaluation

### 7.1 Understanding Model Behavior
**Goal**: Interpret what models learn and how they work

1. [Towards A Rigorous Science of Interpretable Machine Learning](https://arxiv.org/abs/1702.08608) (2017)
   - *Why*: Framework for interpretability research

2. [Axiomatic Attribution for Deep Networks (Integrated Gradients)](https://arxiv.org/abs/1703.01365) (2017)
   - *Why*: Principled attribution methods

3. [LIME: "Why Should I Trust You?" Explaining the Predictions of Any Classifier](https://arxiv.org/abs/1602.04938) (2016)
   - *Why*: Model-agnostic explanations

4. [Methods for Interpreting and Understanding Deep Neural Networks](https://arxiv.org/abs/1706.07979) (2017)
   - *Why*: Comprehensive interpretability overview

5. [SmoothGrad: Removing Noise by Adding Noise](https://arxiv.org/abs/1706.03825) (2017)
   - *Why*: Improving gradient-based visualizations

### 7.2 Model Evaluation & Robustness
**Goal**: Properly evaluate and benchmark models

1. [Forget What You Know about LLMs Evaluations -- LLMs are Like a Chameleon](https://arxiv.org/pdf/2502.07445) (2025)
   - *Why*: Critical analysis of evaluation methodologies

2. [MLE-bench: Evaluating Machine Learning Agents on Machine Learning Engineering](https://arxiv.org/abs/2410.07095) (2024)
   - *Why*: Benchmarking ML engineering capabilities

3. [On Calibration of Modern Neural Networks](https://arxiv.org/abs/1706.04599) (2017)
   - *Why*: Understanding prediction confidence

4. [The Illusion of Thinking](https://ml-site.cdn-apple.com/papers/the-illusion-of-thinking.pdf) (2025)
   - *Why*: Critical examination of reasoning capabilities

---

## Phase 8: Security, Privacy & Safety

### 8.1 Security Frameworks
**Goal**: Build secure AI systems

1. [Pr$εε$mpt: Sanitizing Sensitive Prompts for LLMs](https://arxiv.org/abs/2504.05147) (2025)
   - *Why*: Protecting sensitive information in prompts

2. [Enterprise-Grade Security for the Model Context Protocol (MCP): Frameworks and Mitigation Strategies](https://arxiv.org/pdf/2504.08623) (2025)
   - *Why*: Security frameworks for AI applications

3. [A2AS: Agentic AI Runtime Security and Self-Defense](https://arxiv.org/pdf/2510.13825) (2025)
   - *Why*: Runtime security for AI agents

### 8.2 Adversarial Robustness
**Goal**: Understand and defend against adversarial attacks

1. [Large Language Models are Unreliable for Cyber Threat Intelligence](https://arxiv.org/abs/2503.23175) (2025)
   - *Why*: Understanding LLM limitations in security contexts

---

## Phase 9: Advanced Topics & Frontiers

### 9.1 Automated AI Research
**Goal**: AI systems that can do research

1. [The AI Scientist: Towards Fully Automated Open-Ended Scientific Discovery](https://arxiv.org/pdf/2408.06292) (2024)
   - *Why*: Fully automated scientific discovery

2. [An AI system to help scientists write expert-level empirical software](https://arxiv.org/pdf/2509.06503) (2025)
   - *Why*: AI for scientific software development

3. [AlphaGo Moment for Model Architecture Discovery](https://arxiv.org/pdf/2507.18074) (2025)
   - *Why*: Automated neural architecture search

### 9.2 Specialized Applications
**Goal**: Apply AI to specific domains

1. [Stable Audio Open](https://arxiv.org/pdf/2407.14358) (2024)
   - *Why*: Audio generation with diffusion models

2. [Breaking the Molecular Dynamics Timescale Barrier Using a Wafer-Scale System](https://arxiv.org/pdf/2405.07898) (2024)
   - *Why*: Specialized hardware for scientific computing

3. [TabPFN: A transformer that solves small tabular classification problems in a second](https://arxiv.org/pdf/2207.01848v3.pdf) (2023)
   - *Why*: In-context learning for tabular data

### 9.3 Consciousness & AGI
**Goal**: Explore philosophical frontiers

1. [Consciousness in Artificial Intelligence: Insights from the Science of Consciousness](https://arxiv.org/pdf/2308.08708v3.pdf) (2024)
   - *Why*: Understanding consciousness in AI systems

2. [OpenAI o3 System Card](https://arxiv.org/pdf/2411.04996) (2024)
   - *Why*: Advanced reasoning capabilities in modern systems

---

## Phase 10: Probabilistic & Bayesian Approaches

### 10.1 Probabilistic Programming
**Goal**: Build probabilistic models programmatically

1. [A Probabilistic Programming Approach to Probabilistic Data Analysis](https://papers.nips.cc/paper/6060-a-probabilistic-programming-approach-to-probabilistic-data-analysis.pdf) (2016)
   - *Why*: Foundation of probabilistic programming

2. [Picture: An Imperative Probabilistic Programming Language for Scene Perception](https://mrkulk.github.io/www_cvpr15/1999.pdf) (2015)
   - *Why*: Vision with probabilistic programming

3. [MCMC using Hamiltonian dynamics](https://arxiv.org/abs/1206.1901) (2012)
   - *Why*: Core inference algorithm (HMC)

### 10.2 Bayesian Deep Learning
**Goal**: Combine deep learning with Bayesian inference

1. [A Bayesian Framework for Modeling Intuitive Dynamics](https://cocosci.berkeley.edu/tom/papers/collisions.pdf) (2009)
   - *Why*: Bayesian models of physical reasoning

---

## Phase 11: Computer Vision Specialization (Optional)

### 11.1 Vision Architectures
1. [Visualizing and Understanding Convolutional Networks (DeconvNet)](https://arxiv.org/abs/1311.2901) (2013)
2. [Striving for Simplicity: The All Convolutional Net](https://arxiv.org/abs/1412.6806) (2015)
3. [Semantic Segmentation using Adversarial Networks](https://arxiv.org/abs/1611.08408) (2016)

### 11.2 Vision Interpretability
1. [Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps](https://arxiv.org/abs/1312.6034) (2013)
2. [Visualizing Deep Neural Network Decisions: Prediction Difference Analysis](https://arxiv.org/abs/1702.04595) (2017)
3. [Synthesizing the Preferred Inputs for Neurons via Deep Generator Networks](https://arxiv.org/abs/1605.09304) (2016)

---

## Phase 12: Hardware & Systems

### 12.1 Hardware Optimization
**Goal**: Optimize models for specific hardware

1. [High-dimensional on-chip dataflow sensing and routing using spatial photonic networks](https://www.nature.com/articles/s41566-023-01272-3.pdf) (2023)
   - *Why*: Photonic computing for neural networks

2. [A Log-Domain Implementation of the Diffusion Network in Very Large Scale Integration](https://proceedings.neurips.cc/paper_files/paper/2010/file/7bcdf75ad237b8e02e301f4091fb6bc8-Paper.pdf) (2010)
   - *Why*: VLSI implementations of neural networks

---

## Phase 13: Policy & Governance

### 13.1 AI Policy & Standards
**Goal**: Understand AI governance and policy

1. [Dual-User Foundation Models with Widely Available Model Weights](https://www.ntia.gov/sites/default/files/publications/ntia-ai-open-model-report.pdf) (2024)
   - *Why*: Policy considerations for open models

---

## Recommended Reading Strategies

### For Beginners
Start with **Phase 1** and **Phase 2.1**, then explore based on interests.

### For ML Practitioners
Start with **Phase 2** (LLMs), then **Phase 3** (Attention), **Phase 5** (Agents), and **Phase 7** (Evaluation).

### For Researchers
Follow sequentially but focus deeply on **Phase 6** (Theory), **Phase 9** (Frontiers), and your area of interest.

### For Engineers/Practitioners
Prioritize **Phase 2** (Training), **Phase 3** (Efficiency), **Phase 7** (Evaluation), and **Phase 8** (Security).

### For Security Specialists
Follow **Phase 1-2** for foundations, then deep dive into **Phase 8**.

---

## Additional Resources

- **Deadlines**: [Academic Conferences](https://aideadlin.es/?sub=ML,CV,CG,NLP,RO,SP,DM,AP,KR,HCI)
- **Full Chronological List**: See [by-date.md](by-date.md)
- **Topical Summary**: See [README.md](README.md)

---

## Notes on Learning

- **Don't read linearly**: Papers build on each other, so refer back to earlier papers as needed
- **Implement as you learn**: Try to implement key concepts from papers
- **Join study groups**: Discuss papers with others
- **Take notes**: Summarize key insights and connections between papers
- **Focus on intuition first**: Understand the "why" before diving into mathematical details
- **Revisit papers**: Papers reveal more insights on second or third readings

---

*Last updated: October 2025*
