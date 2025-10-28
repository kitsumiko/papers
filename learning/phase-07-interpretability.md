# Phase 7: Interpretability & Analysis

[← Back to Learning Path](../learning-path.md) | [← Previous: Phase 6](phase-06-architectures.md)

**Phase Overview**: Neural networks are often treated as black boxes, but what's really happening inside? This phase dives into interpretability—the science of understanding and explaining model behavior. You'll learn about mechanistic interpretability that reverse-engineers learned algorithms, activation analysis techniques like sparse autoencoders that reveal hidden structure, theoretical frameworks like the Neural Tangent Kernel that explain why deep learning works, and methods for detecting when models are truthful or deceptive. As AI systems become more powerful and deployed in high-stakes settings, interpretability becomes crucial for trust, debugging, and safety.

## 7.1 Understanding Model Behavior
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

## 7.2 Model Evaluation & Robustness
**Goal**: Properly evaluate and benchmark models

1. [TruthfulQA: Measuring How Models Mimic Human Falsehoods](https://arxiv.org/abs/2109.07958) (Lin et al., 2022)
   - *Why*: **Factuality benchmark** - reveals that larger models can be less truthful; critical for understanding model limitations

2. [Forget What You Know about LLMs Evaluations -- LLMs are Like a Chameleon](https://arxiv.org/pdf/2502.07445) (2025)
   - *Why*: Critical analysis of evaluation methodologies

3. [MLE-bench: Evaluating Machine Learning Agents on Machine Learning Engineering](https://arxiv.org/abs/2410.07095) (2024)
   - *Why*: Benchmarking ML engineering capabilities

4. [On Calibration of Modern Neural Networks](https://arxiv.org/abs/1706.04599) (2017)
   - *Why*: Understanding prediction confidence

5. [The Illusion of Thinking](https://ml-site.cdn-apple.com/papers/the-illusion-of-thinking.pdf) (2025)
   - *Why*: Critical examination of reasoning capabilities

---

**Next**: [Phase 8: Security, Privacy & Safety →](phase-08-security.md)
