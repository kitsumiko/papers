# Optimization & Energy-Based Methods

[← Back to Learning Path](../learning-path.md) | [← Prev: Architectures](architectures.md) | [Next: Interpretability →](interpretability.md) | [📖 Glossary](glossary.md)

**Overview**: Almost every model in this collection gets its weights by minimizing a loss, and many solve problems by minimizing something else at inference time. This area is about optimization itself. It covers the optimizers that drive training, from first-order [gradient descent](glossary.md#gradient-descent) and [SGD](glossary.md#sgd-stochastic-gradient-descent) to second-order curvature methods and learned predictors of weight updates. It covers energy-based models, which turn inference into descent on a learned energy function and descend directly from the Hopfield and Boltzmann work in [Classics](classics.md). It also covers learned and LLM-driven optimizers that attack combinatorial and black-box problems where gradients aren't available. Together these explain why training works as well as it does, and they suggest that "reasoning" is sometimes just a well-shaped objective plus a good way to descend it.

## Training Dynamics & Optimizers
**Goal**: Understand how optimizers shape training speed and stability

1. [The Potential of Second-Order Optimization for LLMs: A Study with Full Gauss-Newton](https://arxiv.org/pdf/2510.09378) (2025)
   - *Why*: **Second-order methods revisited for LLMs** - makes full Gauss-Newton optimization tractable at LLM scale through careful implementation; shows curvature-aware updates converge in fewer steps than Adam on language modeling tasks; quantifies the gap between first- and second-order methods to motivate future optimizer research

2. [Accelerating Training With Neuron Interaction And Nowcasting Networks](https://arxiv.org/pdf/2409.04434) (2024)
   - *Why*: **Predicting weight updates to skip training steps** - trains a small auxiliary network to predict future gradient updates by modeling neuron interactions; enables "nowcasting" of weight trajectories to skip multiple optimization steps; achieves training speedups by reducing the total number of forward-backward passes needed for convergence

## Energy-Based Models & Optimization as Inference
**Goal**: Solve problems by descending a learned energy function

1. [Classical Energies Learn to Reason 1,000x Faster than Looped Models](https://arxiv.org/abs/2611.17309) (Hoover et al., 2026)
   - *Why*: **Energy descent as a competitor to looped reasoners** - learns a version of Hopfield & Tank's 1980s constraint energy (CRAM) and solves puzzles by descending it, which gives built-in early stopping, compute that adapts to difficulty, and interpretable weights that recover the puzzle's constraints; it solves 99.996% of Sudoku-Extreme and 100% of Maze-Unique after 42 ms and 132 ms of single-GPU training (at least 1,000x faster than looped models), and the authors read this as a warning that such easily solved benchmarks say little about how looped reasoning scales.

## Learned & LLM-Driven Optimization
**Goal**: Use learned models and accelerators to search solution spaces without gradients

1. [Large Language Models as Optimizers](https://arxiv.org/abs/2309.03409) (Yang et al., 2023)
   - *Why*: Introduces OPRO (Optimization by PROmpting), in which an LLM proposes new solutions from a prompt listing past solutions and their scores, so the model acts as a gradient-free optimizer driven by natural language. Applied to prompt search, it finds instructions that beat human-written prompts by up to 8% on GSM8K and up to 50% on Big-Bench Hard, and it is the source of the well-known "Take a deep breath and work on this problem step-by-step" prompt.

2. [cuGenOpt: A GPU-Accelerated General-Purpose Metaheuristic Framework for Combinatorial Optimization](https://arxiv.org/abs/2603.19163) (Liu, 2026)
   - *Why*: Demonstrates a "one block evolves one solution" CUDA architecture with hardware-aware resource management across GPU generations (T4, V100, A800), outperforming general MIP solvers by orders of magnitude on combinatorial optimization problems.

---

**Related**: [Architectures](architectures.md) | [Reasoning](reasoning.md) | [Hardware](hardware.md)
