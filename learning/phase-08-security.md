# Phase 8: Security & Robustness

[← Back to Learning Path](../learning-path.md) | [← Previous: Phase 7](phase-07-interpretability.md) | [📖 Glossary](glossary.md)

**Phase Overview**: AI systems face unique security challenges that don't exist in traditional software. This phase examines the threat landscape: [adversarial examples](glossary.md#adversarial-example) that fool models with imperceptible perturbations, [jailbreaking](glossary.md#jailbreaking) techniques that bypass safety guardrails, [prompt injection](glossary.md#prompt-injection) attacks that hijack model behavior, and [data poisoning](glossary.md#data-poisoning) that corrupts training. You'll also learn about defenses like [adversarial training](glossary.md#adversarial-training) and certified robustness. As AI systems control increasingly important decisions—from content moderation to autonomous vehicles—understanding their vulnerabilities and defenses becomes critical for responsible deployment.

## 8.1 Adversarial Attacks
**Goal**: Build secure and aligned AI systems

1. [Training language models to follow instructions with human feedback (InstructGPT)](https://arxiv.org/abs/2203.02155) (Ouyang et al., 2022)
   - *Why*: **The InstructGPT paper** - demonstrates fine-tuning with RLHF for alignment; 1.3B model preferred over 175B GPT-3

2. [Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073) (Bai et al., 2022)
   - *Why*: **Central to LLM safety** - self-improvement through AI feedback (RLAIF) without human labels; reduces need for oversight

3. [Pr$εε$mpt: Sanitizing Sensitive Prompts for LLMs](https://arxiv.org/abs/2504.05147) (2025)
   - *Why*: Protecting sensitive information in prompts

4. [Enterprise-Grade Security for the Model Context Protocol (MCP): Frameworks and Mitigation Strategies](https://arxiv.org/pdf/2504.08623) (2025)
   - *Why*: Security frameworks for AI applications

5. [A2AS: Agentic AI Runtime Security and Self-Defense](https://arxiv.org/pdf/2510.13825) (2025)
   - *Why*: Runtime security for AI agents

6. [Breaking Agent Backbones: Evaluating the Security of Backbone LLMs in AI Agents](https://arxiv.org/abs/2510.22620) (Bazinska et al., 2025)
   - *Why*: Systematic security evaluation of LLM backbones in AI agents; introduces threat snapshots framework and b³ benchmark with 194,331 adversarial attacks; reveals reasoning capabilities improve security while model size doesn't correlate

7. [GPTFuzzer: Red Teaming Large Language Models with Auto-Generated Jailbreak Prompts](https://arxiv.org/pdf/2309.10253) (Yu et al., 2023)
   - *Why*: Automated jailbreak generation framework inspired by AFL fuzzing; achieves 90%+ attack success rates against ChatGPT and Llama-2 by mutating seed templates; demonstrates scalability and adaptability for red-teaming LLMs

## 8.2 Adversarial Robustness
**Goal**: Understand and defend against adversarial attacks

1. [Large Language Models are Unreliable for Cyber Threat Intelligence](https://arxiv.org/abs/2503.23175) (2025)
   - *Why*: Understanding LLM limitations in security contexts

2. [SEC-bench: Automated Benchmarking of LLM Agents on Real-World Software Security Tasks](https://openreview.net/pdf?id=QQhQIqons0) (Lee et al., 2025)
   - *Why*: First fully automated benchmarking framework for evaluating LLM agents on authentic security engineering tasks; introduces multi-agent scaffold for constructing verified vulnerability datasets with reproducible PoCs and patches; reveals significant performance gaps (18% PoC generation, 34% vulnerability patching) highlighting need for specialized security-focused agent architectures

---

**Next**: [Phase 9: Advanced Topics & Frontiers →](phase-09-advanced.md)
