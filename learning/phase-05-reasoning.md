# Phase 5: Reasoning & Alignment

[← Back to Learning Path](../learning-path.md) | [← Previous: Phase 4](phase-04-retrieval.md) | [📖 Glossary](glossary.md)

**Phase Overview**: Raw language models don't naturally follow instructions or break down complex problems step-by-step. This phase covers the crucial techniques that transform base models into helpful, harmless assistants: [reinforcement learning from human feedback (RLHF)](glossary.md#rlhf-reinforcement-learning-from-human-feedback) that aligns models with human preferences, [chain-of-thought prompting](glossary.md#chain-of-thought-prompting) that enables multi-step reasoning, and [constitutional AI](glossary.md#constitutional-ai) approaches that encode ethical principles. You'll also learn about [agent](glossary.md#agent) frameworks like [ReAct](glossary.md#react-reasoning-and-acting) that give models the ability to use tools and take actions. These methods are what make modern AI systems genuinely useful and (relatively) safe.

## 5.1 Teaching Models to Reason
**Goal**: Build models that can reason and solve complex problems

1. [Deep Reinforcement Learning from Human Preferences](https://arxiv.org/abs/1706.03741) (Christiano et al., 2017)
   - *Why*: **Introduces RLHF** - the foundation of alignment; learning from human feedback without explicit rewards

2. [Proximal Policy Optimization Algorithms (PPO)](https://arxiv.org/abs/1707.06347) (Schulman et al., 2017)
   - *Why*: **Modern RL training backbone** for LLM fine-tuning; simpler and more stable than TRPO

3. [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629) (Yao et al., 2022)
   - *Why*: **Core to reasoning-agent architectures** - interleaves reasoning traces with actions for better problem-solving

4. [Reflexion: Language Agents with Verbal Reinforcement Learning](https://arxiv.org/abs/2303.11366) (Shinn et al., 2023)
   - *Why*: Agentic self-improvement through linguistic feedback and episodic memory

5. [Hierarchical Reasoning Model](https://arxiv.org/pdf/2506.21734) (2025)
   - *Why*: Brain-inspired hierarchical reasoning

6. [Less is More: Recursive Reasoning with Tiny Networks](https://arxiv.org/pdf/2510.04871) (2025)
   - *Why*: Efficient reasoning with small models

7. [Reinforcement Pre-Training](https://arxiv.org/abs/2506.08007) (2025)
   - *Why*: Using RL during pre-training phase

## 5.2 Agentic Systems
**Goal**: Create autonomous AI agents

1. [A Generalist Agent](https://arxiv.org/pdf/2205.06175) (2022)
   - *Why*: Multi-task agent foundations (Gato)

2. [Executable Code Actions Elicit Better LLM Agents](https://arxiv.org/pdf/2402.01030) (2024)
   - *Why*: Code execution for agents

3. [DynaSaur: Large Language Agents Beyond Predefined Actions](https://arxiv.org/abs/2411.01747) (2024)
   - *Why*: Dynamic action generation

4. [MAS-ZERO: Designing Multi-Agent Systems with Zero Supervision](https://arxiv.org/pdf/2505.14996) (2025)
   - *Why*: Automated multi-agent system design

5. [Small Language Models are the Future of Agentic AI](https://arxiv.org/abs/2506.02153) (2025)
   - *Why*: Makes the case for using smaller, specialized models in agent systems for efficiency and cost-effectiveness; practical deployment considerations

6. [Learning in Stackelberg Mean Field Games: A Non-Asymptotic Analysis](https://arxiv.org/pdf/2509.15392) (Zeng et al., 2025)
   - *Why*: Introduces AC-SMFG, a single-loop actor-critic algorithm for hierarchical multi-agent systems with a leader and infinite population of followers; first Stackelberg MFG algorithm with non-asymptotic convergence guarantees; addresses policy optimization in strategic interactions like optimal liquidation and public policy

7. [Embedded Universal Predictive Intelligence: a coherent framework for multi-agent learning](https://arxiv.org/pdf/2511.22226) (Meulemans et al., 2025)
   - *Why*: Introduces a mathematical framework for embedded agency and prospective learning in multi-agent settings; extends AIXI theory with self-prediction where Bayesian RL agents predict both future inputs and their own actions; enables infinite-order theory of mind and novel forms of cooperation in mixed-motive scenarios

---

**Next**: [Phase 6: Novel Architectures & Theory →](phase-06-architectures.md)
