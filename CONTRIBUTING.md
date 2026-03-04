# Contributing to AI/ML Papers Collection

Thank you for your interest in contributing! This document provides guidelines and instructions for contributing to this curated research paper collection.

## 📋 Table of Contents

- [Ways to Contribute](#ways-to-contribute)
- [Paper Selection Criteria](#paper-selection-criteria)
- [How to Suggest a Paper](#how-to-suggest-a-paper)
- [How to Fix Issues](#how-to-fix-issues)
- [Style Guidelines](#style-guidelines)
- [Code of Conduct](#code-of-conduct)

---

## 🎯 Ways to Contribute

### 1. Suggest New Papers
Found an influential paper that should be included? Open an issue using the paper suggestion template.

### 2. Fix Broken Links
Papers move, URLs change. If you find a broken link, please report it or submit a fix.

### 3. Improve Annotations
The "Why" explanations help readers understand paper significance. Improvements are welcome!

### 4. Add Glossary Terms
Found a term that should be in the glossary? Suggest additions.

### 5. Report Errors
Spotted a typo, incorrect date, or miscategorization? Let us know!

---

## 📚 Paper Selection Criteria

We prioritize papers that meet these criteria:

### Must Have
- [ ] **Influential or foundational** - Highly cited, introduced key concepts, or represents a significant advance
- [ ] **Accessible** - Available via arXiv, conference proceedings, or author's website (open access preferred)
- [ ] **Relevant** - Fits within the AI/ML research landscape

### Strong Preference
- [ ] **Pedagogically valuable** - Clear writing that teaches concepts effectively
- [ ] **Well-scoped** - Focused contribution that's digestible
- [ ] **Reproducible** - Methods described in enough detail to replicate

### Area Fit
Consider which topic area the paper belongs to:

| Area | Focus | Example Papers |
|------|-------|----------------|
| Foundations | Deep learning basics | LeNet, AlexNet, Word2Vec |
| Language Models | LLMs | BERT, GPT-3, Transformers |
| Attention | Efficient attention | FlashAttention, RetNet |
| Retrieval | RAG, retrieval | RAG, Dense Passage Retrieval |
| Reasoning | Reasoning & agents | ReAct, RLHF, Constitutional AI |
| Architectures | Alternative architectures | Mamba, KAN, Neural Turing Machines |
| Interpretability | Interpretability & evaluation | LIME, Integrated Gradients |
| Safety | Security & safety | Prompt injection, Adversarial ML |
| Advanced | Advanced applications | AI Scientist, Multimodal |
| Probabilistic | Probabilistic models | Diffusion, Bayesian methods |
| Vision | Vision & multimodal | ViT, CLIP, SAM |
| Hardware | Hardware & systems | Photonics, VLSI |
| Policy | Policy & governance | GDPR, EU AI Act, NIST RMF |

---

## 📝 How to Suggest a Paper

### Option 1: GitHub Issue (Preferred)

1. Go to [Issues → New Issue](../../issues/new/choose)
2. Select "📄 Paper Suggestion" template
3. Fill in:
   - Paper title and link (arXiv preferred)
   - Authors and publication year
   - Suggested area
   - Why this paper should be included
   - A "Why read this?" annotation (1-2 sentences)

### Option 2: Pull Request

If you're comfortable with Git:

1. Fork the repository
2. Create a branch: `git checkout -b add-paper-<short-name>`
3. Add the paper to the appropriate area file (in `learning/`)
4. Add to `by-date.md` in the correct chronological position
5. Submit a pull request using the template

### Paper Entry Format

```markdown
### Paper Title

📄 [Paper Title](https://arxiv.org/abs/XXXX.XXXXX)

**Why:** One or two sentences explaining why this paper is important and what the reader will learn.

**Key contributions:**
- Contribution 1
- Contribution 2
- Contribution 3
```

---

## 🔧 How to Fix Issues

### Broken Links

1. Identify the broken link
2. Find the correct URL:
   - Check [arXiv](https://arxiv.org)
   - Search [Semantic Scholar](https://semanticscholar.org)
   - Look for author's website or institutional repository
3. Open an issue or submit a PR with the fix

### Typos and Corrections

For small fixes:
1. Edit the file directly on GitHub
2. Submit as a PR with a clear description

For larger changes:
1. Open an issue first to discuss
2. Reference the issue in your PR

---

## 🎨 Style Guidelines

### Markdown Formatting

- Use `###` for paper titles within area files
- Use emoji indicators:
  - 📄 for research papers
  - 🔒 for paywalled content
  - 📜 for policy documents
  - ⚠️ for important notes

### Link Formatting

```markdown
# Preferred: arXiv abstract page
[Paper Title](https://arxiv.org/abs/XXXX.XXXXX)

# Acceptable: Direct PDF
[Paper Title](https://arxiv.org/pdf/XXXX.XXXXX.pdf)

# Also acceptable: Conference proceedings
[Paper Title](https://papers.nips.cc/paper/YYYY/...)
```

### Annotations

Keep "Why" annotations:
- **Concise**: 1-2 sentences maximum
- **Specific**: What will the reader learn?
- **Contextual**: How does this relate to other papers?

Good example:
> **Why:** Introduces the Transformer architecture that became the foundation for modern LLMs. Essential reading for understanding attention mechanisms.

Avoid:
> **Why:** An important paper about neural networks. (Too vague)

---

## 📊 Glossary Contributions

When adding glossary terms:

1. Place in alphabetical order within the appropriate category
2. Use this format:

```markdown
### Term Name

**Definition:** Clear, concise definition.

**Context:** Where/how this term is used.

**Related:** Links to related terms or papers.
```

---

## 🤝 Code of Conduct

### Our Standards

- Be respectful and constructive
- Focus on the technical merits of papers
- Welcome newcomers and help them contribute
- Acknowledge different perspectives on paper importance

### Not Acceptable

- Dismissive or rude comments
- Personal attacks
- Spam or self-promotion
- Off-topic discussions

---

## ❓ Questions?

- **General questions**: Open a [Discussion](../../discussions)
- **Bug reports**: Open an [Issue](../../issues)
- **Paper debates**: Use Discussions to debate paper inclusion/classification

---

## 🙏 Recognition

Contributors are valued! Significant contributions will be acknowledged in:
- Pull request descriptions
- Release notes (for major updates)
- The project README (for sustained contributions)

---

Thank you for helping make this resource better for the AI/ML learning community! 🚀

