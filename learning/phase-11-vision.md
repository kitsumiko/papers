# Phase 11: Vision & Multimodal Systems

[← Back to Learning Path](../learning-path.md) | [← Previous: Phase 10](phase-10-probabilistic.md) | [📖 Glossary](glossary.md)

**Phase Overview**: The transformer revolution didn't stop at text—it transformed computer vision too. This phase explores how [attention mechanisms](glossary.md#attention-mechanism) replaced [convolutional neural networks](glossary.md#cnn-convolutional-neural-network) as the dominant paradigm in vision, how [CLIP](glossary.md#clip-contrastive-language-image-pre-training) bridges vision and language through contrastive learning, and how models like [SAM](glossary.md#sam-segment-anything-model) achieve unprecedented zero-shot image segmentation. You'll see how the same principles that power ChatGPT enable models to understand and generate images, paving the way for truly [multimodal](glossary.md#multimodal-learning) AI systems that can seamlessly work with text, images, and video together.

## 11.1 Vision Transformers

1. [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale (Vision Transformer/ViT)](https://arxiv.org/abs/2010.11929) (Dosovitskiy et al., 2021)
   - *Why*: **Missing but crucial** - applies pure transformers to vision; connects vision and language model architectures

2. [Learning Transferable Visual Models From Natural Language Supervision (CLIP)](https://arxiv.org/abs/2103.00020) (Radford et al., 2021)
   - *Why*: **The bridge to multimodal AI** - learns vision-language alignment from 400M image-text pairs; enables zero-shot transfer

3. [Segment Anything (SAM)](https://arxiv.org/abs/2304.02643) (Kirillov et al., 2023)
   - *Why*: **Pivotal open foundation model** - promptable segmentation with 1B masks; demonstrates vision foundation model capabilities

4. [Visualizing and Understanding Convolutional Networks (DeconvNet)](https://arxiv.org/abs/1311.2901) (2013)
   - *Why*: Understanding what CNNs learn through deconvolution

5. [Striving for Simplicity: The All Convolutional Net](https://arxiv.org/abs/1412.6806) (2015)
   - *Why*: Replacing pooling with learned downsampling

6. [Semantic Segmentation using Adversarial Networks](https://arxiv.org/abs/1611.08408) (2016)
   - *Why*: Applying GANs to segmentation tasks

## 11.2 Vision Interpretability

1. [Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps](https://arxiv.org/abs/1312.6034) (2013)
   - *Why*: Visualizing image classification models

2. [Visualizing Deep Neural Network Decisions: Prediction Difference Analysis](https://arxiv.org/abs/1702.04595) (2017)
   - *Why*: Understanding network decision-making

3. [Synthesizing the Preferred Inputs for Neurons via Deep Generator Networks](https://arxiv.org/abs/1605.09304) (2016)
   - *Why*: Generating optimal inputs for neurons

---

**Next**: [Phase 12: Hardware & Systems →](phase-12-hardware.md)
