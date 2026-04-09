# Knowledge Distillation Experiments

This repository is a collection of experiments around knowledge distillation (KD), vision transformers, and masked image modeling. It serves as a working knowledge base rather than a polished framework.

## Overview

The core focus is exploring different ways to transfer knowledge from larger models to smaller ones, with an emphasis on:

* Distilling large vision models (e.g. DINOv2-L) into lightweight architectures (ViT-Tiny)
* Combining knowledge distillation with masked image modeling
* Training and pretraining ViT-Tiny models on ImageNet
* Testing simple and custom KD optimization strategies

The codebase is modular but experimental in nature. Expect some overlap between scripts and iterative variations.

## Structure

```
experimentation/        # Notebooks for prototyping and testing ideas
src/
  kd_with_masked_image_modelling/
    dataset_loader.py  # Dataset handling
    models.py          # Model definitions (ViT variants, etc.)
    losses.py          # KD + reconstruction losses
    optimisation.py    # Training utilities
    optim_kd.py        # KD-specific optimization logic
    main.py            # Standard KD training entrypoint
    masked_main.py     # Masked image modeling training
    masked_main_kd.py  # KD + masked modeling combined
    vit_tiny.py        # ViT-Tiny implementation
    vit_tiny_masked.py # Masked ViT-Tiny variant
    utils.py           # Helpers
    wandb_trainer.py   # Logging/training wrapper
```

Additional standalone scripts and notebooks are included for:

* Classification experiments
* Attention visualization
* Distillation trials
* ImageNet pretraining

## Key Ideas Explored

### 1. Distillation: Large → Tiny

* Distilling from larger pretrained teachers (e.g. DINOv2-L)
* Target: compact ViT-Tiny models
* Focus on representation transfer rather than exact output matching

### 2. KD + Masked Image Modeling

* Integrating reconstruction objectives into the KD process
* Student learns from:

  * Teacher signals (logits/features)
  * Masked patch reconstruction
* Aimed at improving representation quality under compression

### 3. Simple KD Optimization

* Lightweight KD setups without heavy tuning
* Experiments with:

  * Feature matching
  * Logit-based distillation
  * Hybrid loss formulations

### 4. ViT-Tiny Pretraining

* Scripts for training ViT-Tiny from scratch on ImageNet
* Used both as:

  * Standalone baseline
  * Initialization for distillation

## Usage

Most workflows are script-based:

### Standard KD

```bash
python main.py
```

### Masked Image Modeling

```bash
python masked_main.py
```

### KD + Masked Modeling

```bash
python masked_main_kd.py
```

Configuration is handled inside scripts or via lightweight argument edits.

## Future improvements
* The code base is a little redundant at this point, would love to improve the structure.
* Planning to add the results after distillation from various methods.

## Notes

* This repo is experimental and iterative
* Code is optimized for flexibility, not abstraction
* Some duplication exists across experiments by design
* WandB is used optionally for logging

## Purpose

This is not a library. It’s a record of experiments, ideas, and directions explored around compressing vision models and improving small-model learning through distillation.
