# Dataset Composition and Sample Selection for Reliable Fine-Tuning

This repository studies how the **composition** of a fine-tuning dataset (its size, class balance and selection strategy) affects the reliability of a fine-tuned Vision Transformer. Reliability is measured by in-distribution (ID) accuracy, out-of-distribution (OOD) detection, calibration and feature drift.

The study comprises **633 fine-tuning runs** and covers three dimensions: dataset size, sample selection and class balance.

---

## Table of Contents

- [Key Findings](#key-findings)
- [Experimental Setup](#experimental-setup)
- [Evaluation Metrics](#evaluation-metrics)
- [Detailed Findings](#detailed-findings)
- [Strength of the Evidence](#strength-of-the-evidence)
- [Limitations](#limitations)
- [Hypotheses and Proposed Experiments](#hypotheses-and-proposed-experiments)
- [Response to Supervisory Feedback](#response-to-supervisory-feedback)

---

## Key Findings

1. **Class balance had the largest effect** of the three dimensions studied. Most of the benefit, however, does not require *selecting* balanced data: a class-weighted loss or balanced sampling on the full long-tailed pool recovers it.
2. **ID accuracy and near-OOD detection improve together.** The only clear exception is MNIST, so the trade-off is **between types of OOD data**, not between ID accuracy and OOD detection in general.
3. **The effect of dataset size depends on the training protocol.** Fixed-epoch protocols mostly measure undertraining, whereas the 1,000-step protocol reflects the information in the data.
4. **Representative sampling gives no meaningful benefit** over random selection on a balanced pool. It helps on a long-tailed pool only because it covers more classes.

---

## Experimental Setup

| Component | Details |
|---|---|
| **Model** | ViT-B/16, pretrained on ImageNet-21k and fine-tuned on ImageNet-1k |
| **ID dataset** | CIFAR-100 (50 images per class for validation, the remaining 45,000 as a selection pool) |
| **Near-OOD** | CIFAR-10 |
| **Far-OOD** | SVHN, Textures, MNIST, Gaussian noise |
| **Subset sizes** | 5, 10, 25, 50, 75 and 100% of the pool |
| **Selection methods** | Random, or representative sampling (k-means on pretrained embeddings, keeping the image closest to each cluster centre) |
| **Long-tailed pool** | 9,754 images; the largest class has 112x more images than the smallest |
| **Seeds** | 3 per condition |
| **Training protocols** | 1,000 optimisation steps, 3 epochs, 10 epochs. Additional runs with 250 and 3,000 steps at fixed subset sizes |

### OOD Detection Scores

- **Logit-based:** Maximum Softmax Probability (MSP), Energy
- **Feature-based:** Mahalanobis, k-nearest neighbour (kNN)

---

## Evaluation Metrics

- ID accuracy
- Worst-class accuracy
- AUROC (probability that a random ID image scores higher than a random OOD image)
- FPR at 95% TPR
- Expected Calibration Error (ECE)
- Feature drift (1 minus the linear CKA similarity between fine-tuned and pretrained features)

---

## Detailed Findings

### 1. Effect of Class Balance

- At 5% of the long-tailed pool with random selection, a class-balanced subset raised ID accuracy from **45.2% to 76.3%**, raised few-shot class accuracy by **65.8 percentage points**, and raised Energy AUROC on CIFAR-10 by **0.11**.
- Using the **complete** long-tailed pool with a class-weighted loss or balanced sampling raised ID accuracy from **77.2% to about 85%**. This matches the best balanced subset (84.6% at 25%).
- These process controls left Mahalanobis AUROC almost unchanged (differences below 0.003), while Energy AUROC rose by up to 0.06. Balancing therefore mainly corrects the **classifier output, not the learned features**.
- A natural 5% subset contained only about **72 of 100 classes**, and reweighting cannot help absent classes. The benefit of balanced selection at small sizes is therefore primarily a **class coverage** benefit.

### 2. Relation Between ID Accuracy and OOD Detection

- Within-size Spearman correlation between ID accuracy and CIFAR-10 AUROC: **0.94** (Energy), **0.92** (Mahalanobis), **0.97** (MSP). Similar values held for Textures.
- **MNIST is the exception.** Feature-based scores correlated negatively with ID accuracy (magnitudes 0.64 to 0.70). Longer training reduced Mahalanobis and kNN AUROC on MNIST in all 11 subset conditions.
- Fine-tuning reduced Mahalanobis AUROC on MNIST from **0.89** (pretrained) to **0.73 to 0.81**, while raising it on CIFAR-10 by roughly **0.10 to 0.13**.
- A frozen backbone or reduced learning rate kept MNIST detection near 0.90 but lowered CIFAR-10 Mahalanobis AUROC to 0.81 to 0.90.
- A lower ratio of within-class to between-class feature variance correlated with higher CIFAR-10 AUROC (magnitudes 0.79 to 0.89). Fine-tuning appears to make classes more compact, helping near-OOD separation, while discarding low-level information that distinguishes digits from natural images.

### 3. Dataset Size and Sample Selection

- Going from 5% to 100% of the data raised ID accuracy by **4.0 points** (1,000 steps), **26.5 points** (10 epochs) and **77.5 points** (3 epochs).
- At 5%, 3,000 steps gave the same accuracy as 1,000 steps (86.5%), so 1,000 steps had already converged on small subsets. Fixed-epoch protocols mainly measure undertraining.
- On the balanced pool, representative sampling was within **0.2 points** of random selection under all protocols.
- On the long-tailed pool at 5%, representative sampling raised accuracy from 45.2% to 53.5% because it covered about 80 classes instead of 72.

---

## Strength of the Evidence

**Well supported**

- Class balance improves ID accuracy, few-shot accuracy and CIFAR-10 detection (significant in all ten selection and size conditions).
- ID accuracy and near-OOD detection improve together (more than 400 control runs, all protocols).
- Representative and random sampling yield the same accuracy on a balanced pool.
- 1,000 steps had converged at small subset sizes.

**Tentative**

- The MNIST trade-off rests on a single far-OOD dataset, and Energy AUROC on MNIST was highly seed-sensitive.
- Mechanism results are correlational, not causal.
- Selection and process effects are not fully separable at small sizes, because class coverage is confounded with class balance.
- The comparison to the ImageNet-21k backbone is not yet informative (see below).

---

## Limitations

- Only **three seeds** per condition, and significance tests were **not corrected** for the several hundred comparisons, so some significant differences may be false positives.
- One ID dataset, one architecture and fixed hyperparameters. The shared training configuration did not suit the ImageNet-21k backbone (final loss 1.55, ECE about 0.6).
- The 3-epoch protocol gave the smallest subsets only 53 steps (11 on the long-tailed pool), so those models barely learned. No protocol trained until validation accuracy stopped improving.
- kNN and training-based Mahalanobis detectors were fitted on up to 100 images per class, so the fitting set grew from 2,250 images at 5% to 10,000 at 25% and above. Part of their size effect may reflect this.
- The long-tailed pool was artificially created from balanced CIFAR-100 with one seed, and MNIST and noise images were upscaled from 32 to 224 pixels.

---

## Hypotheses and Proposed Experiments

| Question | Competing hypotheses | Proposed test |
|---|---|---|
| **MNIST trade-off** | **A:** Fine-tuning replaces low-level visual information with semantic information, harming far-OOD detection. **B:** The effect is specific to MNIST (grayscale, heavily enlarged). | Measure Mahalanobis AUROC at intermediate transformer layers and on more far-OOD sets, including grayscale natural images. Loss confined to the last layers and shared across low-level shifts supports A. |
| **Balancing effect** | **C:** Balanced selection helps mainly through class coverage. **D:** The class proportions themselves matter. | Compare balanced subsets with natural subsets forced to contain every class. If forced coverage closes most of the gap, C is supported. |
| **Where balancing acts** | **E:** Balancing improves logit-based detection only by correcting the classifier. **F:** It also changes the features. | Retrain only the classifier with a balanced loss on features from a naturally trained model. If it gives the same Energy AUROC gain, E is supported. |

---

## Response to Supervisory Feedback

- **Training protocol:** added fixed budgets of 3 and 10 epochs for every subset size, a step sweep at fixed subset size, and a regression separating the effect of more images from the effect of more steps.
- **OOD evaluation:** analysed MSP against Energy, added Mahalanobis detectors (fitted on the training subset and on the validation set) and a kNN detector, and added Textures, MNIST and Gaussian noise as OOD sets. A separate analysis tested whether the absence of a trade-off holds across scores, protocols and fine-tuning regimes.
- **Class imbalance:** added a class-weighted loss and balanced sampling (applied to the original subsets and the full long-tailed dataset), tested four further balancing methods, and separated the effect of selecting balanced data from that of balancing the training process using paired contrasts.

---

## Citation / Contact

If you use this work, please cite this repository. For questions, open an issue.
