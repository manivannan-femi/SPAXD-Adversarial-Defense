# MVT-SPAR: MVT-Inspired Spatially Adaptive Singular-Perturbation-Based Adversarial Restoration

[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![Dataset](https://img.shields.io/badge/Dataset-GTSRB-green.svg)](https://benchmark.ini.rub.de/gtsrb_dataset.html)
[![Platform](https://img.shields.io/badge/Platform-Google%20Colab-yellow.svg)](https://colab.research.google.com/)

## 1. Overview

This repository presents MVT-SPAR (MVT-Inspired Spatially Adaptive Singular-Perturbation-Based Adversarial Restoration), a spatially adaptive diffusion-based framework for investigating adversarial robustness in deep learning-based traffic sign classification.

The proposed approach integrates an MVT-inspired path-gradient sensitivity estimation mechanism with spatially adaptive diffusion and a singular-perturbation-inspired partial differential equation (PDE) solver.

The framework aims to suppress adversarial perturbations while preserving important image structures and classification-relevant information.

The proposed restoration method is investigated using three different deep learning architectures:

- ResNet-50
- VGG16
- AlexNet

The experiments are conducted using the German Traffic Sign Recognition Benchmark (GTSRB) dataset.

## 2. Research Objectives

The main objectives of this research implementation are:

- To investigate the vulnerability of deep learning models against adversarial perturbations.
- To estimate spatial sensitivity using an MVT-inspired path-gradient approach.
- To construct spatially adaptive diffusion coefficients using sensitivity and image-gradient information.
- To formulate a singular-perturbation-inspired PDE for adversarial image restoration.
- To implement an iterative Jacobi finite-difference solver.
- To investigate the effectiveness of MVT-SPAR across different CNN architectures.
- To evaluate robustness against white-box attacks and black-box patch occlusion.

## 3. Proposed MVT-SPAR Framework

The proposed framework consists of the following stages:

1. Dataset Preparation: Load, preprocess, and resize GTSRB traffic sign images.
2. Classification Model: Train and fine-tune ResNet-50, VGG16, and AlexNet architectures.
3. Adversarial Generation:Generate adversarial examples using FGSM, PGD-40, BIM, and MIM.
4. Sensitivity Estimation: Estimate spatial sensitivity using an MVT-inspired sampled path-gradient approach.
5. Adaptive Diffusion: Construct a spatially varying diffusion coefficient based on sensitivity and image gradients.
6. PDE-Based Restoration: Restore adversarial images by solving the proposed diffusion equation.
7. Performance Evaluation: Evaluate classification accuracy, image restoration quality, and numerical solver performance.

## 4. Mathematical Formulation

### 4.1 MVT-Inspired Path-Gradient Sensitivity Estimation

Let \(X_{\mathrm{adv}}\) represent an adversarial image.

First, Gaussian smoothing is applied:

\[
X_{\mathrm{smooth}}=G_\sigma(X_{\mathrm{adv}}),
\]

where \(G_\sigma\) denotes the Gaussian smoothing operator.

The residual component is defined as

\[
R=X_{\mathrm{adv}}-X_{\mathrm{smooth}}.
\]

A sampled path is constructed as

\[
X(t)=\operatorname{clip}
(X_{\mathrm{adv}}-tR,0,1),
\quad t\in[0,1].
\]

The MVT-inspired sensitivity estimation is formulated as

\[
S(x,y)=
\frac{1}{K+1}
\sum_{t\in\mathcal T}
\sum_c
\left|
R_c(x,y)
\frac{\partial J(\theta,X(t),y)}
{\partial X_c(x,y)}
\right|,
\]

where:

- \(R_c(x,y)\) denotes the residual component.
- \(J\) represents the classification loss.
- \(\theta\) denotes the model parameters.
- \(K\) represents the number of path intervals.
- \(c\) denotes the image channel.

The normalized sensitivity map is calculated as

\[
S_n(x,y)=
\frac{S(x,y)}
{\max_{x,y}S(x,y)+\delta},
\]

where \(\delta\) is a small positive constant.

The implementation uses \(K=5\), corresponding to six sampled path points.

*Note: This is an MVT-inspired sampled path-gradient estimator, not an exact computation of the unknown Mean Value Theorem point.*

### 4.2 Spatially Adaptive Diffusion Coefficient

The spatially adaptive diffusion coefficient is defined as

\[
D(x,y)=
(1+\alpha S_n(x,y))
\exp(-\beta |\nabla I(x,y)|^2),
\]

where:

- \(S_n(x,y)\) represents normalized sensitivity.
- \(\alpha\) controls sensitivity contribution.
- \(\beta\) controls edge preservation.
- \(\nabla I(x,y)\) represents the spatial image gradient.

The diffusion coefficient is bounded numerically to maintain stability.

### 4.3 Singular-Perturbation-Inspired PDE Model

The restoration process is formulated as

\[
\lambda u(x,y)
-\epsilon^2\nabla\cdot
\left(D(x,y)\nabla u(x,y)\right)
=
\lambda X_{\mathrm{adv}}(x,y),
\]

where:

- \(u(x,y)\) is the restored image.
- \(X_{\mathrm{adv}}(x,y)\) is the adversarial input.
- \(D(x,y)\) is the spatially adaptive diffusion coefficient.
- \(\epsilon\) is the diffusion scaling parameter.
- \(\lambda\) is the reaction parameter.

### 4.4 Numerical Solution

The PDE is discretized using a finite-difference approximation and solved using a batched Jacobi iterative scheme.

The numerical implementation incorporates:

- Face-centred arithmetic averaging of diffusion coefficients.
- Zero-normal-flux boundary conditions.
- TensorFlow-based batched operations.
- Iterative updates.
- Relative PDE residual evaluation.

## 5. Dataset Description

The experiments use the German Traffic Sign Recognition Benchmark (GTSRB), consisting of 43 traffic sign categories.

| Parameter | Configuration |
|---|---|
| Dataset | GTSRB |
| Number of classes | 43 |
| Labelled images discovered | 78,418 |
| Training split | 74,418 |
| Held-out test split | 4,000 |
| Input image dimensions | 224 × 224 |
| Image channels | RGB |
| Random seed | 42 |
| Split method | Stratified sampling |

The dataset is extracted from a ZIP archive and processed using an automated image and label discovery mechanism.

Dataset note: The notebooks create a stratified 4,000-image holdout from the discovered labelled images. This should not be described as the official GTSRB test set. The split manifest is saved for reproducibility.

## 6. Deep Learning Architectures

Three classification architectures are investigated to examine the applicability of MVT-SPAR across different CNN models.

### 6.1 ResNet-50

A pretrained ResNet-50 architecture with ImageNet weights is employed as the feature extraction backbone.

The classification head consists of:

- Global Average Pooling.
- Dense layer with 256 neurons.
- Dropout (0.3).
- Dense layer with 128 neurons.
- Dropout (0.4).
- Dense layer with 64 neurons.
- Dropout (0.3).
- Final 43-class softmax layer.

### 6.2 VGG16

A pretrained VGG16 architecture with ImageNet weights is employed as the feature extraction backbone.

The classification head uses the same dense-layer configuration as ResNet-50.

### 6.3 AlexNet

A custom AlexNet-style CNN architecture is implemented and trained from scratch.

The feature extraction architecture includes:

- Five convolutional layers.
- ReLU activation.
- Max-pooling operations.
- Global Average Pooling.
- Fully connected classification layers.
- Final 43-class softmax output.

Unlike ResNet-50 and VGG16, this AlexNet implementation does not use official ImageNet-pretrained AlexNet weights.

### 6.4 Training Configuration

| Parameter | ResNet-50 | VGG16 | AlexNet |
|---|---:|---:|---:|
| Initial weights | ImageNet | ImageNet | Random |
| Image size | 224 | 224 | 224 |
| Number of classes | 43 | 43 | 43 |
| Head training epochs | 5 | 5 | 5 |
| Fine-tuning maximum epochs | 15 | 15 | 50 |
| Initial learning rate | 0.01 | 0.01 | 0.01 |
| Fine-tuning learning rate | 0.0001 | 0.0001 | 0.0001 |
| Optimizer | SGD | SGD | SGD |
| Momentum | 0.9 | 0.9 | 0.9 |
| Training batch size | 16 | 16 | 16 |

Early stopping, learning-rate reduction, and model checkpointing are incorporated into the training process.

## 7. Adversarial Attack Methods

The trained classification models are evaluated against multiple adversarial attacks.

### 7.1 White-Box Attacks

The following gradient-based attacks are implemented:

1. Fast Gradient Sign Method (FGSM).
2. Projected Gradient Descent (PGD-40).
3. Basic Iterative Method (BIM).
4. Momentum Iterative Method (MIM).

The FGSM formulation is

\[
X_{\mathrm{adv}}=
X+\epsilon\operatorname{sign}
\left(\nabla_X J(\theta,X,y)\right).
\]

The investigated perturbation magnitudes are

\[
\epsilon\in
\{0.01,0.03,0.05,0.10\}.
\]

The iterative attacks use 40 steps.

### 7.2 Black-Box Patch Occlusion

A fixed-size black square is placed at a randomly selected image location.

The patch sizes are

\[
p\in\{2,4,6,8\}.
\]

This experiment investigates classification sensitivity to localized image occlusion. It is not an optimized adversarial patch attack.

## 8. Experimental Results

The following table presents the clean classification accuracies recorded in the uploaded notebooks.

| Classification architecture | Recorded clean test accuracy |
|---|---:|
| ResNet-50 | 97.775% |
| VGG16 | 87.850% |
| AlexNet | 99.900% |

These values are the recorded outputs of the respective notebook runs, not a controlled comparative benchmark.

The notebooks also generate:

- White-box attack accuracy tables.
- Patch occlusion accuracy tables.
- MVT-SPAR restoration evaluation tables.
- Classical restoration comparisons.
- Confusion matrices.
- Sensitivity and diffusion coefficient visualizations.
- Image restoration metrics.

**Defense evaluation note:** The initial configuration uses a 64-image restoration pilot to reduce memory consumption. The complete 4,000-image restoration experiment can be enabled after validating the pilot. Therefore, the clean accuracies above should not be interpreted as post-defense accuracies.

## 9. Evaluation Metrics

### Classification Performance

- Accuracy.
- Macro Precision.
- Macro Recall.
- Macro F1-score.
- Confusion Matrix.
- Attack Success Rate.

### Image Restoration Performance

- Peak Signal-to-Noise Ratio (PSNR).
- Structural Similarity Index Measure (SSIM).
- Mean Squared Error (MSE).
- Mean Absolute Error (MAE).

### Numerical Performance

- Relative PDE residual.
- Sensitivity map distribution.
- Adaptive diffusion coefficient distribution.
- Iterative solver behaviour.

## 10. Memory-Efficient Implementation

The notebooks are designed for Google Colab execution while reducing RAM and GPU-memory consumption.

The main implementation features include:

- Batch-based image loading using TensorFlow `tf.data`.
- Incremental image preprocessing.
- Disk-backed NumPy memory maps for adversarial images.
- Incremental attack generation.
- Checkpoint-based experiment resumption.
- Batched TensorFlow Jacobi PDE solver.
- GPU memory growth configuration.
- Incremental result storage.

These strategies help reduce memory pressure during large-scale adversarial experiments.

## 11. Repository Structure

```text
MVT-SPAR-GTSRB/
│
├── README.md
│
├── notebooks/
│   ├── MVT_SPAR_GTSRB_ResNet50_Crash_Resistant.ipynb
│   ├── MVT_SPAR_GTSRB_VGG16_Crash_Resistant.ipynb
│   └── MVT_SPAR_GTSRB_AlexNet_Crash_Resistant.ipynb
│
├── requirements.txt
│
├── results/
│   ├── whitebox_attack_results.csv
│   ├── patch_attack_results.csv
│   ├── mvt_spar_defense_results.csv
│   └── restoration_comparison.csv
│
├── figures/
│   ├── methodology.png
│   ├── sensitivity_maps.png
│   ├── diffusion_coefficients.png
│   └── restoration_comparison.png
│
└── models/
    ├── resnet50_gtsrb.keras
    ├── vgg16_gtsrb.keras
    └── alexnet_gtsrb.keras
```

The folders for results, figures, and models represent the intended organization. Upload the corresponding generated files when available.

## 12. Installation and Dependencies

The implementation uses Python and TensorFlow with scientific computing libraries.

Install the main dependencies using:

```bash
pip install tensorflow numpy pandas opencv-python
pip install matplotlib scikit-learn scikit-image tqdm
pip install reportlab
```

The notebooks should be executed using compatible versions of the required libraries.

## 13. Running the Experiments

The three notebooks are available in the `notebooks/` directory.

### Execution Procedure

1. Open Google Colab.
2. Open the selected notebook.
3. Mount Google Drive.
4. Upload the GTSRB ZIP archive to Google Drive.
5. Execute the package installation and configuration cells.
6. Extract the dataset and create the stratified data split.
7. Train the selected classification architecture.
8. Generate adversarial examples.
9. Estimate MVT-inspired sensitivity maps.
10. Calculate spatially adaptive diffusion coefficients.
11. Perform PDE-based image restoration.
12. Evaluate classification and restoration performance.
13. Generate experimental tables and visualizations.

### Selecting an Architecture

| Notebook | Architecture |
|---|---|
| `MVT_SPAR_GTSRB_ResNet50_Crash_Resistant.ipynb` | ResNet-50 |
| `MVT_SPAR_GTSRB_VGG16_Crash_Resistant.ipynb` | VGG16 |
| `MVT_SPAR_GTSRB_AlexNet_Crash_Resistant.ipynb` | AlexNet |

## 14. Important Experimental Parameters

The common MVT-SPAR configuration is:

| Parameter | Value |
|---|---:|
| Random seed | 42 |
| Image size | 224 × 224 |
| Number of classes | 43 |
| Attack epsilon values | 0.01, 0.03, 0.05, 0.10 |
| Attack iterations | 40 |
| Patch sizes | 2, 4, 6, 8 |
| MVT path intervals (K) | 5 |
| Jacobi iterations | 40 |
| PDE diffusion parameter | 1.0 |
| PDE reaction parameter | 1.0 |
| Sensitivity parameter (α) | 2.0 |
| Edge parameter (β) | 10.0 |
| Initial defense pilot size | 64 |

The defense limit can be increased to 4,000 images after validating the pilot execution.

## 15. Experimental Outputs

The notebooks save experimental outputs in Google Drive under:

`GTSRB-MVT-SPAR/`

The output directories include:

- `Dataset_Split/`
- `Trained_Models/`
- `Adversarial_Images/`
- `Restored_Images/`
- `Results_CSV/`
- `Figures/`
- `PDF_Report/`
- `Logs/`
- `Checkpoints/`

The generated files include trained models, adversarial examples, restored images, performance tables, figures, and PDF reports.

## 16. Research Limitations and Future Work

Future research directions include:

- Full-scale restoration evaluation across all 4,000 held-out images.
- Mathematical convergence and stability analysis of the Jacobi solver.
- Investigation of sensitivity to MVT path sampling parameters.
- Evaluation against adaptive adversarial attacks.
- Comparison with established adversarial purification methods.
- Computational complexity analysis.
- Evaluation on additional traffic sign and medical imaging datasets.
- Investigation of generalization across different CNN architectures.

## 17. Reproducibility

For reproducible experimentation, the following information should be maintained:

- Dataset version and split manifest.
- Model architecture and pretrained weights.
- Training and fine-tuning configurations.
- Adversarial attack parameters.
- MVT sensitivity parameters.
- PDE discretization and solver settings.
- Random seed.
- Software and hardware configurations.

The notebooks include checkpointing and experiment metadata to support reproducibility and recovery after Colab session interruptions.





## 20. Acknowledgement

The author acknowledges the support of the Visvesvaraya Ph.D. Scheme Phase II.

---

Research Project:MVT-SPAR  
Dataset: GTSRB  
Classification Architectures: ResNet-50, VGG16, AlexNet  
Adversarial Attacks: FGSM, PGD-40, BIM, MIM and patch occlusion  
Proposed Defense: MVT-inspired spatially adaptive singular-perturbation-based diffusion  
Numerical Solver: Jacobi finite-difference method  
Implementation: Python, TensorFlow and Google Colab  
Research Status: Experimental implementation
