# Methodology

This section details the design of the diabetic retinopathy (DR) grading framework. The framework classifies fundus images into five ordinal severity grades defined by the International Clinical Diabetic Retinopathy (ICDR) scale: Grade 0 (No DR), Grade 1 (Mild), Grade 2 (Moderate), Grade 3 (Severe), and Grade 4 (Proliferative DR). The pipeline combines circular retina domain cropping, Ben Graham local-average color subtraction, green-channel Contrast Limited Adaptive Histogram Equalization (CLAHE), a ConvNeXt-Tiny feature backbone augmented with Convolutional Block Attention Modules (CBAM), and a Conditional Ordinal Regression for Neural Networks (CORN) classifier head.

## 1. Problem formulation and ordinal classification via CORN

Standard categorical classification using cross-entropy loss treats DR grades as independent nominal classes, ignoring the inherent severity hierarchy. Conversely, single-output scalar regression imposes a rigid metric distance across grades that may not match physiological disease progression. We resolve this by framing 5-class DR grading as a set of $K = 4$ binary decision thresholds using Conditional Ordinal Regression for Neural Networks (CORN).

Let $\mathcal{D} = \{(x_i, y_i)\}_{i=1}^N$ denote a training dataset of fundus images $x_i$ and ordinal labels $y_i \in \{0, 1, 2, 3, 4\}$. The target space is decomposed into $K = 4$ binary sub-tasks corresponding to the rank thresholds $k \in \{0, 1, 2, 3\}$. For a given threshold $k$, evaluation is strictly conditioned on the sub-population of samples that reached or exceeded severity level $k$:

$$S_k = \{i \in \{1, \dots, N\} \mid y_i \ge k\}$$

For each sample $i \in S_k$, the binary target $t_{i,k}$ indicates whether the severity exceeds threshold $k$:

$$t_{i,k} = \begin{cases} 1 & \text{if } y_i > k \\ 0 & \text{if } y_i = k \end{cases}$$

The neural network outputs $K = 4$ raw logits $z(x) = [z_0, z_1, z_2, z_3] \in \mathbb{R}^4$. Applying the logistic sigmoid function $\sigma(\cdot)$ yields conditional probabilities $p_k(x)$:

$$p_k(x) = P(y > k \mid y \ge k, x) = \sigma(z_k)$$

The unconditional probability of exceeding threshold $k$ is computed by cumulative multiplication over preceding conditional probabilities:

$$P(y > k \mid x) = \prod_{j=0}^{k} p_j(x) = \prod_{j=0}^{k} \sigma(z_j)$$

Because $p_j(x) \in (0, 1)$ for all $j$, the product $\prod_{j=0}^k p_j(x)$ is monotonically non-increasing with respect to $k$:

$$P(y > 0 \mid x) \ge P(y > 1 \mid x) \ge P(y > 2 \mid x) \ge P(y > 3 \mid x)$$

This monotonic ordering holds by mathematical construction regardless of network weights, preventing threshold probability uncrossing during inference.

The CORN loss function $L_{\text{CORN}}$ is the average binary cross-entropy across all valid conditional sub-problems:

$$L_{\text{CORN}}(z, y) = \frac{1}{K} \sum_{k=0}^{K-1} \left( \frac{1}{|S_k|} \sum_{i \in S_k} w_{i,k} \cdot \ell_{\text{BCE}}(z_{i,k}, t_{i,k}) \right)$$

where $\ell_{\text{BCE}}(z, t) = - [t \log \sigma(z) + (1-t) \log (1-\sigma(z))]$, and $w_{i,k}$ is a sample-specific weight derived from class imbalance mitigation.

At inference, the predicted ordinal class grade $\hat{y} \in \{0, 1, 2, 3, 4\}$ is obtained by counting the number of unconditional threshold probabilities exceeding $0.5$:

$$\hat{y} = \sum_{k=0}^{K-1} \mathbb{I}\left(P(y > k \mid x) > 0.5\right)$$

To extract a full 5-class probability distribution $P(y = c \mid x)$ for receiver operating characteristic (ROC) analysis and model ensembling, we evaluate:

$$P(y = c \mid x) = \begin{cases} 
1 - P(y > 0 \mid x) & \text{for } c = 0 \\
P(y > c-1 \mid x) - P(y > c \mid x) & \text{for } c \in \{1, 2, 3\} \\
P(y > 3 \mid x) & \text{for } c = 4
\end{cases}$$

## 2. Fundus image preprocessing pipeline

Fundus images collected across clinical environments vary in illumination, field of view, camera sensor properties, and background padding. We execute a deterministic, multi-stage image preprocessing sequence prior to model training to standardize spatial and photometric parameters.

```
Raw Image -> Boundary Detection & Crop -> Pad to Square -> Resize (384x384)
          -> Ben Graham Subtraction -> Green-channel CLAHE -> Circular Mask
          -> Per-source Normalization
```

### 2.1. Spatial cropping and aspect-ratio preservation
Background pixels outside the circular fundus disc are removed using intensity thresholding. Let $I \in \mathbb{R}^{H \times W \times 3}$ denote an input BGR image. A binary mask $M_{\text{bg}}$ is formed by thresholding the grayscale representation $I_{\text{gray}}$ at intensity $T = \max(1, \lfloor 0.08 \times 255 \rfloor) = 20$:

$$M_{\text{bg}}(u, v) = \mathbb{I}(I_{\text{gray}}(u, v) > T)$$

Small noise specks are filtered using morphological opening followed by morphological closing with a $5 \times 5$ square structuring element. The bounding box $(x, y, w, h)$ surrounding the largest external contour of $M_{\text{bg}}$ is extracted. If $w < 0.1 W$ or $h < 0.1 H$, detection is discarded and the full canvas dimensions are used.

The image is cropped to $I[y:y+h, x:x+w]$. To prevent aspect-ratio distortion of retinal structures, the crop is centered onto a square black canvas of dimension $S \times S$, where $S = \max(h, w)$. The padded image is then resized to $384 \times 384$ pixels using area-based interpolation (`cv2.INTER_AREA`).

### 2.2. Illumination correction and contrast enhancement
Uneven clinical lighting and vignetting are suppressed using Ben Graham local-average color subtraction. A Gaussian-blurred image $\mathcal{G}_{\sigma}(I)$ is computed with standard deviation $\sigma = W_{\text{img}} / 10.0 = 38.4$ pixels. The color-corrected image $I_{\text{BG}}$ is generated by weighted blending:

$$I_{\text{BG}} = \text{clip}\left(4.0 \cdot I - 4.0 \cdot \mathcal{G}_{\sigma}(I) + 128.0, \, 0, \, 255\right)$$

Following local color subtraction, Contrast Limited Adaptive Histogram Equalization (CLAHE) is applied exclusively to the green ($G$) channel of $I_{\text{BG}}$. The green channel provides maximum absorption contrast for hemoglobin, enhancing microaneurysms, hemorrhages, and hard exudates. The CLAHE configuration uses a clip limit of $2.0$ over an $8 \times 8$ tile grid.

To eliminate border artifacts introduced by local filtering, a circular mask of radius $R = \lfloor 384 / 2 \rfloor - 2 = 190$ pixels centered at $(192, 192)$ is applied, setting all exterior pixels to zero.

### 2.3. Per-source normalization
Retinal images exhibit systematic distribution shifts depending on the acquisition device. Rather than pooling global dataset statistics, channel-wise mean $\mu_{\text{source}} \in \mathbb{R}^3$ and standard deviation $\sigma_{\text{source}} \in \mathbb{R}^3$ are computed independently for each image source (EyePACS, APTOS 2019, Messidor-2) in RGB order across normalized float values in $[0, 1]$:

$$I_{\text{norm}} = \frac{I_{\text{RGB}} - \mu_{\text{source}}}{\sigma_{\text{source}}}$$

## 3. Network architecture and attention integration

The primary feature extractor is ConvNeXt-Tiny, pre-trained on ImageNet-1k. ConvNeXt-Tiny consists of four hierarchical stages producing feature maps with channel depths $[96, 192, 384, 768]$.

```
Input (3x384x384) -> ConvNeXt Stages 0 & 1 -> Stage 2 (384 ch) + CBAM
                 -> Stage 3 (768 ch) + CBAM -> Projection Head -> CORN Logits (4)
```

### 3.1. Convolutional Block Attention Module (CBAM)
To focus network capacity on localized retinal lesions rather than peripheral background structures, Convolutional Block Attention Modules (CBAM) are integrated into the final two backbone stages (Stage 2 with 384 channels and Stage 3 with 768 channels). Early backbone stages retain uncalibrated spatial representations to preserve generic edge features.

CBAM sequentially applies channel attention $M_c(F)$ and spatial attention $M_s(F')$ to an input feature map $F \in \mathbb{R}^{C \times H \times W}$.

Channel attention aggregates spatial information using global average pooling and global max pooling:

$$F_{\text{avg}}^c = \text{AvgPool}(F), \quad F_{\text{max}}^c = \text{MaxPool}(F)$$

Both descriptors are processed through a shared multi-layer perceptron (MLP) with reduction ratio $R = 16$ and hidden dimension $C_{\text{hidden}} = \max(\lfloor C / R \rfloor, 8)$:

$$M_c(F) = \sigma\left(W_1 \text{ReLU}(W_0 F_{\text{avg}}^c) + W_1 \text{ReLU}(W_0 F_{\text{max}}^c)\right)$$

where $W_0 \in \mathbb{R}^{C_{\text{hidden}} \times C}$ and $W_1 \in \mathbb{R}^{C \times C_{\text{hidden}}}$. The intermediate feature map is $F' = M_c(F) \otimes F$.

Spatial attention pools $F'$ across the channel axis to generate two 2D feature maps, which are concatenated and convolved with a $7 \times 7$ kernel:

$$M_s(F') = \sigma\left(f^{7\times 7}\left([\text{AvgPool}(F'); \, \text{MaxPool}(F')]\right)\right)$$

The final attention-refined feature map is $F'' = M_s(F') \otimes F'$.

### 3.2. Projection head
Features from Stage 3 ($F'' \in \mathbb{R}^{768 \times 12 \times 12}$) enter a regularized projection head:

1. Adaptive Global Average Pooling ($\mathbb{R}^{768 \times 12 \times 12} \to \mathbb{R}^{768}$)
2. Batch Normalization (1D, 768 channels)
3. Dropout ($p = 0.4$)
4. Linear layer ($768 \to 512$)
5. Mish activation function $\text{Mish}(x) = x \cdot \tanh(\text{softplus}(x))$
6. Batch Normalization (1D, 512 channels)
7. Dropout ($p = 0.4$)
8. Linear layer ($512 \to 4$)

The final output is the vector of raw logits $z \in \mathbb{R}^4$ passed to the CORN loss module.

## 4. Class imbalance mitigation via conditional effective sample weighting

Retinal disease datasets exhibit severe class imbalance; Grade 0 (No DR) comprises the vast majority of samples, while Grade 3 (Severe) and Grade 4 (Proliferative) represent small minority fractions. Weighting the 5 nominal classes globally is suboptimal for ordinal decomposition because the sub-population size $|S_k|$ contracts as threshold $k$ increases.

We resolve class imbalance independently within each binary sub-problem $k \in \{0, 1, 2, 3\}$. For sub-problem $k$, the eligible dataset is $S_k = \{i \mid y_i \ge k\}$. Samples in $S_k$ are split into negative instances ($y_i = k$, count $n_{k,0}$) and positive instances ($y_i > k$, count $n_{k,1}$).

We assign weights using the Effective Number of Samples formulation (Cui et al., CVPR 2019) with hyperparameter $\beta = 0.999$:

$$E(n) = \frac{1 - \beta^n}{1 - \beta}$$

The raw class weight for label $c \in \{0, 1\}$ in sub-task $k$ is:

$$w_{k, c}^{\text{raw}} = \frac{1}{E(n_{k, c})}$$

Weights are normalized such that the mean weight over the two sub-classes equals 1:

$$w_{k, c} = \frac{w_{k, c}^{\text{raw}}}{\frac{1}{2} (w_{k, 0}^{\text{raw}} + w_{k, 1}^{\text{raw}})}$$

For a batch sample $i \in S_k$, $w_{i,k} = w_{k, 1}$ if $y_i > k$, and $w_{i,k} = w_{k, 0}$ if $y_i = k$. This structure prevents gradient starvation on high-severity thresholds without destabilizing early thresholds.

## 5. Data augmentation and stochastic regularization

To prevent overfitting and promote domain invariance across clinical sites, we apply stochastic data augmentations, linear interpolation MixUp, and Exponential Moving Average (EMA) weight tracking.

### 5.1. Augmentation pipelines
Augmentations operate directly on normalized float tensors. Two transformation levels are defined:

* **Light Augmentation** (EyePACS phase):
  - Random rotation in $[ -180^\circ, +180^\circ ]$
  - Horizontal flip ($p = 0.5$)
  - Vertical flip ($p = 0.5$)
  - Random affine translation up to $3\%$ of canvas size
  - Random scaling in $[0.95, 1.05]$
  - Color jitter: brightness factor $0.1$, contrast factor $0.1$
  - Gaussian blur: kernel size 3, $\sigma \in [0.1, 1.0]$

* **Heavy Augmentation** (APTOS fine-tuning phase):
  - Random rotation in $[ -180^\circ, +180^\circ ]$
  - Horizontal flip ($p = 0.5$)
  - Vertical flip ($p = 0.5$)
  - Random affine translation up to $6\%$ of canvas size
  - Random scaling in $[0.90, 1.10]$
  - Color jitter: brightness factor $0.2$, contrast factor $0.2$
  - Gaussian blur: kernel size 3, $\sigma \in [0.1, 1.0]$
  - Random erasing ($p = 0.3$, erase area ratio $0.02 \text{ to } 0.08$, aspect ratio $0.5 \text{ to } 2.0$)

We explicitly exclude CutMix, elastic deformation, and synthetic GAN generation, as these operations warp vessel geometry or corrupt localized microaneurysms essential for accurate DR grading.

### 5.2. Stochastic MixUp
MixUp is applied with probability $p = 0.5$ during training iterations. For a pair of samples $(x_i, y_i)$ and $(x_j, y_j)$, a blend parameter $\lambda$ is sampled from a Beta distribution $\text{Beta}(\alpha, \alpha)$ with $\alpha = 0.2$:

$$\tilde{x} = \lambda x_i + (1 - \lambda) x_j$$

The loss is evaluated as a convex combination of loss targets:

$$L_{\text{batch}} = \lambda L_{\text{CORN}}(f(\tilde{x}), y_i) + (1 - \lambda) L_{\text{CORN}}(f(\tilde{x}), y_j)$$

### 5.3. Exponential Moving Average (EMA)
During model training, an auxiliary set of shadow weights $\theta_{\text{EMA}}$ is maintained alongside active model parameters $\theta$. Following each optimization step, shadow weights are updated using decay coefficient $\gamma = 0.999$:

$$\theta_{\text{EMA}} \leftarrow \gamma \theta_{\text{EMA}} + (1 - \gamma) \theta$$

Validation metrics and final model checkpoints are evaluated exclusively using $\theta_{\text{EMA}}$.

## 6. Three-phase staged training protocol

Training is conducted across three sequential phases to transfer representation knowledge from large-scale screening data (EyePACS) to target clinical distributions (APTOS 2019).

```
Phase 1: Frozen Backbone Pre-training (EyePACS, 5 epochs, Head LR=1e-3, bs=384)
   |
Phase 2: Full Backbone Fine-tuning (EyePACS, 35 epochs, Head LR=1.22e-4, Backbone LR=1.22e-5, bs=48)
   |
Phase 3: Target Domain Adaptation (APTOS 2019, 20 epochs, Head LR=1e-4, Backbone LR=1e-6, bs=32)
```

### 6.1. Optimization parameters and weight decay exemption
Optimization uses AdamW. Per standard regularization practice, weight decay $\lambda = 0.01$ is applied exclusively to 2D Convolution and Linear layer weights. Bias vectors, Batch Normalization parameters, and Layer Normalization parameters are explicitly assigned weight decay $\lambda = 0.0$.

### 6.2. Phase execution details

* **Phase 1: Frozen Backbone Pre-training**
  - Dataset: EyePACS train split ($N = 29,837$)
  - Objective: Align random projection head weights prior to full network optimization.
  - Frozen modules: ConvNeXt-Tiny backbone and CBAM parameters.
  - Duration: 5 epochs, batch size 384.
  - Learning rate: $\eta_{\text{head}} = 1 \times 10^{-3}$, $\eta_{\text{backbone}} = 0.0$.
  - Learning rate schedule: Cosine annealing with 1 epoch linear warmup.

* **Phase 2: Full Backbone Fine-tuning**
  - Dataset: EyePACS train split ($N = 29,837$)
  - Objective: Learn domain-general DR representations.
  - Frozen modules: None (all parameters unfrozen).
  - Duration: 35 epochs, batch size 48.
  - Learning rates: $\eta_{\text{head}} = 1.22 \times 10^{-4}$, $\eta_{\text{backbone}} = 1.22 \times 10^{-5}$ (square-root scaled for batch size 48).
  - Weight decay: $\lambda = 0.01$.
  - Learning rate schedule: Cosine annealing with 2 epochs linear warmup (minimum epochs 10, overfitting patience 10 epochs based on validation QWK).

* **Phase 3: Target Domain Adaptation**
  - Dataset: APTOS 2019 train split ($N = 2,562$)
  - Objective: Adapt feature representations to target clinic camera statistics under heavy augmentation.
  - Frozen modules: None.
  - Duration: 20 epochs, batch size 32.
  - Learning rates: $\eta_{\text{head}} = 1.0 \times 10^{-4}$, $\eta_{\text{backbone}} = 1.0 \times 10^{-6}$ ($10\times$ reduced backbone learning rate to prevent catastrophic forgetting of EyePACS representations).
  - Learning rate schedule: `ReduceLROnPlateau` monitoring validation loss (factor 0.5, patience 3 epochs).

## 7. Inference, test-time augmentation, and model ensembling

### 7.1. Test-Time Augmentation (TTA)
During evaluation, each test image $x$ undergoes four deterministic spatial transformations: original, horizontal flip, vertical flip, and $180^\circ$ rotation. For each transformation $m \in \{1, 2, 3, 4\}$, the network produces conditional probabilities $p_{k, m}(x) = \sigma(z_{k, m})$.

The TTA conditional probability $\bar{p}_k(x)$ is the arithmetic mean across all four augmented passes:

$$\bar{p}_k(x) = \frac{1}{4} \sum_{m=1}^{4} \sigma(z_{k, m})$$

Unconditional exceedance probabilities are derived from $\bar{p}_k(x)$:

$$P_{\text{TTA}}(y > k \mid x) = \prod_{j=0}^{k} \bar{p}_j(x)$$

Class predictions are decoded by thresholding $P_{\text{TTA}}(y > k \mid x)$ at 0.5.

### 7.2. Model ensembling
To improve generalization, predictions are ensembled across two distinct network architectures trained through the identical 3-phase protocol: Model A (ConvNeXt-Tiny + CBAM + CORN) and Model B (EfficientNetV2-S + CBAM + CORN).

Ensembling is executed strictly at the marginal class probability level $P(y = c \mid x)$, rather than averaging raw logits or conditional probabilities. Averaging conditional probabilities across heterogeneous architectures invalidates individual CORN monotonic rank guarantees.

For each model $m \in \{A, B\}$, conditional probabilities are converted to full 5-class distributions $P_m(y = c \mid x)$ using the mapping defined in Section 1. The ensemble probability distribution $\bar{P}(y = c \mid x)$ is:

$$\bar{P}(y = c \mid x) = \frac{1}{2} \left( P_A(y = c \mid x) + P_B(y = c \mid x) \right)$$

The final ensemble prediction is selected by argmax over $\bar{P}(y = c \mid x)$:

$$\hat{y}_{\text{ensemble}} = \arg\max_{c \in \{0, 1, 2, 3, 4\}} \bar{P}(y = c \mid x)$$

### 7.3. Primary evaluation metrics
Model performance is evaluated primarily using Quadratic Weighted Kappa (QWK), the standard metric for clinical DR competition benchmarks. QWK measures inter-rater agreement while penalizing classification errors quadratically based on grade distance:

$$\kappa = 1 - \frac{\sum_{i=0}^{4} \sum_{j=0}^{4} w_{i,j} O_{i,j}}{\sum_{i=0}^{4} \sum_{j=0}^{4} w_{i,j} E_{i,j}}$$

where $O_{i,j}$ is the observed confusion matrix cell count, $E_{i,j}$ is the expected cell count under independence, and the quadratic weighting matrix entry $w_{i,j}$ is:

$$w_{i,j} = \frac{(i - j)^2}{(4 - 0)^2} = \frac{(i - j)^2}{16}$$

Secondary metrics include macro-averaged one-vs-rest AUC-ROC, overall accuracy, macro F1-score, and per-class precision/recall/F1 metrics. External validation is performed on Messidor-2 without fine-tuning (zero gradient updates) to test out-of-distribution generalization.
