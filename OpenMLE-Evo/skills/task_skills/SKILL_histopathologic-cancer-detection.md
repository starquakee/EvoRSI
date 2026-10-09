# Skill: histopathologic-cancer-detection

## 1. Task-Specific Reading
- Predict one probability per pathology patch: whether the center 32x32 px region contains at least one pixel of tumor tissue.
- Treat this as binary RGB histopathology patch classification, not whole-slide diagnosis, segmentation, object detection, or "tumor anywhere in image" classification.
- The outer patch context is available to help classification, but the label is determined only by the center region. This makes augmentation policy unusually important: transformations must preserve the central-label semantics.
- The metric is ROC AUC on predicted probabilities. Optimize ranking of positives above negatives. Do not optimize a hard threshold, accuracy, recall, F1, or class labels.
- The dataset is large enough for full fine-tuning of pretrained image backbones on a single GPU, and the images are small enough that expensive WSI tiling is unnecessary.
- No metadata or patient/group identifiers are described. Use stratified image-level folds unless actual group identifiers are discovered in the data. Do not invent groups from hash-like image ids.
- The Kaggle version is described as duplicate-free relative to the original benchmark. Duplicate cleaning is not a main score lever.
- Dominant score levers:
  - Strong pretrained CNN/ConvNeXt/EfficientNet-style backbones fine-tuned on small pathology patches.
  - Center-preserving geometric augmentation plus strong but realistic stain/color augmentation.
  - Stratified folds and OOF ROC AUC for model selection and blend weighting.
  - Fold averaging, D4-style TTA, and small diverse model ensembles.
  - Avoiding label-damaging crops, shifts, CutMix, or erasing that change or obscure the center region.

## 2. Highest-Expected-Score Strategy
- Converge toward a fold-averaged ensemble of strong pretrained binary classifiers:
  - Train 4-5 stratified folds with a single sigmoid logit and `BCEWithLogitsLoss`.
  - Select checkpoints by validation ROC AUC on sigmoid probabilities.
  - Save OOF logits/probabilities for every model, fold, and preprocessing variant.
  - Average fold predictions, then blend diverse backbones using OOF AUC as the arbiter.
- Model family:
  - Highest-value route: efficient high-resolution CNN-family backbones and modern ConvNeXt-style backbones from `timm`.
  - Strong practical starting choices: `tf_efficientnetv2_s.in21k_ft_in1k`, `tf_efficientnet_b3/b4/b5` noisy-student variants if available, `convnextv2_tiny.fcmae_ft_in1k` or `convnextv2_small.fcmae_ft_in22k_in1k`.
  - Strong later choices: `convnextv2_base.fcmae_ft_in22k_in1k`, EVA-02 Small/Base, or DINO-style frozen/fine-tuned encoders if they fit and pretrained weights are available.
  - Prefer full fine-tuning for the main route. The training set is not tiny, and histology stain/domain adaptation matters.
- Resolution:
  - Start at a pretrained-friendly square resolution such as 224 or 256. The native patch is small, so extreme upsampling can add compute without real detail.
  - Try 320/384 only after the baseline is stable; keep it if OOF AUC improves. For small patches, better augmentation and ensembling often beat simply enlarging interpolated pixels.
  - Do not use WSI tile grids or multi-crop region mining; each example is already a small patch.
- Preprocessing:
  - Load RGB and preserve tissue color. Use ImageNet normalization for ImageNet/IN22k-pretrained backbones.
  - Do not crop to tissue bounding boxes by default. Cropping can move or reframe the center region and make the label less aligned.
  - Use stain robustness through augmentation rather than aggressive deterministic stain normalization as the first route.
- Augmentation:
  - Use center-preserving D4 geometry: horizontal flip, vertical flip, 90-degree rotations, and transpose-style transforms. Histology orientation is invariant and the center remains the center.
  - Use color/stain augmentation heavily enough to cover scanner/staining variation: brightness/contrast, hue/saturation/value, mild gamma, CLAHE with low probability, mild blur/noise.
  - Avoid random resized crop, large translation, strong shift-scale-rotate, CutMix, and central CoarseDropout as default choices because they can make "center tumor" labels inconsistent.
- Inference:
  - Use continuous probabilities. Average logits or probabilities across folds; probability averaging is safer across heterogeneous architectures, logit averaging is fine for same-model folds.
  - Use D4 TTA for final models because flips/rotations preserve center semantics and histology has no fixed orientation.
  - Consider rank averaging only for late heterogeneous blends when calibration differs; AUC depends on ordering.

## 3. Strong First Implementation Plan
- Build one complete PyTorch solution around a serious single-model multi-fold classifier:
  - Backbone: `tf_efficientnetv2_s.in21k_ft_in1k` or `convnextv2_small.fcmae_ft_in22k_in1k`; if memory or speed is tight, use ConvNeXtV2 Tiny or EfficientNet-B0/B1 noisy-student as the fast fallback.
  - Input size: 224 or 256 for the first stable run. Move to 320/384 only if OOF AUC improves.
  - Head: one binary logit. For CNN feature maps, GeM or adaptive average pooling plus dropout works well. For ViT/EVA-style models, use the backbone's correct pooled/token representation.
  - Loss: `BCEWithLogitsLoss`. Start unweighted or with only mild positive weighting if folds show imbalance. AUC usually rewards ranking more than aggressive recall weighting.
  - Optimizer: AdamW with discriminative learning rates, small backbone LR, larger head LR, warmup plus cosine decay, mixed precision, and gradient accumulation only as needed for stable batch size.
- Validation:
  - Use 5-fold `StratifiedKFold` when time permits; use 3 folds for first iteration only if faster feedback is needed.
  - Keep the same deterministic folds across all later variants so OOF AUC comparisons and blending are meaningful.
  - Select best checkpoint by validation ROC AUC, not loss alone.
- Training augmentation:
  - Geometry: random horizontal/vertical flips, random 90-degree rotation, and optional transpose. These preserve the central target.
  - Photometric: brightness/contrast, HSV jitter, gamma, mild blur/noise, low-probability CLAHE. Use moderate parameters; tumor/stroma color and texture are real signal.
  - Regularization: dropout/drop path, weight decay, EMA if easy. Delay MixUp until the baseline is stable; delay CutMix unless implemented in a center-aware or very conservative way.
- Inference/postprocessing:
  - Predict sigmoid probabilities for each test patch.
  - Average fold probabilities.
  - Add D4 TTA after the non-TTA model is validated; average TTA probabilities before fold/model blending.
  - Output unclipped or lightly clipped continuous probabilities only to avoid exact saturation. Never threshold.

## 4. High-ROI Upgrades Across Rounds
- Round 2:
  - Train all 5 stratified folds if the first pass used fewer. Fold averaging is a high-return variance reducer for AUC.
  - Add D4 TTA: identity, horizontal flip, vertical flip, 90-degree rotations/transposes if implementation is clean. Keep it if it improves OOF-style holdout or produces stable predictions.
  - Tune stain/color augmentation strength. Compare mild vs. medium HSV/brightness/gamma/CLAHE. Retain the variant with better OOF AUC, not the one that looks visually stronger.
  - Compare input sizes 224/256/320/384 with the same folds and backbone. Use the smallest size that matches or beats larger sizes; the dataset is patch-level, not high-resolution WSI.
  - Tune imbalance handling: unweighted BCE, mild `pos_weight`, balanced sampler. Keep the version that improves aggregate OOF AUC without worsening fold variance.
  - Add EMA or checkpoint averaging if validation AUC is noisy near the end of training.
- Round 3:
  - Add a complementary backbone on the exact same folds. If the first model is EfficientNet, add ConvNeXtV2; if the first model is ConvNeXt, add EfficientNetV2 or EVA/DINO-style features.
  - Blend OOF/test predictions using simple nonnegative weights optimized for OOF ROC AUC. Equal or near-equal weights are often more robust than tightly overfit weights.
  - Try probability averaging vs. logit averaging vs. rank averaging for heterogeneous models. Select by OOF AUC.
  - Test MixUp with small alpha only if BCE targets are mixed correctly and OOF AUC improves. It can help regularization, but it is not as center-safe as D4 plus color augmentation.
  - Train one seed-repeated model if architecture diversity is exhausted; seed averaging can reduce ranking noise.
- Late round:
  - Build a compact 2-3 architecture ensemble: one efficient CNN, one ConvNeXt-family model, and one transformer/SSL-style model if feasible.
  - Use pseudo-labeling only after a stable 5-fold ensemble exists. Select only extremely confident test predictions, use soft labels or low weights, and validate against OOF AUC. Do not pseudo-label borderline patches aggressively.
  - Explore stain-normalization variants only as ensemble diversity, not as a replacement for validated augmentation. Some normalization can remove discriminative color variation.
  - Consider center-aware CutMix or same-class patch mixing only after the strong ensemble is stable. Generic CutMix is risky because labels refer to the center region.
  - Fine-tune the final blend with OOF predictions. Avoid public-leaderboard chasing; use leaderboard only to choose between OOF-validated families.

## 5. Validation and Metric Optimization
- Use stratified folds on the binary target. Every fold must contain enough positives and negatives for ROC AUC to be stable.
- Do not use random train/validation splits with changing seeds for model comparison. Fixed folds are essential for trustworthy OOF blending.
- Do not use group folds unless a real patient, slide, or source identifier is present. Hash-like image ids alone are not a leakage-safe grouping key.
- Primary validation target is aggregate OOF ROC AUC over all training patches. Fold AUC is secondary and should be used to diagnose variance or preprocessing failures.
- AUC behavior:
  - Preserve continuous logits/probabilities.
  - Do not tune thresholds; thresholds do not affect ROC AUC.
  - Avoid rounding and excessive clipping because tied predictions can harm ranking.
  - Calibration is secondary for a single model, but matters for probability averaging and blend stability.
- Model selection:
  - Select checkpoints by validation AUC, with loss as a tie-breaker for stability.
  - Trust changes that improve OOF AUC across multiple folds or improve aggregate OOF without one fold doing all the work.
  - Treat very small gains from TTA, blend weights, or augmentation tweaks skeptically unless they are consistent under the same folds.
- Blending:
  - Use OOF predictions from each candidate model to choose weights.
  - Blend probabilities for similarly calibrated binary models; try logits or ranks only when OOF says they help.
  - Never train a stacker on in-fold predictions. A stacker is optional here because there is no metadata; if used, it should only consume OOF model predictions.

## 6. Model, Feature, and Preprocessing Priorities
- Highest-priority models:
  - EfficientNetV2-S or EfficientNet-B noisy-student variants for fast, strong texture classification.
  - ConvNeXtV2 Small/Base for a strong CNN-family complement.
  - EVA-02 or DINO-style models as later diversity if pretrained weights and memory allow.
- Highest-priority target/loss choices:
  - Single sigmoid binary output.
  - BCEWithLogitsLoss, selected by ROC AUC.
  - Optional mild positive weighting or balanced sampling only if it improves ranking.
- Highest-priority augmentation:
  - D4 center-preserving geometry: flips, rotations by 90 degrees, transpose.
  - Stain/color robustness: brightness, contrast, hue, saturation, value, gamma, CLAHE with controlled probability.
  - Mild blur/noise to simulate scanner and preparation variation.
- Preprocessing priorities:
  - Preserve RGB tissue texture and color.
  - Normalize for the pretrained backbone.
  - Avoid default tissue-crop logic; patch coordinates and center semantics matter more than removing background.
  - If any deterministic preprocessing is tried, compare raw RGB vs. light color normalization using identical folds.
- Training priorities:
  - Full fine-tuning with pretrained weights.
  - Moderate dropout/drop path and weight decay.
  - Stable effective batch size and cosine schedule.
  - Save OOF predictions and fold checkpoints for every useful run.
- Inference priorities:
  - Fold average first.
  - Add D4 TTA second.
  - Add architecture blend third.
  - Submit continuous probabilities with no thresholding.

## 7. Avoid or Delay
- Avoid treating the target as "tumor anywhere in the patch." Outer-region tumor does not define the label.
- Avoid random resized crops, large translations, arbitrary crops, or strong affine shifts as default augmentation. They can move tissue relative to the center and introduce label noise.
- Avoid generic CutMix early. Pasting tumor tissue into or out of the center region can make mixed labels semantically wrong for this competition.
- Avoid large central CoarseDropout or erasing that covers the center 32x32 region. It removes the label-defining evidence.
- Avoid WSI tiling, segmentation pipelines, MIL, or slide-level aggregation as first-round dependencies. The provided examples are already patch-level classification inputs.
- Avoid external pathology datasets, private labels, manual relabeling, or unavailable pretrained resources as the default plan.
- Avoid metadata/tabular, text, audio, ordinal, or regression task transfers. This is binary image AUC.
- Avoid optimizing accuracy, F1, recall, specificity, or a 0.5 threshold. These are diagnostics only.
- Avoid aggressive stain normalization before establishing a raw RGB baseline. It can erase useful color contrast and reduce ensemble diversity.
- Avoid leaderboard-driven blend or augmentation choices that contradict fixed-fold OOF AUC.
- Delay pseudo-labeling, rank-loss experiments, and complex stacking until a strong fold-averaged model with validated TTA exists.
