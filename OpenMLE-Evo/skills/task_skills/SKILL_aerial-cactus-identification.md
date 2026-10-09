# Skill: aerial-cactus-identification

## 1. Task-Specific Reading
- Predict `has_cactus` for each RGB aerial thumbnail. This is binary image classification on uniformly resized 32 x 32 images, not detection or segmentation.
- The object signal is presence/absence of a specific columnar cactus in a small overhead crop. The image already frames the decision region; no bounding boxes, masks, metadata, or sequence context are provided.
- The metric is ROC AUC on predicted probabilities. Optimize ranking quality between positive and negative images. A fixed threshold, accuracy, and hard labels are secondary diagnostics only.
- Because AUC is rank-based, final predictions should be smooth sigmoid probabilities/logit ranks from models and ensembles. Do not round, threshold, or force class-prior matching for the submission.
- The images are tiny. Upsampling to a pretrained backbone input size is useful for transfer learning, but it does not create new detail. Very high resolutions mostly waste compute and can overfit interpolation artifacts.
- Aerial orientation is not semantically fixed. Horizontal flips, vertical flips, 90-degree rotations, mild shifts, and modest color/brightness changes are label-preserving. Aggressive crops/dropout can remove the cactus entirely and corrupt the label.
- Dominant score levers:
  - Strong pretrained visual backbones adapted to tiny images.
  - Stratified OOF validation scored by ROC AUC, not threshold metrics.
  - Fold averaging and small diverse model/seed blends.
  - Geometric TTA using flips/rotations.
  - Conservative augmentation that preserves small-object evidence.
  - Probability/rank calibration for stable ensembling, not threshold tuning.

## 2. Highest-Expected-Score Strategy
- Converge toward a compact, fold-averaged binary ensemble:
  - Train 4-5 stratified folds with one or two strong `timm` backbones.
  - Use `BCEWithLogitsLoss` on a single logit output.
  - Select checkpoints by validation ROC AUC using sigmoid probabilities.
  - Average logits or probabilities across folds, seeds, and TTA. For AUC, logit averaging can preserve rank sharpness; probability averaging is safer when model calibration differs.
- Model family:
  - Primary route: efficient pretrained CNN/ConvNeXt/EfficientNet-style models. They fit small images well, converge fast, and have strong local texture bias.
  - Strong first choices: `tf_efficientnetv2_s.in21k_ft_in1k`, `tf_efficientnet_b0_ns`/`b1_ns`, `convnextv2_tiny.fcmae_ft_in22k_in1k`, `convnextv2_small.fcmae_ft_in22k_in1k`, or a compact EVA/ViT only as a complementary second model.
  - Avoid making a huge 448px ViT the default. For 32px source images, a 160-224px training size with a strong compact backbone usually gives a better accuracy/time/overfit tradeoff. A 256-336px second-stage run can be tested, but treat it as an upgrade, not the first assumption.
- Preprocessing:
  - Read RGB, resize/upscale to the model input, normalize with ImageNet statistics for pretrained weights.
  - Do not crop foreground by threshold as a default. The whole 32 x 32 crop is the unit of evidence; threshold crops can remove contextual cactus texture or distort negatives.
  - Do not convert to grayscale. Color and vegetation/background contrast are useful for aerial discrimination.
- Augmentation:
  - Use full orientation invariance: random horizontal flip, vertical flip, random rotate 90.
  - Use mild `ShiftScaleRotate` or affine transforms with small shifts/scales. Keep crop scale high enough that the central object is not routinely removed.
  - Use mild brightness/contrast and hue/saturation/value jitter; aerial lighting can vary, but over-strong color augmentation can erase cactus/background cues.
  - Use little or no CutMix. CutMix is risky because a pasted patch can add/remove the cactus while retaining the original label. MixUp can be useful with small alpha because AUC tolerates soft labels.
- Inference:
  - Average all fold outputs.
  - Add 4-way or 8-way TTA using identity, flips, and 90-degree rotations. These transformations match the aerial invariance and are high ROI.
  - Keep final values as continuous probabilities. Optional clipping to a small epsilon range is fine for numeric stability, but do not compress ranks.
- Medal-oriented endpoint:
  - A strong final solution should be a 5-fold efficient CNN/ConvNeXt model with flip/rotation TTA, plus one architecturally different compact model or second seed if OOF AUC improves.
  - Blend candidates using OOF AUC and rank correlation. Prefer models that make different validation errors; two near-identical backbones at the same size often add less than one good TTA/fold setup.

## 3. Strong First Implementation Plan
- Build a complete PyTorch single-file solution around one serious binary classifier:
  - Backbone: start with `tf_efficientnetv2_s.in21k_ft_in1k` at 224px, or `convnextv2_tiny.fcmae_ft_in22k_in1k` at 224px if ConvNeXt weights are available. These are strong enough for the final blend and fast enough for 5 folds.
  - Head: one linear binary logit. Use dropout around 0.1-0.3 in the head if replacing the default classifier.
  - Loss: `BCEWithLogitsLoss`. Use soft labels only for MixUp; otherwise targets are 0/1 floats.
  - Validation: 5-fold `StratifiedKFold` on `has_cactus`; 4 folds is acceptable if using a heavier second model. Store OOF probabilities for all train images.
  - Selection: keep the checkpoint with best validation ROC AUC per fold. Do not select by accuracy or loss alone.
- Input and preprocessing:
  - Resize all images to 224 x 224 for the first run. This is enough for ImageNet-pretrained features while limiting interpolation overfit.
  - Normalize with ImageNet mean/std.
  - Use standard RGB arrays and avoid task-agnostic cleaning that changes the tiny crop content.
- First training augmentation:
  - `RandomResizedCrop` or resize plus light crop with scale around 0.85-1.0. Do not use low scale ranges that drop the cactus.
  - `HorizontalFlip`, `VerticalFlip`, `RandomRotate90`.
  - Mild `ShiftScaleRotate`: small shift, small scale, small-to-moderate angle if not already using 90-degree rotation.
  - Mild brightness/contrast and HSV jitter.
  - Optional light blur/noise at low probability; avoid making 32px imagery mushy after upscaling.
  - Optional MixUp with alpha around 0.1-0.2 and low probability. Skip CutMix in the first implementation.
- Training behavior:
  - AdamW with discriminative learning rates: lower LR for backbone, higher LR for head.
  - Cosine schedule with warmup, mixed precision, gradient clipping, and effective batch size large enough for stable batch statistics.
  - Train enough epochs for validation AUC to plateau; tiny images and compact backbones converge quickly, so use early stopping patience on AUC rather than a fixed long run.
  - EMA is a useful low-risk addition if simple to integrate; validate with the EMA weights.
- First inference:
  - For each fold, predict the test set with sigmoid outputs.
  - Use TTA over identity + horizontal flip + vertical flip + both flips, or extend to the eight dihedral transforms if implemented cleanly.
  - Average TTA outputs within each fold, then average folds.
  - Submit continuous probabilities in the sample order. Do not tune a threshold.

## 4. High-ROI Upgrades Across Rounds
- Round 2:
  - Add full 5-fold training if the first run used fewer folds. For this task, fold averaging is usually a higher-confidence gain than complex architecture changes.
  - Compare input sizes 160, 224, and 256 on the same folds. Keep the size with best OOF AUC; do not assume bigger is better for 32px originals.
  - Try a second compact backbone with a different inductive bias: EfficientNetV2 if the first model was ConvNeXt, or ConvNeXtV2 Small/Tiny if the first was EfficientNet.
  - Tune augmentation intensity: crop scale floor, color jitter strength, MixUp probability/alpha, and whether blur/noise helps. Use OOF AUC, not visual preference.
  - Add EMA or stochastic weight averaging if training curves are noisy and validation AUC improves.
- Round 3:
  - Build an OOF-based blend of two model families or two seeds. Optimize blend weights by OOF AUC with a coarse grid, then keep simple weights unless the gain is clear.
  - Add 8-way flip/rotation TTA if not already used. Average logits if the component models are similarly calibrated; average probabilities when blending heterogeneous backbones.
  - Try a slightly larger but still task-fit model such as `convnextv2_small.fcmae_ft_in22k_in1k`, `eva02_small_patch14_336.mim_in22k_ft_in1k`, or EfficientNet-B1/B2 at 224-256px. Use only if OOF AUC improves after accounting for training variance.
  - Add light label smoothing through soft targets only if it improves AUC. For binary BCE, over-smoothing can flatten useful score extremes.
- Late round:
  - Pseudo-label only after a stable fold ensemble exists. Select test images with very high or very low averaged probabilities, add them with low weight or soft labels, and monitor OOF AUC on original labeled folds. Avoid pseudo-labeling ambiguous mid-probability images.
  - Use rank averaging as a robustness check for the final blend. If probability calibration differs across models but each ranks well, averaging ranks can improve AUC stability.
  - Consider a small custom CNN trained from scratch only as a diversity component, not as the main model. It may capture 32px-native texture, but transfer models should remain the backbone of the solution.
  - Review OOF errors by probability buckets if feasible. Use that to adjust augmentation or input size, not to hand-label or inject external information.

## 5. Validation and Metric Optimization
- Use `StratifiedKFold` on `has_cactus`. The label is binary, and every fold must preserve the positive/negative balance for AUC to be meaningful.
- There is no task-described group key. Do not invent filename-based groups. If a reliable source grouping is later present in the actual files, use group-aware stratification; otherwise deterministic stratified folds are the trusted validation.
- Primary score is OOF ROC AUC from continuous sigmoid probabilities over all training images. Track fold AUCs as variance diagnostics, but compare experiments by aggregate OOF AUC and consistent fold-wise movement.
- AUC does not need threshold tuning. Any threshold search is irrelevant for final scoring and can distract from ranking quality.
- Calibration:
  - BCE gives usable probabilities, but AUC only cares about ordering. A model with slightly worse logloss can be better if its positive/negative ranks separate more cleanly.
  - Temperature scaling rarely changes AUC for a single model because it is monotonic, but it can help probability averaging in heterogeneous ensembles. Treat it as a blend aid, not a core score lever.
  - Avoid clipping predictions aggressively. Clipping can create ties and damage AUC if many values saturate.
- OOF usage:
  - Keep OOF predictions for every fold/model/seed/TTA variant.
  - Use OOF AUC to choose input size, augmentations, model families, blend weights, and pseudo-label confidence cutoffs.
  - Blend on raw OOF probabilities/logits before making a final test prediction. Do not choose models by public leaderboard noise when OOF is consistent.
- When local CV and leaderboard diverge:
  - Trust stratified OOF if the pipeline is deterministic and all preprocessing/inference transforms match between validation and test.
  - Suspect over-strong augmentation, input-size overfit, or public split variance before hand-tuning probabilities.
  - Prefer a stable fold ensemble with slightly lower peak public score over a single model chosen by one leaderboard submission.

## 6. Model, Feature, and Preprocessing Priorities
- Binary output:
  - Use one logit and `BCEWithLogitsLoss`.
  - Keep continuous sigmoid probabilities for OOF/test.
  - Do not use two-class softmax unless needed for a pretrained head convention; one-logit BCE is simpler and metric-aligned.
- Backbones:
  - First tier for this task: EfficientNetV2-S, EfficientNet-B0/B1 noisy-student, ConvNeXtV2 Tiny/Small.
  - Second tier/diversity: EVA-02 Small/Base or another compact ViT-style model if available and validated. Use it to diversify, not to replace all efficient CNNs.
  - Large 384-448px models are optional late experiments because source images are 32px.
- Image size:
  - Start at 224px for pretrained compatibility.
  - Test 160/192 for less interpolation and 256/336 for model-specific sweet spots.
  - Avoid treating 448/512 as a default upgrade; validate it because it may amplify artifacts.
- Augmentation:
  - Highest value: flips, 90-degree rotations, mild shift/scale, conservative crop, mild color jitter.
  - Medium value: low-probability blur/noise, small-alpha MixUp.
  - Risky: CutMix, heavy CoarseDropout, aggressive random erasing, low-scale random crops.
- Regularization:
  - Use weight decay, dropout, drop path when supported, and early stopping on AUC.
  - Use balanced sampling or positive class weighting only if class imbalance is severe in the loaded labels and OOF AUC/recall diagnostics show the minority class is under-ranked.
  - Focal loss is not the default for ROC AUC; try it only if BCE ranks hard positives poorly under strong imbalance.
- Inference:
  - Fold averaging first.
  - Flip/rotation TTA second.
  - Model/seed blending third.
  - Rank averaging and pseudo-labeling only after strong OOF-backed models exist.

## 7. Avoid or Delay
- Avoid object detection, segmentation, saliency masks, tiling, or localization-first systems. The task is image-level binary classification on fixed tiny crops.
- Avoid external datasets, private labels, manual relabeling, or assumptions about the original VIGIA images beyond the provided resized thumbnails.
- Avoid training from scratch as the main plan. A scratch CNN can be a late diversity experiment, but pretrained transfer should dominate.
- Avoid high-resolution escalation by habit. Since the source is 32 x 32, 384-512px training can add compute and overfit interpolation without adding signal.
- Avoid aggressive crops, CutMix, large random erasing, or heavy dropout that can remove the cactus from a positive image or insert cactus-like patches into negatives while preserving the wrong label.
- Avoid threshold tuning, F1 optimization, accuracy-based checkpointing, or hard labels in final predictions. ROC AUC rewards ranking of probabilities.
- Avoid overcomplicated calibration as an early focus. Monotonic calibration does not improve single-model AUC; use it only if it improves ensemble averaging on OOF.
- Avoid filename features or ordering assumptions. The image ID is an identifier, not a modeling signal.
- Avoid blindly applying medical, tabular, text, restoration, or fine-grained multi-class playbook elements. There is no metadata, no ordinal target, no patient grouping, and no pixel-restoration target.
- Delay pseudo-labeling until the 5-fold ensemble is stable. Poor pseudo-labels near 0.5 can flatten ranks and harm AUC.
- Delay large heterogeneous ensembles until one compact backbone with clean folds, TTA, and OOF tracking is working. For this small binary task, two well-validated components often beat many loosely tuned ones.
