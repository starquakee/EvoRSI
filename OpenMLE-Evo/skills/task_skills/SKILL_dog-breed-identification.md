# Skill: dog-breed-identification

## 1. Task-Specific Reading
- Predict one of 120 dog breeds from RGB dog photographs. This is fine-grained, single-label, multiclass image classification, not detection, segmentation, retrieval, multilabel tagging, or ordinal grading.
- The labels are breed names, and many classes differ by subtle head shape, coat texture, size, ear shape, and color pattern. The useful signal is often local and high-frequency; low-resolution crops and aggressive occlusion can destroy the distinction between visually similar terriers, hounds, spaniels, retrievers, and toy breeds.
- The metric is multiclass log loss on submitted class probabilities. Optimize calibrated softmax probability quality, not only top-1 accuracy. The final prediction for every image must be a full 120-class probability vector that sums to 1.
- Training data are limited per breed, while test images are unlabeled. The core challenge is transferring strong pretrained visual representations without overfitting the small fine-grained label set.
- No metadata, groups, masks, boxes, text, or tabular side channels are described. The image is the only modeling signal.
- Dominant score levers:
  - Strong pretrained backbones that preserve fine-grained visual detail.
  - Effective resolution, usually 384-448 px for the first serious runs and 512 px only if validated.
  - Stratified OOF validation scored by multiclass log loss.
  - Probability calibration: temperature scaling, conservative label smoothing, and probability averaging.
  - Fold averaging, seed averaging, TTA, and a small diverse ensemble.
  - Augmentations that preserve dog identity and breed-specific details.

## 2. Highest-Expected-Score Strategy
- Converge toward a calibrated high-resolution ensemble:
  - Train 4-5 stratified folds for each selected model family.
  - Use pretrained `timm` backbones fine-tuned on 120-class softmax CE.
  - Save OOF logits/probabilities for every fold and evaluate aggregate OOF multiclass log loss.
  - Learn calibration, blend weights, and any pseudo-label cutoffs from OOF predictions only.
  - Average calibrated probabilities across folds, TTA views, seeds, and architectures for the final submission.
- Model family:
  - Primary high-ceiling choices: `eva02_base_patch14_448.mim_in22k_ft_in22k_in1k`, `eva02_large_patch14_448.mim_m38m_ft_in22k_in1k` if memory/time allow, `convnextv2_base.fcmae_ft_in22k_in1k`, and `convnextv2_large.fcmae_ft_in22k_in1k`.
  - Strong practical alternatives: `eva02_small_patch14_336.mim_in22k_ft_in1k`, `tf_efficientnetv2_s.in21k_ft_in1k`, `tf_efficientnet_b4`/`b5` noisy-student-style models if available, and ConvNeXt V2 Small/Tiny for fast folds.
  - Diversity component: a DINOv2/DINO-style ViT as frozen features plus a strong MLP head, or lightly fine-tuned if validation supports it. This is especially useful because the labeled set is small and fine-grained.
- Resolution:
  - Fine-grained dog breeds reward more detail than coarse image tasks. Treat 384 or 448 px as the default serious training range.
  - Use progressive resizing when training heavier models: warm up at 224/336, then fine-tune at 384/448. This saves time and reduces early overfit.
  - Test 512 px only after the 384/448 pipeline is stable; the gain depends on whether images contain enough dog detail after cropping/resizing.
- Loss and calibration:
  - Cross-entropy is directly aligned with multiclass log loss. Use hard-label CE as the baseline loss.
  - Avoid heavy label smoothing as a default because it can flatten probabilities and harm log loss. Try no smoothing first, then small smoothing around 0.02-0.05 if OOF log loss improves.
  - MixUp can improve generalization but weakens calibration if too strong. Keep alpha low/moderate and validate by log loss.
  - Temperature scaling on OOF/validation logits is high ROI for log loss, especially for overconfident fine-tuned models.
- Inference:
  - Average softmax probabilities, not argmax labels. If averaging logits, apply only within the same architecture/fold family where calibration is similar; probability averaging is safer across heterogeneous models.
  - Use horizontal-flip TTA as the default. Add multi-crop or mild scale TTA only when it improves OOF/holdout log loss. Do not use vertical flips or arbitrary rotations for natural dog photos unless explicitly validated.
- Medal-oriented endpoint:
  - A strong final solution should be a calibrated 5-fold EVA/EVA-small or ConvNeXt/EfficientNet model at 384-448 px, plus one complementary architecture or frozen SSL feature model.
  - The final ensemble should be chosen by OOF log loss and blend diversity, not by public leaderboard noise or top-1 accuracy alone.

## 3. Strong First Implementation Plan
- Build a single complete PyTorch solution around one high-performing multiclass classifier:
  - Backbone: start with `eva02_base_patch14_448.mim_in22k_ft_in22k_in1k` at 448 px if it fits comfortably, or `convnextv2_base.fcmae_ft_in22k_in1k`/`tf_efficientnetv2_s.in21k_ft_in1k` at 384 px for a faster first full run.
  - Head: replace the classifier with a 120-class linear or small MLP head. For CNN backbones, GeM or adaptive concat pooling can help fine-grained recognition; for ViT/EVA models, use the model's correct pooled token representation.
  - Target/loss: integer breed label with `CrossEntropyLoss`. Start with label smoothing 0.0; compare 0.03-0.05 only after the baseline OOF log loss is available.
  - Folds: 5-fold `StratifiedKFold` on breed labels if time allows; 4 folds is acceptable for the first heavy model. Store OOF logits and probabilities.
  - Selection: checkpoint by validation multiclass log loss. Track top-1/top-5 accuracy only as diagnostics for breed confusion.
- Preprocessing:
  - Load RGB and normalize with ImageNet statistics for pretrained weights.
  - Use square resize/crop to the chosen input size. Random resized crop should retain most of the dog: scale floor around 0.75-0.85 is safer than aggressive small crops.
  - Avoid foreground threshold cropping as a default. Dog photos vary in pose and background; naive crops can cut off breed-defining ears, tail, coat outline, or full-body shape.
- Augmentation:
  - Use horizontal flip, mild random resized crop, small shift/scale/rotate, mild perspective/affine, brightness/contrast, hue/saturation/value jitter, and light blur/noise.
  - Use CoarseDropout/RandomErasing sparingly and late or at low probability; masking the face or coat can remove the exact signal needed for similar breeds.
  - Try MixUp with alpha around 0.1-0.2 and probability below 0.5. Delay CutMix until the baseline is stable; patch mixing can create unrealistic dog parts and hurt calibration if overused.
  - Keep augmentation lighter for EVA/DINO/large ViT models than for smaller CNNs.
- Training:
  - Use AdamW, discriminative learning rates, cosine schedule with warmup, mixed precision, gradient clipping, and an effective batch size around 32-64 through accumulation if needed.
  - Briefly train the head frozen for 1-2 epochs only if full fine-tuning is unstable; otherwise unfreeze early with a small backbone LR.
  - Use dropout around 0.1-0.3 and drop path around 0.1-0.2 for larger models. Do not over-regularize until OOF curves show overfit.
  - Use EMA if simple to integrate, validating EMA weights by log loss.
- First inference:
  - Predict each test image with every fold model.
  - Apply identity + horizontal flip TTA first.
  - Average fold/TTA probabilities.
  - Apply one global temperature learned from OOF logits if it improves OOF log loss, then use the same temperature behavior on test logits before softmax.

## 4. High-ROI Upgrades Across Rounds
- Round 2:
  - Move from a partial-fold run to full 5-fold training; fold averaging usually improves log-loss stability more reliably than another small hyperparameter tweak.
  - Compare 384 vs. 448 px on the same folds. Keep the size with better aggregate OOF log loss, not merely better accuracy.
  - Tune probability regularization: no label smoothing vs. 0.03-0.05, MixUp alpha/probability, low-probability CutMix, dropout, and drop path.
  - Add OOF temperature scaling. Search one scalar temperature per model family; optionally compare per-fold temperatures if fold behavior is stable.
  - Add horizontal-flip TTA and optional 3-crop/5-crop center-scale TTA if inference budget permits.
- Round 3:
  - Add a complementary backbone with different inductive bias. If the first model is EVA/ViT, add ConvNeXtV2 or EfficientNetV2; if the first is CNN-heavy, add EVA or frozen DINO-style features.
  - Build an OOF blend. Optimize simple nonnegative weights against OOF multiclass log loss. Use probability averaging after calibration; avoid overfitted weight vectors when gains are tiny.
  - Try progressive resizing for the strongest model: 224/336 warmup, 384 main training, 448 final fine-tune. This is high ROI when training time is tight.
  - Try a frozen SSL feature pipeline: extract DINO-style features at 336-448, train a regularized MLP/logistic head across folds, and blend if its OOF errors differ from the fine-tuned CNN/ViT.
  - Add EMA/SWA/model-soup style checkpoint averaging if validation log loss improves without making probabilities underconfident.
- Late round:
  - Train one high-ceiling model such as EVA-02 Large or ConvNeXtV2 Large for fewer folds or selected folds if full 5-fold cost is too high; blend only if OOF supports it.
  - Use pseudo-labeling cautiously: select test images with very high max probability from the calibrated fold ensemble, add them as soft labels or low-weight hard labels, and retrain. Do not pseudo-label ambiguous breeds near the decision boundary.
  - Use class-aware CutMix or same-class patch augmentation as a late fine-grained regularizer, not a first-round dependency.
  - Consider knowledge distillation from the fold/model ensemble into one strong student only if inference cost becomes limiting. Optimize student log loss against a mixture of hard labels and teacher soft probabilities.
  - Inspect OOF confusions among similar breeds to adjust augmentation/resolution, but do not manually relabel or inject breed-specific external knowledge.

## 5. Validation and Metric Optimization
- Use stratified folds over the 120 breed labels. The task description does not provide a natural group key, so default to deterministic `StratifiedKFold`; use group-aware stratification only if the actual loaded data includes a reliable source grouping.
- The primary validation number is aggregate OOF multiclass log loss over all training images. Fold-wise log loss is a variance diagnostic; aggregate OOF is the metric for model selection, calibration, and blending.
- Always compare experiments on the same folds and same preprocessing. Fine-grained image tasks can have noisy fold movement because each breed has limited samples.
- Log-loss behavior:
  - Penalizes overconfident wrong predictions heavily. A model with slightly lower accuracy can be better if it assigns less extreme probability to confusing breeds.
  - Requires calibrated full probability distributions. Do not submit hard one-hot predictions, rank-only scores, or unnormalized logits.
  - Clipping can prevent catastrophic numeric extremes, but excessive clipping flattens useful confidence. Use only tiny epsilon clipping after probability generation if needed.
- Temperature scaling:
  - Fit temperature on validation/OOF logits by minimizing NLL/log loss.
  - Use temperature separately per model family if calibration differs. For a final blend, calibrate each component before weight optimization.
  - Do not tune temperature on test predictions or by public leaderboard.
- OOF usage:
  - Save OOF logits and probabilities for every fold, architecture, seed, and TTA setting.
  - Use OOF log loss to choose image size, augmentation strength, label smoothing, MixUp/CutMix settings, calibration, blend weights, and pseudo-label confidence thresholds.
  - When leaderboard and OOF diverge, trust OOF if folds are stratified, preprocessing is consistent, and the improvement is stable across folds. Treat public feedback as noisy unless it agrees with OOF.

## 6. Model, Feature, and Preprocessing Priorities
- Backbones:
  - First serious run: EVA-02 Base at 448 or ConvNeXtV2 Base/EfficientNetV2-S at 384.
  - Fast iteration: EVA-02 Small, ConvNeXtV2 Small/Tiny, EfficientNetV2-S.
  - Final diversity: EVA/ViT + ConvNeXt/EfficientNet + DINO-style frozen or lightly fine-tuned model.
- Resolution:
  - 384-448 px is the main operating range.
  - Progressive resizing is preferred over jumping directly to high resolution.
  - 512 px is a late experiment for models that remain stable and fit the budget.
- Pooling/head:
  - CNNs: GeM or adaptive concat pooling plus dropout can improve local fine-grained cues.
  - ViTs/EVA/DINO: use CLS/mean/pooled representation appropriate to the backbone; do not apply CNN pooling to token sequences.
  - Keep the head simple enough to avoid memorizing rare breed examples.
- Augmentation:
  - Highest value: horizontal flip, moderate crop/scale, mild affine, mild color jitter, low-probability blur/noise.
  - Medium value: MixUp, small CutMix, random erasing, same-class patching.
  - Risky: vertical flips, large rotations, aggressive low-scale crops, heavy erasing over face/body, strong color distortions that alter coat/breed cues.
- Calibration/ensembling:
  - Calibrate logits before heterogeneous blending.
  - Average probabilities across folds and architectures.
  - Optimize simple blend weights on OOF log loss; prefer robust equal or near-equal blends when optimized gains are tiny.
- Class balance:
  - Use stratification first. If class counts are uneven enough to hurt minority breeds, try weighted sampling or mild class-balanced sampling, but validate by log loss because oversampling can overconfidently memorize rare classes.

## 7. Avoid or Delay
- Avoid training from scratch as the main plan. The dataset is small for 120 fine-grained classes; pretrained visual representations should dominate.
- Avoid object detection, segmentation, landmark detection, or breed-part localization as the first solution. They add complexity without described labels and are lower ROI than high-resolution pretrained classification.
- Avoid external dog datasets, private labels, manual relabeling, breed taxonomies, or internet-derived breed knowledge as the default plan.
- Avoid optimizing for accuracy, macro-F1, top-k, or thresholded outputs. The competition rewards multiclass log loss from probabilities.
- Avoid heavy label smoothing by default. It may improve accuracy but can underfit probability extremes and hurt NLL.
- Avoid aggressive augmentations that erase breed evidence: vertical flips, strong rotations, extreme crops, high-probability CutMix, large random erasing, and heavy blur.
- Avoid rank averaging as a default for multiclass log loss. Use calibrated probability averaging unless OOF proves a rank-based transform improves NLL.
- Avoid choosing models by a single public leaderboard score when OOF log loss disagrees.
- Delay pseudo-labeling until a calibrated fold ensemble is stable. Poor pseudo-labels for visually similar breeds can reinforce exactly the mistakes that dominate log loss.
- Delay large multi-model ensembles until one clean fold pipeline with OOF predictions, calibration, and TTA is working. A small calibrated diverse ensemble is preferable to many uncalibrated overfit models.
