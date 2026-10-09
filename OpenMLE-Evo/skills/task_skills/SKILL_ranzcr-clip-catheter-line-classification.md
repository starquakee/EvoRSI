# Skill: ranzcr-clip-catheter-line-classification

## 1. Task-Specific Reading
- Predict catheter and line presence/position labels from frontal chest radiographs. This is medical, multilabel, image-level classification, not object detection, segmentation, ordinal grading, or single multiclass classification.
- The targets are independent binary columns for endotracheal tube, nasogastric tube, central venous catheter, and Swan Ganz catheter findings. The placement states include abnormal, borderline, normal, and incompletely imaged where applicable. A study can have multiple positive labels across device families, so use sigmoid outputs, not softmax over all labels.
- The metric is mean ROC AUC: compute AUC independently for each target column, then average. Optimize rank quality per label. Thresholds, argmax choices, hard labels, accuracy, and class-prior matching do not directly optimize the score.
- The visual signal is small and spatially precise: thin radiopaque line/tube courses and tip locations relative to trachea, carina, mediastinum, stomach, diaphragm, or central venous anatomy. Excessive downsampling, aggressive crops, and heavy erasing can remove the decisive evidence.
- The provided labels include `PatientID`; repeated patient exams are a leakage risk. Use patient-grouped validation whenever possible, while preserving multilabel prevalence across folds.
- Training annotations are available for some line/tube segments. Treat them as optional auxiliary localization information, not as the first solution dependency. The core scoring target is image-level AUC.
- Dominant score levers:
  - High-resolution pretrained vision backbones that can preserve thin line structure.
  - Patient-grouped multilabel validation and reliable OOF per-label AUC.
  - BCE-style multilabel training with imbalance-aware sampling or loss only where it improves AUC.
  - Radiograph-specific preprocessing that preserves grayscale contrast and full chest anatomy.
  - Fold averaging, moderate orientation-safe TTA, and a small diverse backbone ensemble.
  - Late pseudo-labeling or annotation-aided localization only after a strong grouped-CV baseline exists.

## 2. Highest-Expected-Score Strategy
- Converge toward a grouped 4-5 fold ensemble of high-resolution multilabel chest X-ray classifiers:
  - Train one sigmoid head over all evaluated labels using `BCEWithLogitsLoss`.
  - Split by `PatientID` to prevent patient-level leakage; approximate multilabel stratification by a compact label pattern or frequency-aware fold assignment.
  - Select checkpoints by mean validation AUC across target labels, not validation loss alone.
  - Store OOF probabilities for every fold/model and use aggregate OOF mean AUC for model selection, blend weights, TTA decisions, and pseudo-label cutoffs.
- Model family:
  - Strong practical first tier: ConvNeXtV2 Base/Small, EfficientNetV2-S/B-style, MaxViT/CoaT-like hierarchical models, or EVA-02 Base if pretrained weights are available. These are good fits for 40k radiographs and 24 GB VRAM.
  - Final ensemble diversity: combine one strong CNN/hierarchical model with one transformer/EVA-style model and one efficient CNN trained at a different resolution. Diversity matters because device tips, line courses, and global chest context are learned differently.
  - Avoid pure frozen features as the main route unless full fine-tuning is unstable. The dataset is large enough that fine-tuning high-quality pretrained backbones should outperform a linear head-only approach.
- Resolution:
  - Treat 384-512 px as the main operating range. Thin catheter lines often require more than 224 px.
  - Start at 384/448 for a serious first run, then test 512 if memory and time allow. Consider progressive resizing: 320/384 warmup, then 448/512 fine-tune.
  - Preserve the whole chest field. Do not crop tightly to the lungs only; CVC/ETT/NGT course and tip context may extend through neck, mediastinum, diaphragm, and upper abdomen.
- Preprocessing:
  - Load radiographs as grayscale and convert to 3 channels for ImageNet/IN22k-pretrained backbones, or let `timm` adapt `in_chans=1` if using a model where pretrained stem adaptation is reliable.
  - Apply conservative contrast handling: percentile clipping or CLAHE can help line visibility, but validate against raw grayscale because over-enhancement can distort portable X-ray artifacts and device edges.
  - Resize with aspect-ratio preservation plus padding when feasible; square warping is acceptable for a first implementation but can slightly alter anatomy. Avoid random crops that remove the tube tip.
- Loss/imbalance:
  - BCE is well aligned with AUC. Use per-label `pos_weight` or focal BCE only if rare labels under-rank positives on OOF; heavy positive weighting can worsen probability ranking for common normal labels.
  - Use soft targets only for MixUp; otherwise keep binary labels as floats. Do not force mutual exclusivity inside a device family during training because labels are provided as independent binary targets.
- Inference:
  - Average sigmoid probabilities across folds and models. For same-architecture folds, logit averaging is also reasonable; for heterogeneous blends, probability averaging is safer.
  - Use TTA that preserves chest radiograph semantics: horizontal flip is usually acceptable only if validation confirms it; vertical flips and 90-degree rotations are not anatomically valid. Mild resize/crop or multi-scale center TTA is safer than arbitrary orientation changes.
  - Final predictions should remain continuous probabilities. Do not threshold.

## 3. Strong First Implementation Plan
- Build one complete PyTorch solution around a high-resolution multilabel classifier:
  - Backbone: `convnextv2_base.fcmae_ft_in22k_in1k` at 384/448 or `tf_efficientnetv2_s.in21k_ft_in1k` at 384 for speed. If EVA weights are available and fit cleanly, `eva02_base_patch14_448.mim_in22k_ft_in22k_in1k` is a strong first-run alternative.
  - Head: one shared image encoder with an 11-logit sigmoid multilabel head matching the evaluated target columns. Use a simple MLP head with dropout around 0.2-0.4; for CNN feature maps, GeM or adaptive average+max pooling is high value.
  - Loss: `BCEWithLogitsLoss`, initially unweighted or with mild clipped `pos_weight` for very rare labels. Track whether weighting improves mean OOF AUC rather than assuming it helps.
  - Folds: 4 or 5 `PatientID`-grouped folds. If implementing perfect multilabel stratified grouping is too much, use deterministic group folds balanced by label sums and patient counts. Do not use random image-level KFold.
  - Selection: checkpoint by mean validation AUC over all target labels; also watch per-label AUC so rare abnormal labels are not sacrificed.
- Input/preprocessing:
  - Read each image, convert to grayscale, normalize intensities with a robust deterministic method, replicate to 3 channels, resize/pad to the chosen size, and normalize for the pretrained backbone.
  - Keep full radiograph coverage with a modest margin. Avoid foreground crops that can cut off the upper airway, lower abdomen, or catheter tip.
  - Optionally create a second preprocessing variant for later blending: raw grayscale vs. CLAHE/contrast-enhanced grayscale. Do not mix variants inside one experiment without OOF tracking.
- Augmentation:
  - Use light-to-moderate geometric augmentation: small shift/scale/rotate, slight perspective/affine, and high-scale random resized crop only if it never removes the line tip. Keep rotations small.
  - Use photometric augmentation tuned for X-rays: brightness/contrast, gamma, mild noise/blur, and occasional CLAHE. Avoid HSV color jitter as a meaningful signal; the source is grayscale.
  - Use CoarseDropout/RandomErasing sparingly or delay it. Masking a small tube tip can corrupt the label signal.
  - Try MixUp with small alpha after the baseline is stable. Delay CutMix; pasted patches can create impossible catheter anatomy and confuse placement labels.
- Training:
  - Use AdamW, discriminative learning rates, cosine schedule with warmup, mixed precision, gradient clipping, and effective batch size around 32 through accumulation if needed.
  - Train all layers after a short head warmup, or fine-tune from the start with a smaller backbone LR. Freeze batch norm only if small per-GPU batches destabilize CNN training.
  - EMA is a low-risk addition if implemented cleanly; validate EMA checkpoints by mean AUC.
  - Run enough epochs for rare-label validation AUC to stabilize; early stopping on mean AUC is better than stopping on BCE loss.
- First inference:
  - Predict all folds, average probabilities, and write the continuous per-label outputs.
  - Add only identity plus a conservative horizontal-flip or multi-scale TTA if OOF/held-out validation shows improved mean AUC.

## 4. High-ROI Upgrades Across Rounds
- Round 2:
  - Move to full 5-fold patient-grouped training if the first run used fewer folds. This is usually higher ROI than small hyperparameter changes.
  - Compare 384, 448, and 512 px on the same fold definition. Keep the size with best aggregate OOF mean AUC and acceptable rare-label behavior.
  - Tune radiograph preprocessing: raw grayscale, percentile clipping, CLAHE, and aspect-ratio padding. Prefer the variant that improves per-label OOF AUC without hurting CVC/NGT tip-sensitive labels.
  - Tune imbalance handling per label: no weights vs. clipped `pos_weight` vs. focal BCE. Keep the simplest setting unless rare abnormal labels clearly improve in OOF ranking.
  - Add EMA/SWA or checkpoint averaging if validation curves are noisy.
- Round 3:
  - Add a complementary backbone: if the first model is ConvNeXt/EfficientNet, add EVA/ViT/MaxViT; if the first is transformer-heavy, add a CNN/hierarchical model. Train on the same grouped folds for clean OOF blending.
  - Build an OOF blend using simple nonnegative weights optimized for mean per-label AUC. Equal or near-equal weights are often more robust than overfit per-label weights.
  - Try TTA variants: horizontal flip only if validated, multi-scale resize/pad, and small center-crop/uncrop averaging. Average probabilities or logits before blending.
  - Use the partial line annotations as auxiliary training only if the classification ensemble is stable: train an auxiliary segmentation/heatmap head or use annotation masks for attention/crop regularization, then keep it only if OOF AUC improves.
  - Try progressive resizing for the strongest backbone: train at 384, fine-tune at 512 with lower LR.
- Late round:
  - Train 2-3 diverse model families at the best validated resolutions and blend by OOF mean AUC. A strong endpoint is a ConvNeXt/EfficientNet model, an EVA/ViT-style model, and a different-resolution CNN.
  - Add pseudo-labeling only from a high-confidence fold ensemble. Use soft pseudo-label probabilities for hidden/test-like images if available in the execution setting; avoid hard pseudo-labels near 0.5 or for labels with weak OOF AUC.
  - Explore per-label blend weights or rank averaging for AUC. Rank averaging can help when model calibration differs, but validate because it can flatten useful probability separation.
  - Consider label-relationship regularization only as a late auxiliary: device-family normal/borderline/abnormal labels are correlated, but the submitted targets are independent AUC columns, so constraints must not destroy per-label ranking.

## 5. Validation and Metric Optimization
- Use patient-grouped validation. `PatientID` exists and repeated patients can share anatomy, devices, acquisition conditions, and clinical context. Image-level random splits can overstate performance and select brittle models.
- Preserve multilabel prevalence across folds:
  - Create folds at the patient level, balancing counts for each target label.
  - If exact multilabel stratified grouping is unavailable, use a greedy assignment over patient-level label sums or stratify on compact label patterns for common patterns and balance rare labels manually.
  - Keep the fold split fixed across model families so OOF blend comparisons are meaningful.
- Primary validation score is aggregate OOF mean ROC AUC over the evaluated target columns. Fold-wise AUC is a variance diagnostic; aggregate OOF is the model-selection and blending target.
- For AUC:
  - Train with BCE logits and evaluate sigmoid probabilities.
  - Do not tune thresholds; thresholds do not affect the submitted metric.
  - Avoid aggressive clipping or rounding that creates tied scores and weakens ranking.
  - Calibration is secondary for a single model because monotonic transforms preserve AUC, but calibration can matter when averaging heterogeneous probabilities.
- Track per-label AUC and positive counts. A high mean can hide failed rare abnormal labels. Prefer changes that improve several weak labels without large regressions on common labels.
- Use OOF predictions for:
  - choosing resolution, preprocessing, augmentation strength, loss weighting, and backbone family;
  - optimizing blend weights;
  - deciding whether TTA helps;
  - selecting pseudo-label confidence cutoffs.
- When local CV and leaderboard diverge, trust patient-grouped OOF if the split is balanced and experiments use the same folds. Suspect public split variance, patient leakage in weaker CV, or preprocessing mismatch before chasing one leaderboard result.

## 6. Model, Feature, and Preprocessing Priorities
- Backbones:
  - First serious model: ConvNeXtV2 Base/Small, EfficientNetV2-S, or EVA-02 Base at 384-448.
  - Fast iteration: EfficientNetV2-S or ConvNeXtV2 Small/Tiny at 320-384.
  - Final diversity: CNN/hierarchical model plus EVA/ViT-style model plus different-resolution efficient model.
- Resolution and anatomy:
  - Use 384/448 as the starting range; test 512 for final models.
  - Preserve full chest, neck base, and upper abdomen. NGT and ETT position can depend on anatomy outside a tight lung crop.
  - Use aspect-ratio padding if practical; avoid geometry that badly distorts tube tip location.
- X-ray preprocessing:
  - Grayscale replicated to 3 channels is a robust default for pretrained RGB weights.
  - Percentile clipping, gamma/contrast augmentation, and CLAHE are worth validating.
  - Avoid color-specific augmentations as core signal; use intensity transforms instead.
- Outputs and loss:
  - Use one sigmoid logit per target label.
  - BCE is the default; focal/weighted BCE is an experiment for rare labels, not mandatory.
  - Keep OOF/test probabilities for every label, fold, TTA, and model. Threshold never enters final prediction.
- Augmentation:
  - Highest value: small affine transforms, brightness/contrast/gamma, mild blur/noise, robust resize/pad.
  - Medium value: MixUp with small alpha, EMA, progressive resizing.
  - Risky: large random crops, strong erasing, CutMix, vertical flips, 90-degree rotations.
- Optional annotation use:
  - Use line annotations to add an auxiliary localization head, attention supervision, or annotation-aware crop validation only after the image classifier is strong.
  - Do not make segmentation mask prediction the main first-round solution; many samples lack annotations and the metric is image-level AUC.

## 7. Avoid or Delay
- Avoid multiclass softmax over placement states as the main formulation. The competition scores independent binary columns, and multiple device families can be positive in one study.
- Avoid image-level random validation splits. Patient leakage can make local AUC look strong while hurting generalization.
- Avoid optimizing accuracy, F1, hard thresholds, or argmax states. Submit continuous probabilities ranked well for each label.
- Avoid external X-ray datasets, private labels, manual relabeling, or internet-derived medical resources as the default plan.
- Avoid low-resolution 224 px as the final intended route. It is useful for debugging, but thin line and tip evidence needs higher effective resolution.
- Avoid tight lung-only crops, aggressive random resized crops, and large erasing masks that can remove the catheter tip or relevant course.
- Avoid orientation-invalid TTA such as vertical flips or 90-degree rotations. Horizontal flip is not automatic; use it only if OOF improves.
- Avoid over-weighting rare labels blindly. Very large `pos_weight` can distort rankings and reduce mean AUC.
- Avoid spending the first round on segmentation, detection, or anatomical landmark systems. Annotation-aided localization is a late upgrade once grouped-CV classification is working.
- Delay pseudo-labeling until there is a stable fold ensemble with trustworthy OOF AUC. Bad pseudo-labels on borderline/abnormal cases can reinforce the exact errors that matter most.
- Delay complex label-constraint postprocessing. AUC rewards per-label ranking, and hard consistency rules can damage useful independent probabilities.
