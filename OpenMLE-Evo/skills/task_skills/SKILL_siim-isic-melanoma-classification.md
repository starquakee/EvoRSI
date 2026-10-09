# Skill: siim-isic-melanoma-classification

## 1. Task-Specific Reading
- Predict whether each dermoscopic lesion image is malignant melanoma. The submitted value is one continuous probability per `image_name`.
- Treat this as binary medical image classification with structured metadata and patient grouping, not generic object recognition. The important clinical framing is "which lesion is suspicious for this patient," so patient-level context can be signal as well as leakage risk.
- The metric is ROC AUC between predicted probability and the binary target. Optimize ranking of malignant above benign lesions. Do not optimize hard thresholds, accuracy, F1, or class labels.
- The modality is RGB skin lesion imagery plus metadata: patient identifier, sex, approximate age, anatomic site, and training-only diagnosis/malignancy fields. Images are available in medical/image formats; model the visual lesion, not DICOM mechanics.
- The positive class is rare and clinically subtle. A model that ranks a few malignant lesions correctly beats one with better calibration but weaker ordering. Preserve continuous logits/probabilities through every fold, TTA, blend, and stack.
- Dominant score levers:
  - High-resolution fine-tuning of strong pretrained image backbones because melanoma cues are small, color/texture-driven, and fine-grained.
  - Patient-grouped, target-stratified validation to avoid repeated-patient leakage and to measure the exact AUC objective.
  - Metadata fusion or second-stage tabular stacking over OOF image predictions, especially age, sex, anatomic site, and patient-level count/context features.
  - Careful class imbalance handling that improves ranking without flooding validation with false-positive benign lesions.
  - Fold/model averaging, moderate dermoscopy-safe TTA, and OOF-tuned blend weights.

## 2. Highest-Expected-Score Strategy
- Converge toward a patient-grouped ensemble of high-resolution image classifiers plus a metadata/image-prediction stacker:
  - Train 4-5 folds with `patient_id` groups and target stratification as well as possible.
  - Use one sigmoid logit for malignant probability. Train with BCE-style loss; evaluate and select by validation AUC.
  - Save OOF probabilities/logits for every image from every model. These OOF predictions are the foundation for trustworthy stacking, blending, and patient-context features.
  - Train a small tabular model on metadata plus OOF image predictions, then average/blend its test predictions with the raw image ensemble.
- Model family:
  - Strong practical first tier: EfficientNetV2-S/B-style, EfficientNet noisy-student B4/B5, ConvNeXt/ConvNeXtV2 Small/Base, EVA-02 Base/Small, or another available high-quality `timm` pretrained model.
  - Final diversity: combine one high-resolution efficient CNN, one ConvNeXt-family model, and one transformer/SSL-style model if feasible. Diversity improves AUC more reliably than small hyperparameter changes after the first strong model.
  - Prefer full fine-tuning over frozen features for the main route unless data/time instability forces a frozen encoder. The task has enough images and fine-grained domain shift to benefit from careful full-model adaptation.
- Resolution:
  - Start serious training around 384 px; move best models to 448/512 px if memory and time allow.
  - Do not settle on 224 px as the intended final route. Dermoscopy AUC often depends on asymmetric borders, color variegation, pigment networks, and small structures that downsampling weakens.
  - Use progressive resizing when training the strongest models: warm up at lower resolution, then fine-tune at the final resolution with lower learning rate.
- Image preprocessing:
  - Preserve lesion color and surrounding skin context. Avoid aggressive color normalization that erases diagnostic variation.
  - Use a conservative lesion/foreground crop only if it reliably removes black rulers/borders or empty background without cutting off lesion edges. A bad crop is worse than square resizing.
  - Normalize for the pretrained backbone. Photometric augmentation should mimic acquisition variation, not pathology changes.
- Metadata and patient context:
  - Encode sex, age, anatomic site, missingness flags, and patient lesion counts.
  - Use image-model probability/logit as the strongest numeric feature in a CatBoost/LightGBM-style stacker.
  - Add patient-level contextual features computed from predictions at inference: per-patient count, rank of a lesion's image score within patient, max/mean/std score by patient, and difference from patient mean/max. Build analogous features on OOF predictions for training so the stacker sees leakage-safe context.
  - Metadata alone should not dominate final predictions, but it can correct visually ambiguous cases and improve ranking within demographic/site strata.
- Inference:
  - Average sigmoid probabilities or logits across folds. For homogeneous folds, logit averaging is fine; for mixed architectures and stackers, probability/rank blending is safer.
  - Use dermoscopy-valid TTA: horizontal/vertical flips, mild scale/crop, and possibly 90-degree rotations if validation supports invariance. Average probabilities before stacking/blending.
  - Submit continuous probabilities. Do not threshold.

## 3. Strong First Implementation Plan
- Build one complete PyTorch solution with a single strong image backbone and a lightweight metadata stack:
  - Backbone: `tf_efficientnetv2_s.in21k_ft_in1k`, `tf_efficientnet_b4_ns`, `convnextv2_small.fcmae_ft_in22k_in1k`, or `convnextv2_base.fcmae_ft_in22k_in1k` depending on available pretrained weights and speed. Use 384 px for the first robust run; 448 px if it fits comfortably.
  - Head: one sigmoid binary output. For CNN feature maps, GeM or adaptive concat pooling plus dropout is high value. For ViT/EVA-style backbones, use the correct pooled/token representation from the model.
  - Loss: `BCEWithLogitsLoss` with either mild positive weighting or balanced sampling. Validate the imbalance choice by AUC; excessive weighting often improves recall-looking behavior while hurting rank quality.
  - Folds: 4 or 5 `StratifiedGroupKFold`-style folds grouped by `patient_id` and stratified by target. If exact stratified grouping is unavailable, use deterministic grouped folds balanced by malignant counts and total samples.
  - Checkpoint selection: validation AUC on sigmoid probabilities. Keep the best checkpoint per fold and store OOF logits/probabilities.
- Preprocessing and augmentation:
  - Load RGB image, optionally crop only obvious non-lesion border/background with margin, resize to target square, normalize for ImageNet/IN-pretrained weights.
  - Use moderate geometric augmentation: random resized crop with high retained scale, flips, small shift/scale/rotate, and mild perspective/affine only if it does not distort lesions excessively.
  - Use photometric augmentation: brightness/contrast, hue/saturation/value, mild blur/noise, occasional CLAHE. Keep color shifts modest because color is diagnostic.
  - Use CoarseDropout lightly; avoid erasing the entire lesion. Delay CutMix. MixUp with small alpha can help imbalance regularization if BCE targets are mixed correctly.
- Metadata stack in the first implementation:
  - Encode `sex` and `anatom_site_general_challenge` as categoricals, keep `age_approx` numeric, add missing indicators, and add patient lesion count.
  - Train a CatBoost or LightGBM binary classifier per fold or on full OOF-ready features using only leakage-safe OOF image predictions for training.
  - For test, use averaged image predictions to create equivalent features. Add patient score aggregates and within-patient rank/difference features if implemented cleanly.
  - Blend image ensemble and stacker output by OOF AUC. A simple weighted average such as mostly image plus some stacker is a robust starting point; tune weights on OOF, not leaderboard feedback.
- First inference/postprocessing:
  - Average fold predictions from the image model.
  - Run the metadata stacker using the averaged image score and patient/context features.
  - Output blended continuous probabilities clipped only to a sane open interval if needed to avoid exact 0/1 saturation. Clipping is for numerical stability, not thresholding.

## 4. High-ROI Upgrades Across Rounds
- Round 2:
  - Move from 384 to 448/512 px for the best backbone or use progressive resizing. This is usually the highest-return image upgrade for fine-grained dermoscopy.
  - Train all 5 patient-grouped folds if the first run used fewer folds. Fold averaging reduces variance on rare malignant examples.
  - Tune imbalance handling: unweighted BCE vs. clipped `pos_weight` vs. balanced sampler vs. focal BCE. Keep the version with best OOF AUC and stable fold-wise behavior.
  - Compare preprocessing variants: raw resize, conservative lesion crop, and crop plus mild color/contrast normalization. Retain only variants that improve OOF AUC.
  - Add simple TTA: identity plus flips, optionally multi-scale center crop. Keep TTA if it improves OOF/holdout AUC or stable fold predictions.
- Round 3:
  - Add a complementary backbone on the same patient-grouped folds. If the first model is EfficientNet, add ConvNeXt/EVA; if ConvNeXt, add EfficientNet/EVA.
  - Build an OOF blend of model probabilities/logits and the metadata stacker. Use simple nonnegative weights optimized for OOF AUC; equal or near-equal weights are often more robust than overfit weights.
  - Strengthen the stacker: add image logits from each fold/model, patient-level score aggregates, site/age interactions, missingness flags, and categorical handling through CatBoost or LightGBM.
  - Try rank averaging for heterogeneous models. Since AUC depends only on ranking, rank blending can help when calibration differs strongly across architectures.
  - Add EMA/SWA or checkpoint averaging if training curves are noisy and OOF AUC improves.
- Late round:
  - Train a compact 2-3 architecture ensemble at best validated resolutions: efficient high-res CNN, ConvNeXt-family model, and EVA/ViT/SSL-style model if feasible.
  - Use pseudo-labeling only after a stable fold ensemble exists. Select extremely confident test predictions far from the malignant/benign ambiguity region; use soft labels or low weights. Do not pseudo-label borderline lesions aggressively.
  - Explore AUC-oriented pairwise/ranking loss only as an auxiliary to BCE if the implementation is already stable; BCE with good validation and ensembling is the reliable route.
  - Tune blend weights and stacker features using OOF predictions. Avoid public-leaderboard weight chasing, especially because patient/test composition can shift.

## 5. Validation and Metric Optimization
- Use patient-grouped validation. Multiple lesion images can belong to the same patient, and random image-level folds can leak patient appearance, acquisition style, age/site distribution, and patient lesion context into validation.
- Preserve malignant prevalence in every fold. The positive class is rare; a fold with too few positives gives noisy AUC and unreliable model selection.
- Primary validation target is aggregate OOF ROC AUC over all training images. Fold AUC is useful for variance diagnostics, but OOF AUC is the correct basis for model selection, blending, stacker features, and TTA decisions.
- For AUC:
  - Train/evaluate continuous sigmoid probabilities or logits.
  - Do not tune thresholds; thresholds have no effect on submitted AUC.
  - Avoid rounding and avoid excessive clipping that creates tied predictions.
  - Calibration is secondary for a single model because monotonic transforms preserve AUC, but calibration can affect probability averaging and stacker behavior.
- Build stackers only from OOF image predictions. Training a tabular stacker on in-fold image predictions is leakage and will overstate CV.
- Patient-context prediction features must be OOF-safe:
  - For training, compute per-patient ranks/aggregates from OOF predictions only.
  - For test, compute them from averaged test predictions.
  - Do not use true diagnosis, benign/malignant text fields, or target-derived patient aggregates outside the training label.
- Trust improvements that raise OOF AUC across the same grouped folds and do not depend on one unstable fold. If local CV and leaderboard diverge, prefer patient-grouped OOF unless the fold construction is clearly unbalanced.
- Inspect score distributions by target and by metadata strata. Useful changes should improve separation of malignant vs. benign, especially in visually hard, common benign-looking regions, not only shift all scores upward.

## 6. Model, Feature, and Preprocessing Priorities
- Backbones:
  - First serious model: EfficientNetV2-S/B4/B5, ConvNeXtV2 Small/Base, or EVA-02 Small/Base at 384-448.
  - Final blend: one efficient CNN, one ConvNeXt, and one transformer/SSL-style model when feasible.
  - Use pretrained weights whenever available; training dermoscopy classifiers from scratch is not competitive under the budget.
- Resolution:
  - 384 px is the minimum serious starting point.
  - 448/512 px is preferred for final models if batch size, time, and stability allow.
  - Preserve lesion boundary and local texture; do not crop so tightly that surrounding skin/asymmetry context disappears.
- Metadata:
  - High-value raw fields: age, sex, anatomic site, missingness indicators, patient lesion count.
  - High-value derived fields: age bins/interactions with site/sex, site frequency/count encoding, patient image count, image score rank within patient, difference from patient mean/max image score.
  - Best use: second-stage CatBoost/LightGBM stacker fed by OOF image logits/probabilities plus metadata. Direct neural metadata fusion is useful but often less flexible than OOF stacking for this task.
- Augmentation:
  - Highest value: flips, small affine transforms, random resized crop with large scale, brightness/contrast, HSV jitter, mild blur/noise.
  - Medium value: MixUp, light CoarseDropout, EMA, progressive resizing.
  - Risky: large CutMix, extreme color jitter, heavy blur, strong distortions, or crops that remove lesion edges.
- Loss and sampling:
  - BCE is the default for AUC. Focal or weighted BCE is an experiment for class imbalance, not automatically better.
  - Balanced sampling can help expose positives every epoch, but validate that it improves ranking and does not over-rank benign lookalikes.
- Outputs:
  - Keep logits/probabilities per fold/model/TTA.
  - Average/blend continuous outputs.
  - Rank averaging is a legitimate late AUC optimization when model calibrations differ.

## 7. Avoid or Delay
- Avoid image-level random validation. It is leakage-prone when patients have multiple lesion images and will select brittle models.
- Avoid optimizing accuracy, F1, sensitivity at a fixed threshold, or hard malignant/benign labels. The competition rewards probability ranking by ROC AUC.
- Avoid external lesion datasets, private labels, manual relabeling, internet-derived resources, or unavailable pretrained assets as the default plan.
- Avoid using training-only diagnosis or benign/malignant text fields as inference features. They are labels/label proxies, not test-time covariates.
- Avoid metadata-only or tabular-only solutions as the main route. Metadata is a strong complement, but the primary signal is dermoscopic image evidence.
- Avoid making segmentation or lesion detection the first-round dependency. Cropping can help, but a full segmentation pipeline is lower ROI than high-resolution classification plus stacking.
- Avoid DICOM-specific engineering as the core modeling plan unless image quality from provided RGB images is demonstrably poor. The scoring problem is classification, not medical file parsing.
- Avoid aggressive color normalization that removes clinically meaningful pigmentation differences.
- Avoid heavy CutMix/erasing before a stable baseline; it can remove the lesion signal while leaving the label unchanged.
- Delay pseudo-labeling until OOF validation, fold averaging, and metadata stacking are stable.
- Delay complex calibration and threshold logic. Calibration can help stacking, but thresholds do not improve AUC and should not drive final predictions.
