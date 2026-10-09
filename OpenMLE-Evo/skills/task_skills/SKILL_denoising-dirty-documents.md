# Skill: denoising-dirty-documents

## 1. Task-Specific Reading
- The real task is paired grayscale document restoration: map each noisy scanned text image to its clean pixel-intensity image.
- The target is a full image, not a class label, box, mask category, OCR transcript, or text semantic output. Every output pixel is a continuous value in `[0, 1]`, where 0 is black ink and 1 is white paper.
- The metric is RMSE over cleaned grayscale pixel intensities. Optimize pixel fidelity directly; visually pleasing binarization can lose if it moves antialiased text edges or paper tones away from the clean target.
- Training has aligned noisy/clean pairs. This makes supervised image-to-image regression the main route; unsupervised denoising, OCR-aware modeling, or semantic document understanding should not lead.
- Noise is synthetic document degradation: stains, faded spots, wrinkles, page artifacts, and similar low-frequency/structured corruption. Useful signal is mostly local and pixel-aligned: black text strokes, white background, gray antialiasing, and slowly varying paper/noise fields.
- Dominant score levers:
  - Train on paired noisy-clean grayscale crops with identical geometry and no label-destroying transforms.
  - Use a compact high-resolution denoising network that preserves fine text strokes.
  - Align loss and model selection with full-image RMSE, not only patch loss or subjective sharpness.
  - Use conservative intensity postprocessing and clipping to `[0, 1]`.
  - Blend complementary neural and classical outputs only if OOF/full-image RMSE improves.
  - Avoid over-smoothing, hard thresholding, and resizing artifacts that damage character edges.

## 2. Highest-Expected-Score Strategy
- Converge toward a supervised restoration ensemble centered on residual denoising:
  - Train one or more fully convolutional grayscale-to-grayscale models on random patches from noisy/clean pairs.
  - Predict either the clean image directly or, preferably for stability, the residual/noise correction: `clean = noisy + delta`, clipped to `[0, 1]`.
  - Validate by reconstructing complete held-out images and scoring RMSE over all pixels. Patch validation loss is useful for training curves but not sufficient for model selection.
- Primary model family:
  - Start with a compact U-Net/ResUNet or DnCNN-style residual CNN. This task rewards local denoising and edge preservation more than semantic pretrained classification features.
  - Use encoder-decoder skip connections so the model can keep clean text structure while removing low-frequency stains and wrinkles.
  - A strong architecture is a 1-channel residual U-Net with 32-64 base channels, 4 downsampling stages, residual double-conv blocks, bilinear or PixelShuffle upsampling, skip concatenations, and a final `delta` head.
  - Add channel/spatial attention or SCSE blocks only after a stable baseline. They can help separate background artifacts from text but should not make the first script fragile.
- Classical denoising belongs in the system, but not as the only final plan:
  - Use threshold/morphology/median/Gaussian/background-estimation outputs as baselines, auxiliary inputs, or blend members.
  - Particularly useful classical features are a smoothed background estimate, local contrast-normalized image, median-filtered image, and adaptive-threshold mask. Feed them as extra channels only if validation improves over 1-channel raw input.
  - Avoid committing to hard binary cleanup. The target is grayscale intensity, and antialiasing around characters matters under RMSE.
- Loss:
  - Main loss should be MSE because RMSE is monotonic in MSE over pixels.
  - Add a small L1 or Charbonnier term only if it improves full-image RMSE by preserving strokes; keep MSE as the selection metric.
  - Edge/stroke-weighted auxiliary loss can be useful, but use low weight. Over-weighting text pixels can overfit strokes and worsen the many background pixels that dominate global RMSE.
- Training/inference shape:
  - Keep original aspect ratio and pixel grid for inference. The metric is pixel-aligned; resizing away and back is dangerous unless the network is fully convolutional and predictions are returned at original size.
  - Patch training is usually the best compute route. Use patch sizes around 128-256, with enough context to see stains/wrinkles but enough batches to learn many text/background patterns.
  - For inference on larger images, run the model fully convolutionally if memory allows. If tiling is needed, use overlapping tiles and average with smooth window weights to avoid tile seams in RMSE.
- Medal-oriented endpoint:
  - A strong final solution is a fold-averaged or seed-averaged residual U-Net/DnCNN ensemble, trained with MSE-aligned losses on diverse crops, using flip/transpose TTA where geometry is invertible, optionally blended with a validated classical/background-corrected output.
  - The final blend should be optimized on OOF full-image RMSE using simple nonnegative weights. Do not tune by visual inspection alone.

## 3. Strong First Implementation Plan
- Build one complete PyTorch solution around a supervised residual U-Net:
  - Input: noisy grayscale crop scaled to `[0, 1]`.
  - Optional extra channels for first run only if easy: local mean/background estimate and `noisy - background`. If implementation risk is high, use raw 1-channel input first.
  - Output: one-channel residual correction with clean prediction `clip(noisy + delta, 0, 1)`.
  - Architecture: 4-level U-Net, base channels 32 or 48, residual conv blocks, GroupNorm or InstanceNorm instead of BatchNorm, SiLU/ReLU activations, bilinear/PixelShuffle upsampling, skip connections, final 1x1 or 3x3 conv.
  - Keep the model compact enough to train many crops quickly on one 24 GB GPU. A document denoiser trained well beats a giant semantic backbone trained shallowly.
- Crop sampling:
  - Sample random aligned patches from paired noisy and clean images.
  - Use a mixture of patch types: uniform random patches, high-ink/text patches selected by dark-pixel density, and high-noise/background patches selected by local variance or noisy-clean difference when available in training.
  - Include full-width or larger-context crops if image dimensions allow; stains and wrinkles are partly low-frequency, and very tiny crops can learn only local smoothing.
- Augmentation:
  - Safe geometric transforms: horizontal flip, vertical flip, 90-degree rotations/transposes if applied identically to noisy and clean. Text orientation is not semantically relevant for pixel restoration because target undergoes the same transform.
  - Conservative intensity augmentation on the noisy input only can improve robustness: mild brightness/contrast/gamma, small Gaussian noise, low-probability blur. Do not alter the clean target except through shared geometry.
  - Avoid random resized crops that change scale. Document pixel structure and antialiasing are metric-sensitive.
- Loss and optimization:
  - Use MSE on predicted clean pixels as the primary loss.
  - Consider `loss = MSE(clean_pred, clean) + 0.1 * L1(clean_pred, clean)` for a first robust run. If local RMSE worsens, return to pure MSE.
  - Optionally add a tiny gradient loss on Sobel differences to protect text edges, but never let it dominate MSE.
  - Use AdamW or Adam, cosine schedule or reduce-on-plateau, mixed precision, and EMA if simple. Select checkpoints by validation full-image RMSE.
- Validation:
  - Hold out complete images, not random patches from every image. Patch-level random splitting leaks page/font/noise patterns into validation and overstates performance.
  - Use 5 folds if the train image count is small enough to train quickly; otherwise start with one deterministic holdout and later expand to folds/seeds.
  - For each validation image, run the exact inference path, reconstruct at original size, clip to `[0, 1]`, then compute global RMSE over all validation pixels.
- Inference/postprocessing:
  - Predict each test image at original resolution. If tiling, use overlap of at least 32-64 pixels and average overlaps smoothly.
  - Clip final predictions to `[0, 1]`. This is metric-safe because targets lie in that interval.
  - Do not hard threshold the final image. If threshold-like cleanup is tested, blend it softly with the neural prediction and keep it only when OOF RMSE improves.

## 4. High-ROI Upgrades Across Rounds
- Round 2:
  - Add full-image OOF validation if the first run used a single holdout. Use identical image folds for all later experiments so RMSE comparisons are meaningful.
  - Improve crop sampling: oversample text-edge patches, high-noise patches, and mostly-background stained patches. The model must learn both stroke preservation and background whitening.
  - Compare direct clean prediction vs. residual correction with the same architecture. Residual usually stabilizes training because most pixels should remain near the noisy input except corrupted areas.
  - Tune patch size: 128 for speed/local detail, 192-256 for stains/wrinkles/context. Keep the size that wins full-image RMSE, not just patch loss.
  - Add EMA or top-checkpoint weight averaging. Restoration losses are smooth, and averaged weights often reduce pixel noise without inference cost.
  - Try a validated classical postprocess or blend: neural prediction, median/bilateral filtered output, and background-corrected output. Optimize a simple OOF weight such as `a * neural + b * classical + c * noisy`.
- Round 3:
  - Add a second model family for diversity:
    - DnCNN/residual plain CNN with many 3x3 layers and no downsampling for crisp local denoising.
    - A deeper ResUNet with attention/SCSE for structured stains and wrinkles.
    - A lightweight pretrained-encoder U-Net only if pretrained weights are available locally and it improves OOF RMSE; it should not replace the document-specific denoiser by default.
  - Add TTA using invertible transforms: horizontal flip, vertical flip, and transpose/rot90 if the model is fully convolutional. Invert outputs before averaging. Keep TTA only if OOF RMSE improves.
  - Train multiple seeds/folds and average predictions. For pixel RMSE, independent denoisers often average out small artifacts.
  - Add multi-scale context without resizing the final output: feed an additional downsampled/upsampled background channel or use a larger receptive field branch.
- Late round:
  - Optimize blend weights on OOF predictions across neural models, classical outputs, and possibly the raw noisy input. Use nonnegative weights and clipping; avoid high-variance unconstrained weights on few validation images.
  - Add pseudo-labeling only cautiously. Because train already has paired clean targets, pseudo-labeling test images with a teacher can help as self-distillation but can also freeze teacher artifacts. Use soft teacher targets with low weight and retain supervised train pairs as the anchor.
  - Try edge-aware calibration: slight local contrast restoration or black-text sharpening only if global OOF RMSE improves. Many visually sharper outputs overshoot clean antialiased pixels.
  - Add a small validation-driven intensity calibration: fit a global affine correction or per-prediction-bin bias on OOF residuals, then apply to test predictions. Keep it simple to avoid leaderboard overfit.

## 5. Validation and Metric Optimization
- Split by complete image/page. Never put patches from the same source image in both train and validation.
- Use the exact competition metric locally: RMSE over all validation pixels after reconstruction and clipping. Since RMSE is `sqrt(MSE)`, minimizing pixel MSE is metric-aligned, but checkpoint and blend selection should report RMSE on reconstructed images.
- Track RMSE by image and by rough pixel regions:
  - Mostly white/background pixels, which dominate pixel count.
  - Dark text pixels, where over-smoothing or thresholding damages strokes.
  - Edge/gray pixels, where antialiasing errors are common.
  - High-noise/stain areas, where denoising gains are largest.
- Trust experiments that improve aggregate full-image OOF RMSE and do not create large per-image regressions. A tiny global gain caused by one image can be unstable with small document datasets.
- Use OOF predictions for all blend and calibration decisions:
  - Blend continuous pixel predictions before clipping when possible, then clip once.
  - Optimize simple scalar weights by OOF RMSE. Equal weights are often competitive; prefer them unless optimized weights give a clear, fold-stable gain.
  - If using affine calibration, fit on OOF residuals only and keep coefficients conservative.
- Validate inference tiling and TTA on held-out full images. Tile boundary artifacts can be invisible in patch loss but visible in pixel RMSE.
- When local CV and leaderboard differ, prefer image-level OOF unless the validation split is clearly too small or nonrepresentative. Public feedback can be noisy for restoration tasks; do not hand-tune thresholds or blend weights without OOF support.

## 6. Model, Feature, and Preprocessing Priorities
- Highest-value model choices:
  - Residual U-Net: best first complete route for mixed local text and broader page artifacts.
  - DnCNN/residual plain CNN: strong complementary model for local noise and stroke sharpness.
  - ResUNet with attention/SCSE: useful upgrade for structured stain/wrinkle removal.
  - Classical denoisers/background correction: useful as features or blend members, not the sole medal-level plan.
- Highest-value inputs/features:
  - Raw noisy grayscale intensity.
  - Smoothed background estimate from large-kernel blur/median filtering.
  - Local contrast or residual-to-background channel.
  - Optional soft foreground/text mask from adaptive thresholding, used as an input feature or diagnostic, not as a hard final mask.
- Preprocessing priorities:
  - Preserve original pixel grid and grayscale scale.
  - Normalize consistently to `[0, 1]`; train and target must use the same intensity convention as the metric.
  - Use padding modes that avoid artificial dark/white borders in patches and tiles, such as reflection padding.
  - Avoid resizing for training except extracting same-scale crops. If a model needs divisible dimensions, pad and crop back rather than resample.
- Training priorities:
  - Balanced crop sampler over text, background, and artifact-heavy areas.
  - MSE-first objective with optional small L1/edge auxiliary.
  - EMA or checkpoint averaging.
  - Full-image validation and OOF prediction storage.
- Inference priorities:
  - Fully convolutional original-size prediction when possible.
  - Overlap-averaged tiling when needed.
  - Invertible TTA averaged in pixel space.
  - Clip to `[0, 1]` at the end.

## 7. Avoid or Delay
- Avoid image classification backbones with scalar heads. The output is a dense grayscale image, and global semantic features do not solve pixel restoration.
- Avoid OCR, language modeling, character recognition, segmentation labels, or transcript reconstruction as the primary path. The metric never reads text; it scores pixels.
- Avoid hard binarization as the default final output. Clean targets are grayscale, and RMSE penalizes lost antialiasing and paper-tone variation.
- Avoid aggressive smoothing, median filtering, morphological opening/closing, or background whitening unless OOF RMSE improves. These can remove stains but also erase thin strokes.
- Avoid optimizing SSIM, perceptual loss, adversarial loss, or visual sharpness as the main objective. They may produce nicer images with worse pixel RMSE.
- Avoid random patch validation split leakage. It will make the model look strong locally while failing on new pages.
- Avoid scale-changing augmentations and resampling-heavy preprocessing. Pixel alignment and character edge widths matter.
- Avoid external datasets, private clean images, manual labels, or unavailable pretrained restoration weights as the default route.
- Delay pseudo-labeling and self-supervised denoising until a supervised paired-patch denoiser and OOF blend are already strong.
- Delay heavy pretrained ViT/semantic encoders unless there is clear local evidence they improve dense RMSE. For this task, inductive bias and pixel alignment usually beat semantic capacity.
- Avoid public-leaderboard-driven blend weights or threshold choices. Use OOF full-image RMSE as the anchor.
