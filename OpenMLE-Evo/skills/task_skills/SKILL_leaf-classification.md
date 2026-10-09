# Skill: leaf-classification

## 1. Task-Specific Reading
- Predict one of 99 plant species for each leaf specimen. This is balanced, fine-grained, single-label multiclass classification, not detection, segmentation, multilabel tagging, or ordinal prediction.
- The strongest native signal is the provided numeric descriptor table: 64 margin features, 64 shape features, and 64 texture features per image. These are already task-specific morphology/texture summaries and should dominate the first solution.
- Binary leaf images are available, but the labeled set is tiny: approximately 16 samples per species overall, with only a small number of labeled examples per class. A full deep image model can overfit easily. Use images mainly for additional deterministic contour/region features, optional frozen representations, or a carefully validated low-weight model in the ensemble.
- The metric is multiclass logarithmic loss over submitted class probabilities. Optimize calibrated full probability vectors, not top-1 accuracy. One overconfident wrong class can cost more than several modest ranking mistakes.
- Classes are many and visually close. Useful signal comes from subtle outline, margin, shape, venation/texture, and relative descriptor patterns. Preserving feature scale and block structure is more important than generic feature explosions.
- Dominant score levers:
  - Strong preprocessing of the three descriptor blocks: scaling, variance stabilization, and optional block-wise dimensionality reduction.
  - Calibrated classical multiclass models suited to small-N, medium-dimensional data: multinomial logistic regression, RBF/poly SVM, ExtraTrees/RandomForest, kNN, and regularized GBDTs as diversity components.
  - OOF-based probability blending/stacking optimized directly for multiclass log loss.
  - Deterministic image-derived shape features if they add OOF signal beyond the provided descriptors.
  - Conservative probability calibration, clipping, and row normalization.

## 2. Highest-Expected-Score Strategy
- Converge toward a calibrated descriptor-first ensemble with optional image-derived features:
  - Use the 192 provided features as the primary representation.
  - Split features by prefix into margin, shape, and texture blocks; preprocess each block separately before concatenation.
  - Train many low-to-medium variance models on identical stratified folds, collect OOF probabilities, and optimize blend weights against OOF multiclass log loss.
  - Add image-derived contour/region moments only if they improve OOF log loss on the same folds.
  - Treat CNN/frozen image features as ensemble diversity, not the core plan, unless OOF proves image-only predictions are competitive.
- Best model family for this task:
  - Primary: regularized multinomial LogisticRegression on standardized/PowerTransformed descriptors. It is often very strong for small balanced multiclass descriptor data and produces usable probabilities.
  - Primary diversity: SVC/NuSVC with RBF and polynomial kernels, calibrated via fold validation or `probability=True` only when OOF log loss confirms calibration. SVMs can capture curved species boundaries in descriptor space.
  - Local-neighborhood diversity: distance-weighted kNN on scaled descriptors and PCA/SVD variants. With balanced classes and many similar species, neighborhood models can add complementary probability mass to the correct family of leaves.
  - Tree diversity: ExtraTrees/RandomForest on raw and transformed features; LightGBM/XGBoost/CatBoost as regularized multiclass components, but do not let them dominate unless CV supports it. With very few rows per class, deep boosted trees can become overconfident.
  - Optional NN: a small RankGauss-normalized MLP or denoising-autoencoder-initialized MLP can add diversity, but should be a later blend member rather than the main first path.
- Feature strategy:
  - Create multiple descriptor views: raw standardized, log/PowerTransformer, QuantileTransformer/RankGauss, block-wise PCA/SVD, and selected interactions only among high-importance features.
  - Keep block identity. A transform that helps texture may hurt shape; compare all-block scaling with separate block scaling.
  - Add image features from binary masks: area, perimeter, bounding-box aspect, eccentricity, solidity, extent, convex area ratio, orientation, major/minor axes, equivalent diameter, Hu moments, contour Fourier-like summaries if implemented cleanly, and simple skeleton/edge density. Use these as extra tabular columns, not as a replacement for descriptors.
- Inference:
  - Average fold-level probabilities for each model.
  - Calibrate or temperature-scale models only using OOF/validation predictions.
  - Blend probabilities with nonnegative weights optimized on OOF log loss; equal or near-equal robust blends are preferable when optimized gains are tiny.
  - Clip only lightly to prevent numeric extremes, then row-normalize.

## 3. Strong First Implementation Plan
- Build one complete single-file solution around a serious tabular multiclass ensemble:
  - Load train/test descriptor columns and labels.
  - Encode species labels with a stable label encoder matching the submission class order.
  - Define feature blocks by margin, shape, and texture prefixes.
  - Use 5-fold `StratifiedKFold` for first iteration; use 10-fold or repeated 5-fold later if time permits. Every fold should preserve all 99 classes.
- Preprocessing views:
  - View A: `StandardScaler` on all 192 descriptor columns.
  - View B: block-wise `StandardScaler` plus `PowerTransformer` or `QuantileTransformer(output_distribution="normal")`.
  - View C: concatenate scaled descriptors with block-wise PCA/SVD components, e.g. 20-40 components per block if validated.
  - View D: add deterministic binary-image region/contour features if images are easy to parse reliably.
- First model set:
  - Multinomial LogisticRegression with `lbfgs` or `saga`, tuned over several `C` values. Use class-balanced defaults only if fold diagnostics show minority instability; the dataset is nominally balanced.
  - RBF SVC over scaled descriptors with grid values for `C` and `gamma`. Prefer probability outputs that are evaluated through OOF log loss; if raw decision scores are used, calibrate on validation folds.
  - Polynomial SVC or linear SVC/logistic variant for a different boundary shape.
  - ExtraTreesClassifier with many trees, shallow-to-moderate depth, and conservative leaf sizes. It often contributes diverse probabilities even if standalone log loss trails the linear/SVM models.
  - Distance-weighted kNN with several `k` values on scaled/PCA features; keep only if OOF log loss or ensemble selection supports it.
  - Optional LightGBM/XGBoost multiclass model with strong regularization, small leaves/depth, column subsampling, and early stopping. Use it as a blend candidate rather than assuming it is best.
- Training target/loss:
  - Use multiclass objectives and probability predictions aligned to log loss.
  - For neural components, use cross-entropy with no or very small label smoothing. Heavy smoothing can underfit probability peaks in a balanced 99-class task.
- First blend:
  - Store OOF probability matrices for every model/view.
  - Compute aggregate OOF multiclass log loss.
  - Optimize nonnegative blend weights with a simple constrained search or greedy hill climbing on OOF predictions.
  - Use the same weights on averaged test predictions.
  - Apply light clipping such as a small epsilon floor/ceiling, then renormalize each row.

## 4. High-ROI Upgrades Across Rounds
- Round 2:
  - Expand from one scaled feature view to multiple descriptor views: raw scaled, PowerTransformed, RankGauss, block-wise PCA/SVD, and raw plus image-derived region features.
  - Tune the strongest logistic and SVM components on identical folds. For log loss, a well-calibrated lower-accuracy model can beat an overconfident classifier.
  - Add ExtraTrees/RandomForest and kNN components as diversity candidates; include only those selected by OOF blend optimization.
  - Compare 5-fold vs 10-fold stratification. The small per-class count makes 10-fold attractive for final training, but 5-fold gives less noisy validation folds. Trust aggregate OOF more than a single fold.
  - Try per-model temperature scaling or calibration using OOF logits/scores where available. Keep calibration only if it improves OOF log loss.
- Round 3:
  - Add deterministic image-derived features from binary masks. Prioritize robust regionprops and contour moments over fragile handcrafted parsing.
  - Train a small tabular MLP on RankGauss features for ensemble diversity. Keep it compact, heavily regularized, and multi-seed averaged; discard if it is highly correlated and worse.
  - Try a frozen pretrained image encoder or small CNN only as a secondary model. Use binary/3-channel replicated images, strong geometric invariance, and early stopping. Blend only if OOF predictions are complementary.
  - Build a level-2 stack on OOF probabilities: multinomial logistic regression or Ridge on flattened model probability vectors. Use strong regularization and identical outer folds to avoid overfitting OOF noise.
  - Use greedy ensemble selection/hill climbing to choose among all saved OOF prediction sets rather than manually including every model.
- Late round:
  - Run multi-seed repetitions for the best logistic/SVM/tree/MLP configurations and average probabilities.
  - Perform careful feature selection: remove descriptor/image features that consistently worsen OOF, but avoid aggressive dimensionality reduction that erases rare-species cues.
  - Try soft pseudo-labeling only after a very stable calibrated ensemble exists, and only for test samples with extremely high confidence. Use low weights; do not let pseudo-labels change validation folds.
  - Explore class-confusion-aware blends: if OOF shows certain models specialize in margin-like or texture-like confusions, let blend optimization decide their weight rather than applying manual class overrides.
  - Avoid late CNN-heavy detours unless image OOF is already contributing. The tabular descriptors are purpose-built for this competition and usually offer the best return per hour.

## 5. Validation and Metric Optimization
- Use stratified folds by species. There is no described group, time, or patient-like leakage unit; do not invent groups from IDs.
- Because each class has very few labeled examples, validation variance is high. Compare experiments on the same fold indices and prioritize aggregate OOF log loss over fold-wise swings.
- Use OOF probabilities for all metric-sensitive decisions:
  - model hyperparameters,
  - feature view inclusion,
  - calibration/temperature,
  - blend weights,
  - stacking regularization,
  - pseudo-label thresholds.
- Metric behavior:
  - Multiclass log loss rewards assigning calibrated probability to the true class, not only ranking it first.
  - Hard one-hot predictions are dangerous. So are uncalibrated SVM/tree probabilities with exact zeros.
  - Submit full 99-class probabilities. Average probabilities across folds/models; do not average argmax labels.
  - Clip probabilities only after model/blend generation, using a small epsilon. Excessive clipping or uniform smoothing can flatten the signal and hurt the best classes.
- Calibration:
  - Logistic regression may already be reasonably calibrated; SVM, kNN, and tree models often need calibration or blend shrinkage.
  - Use OOF log loss to choose between raw `predict_proba`, temperature scaling, isotonic/sigmoid calibration, and simple probability smoothing.
  - For multiclass calibration, prefer simple global temperature/smoothing or fold-aware calibration. Complex per-class calibration can overfit badly with roughly 10 examples per class.
- Local vs leaderboard:
  - Trust stable OOF improvements that persist across folds and model families.
  - Treat public leaderboard movement as noisy because the train/test sizes are small.
  - If leaderboard and OOF diverge, prefer the ensemble with better calibrated OOF log loss unless the OOF gain is within noise and a simpler model is more robust.

## 6. Model, Feature, and Preprocessing Priorities
- Descriptor preprocessing:
  - Scale numeric columns; most useful models are distance- or margin-based.
  - Process margin, shape, and texture blocks separately before concatenation.
  - Test PowerTransformer/QuantileTransformer because descriptor histograms and shape summaries may be skewed.
  - Keep raw standardized features alongside reduced components; PCA can remove discriminative rare-species detail.
- Descriptor models:
  - Highest priority: multinomial logistic regression with tuned regularization.
  - Highest nonlinear priority: RBF SVM and polynomial SVM with validated probability calibration.
  - Useful diversity: ExtraTrees/RandomForest, kNN on scaled/PCA features, small regularized GBDTs, compact MLP.
  - Avoid letting one unstable high-capacity model dominate the final blend.
- Image-derived features:
  - Extract foreground mask statistics and contour/region moments from binary images.
  - Normalize shape features by image scale where appropriate so size and crop differences do not dominate.
  - Add image features to the tabular matrix and let CV/blending decide their value.
- Image models:
  - If used, keep them simple and heavily regularized: small ConvNeXt/EfficientNet-style fine-tune or frozen feature extractor plus logistic/MLP head.
  - Use leaf-safe augmentations: rotations, flips, small shifts/scales are valid for isolated leaves; avoid distortions that alter margin shape.
  - Expect image-only CNNs to be limited by sample count; their value is complementary error diversity.
- Ensembling:
  - Save every model’s OOF and test probability matrix.
  - Optimize blend weights on log loss with nonnegative constraints.
  - Prefer calibrated probability averaging over rank averaging, because the metric consumes probabilities.
  - Use stacking only with strong regularization and OOF discipline.

## 7. Avoid or Delay
- Avoid a pure CNN-first solution. The labeled image set is tiny and the provided descriptors are purpose-built for species discrimination.
- Avoid ignoring the numeric descriptors or treating all features as generic anonymous columns without block-aware scaling.
- Avoid large modern ViT/EVA-style fine-tuning as the first plan. It is compute-heavy and likely to overfit binary silhouettes with very few examples per species.
- Avoid aggressive feature generation over all pairwise interactions before a strong calibrated baseline exists. With roughly 1k labeled rows and 99 classes, noisy interactions can destroy log-loss calibration.
- Avoid target encoding, group aggregations, temporal features, or entity-style leakage recipes. There are no useful categorical/group/time fields in the task description.
- Avoid optimizing accuracy, macro-F1, top-k accuracy, or hard labels. The objective is multiclass log loss from probabilities.
- Avoid uncalibrated overconfident probabilities from SVMs, trees, kNN, or neural nets. Validate probability quality, not just class ranking.
- Avoid per-class calibration or class-prior manipulation based on tiny fold counts unless aggregate OOF log loss improves clearly.
- Avoid external plant datasets, manual labels, private botanical knowledge, or internet-derived species priors as the default plan.
- Delay pseudo-labeling until after a stable OOF-calibrated ensemble exists. Wrong high-confidence pseudo-labels on similar species will directly worsen log loss.
- Delay complex stacking until base OOF predictions are reliable. A small, well-calibrated blend of complementary descriptor models is a stronger default than a large stack trained on noisy OOF signals.
