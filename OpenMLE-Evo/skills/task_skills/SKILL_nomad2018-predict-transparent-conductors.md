# Skill: nomad2018-predict-transparent-conductors

## 1. Task-Specific Reading
- Treat this as small-data scientific multi-target regression, not generic wide tabular regression. Each row is a candidate transparent conductor material with compact CSV descriptors plus a per-material atomic geometry file.
- Predict two positive continuous targets: `formation_energy_ev_natom` and `bandgap_energy_ev`. The leaderboard score is the mean RMSLE over the two targets, so train and validate primarily as RMSE on `log1p(target)` for each target.
- The two targets are related but not interchangeable. Formation energy is more tied to stability/composition/packing; bandgap is more sensitive to composition, local geometry, and symmetry. Use shared features, but usually train separate target-specific models and blend target-specific predictions.
- The data scale is small, around a few thousand materials. High-capacity neural models on raw atom coordinates are risky as first-line models unless heavily regularized and validated. The highest expected return is from compact, physically meaningful descriptors plus robust tree ensembles.
- The provided modalities are:
  - Low-dimensional structured material descriptors: spacegroup, atom count, relative Al/Ga/In composition, lattice vector lengths, lattice angles.
  - Atomic geometry files: Cartesian coordinates and species for all atoms in the unit cell.
- Dominant score levers:
  - Exact metric alignment through log-space modeling and log-space blending.
  - Geometry-derived features that summarize cell volume, density, composition counts, pair distances, nearest-neighbor structure, coordinate spread, and element-pair environments.
  - Stable repeated CV because a single random split on this small dataset can overstate changes.
  - Diverse but regularized GBDT ensembles, with simple linear/kernel/nearest-neighbor models only as ensemble diversity.
  - Conservative clipping in original target space after inverse transform, calibrated with OOF predictions.

## 2. Highest-Expected-Score Strategy
- Converge toward a target-wise ensemble of LightGBM, XGBoost, CatBoost, Ridge/ElasticNet, ExtraTrees or RandomForest-style models, and possibly a compact MLP, all trained on `log1p` targets and blended in log space by OOF-optimized weights.
- Make feature engineering the center of the solution. The raw CSV is too small and too low-dimensional to fully describe the material. Parse every `geometry.xyz` and create deterministic descriptors:
  - Element counts and fractions for Al, Ga, In, O from geometry; cross-check against CSV composition.
  - Lattice volume from vector lengths and angles; density-like features such as total atoms per volume, oxygen per volume, cation per volume, and per-element density.
  - Lattice shape features: products, ratios, min/max/range/std of vector lengths, angle deviations from 90 degrees, angle products, and volume normalized by length products.
  - Composition interactions: Al/Ga/In/O ratios, cation fractions, entropy-like composition diversity, dominant cation flags, pairwise products of composition fractions, and composition by spacegroup interactions.
  - Coordinate descriptors by element: centroid, std, min, max, range along x/y/z; distances from centroid; within-element and cross-element coordinate separation stats.
  - Pairwise distance descriptors by unordered element pair, especially cation-O and cation-cation pairs: min, max, mean, std, percentiles, nearest-neighbor means, and counts under multiple distance radii. Normalize distances by lattice scale and by volume-derived length.
  - Row-level geometry descriptors: all-pair distance distribution, nearest-neighbor distribution, per-atom average nearest-neighbor distance, coordinate bounding-box volume, and occupancy-like ratios.
- Use structured parsers and numeric geometry operations, not string hacks, but keep descriptors compact. Hundreds of high-signal features are useful; thousands of arbitrary interactions are likely to overfit.
- Model each target separately by default. A shared multi-output MLP or scikit multioutput wrapper can be tried for diversity, but single-target GBDTs usually give better control over hyperparameters, feature importance, clipping, and blend weights.
- For GBDTs, prefer conservative regularization because `n` is small:
  - LightGBM: low learning rate, many estimators with early stopping, moderate leaves, meaningful `min_child_samples`, column and row subsampling, L1/L2 regularization.
  - XGBoost: `hist` or GPU histogram if safe, shallow-to-medium depth, subsampling, regularization, early stopping in constructor.
  - CatBoost: useful for `spacegroup` and discrete/categorical variants; pass categorical features where possible and use moderate depth.
- Add diversity through:
  - Different feature sets: raw+cell, raw+geometry, raw+geometry+interactions, selected features.
  - Different model families: LGB/XGB/Cat/Ridge/ExtraTrees/KNN target features.
  - Different seeds and repeated folds.
  - Target transforms beyond `log1p` only after `log1p` baseline is stable; RMSLE makes `log1p` the anchor.
- Final inference should average or stack predictions in log space, then inverse-transform with `expm1`, enforce non-negativity, and apply only OOF-validated clipping. Because the score is the mean of two RMSLEs, optimize and monitor each target separately before averaging.

## 3. Strong First Implementation Plan
- Build one complete script around a serious first-round pipeline:
  - Load CSV descriptors.
  - Parse geometry files for train and test ids.
  - Construct a single feature table with raw CSV, lattice/composition features, geometry descriptors, and pairwise distance summaries.
  - Train target-wise 5- or 10-fold shuffled KFold models on `log1p` targets.
  - Produce OOF predictions and test predictions for each target and model.
  - Blend models per target using OOF RMSE in log space; start with weighted average or Ridge meta-model on OOF predictions if there are at least 3 diverse base models.
- First feature set should include:
  - Raw numeric columns and encoded `spacegroup`.
  - `spacegroup` as categorical for CatBoost/LightGBM and label-encoded for XGBoost.
  - Cell volume, density, length ratios, angle deviations, and length-angle products.
  - Estimated or parsed element counts/fractions for Al/Ga/In/O.
  - Composition ratios and products: Al/Ga, Al/In, Ga/In, cation/O, each cation fraction times density, and dominant-cation indicators.
  - Geometry descriptors: per-element coordinate stats, all-pair distance stats, element-pair distance stats, nearest-neighbor stats, and distance histogram counts at several radii chosen from training quantiles rather than arbitrary physical constants.
- First model suite:
  - LightGBM per target as primary model, with `objective=regression`, RMSE metric, CPU device, early stopping, and regularized leaf settings.
  - XGBoost per target as a second strong tree model, using squared error on log targets and constructor-level early stopping.
  - CatBoost per target as a categorical-aware model; include raw or string `spacegroup` and optionally binned composition/size categoricals.
  - Ridge or ElasticNet on standardized numeric features plus one-hot `spacegroup`; this is not a standalone top model but can improve blend stability.
  - Optional ExtraTrees/RandomForest only if it adds OOF diversity; cap depth or use enough trees to reduce noise.
- Use 10-fold KFold if runtime is comfortable; otherwise 5-fold is acceptable for the first script. Keep identical fold splits across targets and models for clean OOF blending.
- Optimize the first blend simply:
  - Compute OOF log predictions for each model and target.
  - Use non-negative weights or Ridge on OOF predictions per target, minimizing RMSE in log space.
  - If a model is highly correlated and worse, downweight or exclude it. Do not force equal weights.
- Postprocess:
  - Inverse-transform only after blending.
  - Clip to at least zero.
  - Consider upper clipping near the training target maximum or high quantile only if OOF RMSLE improves; do not hard-clip aggressively without evidence.

## 4. High-ROI Upgrades Across Rounds
- Round 2:
  - Strengthen geometry features. Add periodic-aware or cell-normalized pair distances if feasible, plus per-element nearest-neighbor statistics and distance histogram counts by element pair.
  - Add OOF-safe nearest-neighbor target aggregate features in descriptor space: for each validation/test row, compute target means/stds from the nearest training rows using standardized composition+cell+geometry features. Do this within folds to avoid leakage.
  - Tune LightGBM and XGBoost separately for each target with conservative search ranges. Formation energy and bandgap may prefer different depths and regularization.
  - Switch from one random KFold to repeated KFold for evaluation stability. Average model predictions over several seeds only after validating the feature set.
- Round 3:
  - Build a per-target ensemble library: multiple LGB/XGB/Cat configurations, Ridge/ElasticNet, ExtraTrees, KNN-feature GBDT, and a compact tabular MLP trained on standardized features.
  - Optimize blend weights per target on OOF log predictions. Prefer Ridge or constrained weight optimization over a tree meta-learner; a high-capacity stacker can overfit 3k rows.
  - Add feature selection variants: train models on all features, top feature-importance subset, and noisiest-interaction-pruned subset. Blend variants if OOF correlations are not too high.
  - Explore target-specific clipping and monotone sanity checks in original space, but keep clipping mild.
- Late round:
  - Use repeated-fold, multi-seed bagging for the best models and average in log space.
  - Try pseudo-labeling only as an experiment: add test predictions with low ensemble disagreement to training with small weight, retrain, and keep only if repeated CV and public feedback both support it. This is not a default plan.
  - Try a compact graph/point-cloud neural model only after the engineered-feature ensemble is strong. With a few thousand samples, raw-coordinate deep learning has high variance; its best role is an ensemble member if OOF improvement is real.
  - Calibrate final blend weights using stable OOF across repeats, not a single lucky fold. If public leaderboard disagrees with CV, use public feedback only to choose among already CV-credible candidates.

## 5. Validation and Metric Optimization
- Use shuffled KFold because the task description does not define time order, users, or groups. Since rows are independent materials, random KFold is appropriate unless exploratory analysis reveals duplicate or near-duplicate structures.
- Prefer 10-fold CV or repeated 5-fold CV for the main comparisons. With only around 3k rows, fold variance can be large; judge upgrades by repeated mean score and per-target consistency.
- Evaluate exactly in metric space:
  - Train on `log1p(y)`.
  - OOF score for each target is RMSE between `log1p(y_true)` and log prediction.
  - Competition score proxy is the mean of the two target RMSEs.
- Keep OOF predictions for every base model and target. Use OOF not only for scoring but also for:
  - Blend weight optimization.
  - Checking target-specific model usefulness.
  - Clipping calibration.
  - Residual analysis by composition, spacegroup, atom count, and target magnitude.
- Stratification is not required, but for more stable folds you can create bins from target quantiles or from a combined target score and use those bins for fold assignment. This is useful if one target has a skewed distribution.
- Avoid leakage in feature engineering:
  - Geometry and row-level descriptors are safe because they use only row inputs.
  - Target encodings by `spacegroup` or binned composition must be OOF-safe.
  - Nearest-neighbor target aggregates must exclude validation rows during fold generation.
- Trust repeated OOF more than one public leaderboard result. If local CV and leaderboard diverge, inspect whether test composition/spacegroup distribution differs from train; prefer changes that improve both targets or improve the weaker target without sacrificing the other.
- Because RMSLE punishes multiplicative error and invalid negative predictions, blend in log space and enforce non-negative original predictions. Do not train on raw targets and then hope RMSLE aligns.

## 6. Model, Feature, and Preprocessing Priorities
- Highest-value model priorities:
  - Target-wise LightGBM as the main workhorse.
  - XGBoost and CatBoost for ensemble diversity.
  - Ridge/ElasticNet on standardized features for smooth extrapolation and blend stabilization.
  - ExtraTrees or RandomForest only if OOF correlation with GBDTs is meaningfully lower.
  - Compact MLP only as a later diversity model, not as the first primary route.
- Highest-value feature priorities:
  - Lattice volume and density features; these are likely stronger than raw vector lengths alone.
  - Composition features for Al/Ga/In/O and cation ratios; include interactions with density and spacegroup.
  - Spacegroup categorical treatment, frequency/count encoding, and OOF-safe target encoding if validated.
  - Element-pair distance descriptors, especially O-related local environments because transparent-conductor properties depend on oxide structure.
  - Nearest-neighbor geometry features by atom type, normalized by lattice scale.
  - Distance histogram features based on quantile-derived bins, both all-pair and per element-pair.
  - Group-level aggregates by `spacegroup`: numeric feature mean/rank/difference from group mean, created without target leakage.
  - KNN target aggregate features in standardized feature space, created OOF-safe.
- Preprocessing priorities:
  - Preserve raw continuous features for tree models; do not over-standardize for GBDTs.
  - Standardize numeric features for Ridge, KNN, and MLP.
  - Encode categorical spacegroup differently by model: raw categorical for CatBoost, categorical/label for LightGBM, numeric label or one-hot for XGBoost/Ridge as appropriate.
  - Use missing indicators only if actual missingness appears; the described data should mostly be complete.
  - Clip or winsorize engineered ratios only to prevent numerical explosions; avoid removing rows unless values are impossible.

## 7. Avoid or Delay
- Avoid treating this as plain CSV-only tabular regression for the final solution. Raw CSV models are a useful sanity check, but the geometry files are a major signal source.
- Avoid image, text, audio, and large transformer recipes. Nothing in the task is pixel, language, or audio data.
- Avoid a raw-coordinate graph neural network as the first implementation. It is attractive for materials data, but the dataset is small and the available guidance favors robust tabular ensembles; use neural geometry models only after a stable engineered-feature ensemble exists.
- Avoid metric-mismatched raw-scale training as the main objective. RMSLE requires log-space optimization or an equivalent squared-log objective.
- Avoid leaky target encoding, full-train nearest-neighbor target means, or feature selection decisions made using validation targets outside the fold protocol.
- Avoid overproducing arbitrary polynomial features before strong physical descriptors. Products and ratios should be composition-, lattice-, or density-motivated.
- Avoid aggressive clipping to the training range unless OOF validates it. Mild non-negativity is mandatory; upper clipping is a tunable postprocess.
- Avoid relying on external materials databases, pretrained chemistry resources, private labels, or manual labels. The default solution should use only provided CSV and geometry inputs.
- Delay pseudo-labeling, high-capacity stacking, and neural coordinate models until repeated CV shows the engineered-feature GBDT ensemble has plateaued.
