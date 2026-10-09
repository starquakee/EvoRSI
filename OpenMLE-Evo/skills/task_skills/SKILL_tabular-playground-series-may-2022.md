# Skill: tabular-playground-series-may-2022

## 1. Task-Specific Reading
- This is a simulated manufacturing-control binary classification task. Predict the probability that each machine-state row belongs to target `1`.
- The data is tabular with normalized continuous features and categorical features. The description explicitly emphasizes feature interactions, so the winning route is not just stronger base learners; it is interaction extraction plus robust ensembling.
- The evaluation metric is ROC AUC. AUC rewards correct ranking of positive rows above negative rows, not calibrated probabilities and not a decision threshold. Final outputs should be continuous probabilities or rank-like scores. Do not tune a 0.5 threshold or optimize hard labels.
- Treat `id` as an identifier by default. It may be useful only for diagnostics such as checking train/test distribution or row-order artifacts; do not make it a modeling feature unless a fold-safe experiment proves signal.
- Expect the strongest raw signal to come from:
  - GBDT split discovery on continuous features.
  - Native categorical handling and fold-safe encodings.
  - Explicit pairwise and small higher-order interactions across categorical and numeric blocks.
  - Blending model families whose rankings differ.
  - OOF-optimized rank/probability aggregation.
- Dominant score levers:
  - A 5-fold or 10-fold StratifiedKFold OOF framework using identical folds for every model.
  - LightGBM, XGBoost, and CatBoost variants with different depth/regularization profiles.
  - High-value feature interactions: categorical counts/frequencies, categorical pair encodings, structured token decomposition for string categoricals if present, and arithmetic interactions among important continuous features.
  - Rank-based blending and weight selection on OOF AUC.
  - Conservative late-round pseudo-labeling or stacking only after a stable OOF ensemble exists.

## 2. Highest-Expected-Score Strategy
- Converge toward a diverse GBDT ensemble trained on raw features plus a compact, validated interaction feature set. This task is pure tabular AUC; GBDTs should be the backbone, with neural models used only for diversity after the tree ensemble is strong.
- Use three primary families:
  - LightGBM binary GBDT on CPU as the fast iteration workhorse. Train multiple variants: one moderately deep, one shallow/regularized, and possibly one DART/GOSS-style diversity run if time allows.
  - XGBoost binary logistic histogram models as the second family. Vary depth and `min_child_weight` to produce smoother and sharper rankers. Keep early stopping configured in the estimator constructor.
  - CatBoostClassifier as the categorical specialist. Prefer native categorical features for object/string/low-cardinality columns, and test whether selected low-cardinality integer-like features improve as categoricals.
- Optimize the system around AUC:
  - Train with binary logloss/objective because it gives stable gradients and useful probabilities.
  - Evaluate and select by OOF ROC AUC.
  - Blend by OOF AUC, not validation logloss.
  - Prefer rank averaging when models have different calibration but good ranking.
- Engineer interaction features in controlled waves:
  - Start with all raw non-id columns.
  - Identify categorical columns by dtype and low/medium cardinality. For string-valued categoricals, extract simple structural features if tokens are composed of repeated characters, fixed positions, length, counts, or other parseable symbols. Do not leave a structured categorical token only as one opaque label.
  - Add count/frequency encodings for categorical columns and selected categorical pairs using train+test feature values only.
  - Add fold-safe target encodings for individual categoricals and selected pairs only when their cardinality/support is high enough and OOF AUC improves. Use smoothing; never compute target encodings globally before CV.
  - For continuous features, use an initial feature-importance pass to choose top columns, then add limited pairwise products, differences, absolute differences, ratios with epsilon, squared terms, and row-level summaries. Keep the interaction set compact.
- Use the validation framework as the control plane:
  - Identical stratified folds across every model and feature experiment.
  - OOF predictions saved for each candidate model.
  - Test predictions averaged across folds.
  - Blend weights, rank blending, stacking, and pseudo-label selection all decided from OOF behavior.
- The medal-oriented endpoint should be a blend of several strong OOF rankers: multiple LightGBM variants, multiple XGBoost variants, CatBoost with native categorical treatment, and optionally a RankGauss MLP or logistic/polynomial model only if it improves ensemble AUC despite weaker standalone performance.

## 3. Strong First Implementation Plan
- Build one complete first script around a serious 5-fold StratifiedKFold ensemble. Use 5 folds for the first pass; move to 10 folds or repeated seeds later only after the feature set is stable.
- Preprocessing:
  - Drop target from training features and exclude `id` from default model features.
  - Split columns into continuous, categorical, and low-cardinality numeric candidates from the observed dataframe, not from assumptions.
  - Leave missing values for tree models; add missing indicators only for columns with nontrivial missingness.
  - For LightGBM and XGBoost, label-encode categorical columns consistently over train+test. Also add count and normalized frequency features.
  - For CatBoost, keep raw categorical values as strings or categorical codes and pass categorical feature indices natively.
- First-round feature set:
  - Raw continuous and categorical features.
  - Count/frequency encodings for all categorical columns and low-cardinality numeric columns.
  - Pairwise categorical interaction features for the most plausible categorical columns: concatenated pair labels for CatBoost, count/frequency encodings for all GBDTs, and target encodings only inside folds.
  - If any string categorical column has internal structure, add per-position symbols, token length, number of unique symbols, repeated-symbol counts, and symbol-frequency summaries. This is often higher value than treating the whole string as a single category.
  - Continuous summaries: row mean/std/min/max over normalized continuous columns, and optionally positive/negative counts if values are centered around zero.
  - Pairwise arithmetic features among the top 8-12 continuous features from a quick LightGBM importance pass: product, difference, absolute difference, ratio, and squared terms. Keep only features that improve OOF or have stable nonzero importance.
- First-round models:
  - LightGBM binary GBDT with low learning rate, high estimator cap, early stopping, CPU device, row/column subsampling, and L1/L2 regularization. Use `metric='auc'` for monitoring while training with binary objective.
  - XGBoost binary logistic histogram model with a different depth/regularization profile. Use `eval_metric='auc'` or logloss plus OOF AUC selection.
  - CatBoost binary classifier using native categoricals, ordered-style target statistics, depth around moderate values, early stopping, and enough iterations for low learning rate.
- Inference and blending:
  - Store OOF probabilities and fold-averaged test probabilities for every model.
  - Start with equal average and rank average of LightGBM, XGBoost, and CatBoost. Choose the better OOF AUC aggregation.
  - Optimize nonnegative blend weights on OOF AUC with a small grid, coordinate search, or hill-climbing. Include a model only if it improves OOF blend AUC or materially diversifies rankings.
  - Final output should be the weighted blended probability/rank score. Thresholding is irrelevant for AUC.

## 4. High-ROI Upgrades Across Rounds
- Round 2:
  - Expand and ablate interaction features. Prioritize categorical pair count/frequency features, structured string-token decomposition if present, and fold-safe target encodings for selected individual and pair categorical columns.
  - Tune LightGBM first: `num_leaves`, `min_data_in_leaf`, feature fraction, bagging fraction, and L1/L2 regularization. Train one higher-capacity and one smoother variant.
  - Add XGBoost variants with different `max_depth` and `min_child_weight`; keep one shallow smoother model and one deeper interaction-heavy model.
  - Test CatBoost with alternate categorical sets: object/string only, object plus low-cardinality integers, and raw label-encoded features. Keep by blend AUC, not standalone AUC.
  - Use feature importance across folds to remove noisy generated features and identify the next small set of numeric interactions.
- Round 3:
  - Move from 5 folds to 10 folds or add 2-3 seeds for the best model configurations if runtime allows. For AUC, multi-seed averaging mainly stabilizes ranking in near-tie regions.
  - Build a larger OOF library: 2-4 LightGBM variants, 2-3 XGBoost variants, 1-2 CatBoost variants, and optionally one simple RankGauss MLP or regularized logistic model with interaction/one-hot features for diversity.
  - Use rank-based hill climbing over OOF predictions. Let weaker but less-correlated models enter only when the OOF ensemble improves.
  - Try target-encoding variants with different smoothing and support cutoffs. Low-support categories can overfit AUC; require consistent fold gains.
  - Run adversarial validation only as a diagnostic. If train/test separability is high, downweight or remove features that primarily distinguish train from test unless they also improve OOF AUC.
- Late round:
  - Use stacking only after clean OOF predictions exist for several diverse base models. Prefer a very regularized logistic regression or ridge-style meta-learner on OOF prediction columns; compare against simple weighted rank blending.
  - Try pseudo-labeling only with very high-confidence test rows and only if train/test distributions look close. Add pseudo-labeled rows to training folds, never validation folds, and keep it if OOF-style simulations or public/private-risk reasoning are favorable.
  - Consider CatBoost baseline boosting or residual-style second-stage models only after the main ensemble plateaus.
  - Increase fold count/seeds for the final chosen feature set rather than adding broad unvalidated feature explosions.

## 5. Validation and Metric Optimization
- Use StratifiedKFold with shuffle as the primary validation because the task is binary classification with no stated groups or temporal ordering. Stratification stabilizes positive-rate balance and AUC variance across folds.
- Keep the exact same splits for every feature set, model family, and ensemble experiment. Otherwise small AUC differences are indistinguishable from split noise.
- Trust full OOF ROC AUC over single-fold results. Fold-level AUC variance should guide whether a claimed gain is real; changes smaller than the observed fold noise should be treated as inconclusive.
- Use OOF predictions for every metric-sensitive decision:
  - Feature inclusion.
  - Target-encoding smoothing and columns.
  - Blend weights.
  - Rank averaging versus probability averaging.
  - Stacking model selection.
  - Pseudo-label thresholds.
- AUC-specific optimization:
  - Do not tune a classification threshold.
  - Do not choose models by accuracy or F1.
  - Calibration is secondary. A model with worse logloss can be better if it ranks positives higher.
  - Rank-transforming each model's OOF/test predictions before blending is often robust when models have different probability scales. Compare raw-probability blending and rank blending on OOF AUC.
  - Clipping extreme probabilities usually does not help AUC unless it prevents numerical issues in logit transforms; avoid unnecessary clipping as a score strategy.
- If local CV and public leaderboard disagree:
  - Prefer OOF when experiments use identical folds and consistent feature generation.
  - Check whether the public result changed from calibration/rank effects; for AUC, monotonic transforms should not change single-model ranking but can affect blended rankings.
  - Avoid chasing one public jump with high-variance target encodings or row-order features unless repeated OOF evidence supports it.

## 6. Model, Feature, and Preprocessing Priorities
- Highest-priority models:
  - LightGBM binary GBDT on CPU, several regularization/capacity settings.
  - XGBoost binary logistic histogram models, with early stopping configured correctly and depth/min-child diversity.
  - CatBoost binary classifier with native categorical handling.
  - Optional late diversity: RankGauss MLP with embeddings for categorical features, or regularized logistic regression on one-hot/polynomial interactions.
- Highest-priority feature work:
  - Preserve all raw non-id columns.
  - Robustly identify categorical and low-cardinality numeric columns.
  - Count/frequency encode individual categoricals and selected categorical pairs.
  - Decompose structured string categoricals if any exist: position-wise symbols, length, unique-count, repeated-symbol counts, and symbol frequencies.
  - Add OOF target encodings for high-value categorical columns and pairs with smoothing and support checks.
  - Add compact numeric interactions only among important continuous columns: product, difference, absolute difference, ratio, square, and row summaries over normalized numeric blocks.
  - Use fold importance stability to prune generated noise.
- Highest-priority preprocessing:
  - Leave continuous normalized values mostly unchanged for trees.
  - Use RankGauss only for neural or linear diversity models, not as mandatory GBDT preprocessing.
  - For categorical encoding, fit transformations without target leakage. Feature-only encodings such as counts may use train+test values; target encodings must be OOF for train and full-train for test after validation.
  - Treat class imbalance carefully. For AUC, heavy class weights can distort ranking; test `scale_pos_weight` or class weights only if OOF AUC improves.
- Highest-priority ensemble behavior:
  - Average fold predictions within a model.
  - Compare probability average, logit average, and rank average across models.
  - Optimize blend weights on OOF AUC with nonnegative weights.
  - Add models for diversity, not just standalone score.

## 7. Avoid or Delay
- Avoid generic image, text, audio, sequence, or time-series methods. This is pure tabular binary classification.
- Avoid weak single-model baseline positioning as the final strategy. A serious solution should quickly reach a multi-family GBDT blend.
- Avoid external datasets or original-source data as a default. The task description only guarantees the provided simulated train/test files.
- Avoid using `id` or row order as a feature by default. Treat any row-order signal as risky until validated rigorously.
- Avoid global target encoding, target encoding before splitting, or pair target encodings without OOF construction.
- Avoid broad brute-force feature explosions in the first script. Interaction discovery matters, but uncontrolled thousands of arithmetic/category crosses can add noise and runtime without reliable AUC gain.
- Avoid optimizing logloss, accuracy, F1, or thresholds as the final objective. AUC is threshold-free and ranking-based.
- Avoid over-calibration work early. Calibration can improve logloss while leaving AUC unchanged or hurting blended ranking.
- Avoid neural tabular models before a strong GBDT ensemble exists. Use them late as diversity only.
- Avoid stacking before the OOF prediction library is clean and diverse. Weighted or hill-climbed rank blending is the safer high-ROI ensemble first.
- Avoid pseudo-labeling early. It can amplify public-distribution artifacts and should be late-round, high-confidence, and OOF-justified.
- Avoid trusting one leaderboard move, one fold, or one random seed for feature interactions and postprocessing decisions. Require stable OOF AUC improvements.
