# ML Failure Prevention Skill: Plan-Code Version

## Purpose

Read this skill before planning and while writing a one-shot ML solution script.

Each round follows this workflow:

1. Use this skill before coding to make an explicit contract plan.
2. Write one complete ML solution script.
3. Run the script in the sandbox.

The script is an end-to-end Kaggle-like solution: data loading, preprocessing, validation, modeling, inference, and writing the required submission file.

The purpose is to prevent brittle assumptions that break otherwise strong, high-scoring approaches, while preserving ambition in modeling, feature engineering, ensembling, deep learning, pretrained encoders, and modality-specific methods.

This skill is not a request to expose hidden chain-of-thought. The pre-code plan should be a concise, visible contract checklist: what the code must preserve, align, validate, and output.

---

## Core Principle

Build ambitious but contract-aware solutions.

Code design MUST be driven by:

- observed data schema;
- fixed-sandbox API compatibility;
- feature/model compatibility;
- valid split assumptions;
- explicit array/tensor/container interfaces;
- modality-aware preprocessing;
- output-schema construction by design.

Do not weaken modeling just to avoid contract work. Strong models, deep learning, ensembling, pretrained encoders, rich feature engineering, and modality-specific methods are encouraged when appropriate, but their data, target, split, tensor, and output contracts MUST be satisfied.

---

## Workflow Requirements

### A. Before Coding: Explicit Contract Plan

Before writing code, produce a concise plan that answers the relevant contracts below.

The plan MUST identify:

- available files and how schema will be derived from loaded data;
- target, identity, feature, metadata, auxiliary, join-key, and output roles;
- model family, feature representation, target encoding, objective/loss/metric, and prediction representation;
- train/test feature alignment strategy;
- validation split strategy and why it is valid for the target/data structure;
- dataframe/numpy/sparse/tensor/list/dict boundaries, including shape, dtype, device, and row order;
- modality-specific irregularities and how preprocessing will handle them;
- output schema, row identity, class/order mapping, and prediction-to-submission conversion.

The plan MUST NOT rely on remembered Kaggle patterns, common column names, common APIs, common tensor shapes, common label spaces, or common output formats.

### B. During Coding: Make Contracts Concrete

While writing code, relevant contracts MUST be implemented through code structure, explicit transformations, checks, alignment logic, or safe fallbacks.

A runtime guard must not turn a recoverable contract violation into a full round failure. If the script can choose a semantically valid fallback, it must do so and log it. Only raise when continuing would create invalid labels, misaligned rows, or semantically wrong output.

Do not merely mention safeguards in comments. If a contract matters, the script must preserve it, check it, or make violation impossible by construction.

---

# Failure Prevention Contracts

## 1. Environment and API Contract

The runtime sandbox dependencies are fixed. Do not suggest upgrading, installing, or changing them. Write code that works with the available dependency set.

Requirements:

- Do not rely on remembered latest-version APIs.
- Avoid fragile keyword arguments, obscure callbacks, optional packages, and backend-specific dtype/device features unless guarded.
- Use stable, widely supported APIs and simple call patterns.
- When using trainers, gradient boosting libraries, tokenizers, image/audio decoders, acceleration backends, or deep learning frameworks, assume version-specific conveniences may be absent.
- Prefer in-script compatible paths: broadly supported arguments, isolated optional imports, no obscure callback protocols, and explicit training/inference calls.
- This contract is not a reason to avoid ambitious libraries; it is a reason to avoid brittle API assumptions.

Checklist:

- [ ] Code does not require changing sandbox dependencies.
- [ ] Optional imports or optional backends are isolated.
- [ ] Advanced libraries use stable, widely supported APIs.
- [ ] Training and inference calls are explicit enough to survive version differences.
- [ ] No modeling strength is lost merely because a robust API path is needed.

---

## 2. Data Schema Contract

Derive schema from loaded files, not from memory, prompt patterns, or generic task templates.

Treat columns, indexes, join keys, generated features, row identity, table shape, and train/test feature alignment as design contracts.

Requirements:

- Load and inspect actual files before relying on concrete columns or shapes.
- Identify target, identity, feature, metadata, auxiliary, join-key, and output roles from data.
- Preserve identity fields needed for prediction output.
- Track operations that create, rename, drop, reorder, aggregate, filter, or duplicate fields or rows.
- Merges, groupbys, pivots, encodings, filtering, batching, concatenation, and index resets must not silently destroy row identity or output alignment.
- Sanitize generated feature names when downstream estimators impose naming constraints.
- Avoid hard-coded concrete column assumptions unless derived from loaded files or created by the script.

Checklist:

- [ ] Schema is derived from loaded files.
- [ ] Target, identity, feature, metadata, auxiliary, and output roles are explicit.
- [ ] Test row identity and row order are preserved.
- [ ] Joins/groupbys/pivots/encodings/filtering/index resets preserve output identity.
- [ ] Generated feature names are safe for the chosen estimator.
- [ ] The code does not assume stale intermediate dataframe schemas.

### Train/Test Feature Alignment

Construct train and test features symmetrically.

Requirements:

- Fit encoders, vectorizers, imputers, scalers, feature selectors, and related preprocessing artifacts on training data as appropriate.
- Transform validation/test data into the same feature space.
- Explicitly align dataframe columns, sparse matrices, vocabularies, categorical levels, generated dummy variables, feature names, feature order, and row order before modeling or prediction assembly.
- Handle non-numeric/object data, mixed-type columns, missing values, infinite values, date-like values, high-cardinality categoricals, sparse/dense assumptions, and dataframe/numpy/tensor expectations.
- Do not pass raw strings or object arrays into numeric estimators or tensor models unless the model explicitly supports them.

Checklist:

- [ ] Train/test feature construction is symmetric.
- [ ] Preprocessing artifacts are fitted on train and reused for validation/test.
- [ ] Train/test feature spaces are explicitly aligned.
- [ ] Feature order and row order are preserved.
- [ ] Missing, infinite, object, categorical, date-like, sparse, and dense cases are handled.
- [ ] Feature representation is compatible with the chosen model.

---

## 3. Feature, Target, and Model Compatibility Contract

Choose the model, feature representation, target encoding, objective, loss, metric, sparsity, missingness, dtype, and value domain as one compatible system.

Strong models only help if the produced data satisfies their contracts.

Requirements:

- Decide whether the task is regression, binary classification, multiclass, multilabel, multioutput, ranking, segmentation, sequence labeling, detection, survival, or another structured prediction task.
- Align target encoding with objective, loss, metric, prediction shape, and output format.
- Preserve class mappings across train, validation, inference, ensembling, and output construction.
- Distinguish labels, logits, probabilities, ranks, masks, boxes, token labels, strings, and continuous values.
- Convert internal model outputs into the representation required by validation and submission.
- Use ambitious estimators, pretrained encoders, deep models, or ensembles when appropriate, but make preprocessing and target mapping satisfy their requirements.

Checklist:

- [ ] Task type is explicit and derived from actual target/output structure.
- [ ] Feature representation matches model input requirements.
- [ ] Target encoding matches objective/loss/metric.
- [ ] Prediction shape and representation match validation and output format.
- [ ] Class mappings, target mappings, thresholds, and inverse transforms are persistent where needed.
- [ ] Logits, probabilities, labels, ranks, masks, boxes, strings, and continuous outputs are not mixed casually.

---

## 4. Validation Split Contract

Design validation from the target distribution and data-generating structure.

Do not blindly apply stratification, folds, random splits, or grouped splits because they are common templates.

Requirements:

- Inspect target support and data structure before choosing validation.
- Rare classes, missing classes in folds, single-class partitions, multilabel targets, grouped observations, repeated identities, temporal order, sequential dependence, and small-sample conditions require appropriate split design.
- Use stratification only when each stratum has enough support.
- Use group-aware splits when identity or entity leakage matters.
- Use time/order-aware splits when future-like validation matters.
- For multilabel or structured targets, choose a validation strategy that reflects the actual target representation.
- Do not skip validation entirely for safety.
- If full cross-validation is invalid for the data support, use a valid alternative that still informs model selection, thresholding, ensembling, and feature decisions.

Checklist:

- [ ] Validation choice is based on target distribution and data structure.
- [ ] Stratification is used only when support is sufficient.
- [ ] Group leakage and repeated identities are considered.
- [ ] Time/order dependence is considered.
- [ ] Multilabel or structured targets use a compatible validation strategy.
- [ ] Empty or invalid folds are avoided.
- [ ] Validation remains useful for high-score modeling decisions.

---

## 5. Array, Tensor, and Batch Interface Contract

Every conversion boundary needs explicit container, shape, rank, dtype, device, and row-order semantics.

This includes transitions among dataframes, numpy arrays, sparse matrices, tensors, Python lists, dictionaries, datasets, dataloaders, prediction arrays, and model-specific batches.

Requirements:

- Do not assume indexes, row order, feature order, sparse structure, rank, or dimensions survive conversion automatically.
- Preserve identity and alignment when moving from dataframe features to arrays, sparse to dense, tensors to numpy, per-fold predictions to ensembles, and batches back to full test predictions.
- For variable-size samples, design collation before training: pad, resize, crop, mask, bucket, or keep structured batches as the model requires.
- For deep learning, align input tensors, labels, masks, losses, auxiliary tensors, dtype, and device.
- For prediction outputs, distinguish binary, multiclass, multilabel, regression, multioutput, sequence, segmentation, and detection shapes.
- Concatenate fold, batch, and ensemble predictions only after row order, dimensions, and target semantics match.

Checklist:

- [ ] Dataframe/numpy/sparse/tensor/list/dict boundaries are explicit.
- [ ] Shape, rank, dtype, device, feature order, and row order are defined where needed.
- [ ] Variable-size samples have deliberate collation.
- [ ] Deep learning tensors, labels, masks, losses, auxiliary inputs, dtype, and device are aligned.
- [ ] Batch predictions map back to the correct test rows.
- [ ] Fold and ensemble predictions are combined only after semantic and dimensional alignment.

---

## 6. Modality Edge-Case Contract

Generic recipes break on modality irregularities. Modality-aware preprocessing should enable stronger models by handling irregular samples before they reach the model.

Requirements:

- Text and sequence data: expect empty text, unusual Unicode, special formatting, long documents, truncation risk, token-level labels, padding, masks, and token-label alignment issues.
- Tabular data: expect mixed numeric/string fields, date-like fields, high-cardinality categoricals, missing categories, and train/test category drift.
- Image and video: expect channel variation, variable resolution, corrupt or unreadable samples, frame sampling issues, channel-order differences, and labels stored in separate metadata.
- Audio: expect sample-rate variation, short clips, silence, stereo/mono differences, decode failures, and window/frame assumptions.
- Graph and multimodal data: expect nested heterogeneous records, multiple linked files or tables, and metadata-label alignment requirements.
- Use modality-specific decoders, normalization, augmentation, collation, masking, metadata joins, and label alignment deliberately.
- Do not ignore rich modalities solely because their plumbing is harder; make the plumbing robust enough to support the strongest feasible approach.

Checklist:

- [ ] Relevant modality irregularities are identified before coding.
- [ ] Text/sequence edge cases are handled when relevant.
- [ ] Tabular mixed types, dates, categoricals, missing categories, and train/test drift are handled.
- [ ] Image/video channel, resolution, corruption, frame, and metadata-label issues are handled.
- [ ] Audio sample rate, short clip, silence, stereo/mono, decode, and window/frame issues are handled.
- [ ] Graph/multimodal nested records and metadata-label alignment are handled.
- [ ] Modality plumbing supports strong modeling rather than avoiding the modality.

---

## 7. Output Schema Contract

The output format is a design constraint from the start, not an afterthought.

Requirements:

- Preserve test row identity, row order, required columns, label/class mapping, target representation, value domain, and required probability/rank/string/structured-output format throughout the pipeline.
- Design preprocessing, batching, filtering, inference, fold aggregation, ensembling, thresholding, inverse transforms, clipping, rounding, and postprocessing so they preserve the external output contract.
- Keep label encoders, class order, probability columns, thresholds, sequence-token mappings, masks, boxes, or structured prediction mappings explicit and reusable at prediction time.
- Internal model outputs are not automatically valid external predictions.
- Convert logits, probabilities, labels, ranks, regression values, masks, boxes, token labels, or strings into the exact required representation while keeping row identity intact.
- Do not leave output assembly as a final mechanical write step.

Checklist:

- [ ] Output schema is considered before training.
- [ ] Test row identity and row order survive the full pipeline.
- [ ] Required columns and column order are preserved.
- [ ] Class/label mapping and target representation are explicit.
- [ ] Thresholding, inverse transforms, clipping, rounding, and postprocessing preserve the external contract.
- [ ] Fold aggregation and ensembling preserve row order and class/order semantics.
- [ ] Internal predictions are converted into the exact required external format.
- [ ] Final submission construction is validated before writing.

---

## Compact Mandatory Heuristics

- MUST NOT assume schema, API, dtype, shape, label space, row order, class order, or output format from memory.
- MUST load data first, derive schema, and make target, identity, feature, and output roles explicit.
- MUST choose model and representation as one contract: features, target, objective, metric, and output must agree.
- MUST keep row identity alive through preprocessing, splitting, batching, prediction, ensembling, and output writing.
- MUST construct train/test features symmetrically and explicitly align feature spaces.
- MUST treat categorical mappings, class order, token labels, thresholds, and inverse transforms as persistent artifacts.
- MUST choose validation based on target support, grouping, multilabel structure, and time/order constraints.
- MUST define every dataframe/numpy/sparse/tensor/list/dict boundary by container, shape, dtype, device, and row order.
- MUST design collation, padding, resizing, masking, and batching before using variable-size or structured samples.
- MUST use advanced libraries and modality-specific models when valuable, but avoid fragile version-specific APIs.
- MUST build output construction around the required external contract before training.
- MUST prefer robust ambitious plumbing over simpler modeling chosen only to avoid implementation risk.

---

## What This Skill Is Not

- Not a traceback cookbook or a list of post-failure fixes.
- Not a reason to avoid ambitious models, deep learning, ensembling, feature engineering, or modality-specific methods.
- Not a request to upgrade, install, or change sandbox dependencies.
- Not post-hoc debugging advice or merely a final submission-only check.
- Not benchmark-specific and not tied to particular tasks, datasets, columns, paths, labels, error messages, or leaderboard quirks.
- Not a resource-conservatism rule derived from short-run failures; manage resources as part of sound modeling, not as a reason to default to weak baselines.
