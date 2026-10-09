# Skill: text-normalization-challenge-russian-language

## 1. Task-Specific Reading
- This is Russian token-level text normalization for TTS/ASR style output. Each row is one token inside a sentence. Predict the exact normalized `after` string for every test token.
- The target is not a document label and not open-ended sentence generation. It is a token transformation problem where class, token shape, sentence context, and Russian inflection determine the correct spoken form.
- Training includes a `class` column, but test intentionally omits it. Treat class recovery as a first-class modeling problem. Use train classes to learn both class priors and class-specific normalization conventions, then infer test classes with regex rules plus a learned sparse classifier.
- Evaluation is total token accuracy. A token is correct only when the full predicted string exactly matches the expected normalized string. Partial numeric correctness, plausible Russian wording, or semantically equivalent alternatives get no credit if spelling, spacing, case convention, silence marker, `_trans` suffix convention, or morphology differs.
- The Russian data includes transliterated outputs for foreign names and letters, marked with `_trans` on output tokens. Preserve that convention exactly. Do not replace it with normal Cyrillic spelling, Latin spelling, or a generic phonetic approximation.
- Dominant score levers:
  - Near-perfect handling of unchanged tokens and punctuation/silence, because they usually dominate row count.
  - High-purity dictionaries from observed `before -> after` and `(class, before) -> after` mappings.
  - Accurate test class inference for ambiguous written forms: numbers, dates, times, measures, currencies, abbreviations, initials, foreign names, digit sequences, and punctuation.
  - Russian morphology-aware verbalization. Numeric, date, unit, and currency outputs need the case, gender, and ordinal/cardinal form learned from train conventions.
  - Conservative arbitration. Do not let a generic sequence model overwrite high-confidence dictionary or rule outputs.

## 2. Highest-Expected-Score Strategy
- Converge toward a hybrid dictionary, finite-state grammar, class classifier, and candidate-ranker system. For this task, a grammar-like cascade is the highest-expected route because the metric rewards exact convention matching over fluent generation.
- Build around candidate generation and candidate selection:
  - Candidate sources: exact memorization, class-conditional memorization, context-conditioned memorization, deterministic class rules, fuzzy nearest-neighbor retrieval, and a compact learned fallback.
  - Store frequency, purity, class distribution, and context support for every memorized mapping.
  - Select outputs by calibrated source reliability, not by surface plausibility. High-count pure dictionary entries and validated deterministic rules should outrank neural outputs.
- Use train `class` labels aggressively:
  - Learn token class from raw `before`, neighboring tokens, sentence position, token shape, regex flags, and local punctuation.
  - Train a character n-gram classifier over token-only and context-enriched strings. Sparse linear models are a strong fit because class depends heavily on morphology, digits, punctuation, and prefixes/suffixes.
  - Combine classifier probabilities with deterministic rule candidates. Regex should force a class only when validation precision is extremely high; otherwise it should add candidate classes.
- Treat deterministic normalization as the main model:
  - Mine the exact Russian words used for months, currencies, units, punctuation, zero, decimal separators, signs, ordinal endings, and digit names from train.
  - Implement class-specific grammars for cardinals, ordinals, decimals, dates, times, money, measures, digit strings, telephone/address-like strings, letters, abbreviations, electronic-like tokens, foreign-name transliteration fragments, punctuation, and plain tokens.
  - Prefer train-observed spellings and phrase order over hand-written linguistic intuition whenever they conflict.
- Use a learned sequence model only as a fallback and diversity source:
  - A small character-level encoder-decoder or Transformer trained from scratch on changed tokens can help rare abbreviations, foreign names, mixed alphanumeric tokens, and irregular transliterations.
  - Condition it on predicted class and local context, but keep it behind validated dictionary/rule candidates unless OOF evidence shows a class-specific lift.
  - Do not make a downloaded pretrained seq2seq model the default plan. The environment guarantees libraries, not external weights, and exact Russian convention matching is better served by train-derived rules.
- Final convergence should use full-train dictionaries, full-train class models, class-specific rule configurations selected by OOF accuracy, and a conservative candidate ranker or precedence table.

## 3. Strong First Implementation Plan
- Build a serious first script as a deterministic cascade with ML-assisted class inference.
- Data representation:
  - Reconstruct each sentence in token order. For every token, keep raw `before`, previous/next one or two tokens, sentence start/end flags, token length, shape string, digit groups, casing flags, separator flags, punctuation-only flag, and mixed-script/mixed-alphanumeric indicators.
  - Learn the exact unchanged-token convention and punctuation/silence convention from train. Preserve those outputs; do not assume that unchanged means copying `before`.
  - Keep all tokens in validation diagnostics, but separately track changed-token accuracy so rare but important normalization classes are not hidden by plain-token volume.
- Validation:
  - Split by `sentence_id`, not by row. Row-random splits leak context and overstate memorization.
  - Use one robust grouped holdout for the first round; use grouped folds once the cascade is stable.
  - Score total exact token accuracy, changed-token accuracy, and per-class accuracy.
- Class inference:
  - Use high-precision regex features for punctuation, alphabetic plain tokens, integer forms, decimal forms, signed values, slash/dash dates, colon times, currency-marked values, unit-suffixed values, Roman-like tokens if present, mixed letters/digits, all-caps abbreviations, initials, URLs/emails/electronic strings, and digit-sequence patterns.
  - Train a sparse class classifier with character n-grams from `before`, lowercased copies, token shape, `prev before next`, and regex flags. Logistic regression, linear SVM, or SGD with calibrated probabilities is appropriate.
  - Use top class probabilities rather than only argmax. Generate candidates for the top 2-3 plausible classes when ambiguity is high.
- Candidate generation:
  - Exact map: if `before` has a dominant `after` with high count and high purity, emit it.
  - Class map: if `(predicted_class, before)` has a dominant output, prefer it over token-only memorization for ambiguous strings.
  - Context map: for ambiguous tokens, add candidates from `(prev, before, next)`, `(prev, before)`, and `(before, next)` mappings with purity thresholds.
  - Rule outputs: generate normalized strings for likely classes, including multiple variants for conventions that validation must choose.
  - Fallback outputs: unchanged convention for plain-like tokens, silence convention for punctuation-like tokens, class-conditioned character spelling or seq2seq for rare changed classes.
- Rule priorities:
  - Plain and punctuation must be conservative. False normalization of frequent words costs more than missing a few rare transformed tokens.
  - Numbers: implement Russian integer verbalization with gender and case variants; learn from train when to use cardinal, ordinal, year-like, digit-by-digit, or grouped reading.
  - Dates: parse numeric and month-name patterns; generate month forms and ordinal/cardinal year phrasing according to train conventions. Russian genitive/date forms matter.
  - Time: handle `hh:mm`, leading zeros, minute forms, and separator wording by train convention.
  - Money and measures: split numeric value from currency/unit; mine unit/currency spoken forms and plural/case behavior from train; avoid English-style assumptions.
  - Letters, abbreviations, electronic, and transliteration: rely heavily on train-derived dictionaries and character spelling tables, including `_trans` handling.
- Selection:
  - First use high-confidence exact or class-conditional dictionary candidates.
  - Else use validated rule output for the most likely class.
  - Else use context dictionary or high-similarity retrieval.
  - Else use learned fallback only for classes where validation shows it beats unchanged/rule fallback.
  - Track OOF accuracy by candidate source and class, then hard-code source precedence per class for the final script.

## 4. High-ROI Upgrades Across Rounds
- Round 2:
  - Rebuild the cascade with OOF-safe dictionaries. For each fold, build mappings only from the training fold, generate all candidates on the validation fold, and record source performance.
  - Tune dictionary purity/count thresholds separately for token-only, class-conditional, and context maps. High-frequency ambiguous tokens should not be memorized blindly.
  - Expand rule exceptions from validation errors: months, abbreviations, initials, units, currencies, punctuation, foreign-name fragments, and high-count numeric formats.
  - Grid-search class-specific convention switches for numeric/date/time outputs: gender, case, year phrasing, leading zero handling, decimal separator wording, and digit-vs-cardinal reading.
  - Improve class prediction with an ensemble of char n-gram models using different contexts: token-only, neighbor-window, and sentence-shape summary.
- Round 3:
  - Add an OOF candidate ranker. Features should include source type, class probability, dictionary count/purity, context support, regex flags, candidate length, whether candidate appeared in train, similarity to nearest train token, and unchanged/silence flags.
  - Add fuzzy retrieval within predicted class using character TF-IDF nearest neighbors. Accept only high-similarity, class-consistent mappings; it should rescue misspellings and rare variants, not override clean rules.
  - Train a compact character-level seq2seq fallback on changed tokens with input formatted as class plus local context plus current token. Use greedy or small-beam decoding and only enable it for classes where OOF source accuracy improves.
  - Use multi-fold OOF class probabilities and candidate predictions to set final arbitration thresholds, then retrain components on all labeled data.
- Late round:
  - Ensemble several cascades differing in class classifier, dictionary thresholds, and rule-variant choices. Combine string outputs by OOF-weighted voting or by ranker features; do not average model probabilities into unsupported strings.
  - Pseudo-label only low-risk test classes for class-model adaptation, not normalization targets. High-confidence class pseudo-labels can stabilize test-like shape priors without inventing new outputs.
  - Add targeted manual grammar expansion only when it fixes repeated OOF errors or high-support train mappings.
  - Consider a second neural fallback or checkpoint averaging only after dictionary/rule coverage and class inference are saturated.

## 5. Validation and Metric Optimization
- Use sentence-grouped validation. It prevents the same local sentence context from appearing in both train and validation and gives a more honest estimate of context memorization.
- During CV, rebuild dictionaries inside each fold. Full-train dictionaries used on validation leak exact row information and make memorization look stronger than it will be on unseen test rows.
- Optimize total exact token accuracy, but inspect:
  - Changed-token accuracy, to measure real normalization quality.
  - Per-class accuracy, to identify which rule families move score.
  - Source-by-class accuracy, to decide whether dictionary, rule, retrieval, or neural fallback should win.
  - Confusion between class predictions, because many normalization errors are actually class errors.
- Metric-aligned behavior:
  - Exact string conventions dominate. Tune spacing, markers, suffixes, Russian word forms, transliteration suffixes, and silence outputs from train.
  - Prefer a high-confidence unchanged output over speculative normalization for alphabetic tokens unless class evidence is strong.
  - For numeric-looking tokens, class probability should decide between cardinal, date, time, digit sequence, address-like, and measure/currency handling.
  - Candidate thresholds should maximize token accuracy, not macro class accuracy. Macro/per-class metrics are diagnostics.
- If local CV and public leaderboard diverge, trust leakage-safe OOF by class first. Then inspect train/test shift in token shapes and predicted classes. Avoid broad rule changes that reduce OOF on high-volume plain or punctuation classes.

## 6. Model, Feature, and Preprocessing Priorities
- Highest-value components:
  - Exact token dictionaries with count and purity metadata.
  - Class-conditional dictionaries using train `class`.
  - Context dictionaries for ambiguous recurring tokens.
  - Sparse character n-gram class classifier with local context.
  - Russian numeric/date/time/money/measure grammars tuned to train conventions.
  - Transliteration and character spelling dictionaries, including `_trans` outputs.
  - Candidate source confidence and class-specific arbitration.
- Highest-value class/ranker features:
  - Raw token, lowercase token, prefixes, suffixes, character n-grams, and whole-token identity.
  - Shape pattern for digit/letter/case/punctuation sequences.
  - Counts of digits, Cyrillic-like letters, Latin letters, uppercase letters, separators, periods, commas, slashes, colons, hyphens, plus/minus signs, and currency/unit-like markers.
  - Regex flags for integer, decimal, signed number, date-like, time-like, unit-suffixed, currency-marked, all-caps, initial-like, mixed alphanumeric, URL/email-like, digit-sequence, and punctuation-only tokens.
  - Previous and next raw tokens and shapes; sentence beginning/end flags.
  - Candidate metadata: source type, dictionary frequency, mapping purity, predicted class probability, nearest-neighbor similarity, candidate length, and unchanged/silence indicator.
- Preprocessing priorities:
  - Preserve raw text. Do not lowercase, strip punctuation, remove suffix markers, or transliterate inputs destructively. Use normalized copies only as features.
  - Preserve sentence order and token ids for context-aware features.
  - Normalize generated candidate whitespace only to the exact convention learned from train.
  - Mine all surface conventions from train rather than imposing external grammar preferences.
- Model priorities:
  - Classical sparse ML is the best first learned component for class recovery.
  - Deterministic dictionaries and rules should carry most normalization.
  - Neural seq2seq should be compact, character-level, class-conditioned, and used as a fallback.

## 7. Avoid or Delay
- Avoid treating the task as generic Russian text generation. A fluent normalized phrase is wrong if it differs by one word form, suffix marker, spacing choice, or silence convention.
- Avoid a transformer-only first solution. It is likely to produce plausible but convention-mismatched outputs and can damage high-frequency unchanged tokens.
- Avoid relying on external datasets, private grammars, downloaded pretrained weights, manual labels, or internet resources as the default strategy.
- Avoid English text-normalization assumptions. Russian morphology, date forms, unit/currency inflection, and transliteration conventions are central.
- Avoid row-random validation, full-train dictionaries during validation, and leaderboard-only tuning. These create false confidence.
- Avoid over-normalizing alphabetic tokens. Plain tokens are frequent, and false positives can erase gains from rare classes.
- Avoid aggressive cleaning: no punctuation stripping, token merging, casing destruction, Unicode simplification, or removal of `_trans` markers.
- Avoid one global precedence order. Dictionary, rule, retrieval, and neural reliability differ by class.
- Avoid optimizing changed-token or macro class accuracy at the expense of total exact token accuracy. Use those views to guide fixes, not to choose final thresholds blindly.
- Delay broad pseudo-labeling, large neural models, and cascade ensembling until a strong OOF-validated dictionary/rule/classifier system exists.
- Delay one-off manual exceptions unless they cover repeated validation failures or high-support train mappings. Single-case exceptions overfit easily and rarely move total score.
