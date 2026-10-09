# Skill: text-normalization-challenge-english-language

## 1. Task-Specific Reading
- This is English token-level text normalization for TTS/ASR style output. Each row is one token inside a sentence. Predict the exact normalized `after` string for every test token.
- The target is not a document label and not free-form sentence generation. It is a token transformation problem where the same surface token can require different spoken forms depending on token class and local context.
- Training includes a token `class`; test intentionally omits it. Treat class recovery as a central subtask. Use train labels to learn both class priors and class-specific normalization conventions, then infer test class with a hybrid regex plus ML classifier.
- Evaluation is total token accuracy. A prediction is correct only if the whole output string matches exactly. Partial numeric correctness has no value if wording, spacing, punctuation, pluralization, silence markers, or unchanged-token convention differs.
- The dominant score levers are:
  - Near-perfect handling of unchanged/plain tokens and punctuation, because they usually dominate row count.
  - High-coverage dictionaries from observed `before -> after` mappings, especially for names, abbreviations, symbols, acronyms, and recurring written forms.
  - Deterministic class-specific normalization for numbers, dates, times, money, measures, ordinals, digit sequences, addresses, letters, and electronic-like tokens.
  - Accurate test class inference, since class decides whether `1200` is a cardinal, year, address, time fragment, telephone component, or digit sequence.
  - Conservative arbitration: high-confidence dictionary/rule outputs should not be overwritten by a generic model that produces plausible but convention-mismatched English.
- Preserve the exact training convention for unchanged tokens and silence/punctuation. Do not assume whether unchanged tokens are represented by the original token or by a marker; infer and copy the convention from train.

## 2. Highest-Expected-Score Strategy
- Converge toward a hybrid finite-state, dictionary, and learned-ranker system. For this competition, a strong grammar-like cascade should beat a transformer-only route because the metric rewards exact convention matching, not semantic plausibility.
- Build the system around candidate generation and candidate selection:
  - Generate candidates from exact memorization, class-conditional memorization, context-conditioned memorization, regex/rule normalizers, and a learned fallback.
  - Assign confidence to each source. Exact high-purity dictionary and validated deterministic rules should outrank neural or fuzzy candidates.
  - Train a candidate ranker or use calibrated priority rules to choose among candidates when multiple plausible normalizations exist.
- Use the provided training class labels aggressively:
  - Learn class distributions by token shape, casing, punctuation pattern, surrounding token patterns, and sentence position.
  - Train a character n-gram classifier for class prediction using `before`, previous/next tokens, simple shape features, and regex feature flags.
  - Combine classifier probabilities with deterministic class overrides for obvious patterns. Regex should override only when precision is extremely high; otherwise it should add candidate classes.
- Treat deterministic normalization as the core model:
  - Implement an English number verbalizer and tune its variants against validation: year-style reading, `and` usage, comma handling, zero versus oh, decimal digit reading, ordinal suffixes, negative signs, fractions, and large magnitudes.
  - Implement class-specific grammars for dates, time, money, measures, digit strings, telephone-like strings, addresses, letters, electronic tokens, punctuation, and plain tokens.
  - Mine unit, currency, month, abbreviation, symbol, and letter-name dictionaries from train. Favor train-observed spellings over hand-written assumptions.
- Use learned sequence modeling only as a fallback and diversity source:
  - A small character-level encoder-decoder or Transformer trained from scratch on non-trivial `before/context/class -> after` pairs can help for rare classes and unseen abbreviations.
  - Keep its outputs behind dictionary/rule outputs unless OOF evidence proves it wins for a class or confidence band.
  - Do not spend the first round fine-tuning a large external pretrained seq2seq model unless weights are already available and the rule/dictionary baseline is stable.
- Use validation to choose conventions, not just model checkpoints. Many decisions are discrete formatting choices: hyphen versus space, dollar phrasing, year phrasing, leading-zero time phrasing, plural unit naming, and whether punctuation emits silence.
- Final inference should use artifacts trained on all labeled data after validation decisions are fixed: full-train dictionaries, full-train class classifier, full-train fallback model if used, and rules configured by OOF/holdout analysis.

## 3. Strong First Implementation Plan
- Build a serious first script as a deterministic cascade with ML-assisted class inference.
- Data representation:
  - Reconstruct sentence context with previous token, next token, two-token windows, sentence position, token length, character classes, casing pattern, digit groups, punctuation flags, and regex indicators.
  - Separate tokens whose `after` equals the unchanged convention from tokens that require real normalization, but keep both in training diagnostics because total accuracy depends on both.
- Validation:
  - Split by `sentence_id` so local context examples do not leak across train and validation.
  - Use one robust holdout or several folds if time allows. Report total token accuracy, changed-token accuracy, and per-class accuracy.
- Class inference:
  - Start with high-precision regex rules that return candidate classes for obvious forms: punctuation, alphabetic plain words, numeric strings, decimal forms, currency-prefixed/suffixed strings, time-like `:` forms, date-like slash/dash forms, ordinal suffixes, measures with units, emails/URLs, telephone-like digit patterns, and letter/digit mixtures.
  - Train a linear character n-gram model for class prediction. Use char n-grams of `before`, char n-grams of `prev before next`, and sparse shape features. Logistic regression, linear SVM, or SGD with calibrated probabilities is enough and fast on large token data.
  - Blend rule candidates and ML probabilities. If a rule is precision-proven on validation for a pattern, force the class; otherwise keep top 2-3 class candidates for normalization candidate generation.
- Candidate generation:
  - Exact map: if a `before` token has one overwhelmingly dominant `after`, use it. Track purity and count; avoid blindly memorizing ambiguous tokens.
  - Class map: for predicted class and token, use the majority train `after` for that `(class, before)` pair when available.
  - Context map: for ambiguous tokens, add majority mappings for `(prev, before, next)` and lighter variants such as `(prev, before)` and `(before, next)`.
  - Rule outputs: generate class-specific normalized strings, sometimes multiple variants where conventions differ.
  - Fallback output: unchanged convention for plain tokens; learned/fuzzy candidate for rare normalized classes only.
- Rule priorities:
  - Plain/unchanged and punctuation must be extremely conservative. Do not normalize alphabetic words unless train evidence or class prediction strongly supports a non-plain class.
  - Cardinal/decimal/ordinal: implement English integer and ordinal verbalization; validate wording for large numbers, commas, signed numbers, leading zeros, and decimal-point digit spelling.
  - Date/time: parse common numeric and month-name patterns; tune year reading and AM/PM spelling from train. For times, handle minutes below 10 and cases where the correct form may be clock-style rather than cardinal.
  - Money/measure: split numeric value from unit/currency; mine unit and currency spoken forms from train; handle singular/plural and cents/subunits by validation convention.
  - Digit/telephone/address/letters: prefer character-by-character or grouped reading only when class evidence supports it. These classes are easy to confuse with cardinal numbers.
  - Electronic/verbatim/abbreviation: rely heavily on train-derived symbol and acronym dictionaries; use rules only for obvious separators and individual characters.
- Selection:
  - First choose the highest-confidence exact or class-conditional dictionary candidate.
  - Else choose validated rule output for the most likely class.
  - Else use context dictionary.
  - Else use fallback model or unchanged convention depending on class probabilities.
  - Record OOF candidate source accuracy by class, then hard-code source precedence per class.

## 4. High-ROI Upgrades Across Rounds
- Round 2:
  - Convert the first cascade into an OOF candidate-ranking setup. For validation rows, generate all candidates without using that row in dictionaries; train a simple ranker over candidate features such as source type, class probability, dictionary purity/count, regex flags, edit/shape similarity, and whether candidate appeared in train.
  - Mine rule exceptions from per-class validation errors. Add high-support exception maps for months, units, currencies, abbreviations, titles, symbols, and recurring electronic fragments.
  - Add convention tuning grids for number/date/time variants. Pick variants by class-level OOF accuracy rather than intuition.
  - Improve class prediction with an ensemble of character n-gram classifiers using different contexts: token-only, token plus neighbor tokens, and token plus sentence-level shape summary.
- Round 3:
  - Add a small character-level seq2seq fallback trained on changed tokens, with input formatted as predicted class plus local context plus current token. Use it only for classes where OOF shows positive lift over rules.
  - Add fuzzy retrieval for rare tokens: character TF-IDF nearest neighbors within predicted class; accept only when similarity is very high or when the neighbor mapping is class-consistent.
  - Use multi-fold OOF predictions for class probabilities and ranker training, then retrain final components on all data.
  - Add class-specific fallback policies. For plain/punctuation, fallback should be unchanged/silence convention. For electronic/verbatim/letters, fallback can be character spelling. For numeric classes, fallback should be the safest validated number grammar.
- Late round:
  - Ensemble several cascades that differ in class classifier, rule variant set, and candidate precedence. Since outputs are strings, combine by OOF-weighted voting or candidate ranker features rather than averaging.
  - Pseudo-label only low-risk test classes for self-training the class classifier, not the normalization target. Use high-confidence class labels to stabilize class priors on test-like shapes.
  - Add targeted manual grammar expansion based on validation confusion, but only when it improves OOF and does not reduce high-volume class accuracy.
  - Consider checkpoint averaging or a second small neural fallback only after deterministic coverage is saturated.

## 5. Validation and Metric Optimization
- Use sentence-grouped validation. Random row splits overestimate context dictionaries and can hide failures on unseen sentences.
- For dictionaries during CV, rebuild maps inside each fold. A full-train dictionary used to score validation leaks exact row information and inflates trust in memorization.
- Optimize total exact token accuracy, but diagnose with changed-token accuracy and per-class accuracy:
  - Total accuracy tells leaderboard alignment.
  - Changed-token accuracy exposes whether the system is doing real normalization.
  - Per-class accuracy tells which grammar expansion has the highest return.
- Use OOF predictions for three decisions:
  - class classifier calibration and regex override thresholds,
  - dictionary purity/count thresholds,
  - candidate source precedence or ranker training.
- Tune source precedence by class. A rule can be best for cardinals but worse than memorization for abbreviations; a seq2seq fallback can help electronic/verbatim but hurt dates.
- Metric-aligned behavior:
  - Never output a near-synonym if train convention says another string. Exact spacing and token words matter.
  - Prefer high-confidence unchanged output over speculative normalization for ambiguous alphabetic tokens. False positives on frequent plain tokens are expensive.
  - For rare classes, only accept complex grammar outputs when class probability and pattern match are strong enough to beat unchanged fallback in OOF.
- If local CV and leaderboard diverge, trust leakage-safe per-class OOF over public noise, then inspect train/test distribution shift in token shapes and predicted classes. Do not chase a public score with broad rule changes that reduce OOF on high-count classes.

## 6. Model, Feature, and Preprocessing Priorities
- Highest-value components:
  - Full-train exact dictionaries with purity and count metadata.
  - Class-conditional dictionaries using train labels.
  - Character n-gram class classifier with local context.
  - English numeric verbalizer with validated variants.
  - Date, time, money, measure, digit, telephone, address, letters, electronic, punctuation, and abbreviation rules.
  - Candidate source confidence and class-specific arbitration.
- High-value features for class prediction and ranking:
  - Raw token, lowercased token, character n-grams, prefix/suffix n-grams.
  - Shape string: digit/alpha/case/punctuation pattern.
  - Counts of digits, letters, uppercase letters, separators, currency-like symbols, slashes, colons, periods, commas, hyphens, plus/minus signs.
  - Regex flags for numeric, decimal, ordinal, date-like, time-like, unit-suffixed, all-caps, mixed alnum, URL/email-like, telephone-like, and punctuation-only patterns.
  - Previous and next token shape and raw text; sentence beginning/end flags.
  - Candidate metadata: dictionary frequency, mapping purity, class probability, rule class, nearest-neighbor similarity, and whether candidate is unchanged.
- Preprocessing priorities:
  - Preserve raw tokens. Do not lowercase or strip punctuation before normalization; use normalized copies only as features.
  - Preserve sentence order and token ids for context.
  - Normalize internal whitespace only in generated candidate strings, and only to the convention learned from train.
  - Mine all spelling conventions from train: silence marker, unchanged convention, currency names, unit names, zero/oh, year format, decimal point wording, and separator handling.
- Model priority:
  - Classical sparse ML for class prediction is the best first learned component because it is fast, robust, and trained entirely from provided data.
  - Rule and dictionary candidates should carry the main normalization burden.
  - Neural seq2seq should be compact, character-level, and class-conditioned if added; it is a fallback, not the primary system.

## 7. Avoid or Delay
- Avoid treating the task as generic text generation. A fluent output that differs by one word, hyphen, or silence marker is wrong.
- Avoid a transformer-only first solution. It will learn plausible normalization patterns but often miss exact competition conventions and high-frequency unchanged behavior.
- Avoid relying on external datasets, private grammars, downloaded pretrained weights, manual labels, or internet resources as the default plan.
- Avoid row-random validation and full-train dictionaries during validation. Both create false confidence.
- Avoid over-normalizing plain alphabetic tokens. In token accuracy, many small false positives can erase gains from rare difficult classes.
- Avoid aggressive text cleaning, lowercasing, punctuation stripping, Unicode simplification, or token merging that changes the surface forms needed by the grammar.
- Avoid one global rule precedence order. Source reliability is class-dependent.
- Avoid optimizing macro class accuracy at the expense of total token accuracy. Use macro/per-class metrics for diagnosis, not as the final objective.
- Delay complex neural models, pseudo-labeling, and cascade ensembling until the dictionary/rule/classifier baseline is validated and error buckets are clear.
- Delay broad manual exception additions unless they are supported by repeated validation errors or high-count train mappings. Single-case exceptions are easy to overfit and rarely move total accuracy.
