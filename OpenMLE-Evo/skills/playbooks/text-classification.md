# Text Classification Category Playbook

This playbook is the full method library for Kaggle-style `text_classification`
tasks. It is intended as direct reading material for a skill-generating agent:
after reading a task description, the agent should route the task, select the
right validation strategy, and extract only the applicable recipes.

Do not use this playbook to justify leaderboard probing, hidden labels, public-test
label inference, challenge exploits, sample-submission label updates, manual
hidden-test labeling, or disallowed external data.

## 1. Task Routing

### 1.1 Closed-set text classification

Use when each row contains text and the target is a fixed binary, multiclass, or
multilabel label set.

Default plan:

- grouped or stratified folds depending on leakage axis
- exact metric and submission validator
- DeBERTa-v3 baseline plus TF-IDF linear baseline
- OOF/test prediction saving
- OOF threshold/calibration/rank averaging
- seed/fold/model ensemble if metric is noisy

### 1.2 Short social text

Use for tweets, questions, comments, short reviews, and social posts.

High-value details:

- keep punctuation, casing, hashtags, emojis, URLs, and handles until tested
- tune thresholds on OOF for F1/Fbeta
- use char n-grams for typos, obfuscation, and short texts
- use domain backbones or tokenizers when available
- validate by keyword/source/language/rule/prompt when those define hidden test
  distribution

### 1.3 Pairwise classification

Use for duplicate questions, NLI-like labels, answer equivalence, authorship
pairs, impostor detection, pronoun A/B/Neither, and pairwise verification.

Default plan:

- build inputs jointly, not as two independent documents
- include both transformer cross-encoder features and classical pair features
- enforce or test A/B symmetry
- split by shared question, prompt, article, author, entity, or graph component
- calibrate probabilities if the metric is logloss

### 1.4 Moderation and rule-conditioned classification

Use when the label asks whether a comment violates a rule, policy, toxicity
class, or moderation target.

Default plan:

- format as `rule + comment -> Yes/No probability` when a rule text exists
- use DeBERTa/XLM-R and constrained-logit LLM classifiers
- drop shortcut metadata when it encodes community/source rather than behavior
- group validation by rule/source/language/comment duplicate
- normalize per-rule scores only with OOF support

### 1.5 AI-generated text and hallucination detection

Use for human-vs-generated text, AI source classification, generated essay
detection, or answer hallucination detection.

Default plan:

### 1.6 MAP@k, multiple-choice, and retrieval-assisted classification

Use text classification directly when the candidate set is closed and all
evidence is in the row. Route to retrieval/RAG as primary when external facts,
large label pools, unseen labels, article corpora, or candidate generation
dominate.

Default plan:

- validate with the exact MAP@k/MRR behavior
- score all candidates/options and sort
- use bi-encoder retrieval for candidate recall
- use DeBERTa/Qwen/Gemma/Llama cross-encoder or option-token logits for final
  precision
- keep top-k formatting and label mappings under tests

### 1.7 Long documents, token spans, and extraction route-outs

If the final output is spans, offsets, BIO labels, `predictionstring`, or
character ranges, ordinary text classification is not the primary playbook. Route
to NER/extraction when available.

This playbook still contributes:

- document-group validation
- DeBERTa/Longformer/BigBird model priorities
- offset-preserving tokenization
- candidate reranking with LightGBM or classifiers
- threshold and ensemble logic

### 1.8 Scientific/protein and CTF route-outs

Protein function prediction is not natural-language classification. Inputs are FASTA protein
sequences and outputs are Gene Ontology labels. Use protein-sequence and ontology modeling; only generic multilabel mechanics transfer.

Ciphertext, prompt extraction, exploit, and CTF tasks are route-out tasks. Preserve only validation and leakage warnings, never exploit recipes.

## 2. First-Pass Checklist

Before modeling, extract:

- prediction unit: row, text, pair, option, document, discourse segment,
  rule-comment pair, prompt-response pair, article-candidate pair, or top-k
  candidate list
- target shape: binary, multiclass, multilabel, ordinal, pair winner, top-k,
  probability columns, token/char spans, or generated strings
- metric: accuracy, logloss, AUC, F1/Fbeta, MAP@k, MRR, QWK, micro span F1,
  character F1, or custom utility
- split axis: document, prompt, question, article, patient, source, author,
  language, rule, generator, user, group, duplicate cluster, or time
- text fields: title, body, rule, prompt, answer, option, candidate label text,
  metadata, source, language, question id
- legal external channels: pretrained models, public corpora, generated data,
  translations, unlabeled test text, previous competitions, article corpora
- submission contract: probability columns, hard labels, top-k strings, one row
  per option, one row per candidate, one row per token/span

Then create:

- `folds.csv`
- exact metric helper or a faithful approximation
- submission validator
- one classical baseline with OOF predictions
- one transformer baseline with OOF predictions
- train/test drift and duplicate audit

## 3. Validation And Leakage Control

### 3.1 Stratified and grouped folds

Use stratification for ordinary class imbalance, but use group-aware folds when
an entity can leak. Text duplicates and near-duplicates should be clustered
before fold creation.

```python
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

def make_text_folds(df, label_col, group_col, n_splits=5, seed=42):
    df = df.copy()
    df["fold"] = -1
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True,
                              random_state=seed)
    for fold, (_, va) in enumerate(cv.split(df, df[label_col],
                                            groups=df[group_col])):
        df.loc[df.index[va], "fold"] = fold
    for fold in range(n_splits):
        tr_groups = set(df.loc[df.fold != fold, group_col])
        va_groups = set(df.loc[df.fold == fold, group_col])
        assert not (tr_groups & va_groups)
    return df
```

If group stratification drops rare classes from some folds, repair folds
manually or use a stronger holdout for model selection.

### 3.2 Split by the hidden-test unit

Common examples:

- short-text and question-pair pairs: question ids and question graph components
- Feedback/PII: essay or document id
- clinical span extraction: patient note id, usually with case stratification
- dataset-mention extraction: article id
- AI-generated text: prompt, source, generator/model family
- toxicity classification/community rules: rule and comment duplicate/source family
- preference modeling/LMSYS: prompt/conversation id
- multilingual toxicity: language/source site
- authorship: author/source
- math misconceptions: question/construct/subject id

When hidden test contains new prompts, rules, languages, or labels, random row
CV is often a trap.

### 3.3 Duplicate and conflict audit

Audit exact and normalized duplicates. In social text, duplicated strings can
have conflicting labels. In pair tasks, the same question can appear in many
pairs. In CTF/cipher tasks, duplicate maps were used as challenge-specific
logic; do not transfer that as a normal recipe.

Practical audit:

- exact text duplicate count by label
- normalized duplicate count after whitespace/case/URL normalization
- pair reversal duplicates
- train/test duplicate presence, used only if allowed by the rules
- label conflicts per duplicate cluster
- near-duplicate embeddings for long generated text

### 3.4 Metric-first validation

Choose validation artifacts by metric:

- F1/Fbeta: OOF probabilities and threshold search
- logloss: OOF probabilities and calibration plots
- AUC: OOF ranks and score calibration by group/language/rule
- MAP@k: OOF class/option probabilities and top-k construction
- pair accuracy: OOF A/B and swapped predictions
- span F1: character/span-level OOF predictions, not just token loss

## 4. Text Preprocessing And Tokenization

### 4.1 Do not overclean by default

Use task-fit preprocessing:

- pretrained word embeddings: normalize toward embedding coverage
- short sentiment/reviews: preserve punctuation and stopwords
- tweets: preserve hashtags and handles unless the model cannot consume them
- math problems: preserve numbers, LaTeX, symbols, and option markers
- ingredients: join ingredient lists and use n-grams for multiword ingredients
- Bangla/Arabic/non-English: normalize Unicode and script-specific artifacts
- AI-text detection: keep an original route and optionally add a heavily cleaned
  route only as ensemble diversity

### 4.2 Embedding coverage audit

For cleaning toward pretrained embedding
coverage: do not lowercase automatically, inspect OOV words, try case/plural and
punctuation variants, then decide how to initialize remaining OOV rows.

```python
from collections import Counter
import numpy as np

def embedding_coverage(texts, embedding_index, tokenizer=str.split):
    counts = Counter()
    for text in texts:
        counts.update(tokenizer(text))
    total_tokens = sum(counts.values())
    hit_tokens = sum(c for w, c in counts.items() if w in embedding_index)
    hit_vocab = sum(1 for w in counts if w in embedding_index)
    return {
        "vocab_coverage": hit_vocab / max(len(counts), 1),
        "token_coverage": hit_tokens / max(total_tokens, 1),
        "top_oov": [(w, c) for w, c in counts.most_common(1000)
                    if w not in embedding_index][:50],
    }

def get_embedding(word, embedding_index):
    for cand in (word, word.lower(), word.rstrip("s"), word.rstrip("'s")):
        if cand in embedding_index:
            return embedding_index[cand]
    return None
```

### 4.3 Offset-preserving tokenization

For long documents and span tasks, `text.split()` can drop newlines and break
offset alignment. Use fast tokenizers with offsets, and keep character-level
maps when the submission uses token ids or character ranges.

```python
o = tokenizer(
    examples["text"],
    truncation=True,
    padding=True,
    return_offsets_mapping=True,
    max_length=max_length,
    stride=stride,
    return_overflowing_tokens=True,
)
sample_mapping = o["overflow_to_sample_mapping"]
offset_mapping = o["offset_mapping"]
```

### 4.4 Dynamic batch padding

For RNN/embedding models or variable-length transformers, pad to the batch or a
batch percentile rather than the global max when runtime matters. Use batch-level padding and 95th-percentile truncation to
fit more models into runtime.

```python
def trim_batch_to_percentile(sequences, pad_id=0, pct=95):
    lengths = [len(x) for x in sequences]
    max_len = int(np.percentile(lengths, pct))
    max_len = max(max_len, 1)
    out = np.full((len(sequences), max_len), pad_id, dtype=np.int64)
    for i, seq in enumerate(sequences):
        seq = seq[:max_len]
        out[i, :len(seq)] = seq
    return out
```

## 5. Classical Baselines That Still Matter

### 5.1 TF-IDF plus linear models

A strong first baseline is often:

- word n-grams `(1, 2)` or `(1, 3)`
- char n-grams `(3, 5)` or `(3, 6)` for noisy text
- `LogisticRegression`, `LinearSVC`, `SGDClassifier(loss="modified_huber")`,
  `MultinomialNB`, or NB-SVM
- calibrated probabilities when logloss or blending requires them

Use this baseline for:

- short text
- sentiment
- toxicity
- AI-text artifacts
- language/source diagnostics
- ensemble diversity with transformers

```python
def dummy(x):
    return x

vectorizer = TfidfVectorizer(
    ngram_range=(3, 5),
    lowercase=False,
    sublinear_tf=True,
    analyzer="word",
    tokenizer=dummy,
    preprocessor=dummy,
    token_pattern=None,
    strip_accents="unicode",
)
vectorizer.fit(tokenized_texts_test)
vocab = vectorizer.vocabulary_

vectorizer = TfidfVectorizer(
    ngram_range=(3, 5),
    lowercase=False,
    sublinear_tf=True,
    vocabulary=vocab,
    analyzer="word",
    tokenizer=dummy,
    preprocessor=dummy,
    token_pattern=None,
    strip_accents="unicode",
)
tf_train = vectorizer.fit_transform(tokenized_texts_train)
tf_test = vectorizer.transform(tokenized_texts_test)
```

### 5.2 NB-SVM and Naive Bayes diversity

Naive Bayes alone is often poorly calibrated, but NB log-count ratios remain
useful for sparse text and short phrases.

```python
import numpy as np
from sklearn.linear_model import LogisticRegression

def nb_log_count_ratio(X, y, alpha=1.0):
    p = X[y == 1].sum(axis=0) + alpha
    q = X[y == 0].sum(axis=0) + alpha
    r = np.log((p / p.sum()) / (q / q.sum()))
    return np.asarray(r).ravel()

r = nb_log_count_ratio(X_train, y_train)
clf = LogisticRegression(C=4.0, max_iter=1000)
clf.fit(X_train.multiply(r), y_train)
pred = clf.predict_proba(X_valid.multiply(r))[:, 1]
```

### 5.3 Text statistics and stylometry

Useful features for LightGBM/CatBoost/linear stacks:

- text length, word count, sentence count
- punctuation counts, uppercase ratio, digit count, URL/handle count
- unique-token ratio, repeated punctuation, emoji counts
- language/script indicators
- readability and stopword ratios
- char n-gram similarity for authorship/impostor tasks
- prompt/answer length ratios for hallucination and preference tasks

## 6. Transformer Classification

### 6.1 DeBERTa-first English route

For English classification, start with:

- `microsoft/deberta-v3-base` for iteration
- `microsoft/deberta-v3-large` for serious runs
- max length based on length quantiles and truncation risk
- AdamW, cosine schedule, warmup, mixed precision
- stratified/group folds
- OOF logits/probabilities saved

Use `roberta-large`, `bert-large-cased/uncased`, `deberta-v2-xlarge`,
`deberta-v2-xxlarge`, or `electra-large` as diversity if compute allows.

### 6.2 Long text and windowing

For documents longer than the backbone limit:

- mark the target span/segment if known
- try head-only, tail-only, middle, and head+tail views
- use sliding windows for boundary tasks
- consider `longformer-large-4096` or `bigbird-roberta-large`
- aggregate window logits with mean/max/attention or a second-stage model

For Feedback-style span tasks, route to extraction primary. Still preserve the
windowing and offset lessons.

### 6.3 Segment classification with whole-document context

Feedback Effectiveness is a useful pattern when the row is a known discourse
segment but context matters:

- input the whole essay
- wrap the target discourse in markers such as `[START] ... [END]`
- optionally use type-specific markers
- pool hidden states over the marked segment
- add discourse-type auxiliary loss or features
- split by essay id

This transfers to tasks where labels apply to spans/segments already provided in
the data, not to unknown boundary discovery.

### 6.4 LLM classifiers and constrained logits

For binary or small-class tasks with LLM backbones, prefer constrained logits to
free generation:

- format the prompt with all needed fields
- train or infer a single target token such as `Yes/No`, `A/B`, `0/1`
- compute logits only over valid label tokens
- use deterministic forward passes, not text generation, when possible
- include swapped-order TTA for pair/preference tasks

```python
token_ids = [processor.get_vocab()[word] for word in ["True", "False"]]
if any(token_id == processor.get_vocab()["<unk>"] for token_id in token_ids):
    raise ValueError("One target class is not in the vocabulary.")

pre = processor(text=prompts, return_tensors="pt", padding=True,
                truncation=True, max_length=512).to(device)
with torch.no_grad():
    outputs = model(**pre)
logits = outputs.logits[:, -1, token_ids]
probabilities = torch.softmax(logits, dim=-1)
positive_probs = probabilities[:, 0]
```

```python
peft_config = LoraConfig(
    r=64,
    lora_alpha=16,
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.SEQ_CLS,
    inference_mode=False,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)
```

### 6.5 Model names by route

English closed-set:

- `microsoft/deberta-v3-base`
- `microsoft/deberta-v3-large`
- `roberta-large`
- `bert-large-uncased`
- `deberta-v2-xlarge`, `deberta-v2-xxlarge`

Multilingual:

- `xlm-roberta-large`
- `microsoft/mdeberta-v3-base`
- `sentence-transformers/LaBSE`
- language specialists: BanglaBERT, CAMeL-BERT, RuBERT, BETO, BERTurk,
  CamemBERT

LLM/logit/reranker:

- `Qwen2.5-14B-Instruct`, `Qwen2.5-32B-Instruct`,
  `Qwen2.5-72B-Instruct`
- `Qwen3-8B`, `Qwen3-14B`, `Qwen3-32B`
- `Gemma2-9B`, `Gemma2-27B`
- `Mistral-7B`, `Llama-3.x`, `Phi-4`

Retrieval embeddings:

- `all-MiniLM-L6-v2`
- `intfloat/e5-base-v2`, `intfloat/e5-large-v2`
- `thenlper/gte-base`, `thenlper/gte-large`
- `BAAI/bge-base-en-v1.5`, `BAAI/bge-large-en-v1.5`
- `stella_en_1.5B_v5`, `SFR-Embedding-2_R`, `Linq-Embed-Mistral`

## 7. Thresholds, Calibration, And Ranking

### 7.1 F1 threshold search

For F1/Fbeta, a threshold chosen on one validation split can be unstable. Use
OOF predictions across folds and prefer thresholds whose F1 degradation is
small across folds.

```python
from sklearn.metrics import f1_score
import numpy as np

def find_stable_f1_threshold(y_true, pred, thresholds=None):
    if thresholds is None:
        thresholds = np.arange(0.01, 1.0, 0.01)
    scores = [(f1_score(y_true, pred > t), t) for t in thresholds]
    return max(scores, key=lambda x: x[0])
```

For Fbeta:

```python
from sklearn.metrics import fbeta_score

def best_fbeta_threshold(y_true, pred, beta=1.0):
    thresholds = np.arange(0.01, 1.0, 0.01)
    return max((fbeta_score(y_true, pred > t, beta=beta), t)
               for t in thresholds)
```

### 7.2 Rank averaging

Rank averaging was repeatedly useful when model probability scales differed:
short-text and question-pair used rank predictions for stable F1, and many AUC/MAP tasks
benefit from rank blends.

```python
import numpy as np
from scipy.stats import rankdata

def rank_average(preds):
    ranks = [rankdata(p) / len(p) for p in preds]
    return np.mean(ranks, axis=0)
```

Use probability averaging for calibrated logloss unless rank blending improves
OOF logloss after calibration.

### 7.3 MAP@k

Multiple-choice and top-k label tasks require exact ranking logic.

```python
import numpy as np

def map3_from_probs(probs, y):
    top3 = np.argsort(-probs, axis=1)[:, :3]
    score = (
        (top3[:, 0] == y) * 1.0
        + (top3[:, 1] == y) * 0.5
        + (top3[:, 2] == y) / 3.0
    ).mean()
    return score
```

For MAP@k submissions, test:

- class id to label string mapping
- no duplicate labels per row
- correct separator and ordering
- whether probabilities or logits are sorted after any group normalization

### 7.4 Probability calibration

Use calibration when the metric is logloss, Brier, or threshold-sensitive F1:

- Platt scaling/logistic calibration on OOF
- isotonic only when OOF size is large enough
- temperature scaling for neural logits
- per-language/per-rule calibration only if OOF groups support it
- light clipping for logloss, not public-LB class-count fitting

## 8. Pairwise, Coreference, And Authorship

### 8.1 Pair input construction

Pair tasks need pair features. Do not train independent document classifiers and
subtract predictions unless that is only one auxiliary feature.

```python
def pair_text(a, b, sep="[SEP]"):
    return f"{a} {sep} {b}"

def make_pair_features(f_a, f_b, sim_features):
    feats = {}
    for name, va in f_a.items():
        vb = f_b[name]
        feats[f"{name}_a"] = va
        feats[f"{name}_b"] = vb
        feats[f"{name}_diff"] = va - vb
        feats[f"{name}_absdiff"] = abs(va - vb)
    feats.update(sim_features)
    return feats
```

### 8.2 short-text and question-pair graph features

For question pairs, combine text mining, neural cross-encoders, and graph
features built from the train+test question graph. This is transductive; use
only if the competition rules allow test graph structure.

```python
from collections import defaultdict

def question_graph_features(df):
    neighbors = defaultdict(set)
    for q1, q2 in zip(df["question1"], df["question2"]):
        neighbors[q1].add(q2)
        neighbors[q2].add(q1)
    out = []
    for q1, q2 in zip(df["question1"], df["question2"]):
        n1, n2 = neighbors[q1], neighbors[q2]
        out.append({
            "q1_freq": len(n1),
            "q2_freq": len(n2),
            "shared_neighbors": len(n1 & n2),
            "union_neighbors": len(n1 | n2),
        })
    return out
```

Add:

- TF-IDF cosine
- edit distance, LCS, token Jaccard
- prefix/suffix/question-word indicators
- length and punctuation differences
- graph degree, neighbor intersection, connected-component features
- XGBoost/LightGBM over features
- transformer cross-encoder such as DeBERTa or ESIM-style NLI model

### 8.3 Symmetry and A/B swap

For duplicate and preference tasks:

- train with `(A, B)` and `(B, A)` if label semantics permit
- at inference, average or enforce consistency
- for preference tasks, swap responses and invert probabilities

```python
def swap_average(p_ab, p_ba_for_b_wins):
    # p_ab is P(A wins) from A/B order.
    # p_ba_for_b_wins is P(B wins) from swapped B/A order.
    return 0.5 * (p_ab + p_ba_for_b_wins)
```

### 8.4 Coreference and candidate spans

Gendered pronoun tasks are not plain document classification. Preserve:

- pronoun offset
- candidate A/B offsets
- URL/title/context features
- distance buckets between pronoun and candidates
- BERT embeddings for candidate A, candidate B, and pronoun
- row-normalized 3-class probabilities with clipping

```python
pred = np.clip(pred, 1e-3, 1 - 1e-3)
pred = pred / pred.sum(axis=1, keepdims=True)
```

## 9. Moderation, Toxicity, And Rule Classification

### 9.1 Rule-conditioned input

For community-rule tasks, the row is not just a comment. The rule text and
examples define the target.

```python
SYS_PROMPT = """
You are given a comment on reddit. Your task is to classify if it violates the
given rule. Only respond Yes/No.
"""

text = f"""
Rule: {row.rule}

1) {row.positive_example_1}
Violation: Yes

2) {row.negative_example_1}
Violation: No

3) {row.body}
"""
messages = [
    {"role": "system", "content": SYS_PROMPT},
    {"role": "user", "content": text},
]
prompt = tokenizer.apply_chat_template(
    messages, add_generation_prompt=True, tokenize=False
) + "Answer:"
```

The exact prompt should be task-specific. The transferable pattern is: include
the rule, positive/negative examples when provided, the target comment, and
score only the label token.

### 9.2 Drop shortcut metadata

- keep metadata if it is an official stable covariate and validation supports it
- drop metadata if it leaks source/community rather than behavior
- test with group/source holdout rather than public LB alone

### 9.3 Per-rule score normalization

For column-averaged AUC or per-rule distribution shifts, normalize per rule only
after OOF validation:

```python
def minmax_by_group(df, score_col, group_col):
    out = df[score_col].copy()
    for _, idx in df.groupby(group_col).groups.items():
        x = df.loc[idx, score_col]
        denom = max(x.max() - x.min(), 1e-12)
        out.loc[idx] = (x - x.min()) / denom
    return out
```

Do not tune per-rule shifts from public leaderboard feedback.

### 9.4 Multilingual toxicity

When train is English but test/validation are multilingual:

- use `xlm-roberta-large` or `mdeberta-v3`
- consider translated training data if rules allow
- add monolingual specialists for languages with strong models
- validate per language or leave-language/source out
- calibrate global AUC carefully because scores are compared across languages
- blend multilingual and monolingual models by language only when language ID is
  reliable and validation supports it

## 10. AI-Generated Text And Hallucination Detection

### 10.1 Data mix is the model

generated-text detection-like tasks are often won by broad, balanced, source-aware data, not by a
single architecture. Useful data dimensions:

- human source corpora
- generated text from many LLM families
- prompts with and without source text
- rewrites, continuations, paraphrases
- obfuscation and adversarial augmentations
- prompt/source/generator labels for validation diagnostics

Avoid:

### 10.2 Strong model mix

- `microsoft/deberta-v3-large` sequence classification
- DeBERTa custom tokenizer plus MLM route for hidden-test style
- Mistral-7B QLoRA sequence classification
- Ghostbuster-style token probability features
- TF-IDF char/word n-gram artifact models
- rank-average or weighted ensemble

For hallucination detection, include both prompt/question and answer. Preserve
the beginning/end segments that carry instruction and answer behavior under
length limits.

### 10.3 Source-aware validation

Use one or more:

- prompt-held-out
- generator-family-held-out
- source-corpus-held-out
- question-held-out
- instruction-template-held-out
- time/source split if data was collected in waves

Random CV can report near-perfect results while failing hidden prompts.

### 10.4 Artifact defense

Add augmentations only when labels are preserved:

- random capitalization
- whitespace normalization variants
- punctuation/Unicode noise
- typo or character swap variants
- truncation/head-tail views

Keep an original-text model in the ensemble. Overcleaned models can help but
should not replace the original route.

## 11. Multiple Choice, MAP, And RAG-Assisted Classification

### 11.1 Multiple-choice without external evidence

If all information is in the row, use:

- format prompt/question plus all options
- `AutoModelForMultipleChoice` or DeBERTa cross-encoder
- causal LLM option-token logits over `A/B/C/D/E`
- binary option correctness scoring when positional bias is strong
- MAP@k validation and top-k submission tests

### 11.2 Retrieval-assisted science exams

Closed-book classifiers can plateau; retrieval-assisted variants may improve factual coverage.

Transferable RAG pattern:

- build a corpus of title plus chunk text
- use dense embeddings and/or BM25/TF-IDF for recall
- search question plus all options and question plus each option
- train models on retrieved contexts, not just true/oracle contexts
- infer with more context chunks than training only if validation supports it
- sort option probabilities for MAP@k

```python
import numpy as np

def dense_retrieve(query_emb, doc_emb, topk=20, batch=100000):
    # Embeddings are L2-normalized; cosine similarity is matrix product.
    best_scores = np.full((len(query_emb), topk), -np.inf, dtype=np.float32)
    best_idx = np.full((len(query_emb), topk), -1, dtype=np.int64)
    for start in range(0, len(doc_emb), batch):
        scores = query_emb @ doc_emb[start:start + batch].T
        cand_scores = np.concatenate([best_scores, scores], axis=1)
        cand_idx = np.concatenate([
            best_idx,
            np.arange(start, start + scores.shape[1])[None, :].repeat(len(query_emb), 0)
        ], axis=1)
        order = np.argsort(-cand_scores, axis=1)[:, :topk]
        best_scores = np.take_along_axis(cand_scores, order, axis=1)
        best_idx = np.take_along_axis(cand_idx, order, axis=1)
    return best_idx, best_scores
```

### 11.3 Large label pool and misconception tasks

When the label pool is large or unseen:

- treat label text as retrievable candidates
- use bi-encoder for top 32-100+ candidate recall
- rerank with DeBERTa/Qwen/Gemma/Llama cross-encoder
- use listwise LLM reranking for the final top-k if compute allows
- group validation by construct/subject/question
- generate synthetic query-label pairs only if allowed and validated

### 11.4 LLM option scoring

```python
logits = outputs.logits[:, -1, :]
scores = logits[:, choice_token_ids]  # A/B, A..E, or A..z
rank = scores.argsort(dim=-1, descending=True)
```

For binary option correctness, score each option independently and sort the
option probabilities. This reduces positional bias when options are exchangeable.

## 12. Long-Document, PII, Clinical, And Discourse Route-Outs

These tasks live near text classification but often require extraction.

### 12.1 long-document discourse extraction discourse spans

Primary route: NER/token classification with postprocess and candidate rerank.

Useful transferable points:

- split by essay/document
- preserve offsets/newlines
- use DeBERTa/Longformer/BigBird ensembles
- repair BIO beginnings
- lower threshold for high recall, then rerank candidate spans
- use overlap-aware metric implementation

```python
TP = len(tp_pred_ids)
FP = len(fp_pred_ids)
FN = len(unmatched_gt_ids)
my_f1_score = TP / (TP + 0.5 * (FP + FN))
```

Feedback 2021 first place used a two-stage strategy: token model ensemble for
candidate recall, then LightGBM sentence/span prediction over many features.
This is not a normal sentence-classification route; the first stage is boundary
discovery.

### 12.2 PII token detection

Primary route: token classification/NER.

Transferable points:

- DeBERTa-v3-large with long max length is a strong default
- split by document
- use class weights or low `O` weight for rare PII
- add external synthetic essays only if allowed
- tune label-specific thresholds
- heavy postprocess: title-case name filters, repeated-name propagation, email
  `@`, URL length, ID length, street-address newline repair, regex backfill
- ensemble at token/entity level and tune voting thresholds on OOF

```python
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels").to(self.args.device)
        outputs = model(**{k: v.to(self.args.device) for k, v in inputs.items()})
        with torch.no_grad():
            teacher_outputs = self.teacher_model(**inputs)
        loss_student = self.ce(
            outputs.logits.view(-1, self.model.config.num_labels),
            labels.view(-1),
        )
        kd = torch.nn.KLDivLoss(reduction="batchmean")(
            F.log_softmax(outputs.logits / self.temperature, dim=-1),
            F.softmax(teacher_outputs.logits / self.temperature, dim=-1),
        ) * self.temperature ** 2
        loss = self.alpha * loss_student + (1 - self.alpha) * kd
        return (loss, outputs) if return_outputs else loss
```

### 12.3 clinical span extraction clinical spans

Primary route: token/character span extraction over `(patient_note,
feature_text)`.

Transferable pattern: convert token probabilities back to character
probabilities with offsets, threshold characters, and group contiguous indices.

```python
def get_char_probs(texts, predictions, tokenizer):
    results = [np.zeros(len(t)) for t in texts]
    for i, (text, prediction) in enumerate(zip(texts, predictions)):
        encoded = tokenizer(text, add_special_tokens=True,
                            return_offsets_mapping=True)
        for offset_mapping, pred in zip(encoded["offset_mapping"], prediction):
            start, end = offset_mapping
            results[i][start:end] = pred
    return results

def get_results(char_probs, th=0.5):
    results = []
    for char_prob in char_probs:
        idx = np.where(char_prob >= th)[0] + 1
        groups = [list(g) for _, g in itertools.groupby(
            idx, key=lambda n, c=itertools.count(): n - next(c))]
        results.append(";".join(f"{min(g)} {max(g)}" for g in groups))
    return results
```

### 12.4 dataset-mention extraction style candidate type classification

Primary route: extraction/retrieval. Text classification is useful after
candidates are found:

- DOI/accession regex and dictionary extraction
- semantic search over article context
- classify `(article context, candidate id)` as primary/secondary/missing
- split by article id
- filter literature-like DOI prefixes unless context contains data terms
- use Qwen/DeBERTa/LightGBM as candidate type classifiers

## 13. Multilabel And Ontology Notes

For ordinary multilabel text tags:

- use sigmoid heads with BCE/focal/asymmetric loss
- stratify folds by frequent label combinations or use iterative multilabel
  splits
- tune per-class thresholds on OOF
- inspect rare-label support by fold
- average logits/probabilities across folds before thresholding

For protein-function ontology prediction/protein GO tasks, route out to scientific/protein prediction:

- use protein PLMs such as ESM/ESM2/ESM-1b, ProtT5, ProtBERT
- use BLAST/Diamond/Foldseek/structure evidence if allowed
- propagate child GO scores to parents in the GO DAG
- evaluate with protein-function ontology prediction/Fmax-style metrics, not ordinary text F1

Only generic multilabel mechanics belong in this text playbook.

```python
def propagate_parent_scores(scores, child_to_parents):
    # scores: dict go_term -> score for one protein.
    changed = True
    while changed:
        changed = False
        for child, parents in child_to_parents.items():
            if child not in scores:
                continue
            for parent in parents:
                if scores.get(parent, 0.0) < scores[child]:
                    scores[parent] = scores[child]
                    changed = True
    return scores
```

## 14. Ensembling And Stacking

### 14.1 Save OOF artifacts

For every model/fold/save:

- validation row ids
- OOF logits and probabilities
- test logits and probabilities
- fold id, seed, model name, preprocessing variant
- metric at prediction unit
- group ids for calibration and diagnostics

Without OOF artifacts, thresholding and stacking become public-LB guessing.

### 14.2 Diversity that worked locally

Use diversity across:

- transformer backbones
- seeds/folds
- max lengths and truncation views
- original vs lightly cleaned text
- TF-IDF word/char n-grams
- classical pair/graph features
- LLM constrained logits
- retrieval contexts and rerankers
- language-specific specialists
- pseudo-label/no-pseudo variants

### 14.3 OOF stacking

Stack only after base models are stable:

- level 1: DeBERTa, TF-IDF, LLM logits, graph/pair features, metadata
- level 2: LogisticRegression, Ridge, LightGBM, CatBoost, small MLP
- use identical folds or nested folds to avoid leakage
- calibrate stack outputs for logloss
- prefer rank blend for AUC/MAP if calibration is weak

short-text and question-pair Question Pairs used extremely deep stacking, but the transferable recipe
is not "train hundreds of models"; it is "save diverse OOF features and stack
with leakage control."

### 14.4 Pseudo labels and generated data

Use pseudo labels when:

- rules allow unlabeled/test/external data
- grouped CV improves or at least does not degrade
- pseudo labels are soft or confidence-weighted
- original supervised-only models remain in the ensemble
- pseudo-labeled rows do not leak validation folds

- multilingual toxicity test-domain pseudo labels
- AI-text detection datamix and pseudo labels
- preference modeling preference distillation from larger teachers
- PII external synthetic essays for rare labels
- clinical span extraction pseudo-labeled unlabeled notes

Cases where pseudo labels were weak or harmful also exist. Delay them.

## 15. Subtype Playbooks

### 15.1 Sentiment and short review classification

First route:

- stratified/group folds as appropriate
- word+char TF-IDF logistic/LinearSVC baseline
- DeBERTa/RoBERTa/BERT classifier
- preserve punctuation and stopwords
- tune macro-F1 or logloss directly
- blend TF-IDF and transformer probabilities

Language-specific:

- Bangla: Unicode normalization, BanglaBERT, emoji/placeholder handling
- Arabic poems: CAMeL-BERT or Arabic BERT, title+author+poem input, author
  grouped validation
- Portuguese/Brazilian tweets: script/domain tokenization and hashtag handling

### 15.2 short-text and question-pair-style insincere/toxic binary classification

First route:

- DeBERTa-v3-large or strong RNN embedding model
- TF-IDF char/word diversity
- F1 threshold search on OOF
- rank average model outputs before thresholding
- repeated seeds or folds because F1 is noisy
- inspect duplicate/conflicting labels

RNN/embedding route:

- GloVe + Paragram/FastText meta-embedding
- BiLSTM/BiGRU with pooling and small Conv1D
- statistical text features
- batch-level dynamic padding

### 15.3 Disaster tweets

Useful ideas:

- stratify or group by keyword if keyword defines train/test distribution
- audit relabeled duplicate conflicts
- URL/handle treatment should match the chosen tokenizer
- F1 threshold search, not raw 0.5
- TF-IDF baseline plus transformer

Do not preserve "perfect submission" leakage-based recipes.

### 15.4 Question pairs

First route:

- DeBERTa cross-encoder over `q1 [SEP] q2`
- pair features: TF-IDF cosine, token overlap, edit/LCS, length diff
- train/test graph features if allowed
- LightGBM/XGBoost over pair features
- target-prior calibration/rescale if train/test class priors differ and OOF
  supports it
- A/B swap consistency

### 15.5 Preference and chatbot arena

First route:

- input `prompt + response_a + response_b`
- train pairwise CE/KL on A/B winner or soft labels
- Qwen2.5-14B/Gemma2-9B/Qwen3 as LoRA or constrained-logit classifiers if
  feasible
- DeBERTa baseline if compute constrained
- A/B swap TTA and invert predictions
- split by prompt/conversation
- distill from larger teachers only if allowed

Avoid free-form judge explanations in the final scorer.

### 15.6 Hallucination detection

First route:

- include prompt/question/context and answer
- group by question/prompt/template
- DeBERTa/RoBERTa/Qwen classifier
- length/source/prompt artifact diagnostics
- threshold/calibrate based on metric
- add LLM logits/features only after baseline

### 15.7 Rule-conditioned moderation

First route:

- `rule [SEP] comment` DeBERTa/XLM-R
- constrained-logit Qwen/Gemma classifier
- deduplicate `(body, rule)` conflicts
- drop shortcut subreddit/source fields unless validated
- per-rule rank normalization for AUC only if OOF supports it
- ensemble Qwen/Llama/Gemma/DeBERTa/embedding-neighbor scores

### 15.8 Science MCQ and factual classification

First route:

- if evidence is in row: DeBERTa multiple-choice or Qwen option logits
- if evidence is external: dense/BM25 retrieval plus reranker
- score options independently if positional bias is a concern
- MAP@k construction tests
- blend retrieval contexts/backbones

### 15.9 Dataset mention type classification

Primary route is extraction/retrieval. Use classification for:

- candidate type labels
- DOI/accession type
- primary vs secondary evidence
- false-positive filters

Split by article, not candidate row.

## 16. What To Avoid

Avoid:

- treating all text tasks as single-row classification
- row-level CV when groups repeat
- public-LB threshold search or public-LB rank probing
- training on public test labels, sample-submission labels, or hidden labels
- overcleaning that removes task signal
- using test-vocabulary TF-IDF without documenting transductive rule risk
- using LLM free generation when logits over label tokens suffice
- importing CTF/exploit tactics into normal modeling
- making NER/span tasks into sentence classification
- applying WBF/NMS/span fusion to tasks where char-prob blending is better
- using huge LLMs before a correct metric/fold pipeline exists
- pseudo labels before base OOF is stable
- class/count/prior hacks without OOF evidence and rule support
