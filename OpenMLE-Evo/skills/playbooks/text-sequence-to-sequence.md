# Text Sequence-to-Sequence Category Playbook

This playbook is the full method library for Kaggle-style
`text_sequence-to-sequence` tasks. It is intended as direct reading material for
a skill-generating agent: after reading a task description, the agent should
route the task, choose the right validation geometry, then extract only the
applicable recipes.

Do not use this playbook to justify leaderboard probing, hidden labels,
sample-submission target inference, public split solving, judge attacks,
evaluator-token exploits, or disallowed external data.

## 1. Task Routing

### 1.1 Direct text transduction

Use when every row maps input text to output text:

- machine translation
- transliteration
- phonetic transcription or IPA
- OCR/noisy text cleanup
- spelling or grammar correction
- text normalization
- style transfer with a paired target

Default plan:

- split before sentence expansion or augmentation
- normalize source and target deterministically
- use byte/character seq2seq when OOV or rare script dominates
- train a strong ByT5/mT5/T5-style baseline
- decode multiple candidates and rerank by the official metric or a close proxy
- build an exact submission validator before tuning

### 1.2 Low-resource translation

Use when the task has scarce parallel data, domain-specific language, old
documents, noisy transliteration, or inconsistent orthography.

Default plan:

- treat data construction as the main modeling work
- build sentence-level parallel pairs from documents when the test unit is a
  sentence
- keep document/tablet/source identity for grouped validation and deduplication
- use `google/byt5-large` or `google/byt5-xl` when compute allows; use
  `byt5-base` for iteration
- evaluate `eval_loss` and official metric, but distrust BLEU/chrF when the
  validation set is not target-distribution
- use MBR across beams/sampling/model ensemble at inference

### 1.3 Prompt recovery and inverse rewrite generation

Use when the input is an original text plus a rewritten text and the output is a
likely prompt or transformation instruction.

Default plan:

- build a local metric replica first
- generate several plausible short prompts with instruction LLMs
- keep a strong mean/fallback prompt
- score candidates by the official embedding/perplexity metric
- optionally decode or search in embedding space
- validate tokenizer and special-token behavior against the official scorer

Do not copy local token exploits such as EOS-neighbor strings into normal tasks.

### 1.4 Summarization and grounded QA

Use when the output is a natural-language answer or summary and the input is a
document, set of chunks, or query plus corpus.

Default plan:

- choose stuffing, map-reduce, refine, or retrieval based on context length
- evaluate retrieval recall before answer quality
- constrain the answer to the task's required format
- include source-node or evidence inspection during development
- use exact-match or factual-consistency checks when possible

### 1.5 Extraction and matching

Use when the final output is a list of entities, dataset names, citations,
labels, or pipe-separated strings. This is usually not ordinary seq2seq.

Default plan:

- start with normalization, dictionary matching, acronym/long-form extraction,
  and document-frequency propagation
- add NER or span models for recall
- add a classifier/reranker for candidate filtering
- tune thresholds under the exact F-score/Jaccard/exact-match metric
- use generative models only for candidate expansion or fallback

### 1.6 Reasoning answer generation

Use when the output is a final integer/string answer and the model may write
reasoning or code.

Default plan:

- use math-reasoning methods
- use math-specialized or reasoning-specialized LLMs
- generate multiple candidates
- execute code or verify answers when possible
- extract the final answer deterministically
- majority or weighted-vote candidates under a time budget

This is a text output task but not a standard translation/summarization task.

### 1.7 Code repair

Use when the output is a patch, diff, search/replace edits, or code artifact.

Default plan:

- route to code repair / coding-agent guidance if available
- gather repository context and tests
- generate candidate reproduction tests and patches
- dry-run apply, run targeted tests, verify regressions
- abstain when confidence is low

repository-level code repair evidence is included here only as a secondary generation and
verification pattern.

### 1.8 Structured generation

Use when output must parse as JSON, XML, SVG, SQL, code, or another strict
language.

Default plan:

- state the schema and allowed tags/attributes/operators in the prompt
- generate multiple candidates
- parse/render/execute every candidate
- score valid candidates
- return a minimal safe fallback if no candidate passes

### 1.9 Pure search or game tasks

Route out unless there is a genuine seq2seq subproblem.

- constrained-permutation perplexity permutation is combinatorial optimization with an LM
  scorer.
- LLM 20 Questions is an interactive game policy with state tracking.
- LLM judge disagreement tasks are adversarial evaluator problems.

These folders still contribute reusable scorer, validator, state, and reranking
patterns.

## 2. Output Contract First

A sequence-generation task is won or lost on the exact output contract.

Before modeling, implement:

- required columns and row count
- null/empty output handling
- maximum length or byte budget
- allowed characters, tags, attributes, or vocabulary
- duplicate label behavior
- pipe/list separator behavior
- parser/render/execution validation for structured outputs
- metric normalizer: lowercase, punctuation stripping, whitespace collapse,
  Unicode normalization, or official cleaning

For generated strings, save raw and cleaned outputs separately. Debugging is
much easier when every postprocess step is reversible.

### Implementation Pattern: pipe-string normalizer

```python
import re

def clean_text(txt):
    return re.sub(r"[^A-Za-z0-9]+", " ", str(txt).lower()).strip()

def totally_clean_text(txt):
    txt = clean_text(txt)
    return re.sub(r" +", " ", txt)

def make_prediction_string(labels):
    labels = sorted({clean_text(x) for x in labels if clean_text(x)})
    return "|".join(labels)
```

Use this style for entity/list submissions, but replace the regex with the
competition's official normalizer if one is provided.

### Implementation Pattern: exact permutation validator

```python
from collections import Counter

def validate_permutation(base_text, proposed_text):
    base = Counter(str(base_text).split())
    proposed = Counter(str(proposed_text).split())
    if base != proposed:
        raise ValueError("submitted text is not a valid word permutation")
    return " ".join(str(proposed_text).split())
```

Use this only for tasks whose output is explicitly a constrained permutation or
multiset construction.

## 3. Data Construction

### 3.1 Pair alignment beats model novelty

For translation/transduction, the biggest gains usually come from making the
train unit match the test unit.

High-value work:

- split documents into sentence-level pairs when test rows are sentences
- preserve document/tablet/book/source ids during splitting
- detect alignment failures by length ratio, untranslated fragments, repeated
  text, missing target, target/source echo, and unmatched line spans
- re-extract or drop suspicious pairs
- deduplicate conflicting translations for the same source unit
- prefer original target-language translations when the same source appears in
  multiple translated editions

Do not random-split sentence pairs after alignment if many pairs came from the
same document. That leaks source style and content.

### Implementation Pattern: simple sentence aligner

```python
import re
import pandas as pd

def simple_sentence_aligner(df):
    aligned = []
    for _, row in df.iterrows():
        src = str(row["transliteration"])
        tgt = str(row["translation"])

        tgt_sents = [
            t.strip()
            for t in re.split(r"(?<=[.!?])\s+", tgt)
            if t.strip()
        ]
        src_lines = [s.strip() for s in src.split("\n") if s.strip()]

        if len(tgt_sents) > 1 and len(tgt_sents) == len(src_lines):
            for s, t in zip(src_lines, tgt_sents):
                if len(s) > 3 and len(t) > 3:
                    aligned.append({"source": s, "target": t})
        else:
            aligned.append({"source": src, "target": tgt})
    return pd.DataFrame(aligned)
```

This is a starter heuristic, not a final alignment system. In a serious
low-resource task, add document-aware LLM/OCR extraction, dictionary anchors,
line ranges, and manual or automated rechecks when rules allow.

### 3.2 Synthetic data and pseudo labels

Use synthetic or pseudo-labeled data only when rules allow and validation can
detect noise.

Useful cases:

- dictionary gap coverage for rare terms
- LLM-assisted sentence alignment from allowed documents
- teacher translation for unlabeled target-domain text
- heterogeneous teacher/student iteration
- generated reasoning traces filtered by verifier correctness

Risky cases:

- synthetic data with no source grounding
- target-domain test pseudo-labeling under unclear rules
- self-training the same architecture on its own outputs
- generated labels that cannot be checked by metric/verifier

For low-resource historical-language translation, LLM-labeled or extracted data was valuable when
grounded in real documents; purely morphological translation generation could
hurt.

### 3.3 Domain and metadata tokens

Prefix tokens are high-value when they identify a stable condition of the input:

- source language or target language
- dialect or region
- document source
- task type
- style/tone
- speaker or domain

Do not add metadata tokens that are unavailable, noisy, or distribution-shifted
at test time.

### Implementation Pattern: district-guided token

```python
district_tokens = [f"<district_{d}>" for d in sorted(train_df["district"].unique())]
tokenizer.add_special_tokens({"additional_special_tokens": district_tokens})
model.resize_token_embeddings(len(tokenizer))

train_df["input_text"] = (
    "<district_" + train_df["district"].astype(str) + "> "
    + train_df["regional_text"].astype(str)
)
train_df["target_text"] = train_df["ipa"].astype(str)
```

## 4. Normalization And Cleaning

### 4.1 Source normalization

Normalize the input enough to remove accidental surface noise, but preserve
scored distinctions.

Common operations:

- Unicode normalization and whitespace collapse
- OCR artifact repair
- gap/broken-text markers
- subscript/diacritic conventions
- deterministic casing rules
- punctuation standardization
- source-specific aliases and abbreviations
- language tags or task prefixes

For rare scripts and transliterations, use byte-level models if normalization
cannot make the vocabulary stable.

### Implementation Pattern: gap marker normalization

```python
import re
import pandas as pd

class TextPreprocessor:
    def __init__(self):
        self.big_gap = re.compile(r"(\.{3,})")
        self.small_gap = re.compile(r"(xx+|\s+x\s+)")

    def preprocess_one(self, text):
        if pd.isna(text):
            return ""
        text = str(text)
        text = self.big_gap.sub("<big_gap>", text)
        text = self.small_gap.sub("<gap>", text)
        return text

    def preprocess_batch(self, texts):
        return [self.preprocess_one(t) for t in texts]
```

If the original corpus uses non-ASCII ellipsis or special marks, encode them
explicitly in the target task normalizer. Keep This playbook file ASCII-only.

### 4.2 Target postprocessing

Postprocess only what validation supports.

Usually safe:

- strip leading/trailing whitespace
- collapse repeated spaces
- remove repeated words or repeated n-grams when the model loops
- normalize output markers such as `<gap>`
- normalize fractions or numeric format if the official metric expects it
- remove disallowed characters for structured output

Risky:

- broad proper-name replacement
- LLM post-editing without a verifier
- hand-coded public-LB cleanup
- task-specific constants copied from another competition

### Implementation Pattern: light repeated-ngram cleanup

```python
import re

def clean_repetitions(text):
    text = re.sub(r"\s+", " ", str(text)).strip()
    text = re.sub(r"\b(\w+)(?:\s+\1\b)+", r"\1", text)
    for n in range(4, 1, -1):
        pattern = r"\b((?:\w+\s+){" + str(n - 1) + r"}\w+)(?:\s+\1\b)+"
        text = re.sub(pattern, r"\1", text)
    text = re.sub(r"\s+([.,:;])", r"\1", text)
    return text.strip()
```

Use OOF validation to decide whether repeated-ngram cleanup helps or deletes
legitimate repetitions.

## 5. Model Selection

### 5.1 Translation and transduction models

Start here:

- `google/byt5-small/base`: fast iteration, strong OOV handling
- `google/byt5-large`: strong first serious model for low-resource/noisy text
- `google/byt5-xl`: high ceiling when data is clean and inference budget fits
- `google/mt5-base/large`: multilingual subword route when language coverage is
  good
- `google/umt5-base`: multilingual T5 variant for lower-resource tasks
- `csebuetnlp/banglat5`: language-specific T5-style model when language matches
- `MADLAD-400-3B-MT`: translation diversity or ensemble member
- `mBART`, `NLLB`, `M2M100`: translation routes if allowed and language support
  fits

For paired correction/rewrite:

- `t5-base/large`, `flan-t5-base/large`
- `bart-large`, `pegasus`, `long-t5`, `LED` for summarization
- domain-specific T5/BART variants when task text matches training domain

### 5.2 Decoder-only LLMs

Use when:

- training data is small and instruction following matters
- output format is flexible natural language
- task is reasoning, prompt recovery, code, or structured artifact generation
- supervised seq2seq fine-tuning is infeasible

- Gemma 2B/7B/9B/27B
- Mistral 7B, Mixtral 8x7B, OpenChat/Solar
- Phi-2/Phi-4
- Llama 3/3.1 variants
- Qwen2.5/Qwen3 and Qwen2.5-Coder
- DeepSeekMath, DeepSeek-R1-Distill, QwQ
- NuminaMath and OpenMath/Nemotron math models

Prefer decoder-only LLMs after deciding that the task really needs instruction
generation, reasoning, or code behavior. Do not replace a strong ByT5
translation route with a generic chat model just because the task output is
text.

### 5.3 Quantization and serving

Use quantization to fit the time budget, not as an afterthought.

Useful options:

- bitsandbytes 4-bit NF4/FP4 for Hugging Face inference
- GPTQ/AWQ for LLM sampling
- CTranslate2 `int8_float32` for T5/ByT5-style inference
- vLLM, lmdeploy, TensorRT-LLM for high-throughput sampling
- prefix caching for duplicated prompts or branch expansion
- length sorting and bucket batching

Every engine/version can change output distribution, speed, memory, and even
score tie behavior. Validate the exact serving stack.

## 6. Training Recipes

### 6.1 Strong seq2seq fine-tuning baseline

For a supervised paired dataset:

- add task/domain prefix
- tokenize source and target with truncation
- use `DataCollatorForSeq2Seq`
- train with Adafactor or AdamW
- enable `predict_with_generate`
- save OOF generated strings
- score decoded strings using official metric

### Implementation Pattern: Seq2SeqTrainer skeleton

```python
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)

model_name = "google/byt5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

def preprocess_function(examples):
    inputs = [str(x) for x in examples["input_text"]]
    targets = [str(x) for x in examples["target_text"]]
    model_inputs = tokenizer(inputs, max_length=512, truncation=True)
    labels = tokenizer(targets, max_length=512, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

dataset = Dataset.from_pandas(train_df)
tokenized = dataset.map(preprocess_function, batched=True)
split = tokenized.train_test_split(test_size=0.1, seed=42)

args = Seq2SeqTrainingArguments(
    output_dir="./seq2seq_model",
    learning_rate=1e-4,
    optim="adafactor",
    label_smoothing_factor=0.1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    per_device_eval_batch_size=1,
    num_train_epochs=10,
    predict_with_generate=True,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    report_to="none",
)

trainer = Seq2SeqTrainer(
    model=model,
    args=args,
    train_dataset=split["train"],
    eval_dataset=split["test"],
    tokenizer=tokenizer,
    data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
)
trainer.train()
```

Adapt batch size, max length, and epochs to the target hardware. For length
grouping, consider Adafactor `beta1=0.9` or AdamW if gradients become unstable.

### 6.2 Two-stage and three-stage training

Use staged training when domain data and supervised pairs differ:

- Stage 1: continual pretraining on source-domain text or broad generated CoT
- Stage 2: supervised fine-tuning on clean paired data
- Stage 3: final alignment on competition-style high-quality data, TIR traces,
  or filtered pseudo-labels

low-resource historical-language translation and mathematical reasoning both support staged training, but with different goals:

- translation: learn domain orthography and then precise source-target mapping
- math: learn broad reasoning, then tool-integrated or shorter efficient
  reasoning behavior

### 6.3 LoRA and adapter tuning

Use LoRA when:

- full fine-tuning is too expensive
- the model is decoder-only and instruction tuning is enough
- task is small-demo oriented
- you need several small variants for diversity

- LoRA rank 4 to 16 for Gemma/Qwen-style models
- sequence length 256 to 1024 for short QA/translation demos
- 4-bit quantized base plus adapter training
- merge adapter for inference if runtime allows

For serious low-resource translation, full fine-tuning of ByT5/mT5 often has a
clearer path than LoRA on a generic chat model.

## 7. Decoding, Candidate Generation, And Reranking

### 7.1 Beam, sampling, and diversity

Recommended candidate pool:

- deterministic beam candidates for quality
- sampling candidates for diversity
- different checkpoints or data mixes for ensemble diversity
- different prefixes or prompts for LLM diversity

Do not blindly increase candidates. Candidate count only helps when a scorer,
verifier, or voting rule can select better outputs.

### Implementation Pattern: adaptive beam generation

```python
def get_adaptive_beam_size(attention_mask, base_beams=8):
    lengths = attention_mask.sum(dim=1)
    short_beams = max(4, base_beams // 2)
    return short_beams if int(lengths[0]) < 100 else base_beams

with torch.inference_mode():
    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        num_beams=get_adaptive_beam_size(attention_mask, 8),
        max_new_tokens=256,
        length_penalty=1.3,
        early_stopping=True,
    )
```

Length-adaptive decoding is a runtime and diversity heuristic, not a universal
metric optimizer. Validate it against fixed beams.

### 7.2 MBR selection

Minimum Bayes Risk style selection works when there are many plausible
candidates and the final metric rewards consensus.

Useful similarity functions:

- chrF++ for translation
- BLEU or token F1 when the official metric is n-gram based
- Jaccard for label/entity strings
- sentence embedding cosine for prompt recovery
- length and source-coverage rewards when the metric is sensitive to omission

### Implementation Pattern: chrF++ MBR

```python
import sacrebleu

chrfpp = sacrebleu.metrics.CHRF(word_order=2)

def dedup_keep_order(xs):
    seen, out = set(), []
    for x in xs:
        x = str(x).strip()
        if x and x not in seen:
            seen.add(x)
            out.append(x)
    return out

def mbr_pick(candidates, pool_cap=32):
    candidates = dedup_keep_order(candidates)[:pool_cap]
    if not candidates:
        return ""
    if len(candidates) == 1:
        return candidates[0]

    best_i, best_score = 0, -1.0
    for i, cand in enumerate(candidates):
        score = 0.0
        for j, other in enumerate(candidates):
            if i != j:
                score += chrfpp.sentence_score(cand, [other]).score
        score /= max(1, len(candidates) - 1)
        if score > best_score:
            best_i, best_score = i, score
    return candidates[best_i]
```

### 7.3 Perplexity reranking

Perplexity reranking is useful when:

- official metric is perplexity
- choosing among paraphrases where fluency matters
- prompt recovery has a candidate prompt list and the rewritten text is known

It is weak when semantic faithfulness matters and the LM prefers generic text.

### Implementation Pattern: candidate prompt perplexity

```python
import torch
import numpy as np
from torch import nn

class PerplexityLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, logits, labels):
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        losses = []
        for i in range(labels.shape[0]):
            losses.append(self.loss_fn(shift_logits[i], shift_labels[i]))
        return torch.stack(losses)

def format_prompt(row, prompt):
    return (
        "<start_of_turn>user\n"
        f"{prompt}\n{row['original_text']}<end_of_turn>\n"
        "<start_of_turn>model\n"
        f"{row['rewritten_text']}<end_of_turn>"
    )

def choose_lowest_loss_prompt(row, rewrite_prompts, model, tokenizer):
    samples = [format_prompt(row, p) for p in rewrite_prompts]
    inputs = tokenizer(samples, return_tensors="pt", padding=True, truncation=True).to(model.device)
    with torch.no_grad():
        logits = model(**inputs).logits
    labels = inputs["input_ids"].clone()
    labels.masked_fill_(~inputs["attention_mask"].bool(), -100)
    losses = PerplexityLoss()(logits, labels).detach().cpu().numpy()
    return np.array(rewrite_prompts)[np.argsort(losses)][0]
```

Always confirm whether lower CE/perplexity correlates with the official score.

## 8. Low-Resource Translation Playbook

### 8.1 Data-first sequence

1. Inspect train/test unit: sentence, line, paragraph, full document.
2. Create sentence-level pairs only when they preserve meaning.
3. Normalize source orthography.
4. Deduplicate by physical source/document id.
5. Build one clean validation fold from unseen source groups.
6. Fine-tune ByT5-base for quick feedback.
7. Scale to ByT5-large/xl and/or one diverse model.
8. Decode with beams plus sampling.
9. MBR/rerank with chrF++ and source-coverage checks.
10. Add quantized inference, length sorting, and parallel postprocessing.

### 8.2 Data cleaning checks

High-value filters:

- source length too long for target
- target length too short or too long
- repeated substrings
- target appears to be untranslated source
- source contains OCR junk
- target contains non-target language
- missing gap markers or mismatched line numbers
- duplicate source with conflicting target

### Implementation Pattern: length mismatch filter

```python
def add_length_diagnostics(df, src_col="source", tgt_col="target"):
    out = df.copy()
    out["src_len"] = out[src_col].astype(str).str.len()
    out["tgt_len"] = out[tgt_col].astype(str).str.len()
    out["len_diff"] = (out["tgt_len"] - out["src_len"]).abs()
    out["len_ratio"] = out["tgt_len"] / out["src_len"].clip(lower=1)
    return out

diag = add_length_diagnostics(train_pairs)
suspect = diag[(diag["len_diff"] > 100) | (diag["src_len"] > 500)]
clean = diag.drop(index=suspect.index)
```

Treat threshold values as task-dependent. Use this to identify rows for
re-extraction, not as a blind deletion rule.

### 8.3 Inference runtime

Translation systems often benefit from fitting more candidates into the time limit.

Useful patterns:

- sort test by source length
- bucket by length to reduce padding
- preconvert models to CTranslate2 when supported
- use 4 CPU cores for MBR/postprocess if available
- save periodic checkpoints for long inference
- use BF16/FP16/INT8 only after parity check

## 9. Prompt Recovery Playbook

Prompt recovery is a metric-aware generation problem. The recovered prompt need
not be human-perfect if the embedding metric rewards a different string.

### 9.1 Candidate sources

Combine:

- short generic rewrite prompts
- LLM-predicted prompts from original/rewrite pairs
- few-shot prompt templates
- cluster-specific fallback prompts
- mean prompts optimized on validation
- small suffixes/tags that improve local metric
- embedding-decoded token strings if the metric is embedding-based

### 9.2 Candidate cleanup

Useful cleanup:

- strip numbered lists and quotes
- keep only first sentence when output should be short
- remove "Here is..." preambles
- enforce a minimum length fallback
- concatenate diverse prompt phrasings only if metric rewards it

### Implementation Pattern: LLM prompt recovery generation

```python
def recover_prompt(original_text, rewritten_text, model, tokenizer, max_in=1024, max_out=100):
    prompt = (
        "Instruct: Original Text:"
        f"{original_text}\n"
        "Rewritten Text:"
        f"{rewritten_text}\n"
        "Write a prompt that was likely given to the LLM to rewrite original text "
        "to rewritten text.\nOutput:"
    )
    inputs = tokenizer(prompt, max_length=max_in, truncation=True, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    outputs = model.generate(
        **inputs,
        do_sample=False,
        max_length=inputs["input_ids"].shape[1] + max_out,
        pad_token_id=tokenizer.pad_token_id,
    )
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text.split("Output:", 1)[-1].strip()
```

For final selection, score generated prompts against the official metric or
proxy. Do not assume the semantically best prompt is the metric-best prompt.

### 9.3 Metric exploit quarantine

Prompt-recovery tasks may contain special-token and tokenizer attacks. Treat
them as lessons:

- scorer implementation details matter
- Hugging Face and TensorFlow tokenizers can disagree
- local metric parity must be verified
- exploit strings are not general NLP knowledge

Do not include these strings in a generated task skill unless the task is
explicitly about adversarial metric analysis.

## 10. Extraction, Matching, RAG, And QA

### 10.1 Extraction is not free generation

For entity/list outputs, use this order:

1. exact dictionary matching against known labels
2. normalized matching with raw and cleaned text
3. acronym/long-form extraction
4. NER/span model for new mentions
5. classifier/reranker to remove false positives
6. document-frequency propagation for high-confidence long forms
7. generative model only for uncertain or long-tail candidates

False positives often hurt more than missed rare entities under F0.5 or
precision-weighted metrics.

### Implementation Pattern: literal matching stack

```python
def literal_match_labels(paper_sections, all_labels):
    raw_text = ". ".join(section["text"] for section in paper_sections).lower()
    clean = totally_clean_text(raw_text)
    hits = set()
    for label in all_labels:
        label = str(label).lower()
        if label in raw_text or clean_text(label) in clean:
            hits.add(clean_text(label))
    return hits
```

Stack `dataset_title`, `dataset_label`, `cleaned_label`, curated external
labels, and manual labels only if the task rules allow them.

### 10.2 NER span route

For new entities, convert exact labels to BIO spans and train a token classifier.
Use it as candidate recall, not necessarily the final scorer.

### Implementation Pattern: BIO tagging from exact labels

```python
def find_sublist(big_list, small_list):
    positions = []
    for i in range(len(big_list) - len(small_list) + 1):
        if small_list == big_list[i : i + len(small_list)]:
            positions.append(i)
    return positions

def tag_sentence(sentence, labels):
    words = sentence.split()
    tags = ["O"] * len(words)
    is_positive = False
    for label in labels:
        label_words = label.split()
        for pos in find_sublist(words, label_words):
            is_positive = True
            tags[pos] = "B"
            for i in range(pos + 1, pos + len(label_words)):
                tags[i] = "I"
    return is_positive, list(zip(words, tags))
```

### 10.3 RAG route

For grounded QA:

- chunk documents with metadata
- embed chunks
- retrieve top-k with BM25/vector/hybrid search
- rerank with cross-encoder or BGE-style reranker
- generate answer from retrieved context only
- return fallback when context is weak
- save source nodes for audits

### Implementation Pattern: minimal DSPy RAG

```python
import dspy

class RAG(dspy.Module):
    def __init__(self, num_passages=3):
        super().__init__()
        self.retrieve = dspy.Retrieve(k=num_passages)
        self.generate_answer = dspy.ChainOfThought("context, question -> answer")

    def forward(self, question):
        context = self.retrieve(question).passages
        pred = self.generate_answer(context=context, question=question)
        return dspy.Prediction(context=context, answer=pred.answer)
```

### 10.4 Summarization routes

Use:

- stuffing for short documents
- map-reduce for long independent chunks
- refine for ordered documents where later chunks update a running summary
- RAG summarization when only query-relevant chunks matter

Preserve headings, section titles, and metadata in chunk text when they help the
answer.

## 11. Reasoning Answer Generation

Math and reasoning tasks are not solved by ordinary seq2seq training unless the
model has learned the reasoning domain.

### 11.1 First strong route

- choose a math/reasoning-specialized model
- use multiple candidate generations
- include code prompts when arithmetic or enumeration is common
- run code in a sandbox
- parse final answer from code output or `boxed{...}`
- vote answers, not rationales
- stop early when consensus is strong

### Implementation Pattern: tool-integrated loop

```python
def tir_round_inference(problem, num_sequences, temperature, rounds):
    prompt = problem + "\nPlease integrate reasoning with programs and put the final answer within \\boxed{}."
    texts = [prompt for _ in range(num_sequences)]
    answers, traces = [], []

    for _ in range(rounds):
        generated = llm_generate(texts, temperature=temperature, max_tokens=1024)
        next_texts = []
        for text, new_text in zip(texts, generated):
            completed = False
            if "```python" in new_text:
                try:
                    code = new_text.split("```python")[-1].split("```")[0]
                    code_result = run_python_sandbox(code)
                    result = int(float(str(code_result).strip()))
                    if result >= 0:
                        answers.append(result % 1000)
                        traces.append(text + new_text)
                        completed = True
                except Exception:
                    code_result = "Code execution failed or output is not an integer."
            if not completed:
                next_texts.append(text + new_text + "\n```output\n" + str(code_result) + "\n```")
        texts = next_texts
        if not texts:
            break
    return answers, traces
```

The exact sandbox and parser are task-specific. Never execute untrusted code
outside a sandbox.

### 11.2 Voting

Start with majority vote. Upgrade when there is a verifier or reward model:

- weighted majority by reward score
- geometric mean times count to avoid one high false-positive score
- downweight suspicious frequent wrong answers such as `0`
- ignore invalid, missing, or non-integer answers
- penalize answers copied from prompt when that is a known failure

### Implementation Pattern: weighted geometric vote

```python
import math

def weighted_geometric_mean_times_num(answers, scores):
    buckets = {}
    for answer, score in zip(answers, scores):
        if answer == 0:
            score /= 10.0
        buckets.setdefault(answer, []).append(max(float(score), 1e-9))

    best_answer, best_weight = None, -1.0
    for answer, vals in buckets.items():
        weight = len(vals) * math.prod(vals) ** (1.0 / len(vals))
        if weight > best_weight:
            best_answer, best_weight = answer, weight
    return best_answer
```

Validate every heuristic on local representative problem sets across multiple
seeds.

### 11.3 Time budget

Use dynamic budgets:

- fixed base seconds per problem
- unused-time buffer
- fewer samples when time is low
- early stop at 4/5, 5/7, or mathematically unbeatable consensus
- stop stragglers after most candidates finish
- reduce max tokens for easier problems or late run

## 12. Code Repair And Patch Generation

Code repair tasks require coding-agent methods. Keep this
section for reusable seq2seq output-control and verification patterns.

### 12.1 High-confidence pipeline

1. Decide whether the issue is worth attempting.
2. Gather repository tree and likely files.
3. Retrieve snippets by structured search strings, AST symbols, traceback
   locations, and related tests.
4. Generate reproduction tests.
5. Keep only tests that reproduce the issue without unrelated failures.
6. Generate multiple candidate patches.
7. Validate patch format and dry-run apply.
8. Run reproduction tests and targeted regression tests.
9. Submit the best verified patch or abstain.

Wrong patches can be heavily penalized. Abstention is a modeling decision.

### Implementation Pattern: structured file-query parser

```python
import re
import xml.etree.ElementTree as ET

def extract_file_query(xml_content):
    parsed = {}
    matches = re.findall(r"<root>(.*?)</root>", xml_content, re.DOTALL)
    for match in matches:
        try:
            root = ET.fromstring("<root>" + match + "</root>")
            for entry in root.findall("entry"):
                filepath = entry.findtext("filepath", default="").strip()
                strings = []
                container = entry.find("strings_to_search")
                if container is not None:
                    for node in container.findall("string_to_search"):
                        if node.text:
                            strings.append(node.text.strip())
                if filepath:
                    parsed[filepath] = strings
        except Exception:
            return {}
    return parsed
```

Structured outputs reduce parsing errors, but still need robust fallbacks.

### Implementation Pattern: patch extraction and dry-run gate

```python
import re
from unidiff import PatchSet

def extract_patch_string(text):
    matches = re.findall(r"\n```diff\n(.*?)\n```", text, re.DOTALL)
    if not matches:
        return None
    return matches[-1].strip() + "\n"

def patch_is_parseable(patch_string):
    if not patch_string:
        return False
    try:
        patch = PatchSet(patch_string)
        return any(file for file in patch)
    except Exception:
        return False
```

Run the actual `patch --dry-run` or `git apply --check` in the target
environment before trusting a patch.

### 12.2 Search/replace over raw diff

- exact file path
- contiguous old block
- fully indented replacement
- convert to diff after matching
- reject if the search block is ambiguous or absent

This is especially useful for a downstream Codex-level agent that can apply
edits programmatically.

## 13. Perplexity And Permutation Optimization

constrained-permutation tasks are not ordinary generation. The output is constrained by a
multiset, and the metric is a local LM score.

### 13.1 Build the scorer first

Validate:

- exact tokenizer and model path
- BOS/EOS handling
- padding ignore index
- quantization parity
- batch size effects
- whitespace normalization
- legal permutation check

### Implementation Pattern: batched perplexity scorer

```python
import math
import torch

PAD_TOKEN_LABEL_ID = torch.nn.CrossEntropyLoss().ignore_index

def get_perplexity(model, tokenizer, texts, device, batch_size=32):
    if isinstance(texts, str):
        texts = [texts]
        single = True
    else:
        single = False

    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    out = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        with torch.no_grad():
            marked = [f"{tokenizer.bos_token}{x}{tokenizer.eos_token}" for x in batch]
            inputs = tokenizer(marked, return_tensors="pt", add_special_tokens=False, padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            logits = model(**inputs, use_cache=False).logits
            labels = inputs["input_ids"].clone()
            labels[labels == tokenizer.pad_token_id] = PAD_TOKEN_LABEL_ID

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            loss = loss.view(len(batch), -1)
            valid_len = (shift_labels != PAD_TOKEN_LABEL_ID).sum(dim=-1)
            seq_loss = torch.sum(loss, dim=-1) / valid_len
            out.extend([math.exp(float(x)) for x in seq_loss.cpu()])
    return out[0] if single else out
```

### 13.2 Search patterns

Useful:

- brute force for tiny samples
- stopwords/function words first as starting state
- sorted content words or two sorted blocks
- word insert
- phrase insert
- phrase swap or double-bridge
- simulated annealing with batched candidates
- iterated local search with kicks
- score caching
- distance-matrix pruning from good local optima

Avoid assuming this route generalizes to normal text generation. It is a search
problem with an LM as scoring oracle.

## 14. Structured Generation And Creative Artifacts

### 14.1 Generate, sanitize, render, score

For JSON/XML/SVG/code:

- prompt with allowed schema or tags
- extract the last valid fenced block or root element
- parse strictly
- remove disallowed fields
- render/execute if applicable
- score or rank valid candidates
- return a known-valid fallback on timeout or parse failure

### Implementation Pattern: SVG sanitizer and fallback

```python
import re
from lxml import etree

def enforce_svg_constraints(svg_string, constraints, default_svg):
    try:
        parser = etree.XMLParser(remove_blank_text=True, remove_comments=True)
        root = etree.fromstring(svg_string, parser=parser)
    except etree.ParseError:
        return default_svg

    elements_to_remove = []
    path_regex = re.compile(
        r"^(?:[MmZzLlHhVvCcSsQqTtAa]\s*"
        r"(?:-?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?"
        r"(?:[\s,]+-?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?)*)?\s*)+$"
    )

    for element in root.iter():
        tag = etree.QName(element.tag).localname
        if tag not in constraints.allowed_elements:
            elements_to_remove.append(element)
            continue
        allowed_attrs = set(constraints.allowed_elements[tag]) | set(
            constraints.allowed_elements["common"]
        )
        for attr in list(element.attrib):
            name = etree.QName(attr).localname
            if name not in allowed_attrs:
                del element.attrib[attr]
        if tag == "path":
            d = element.get("d")
            if not d or not path_regex.match(d):
                elements_to_remove.append(element)

    for element in elements_to_remove:
        if element.getparent() is not None:
            element.getparent().remove(element)
    try:
        return etree.tostring(root, encoding="unicode")
    except Exception:
        return default_svg
```

Validate the rendered artifact too. A parseable SVG can still fail conversion or
score poorly.

### 14.2 Text-to-SVG route

Non-exploit reusable pattern:

- generate multiple simple, flat vector-like bitmaps or SVGs
- prefer solid fills, few colors, clear shapes, hard boundaries
- vectorize with contours or a vectorizer
- compress paths and respect byte budget
- rerank by official or proxy VQA/aesthetic scorer
- use fallback SVG

## 15. Interactive Dialogue And Game Agents

LLM 20 Questions is a stateful policy task. Reusable sequence-generation lessons:

- maintain explicit state from previous questions/answers
- summarize knowledge to reduce context burden
- coerce answer outputs to legal values
- use rule handlers for spelling, lexicographic, list, and arithmetic questions
- use entropy or probability-mass splits for question selection
- use LLM fallback only after deterministic rules fail

### Implementation Pattern: yes/no answer coercion

```python
def coerce_yes_no(text):
    text = str(text).strip().lower()
    if text.startswith("yes") or " yes" in text[:12]:
        return "yes"
    if text.startswith("no") or " no" in text[:12]:
        return "no"
    return "no"
```

For tasks with strict output alphabets, never trust raw LLM text directly.

### Implementation Pattern: balanced question selection

```python
def choose_balanced_question(candidate_table, used_questions):
    best_q, best_balance = None, -1.0
    n = len(candidate_table)
    for q in candidate_table.columns:
        if q in used_questions:
            continue
        yes_count = (candidate_table[q] == 1).sum()
        frac = yes_count / max(1, n)
        balance = min(frac, 1.0 - frac)
        if balance > best_balance:
            best_q, best_balance = q, balance
    return best_q
```

For probability-weighted keyword lists, split by probability mass, not raw count.

## 16. Validation And Metric Optimization

### 16.1 Split geometry

Use the strongest leakage-safe split the data supports:

- document/book/tablet split for translation/extraction
- publication/source split for scientific papers
- dialect/region stratification for IPA/transcription
- prompt-template/source text split for prompt recovery
- problem-family split for math
- repository split for code repair
- topic/episode split for game/dialogue tasks

Split before expansion, chunking, sentence alignment, pseudo-labeling, or
generated examples.

### 16.2 Metric replicas

Implement the local metric early:

- BLEU/chrF/edit/WER/CER for transduction
- exact match and normalizer for extraction/answers
- embedding cosine for prompt recovery
- perplexity for LM-scored tasks
- parser/render/execution for structured outputs
- pass/fail tests for code repair
- retrieval recall and answer exactness for RAG

Save OOF raw outputs, cleaned outputs, metric components, and failure reasons.

### 16.3 Candidate selection from OOF

Choose models and postprocessing by OOF:

- single-model decoded score
- ensemble or MBR gain
- fallback usage rate
- invalid output rate
- per-length/domain/source failure slices
- public/private-like split diagnostics
- runtime and timeout rate

Do not choose a postprocess solely because it improved public LB once.

## 17. High-ROI Upgrades Across Rounds

Round 2 ideas:

- stronger normalization and data diagnostics
- byte-level model if OOV is high
- task/dialect/source prefix tokens
- official metric helper and submission validator
- beam plus sampling candidate pool
- MBR or simple reranker
- length sorting and bucket batching
- extraction dictionary plus NER recall stack

Round 3 ideas:

- larger ByT5/mT5 or one diverse model family
- additional allowed domain data
- re-extract suspicious document pairs
- pseudo labels filtered by agreement/verifier
- reward/verifier/voting for answer tasks
- structured parser/rerender/rerank loop
- LoRA variants for prompt diversity
- OOF blend search across model/data variants

Late round ideas:

- CTranslate2 or quantized engine conversion
- parallel MBR/reranking
- metric-specific threshold sweeps
- high-confidence fallback or abstention rules
- error-slice specialist models
- dynamic time budget and early stopping
- task-specific small lexicons or format repair

## 18. Avoid Or Delay

Avoid:

- generic "fine-tune a transformer" advice without output validator
- random row CV after document expansion
- training on noisy synthetic data before cleaning real pairs
- direct generative extraction when search/NER/dictionary is stronger
- broad LLM post-editing without metric verification
- one-sample reasoning answers
- direct unified diff generation without parse/dry-run checks
- structured output without parser and fallback
- hidden judge attacks and special-token exploits in ordinary tasks
- public split solving, leaderboard probing, or score-equation tuning

Delay:

- pseudo-labeling
- large LLMs
- reward models
- complex agent loops
- heavy ensembling
- online adaptation
- differentiable artifact optimization

until the first serious route, validation, metric replica, and output validator
are stable.
