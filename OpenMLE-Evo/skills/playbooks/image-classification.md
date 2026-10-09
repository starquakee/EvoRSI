# Image Classification Category Playbook

This playbook is a method library for Kaggle image-classification tasks. It is
written for a later skill-generation agent: read the task description first,
route to the relevant sections, and extract a compact task-specific skill. Do
not copy the entire playbook into a final skill.

## 1. What Counts As Image Classification

Use this playbook when the final target is a class, probability, ordinal grade,
multilabel vector, retrieval rank, or verification score derived from images or
image-like inputs.

Covered inputs include:

- ordinary RGB images
- small grayscale symbols and digits
- fine-grained plant, animal, product, and landmark images
- medical DICOM slices, CT/MRI studies, and X-rays
- whole-slide histology and tissue microarrays
- dermoscopy with patient metadata
- multi-channel microscopy
- keypoint sequences and strokes treated as image-like classification
- giant images decomposed into detections/crops before a final class/count
- camera/generator forensic images
- pairwise identity or kinship verification

Route out if the final submission is primarily masks, boxes, points, text,
programs, transformed grids, or an interactive policy. Use this playbook only for
secondary classifier gates or feature extraction in those cases.

## 2. Safety Boundary

- public/private leaderboard probing
- public-test label recovery
- sample-submission labels as training labels
- public-test class-count fitting
- old-rule loopholes
- manually labeled hidden test data
- disallowed external data
- submission-row leakage
- same-patient/source leakage across folds

The safe default is to build robust validation and
avoid public-LB overfitting, not to reproduce the exploit.

## 3. Output Contract First

Before modeling, write down:

- prediction unit: image, patient, study, slide, tile, pair, sequence, zone,
  label row, or top-k candidate
- target type: single-label, multilabel, ordinal, binary, retrieval rank,
  pair probability, outlier class, or custom weighted group
- metric: accuracy/logloss/AUC/pAUC/QWK/F1/F2/mAP/GAP/balanced accuracy/custom
- group keys: patient, study, series, slide, signer, writer, source, device,
  family, product, duplicate cluster, location, or session
- legal data channels: external data, previous competition data, unlabeled test
  data, metadata, image EXIF, segmentation masks, annotations, or none
- submission constraints: probability columns, class IDs, space-delimited top-k,
  one row per image-class, one row per study-label, TFLite, runtime limit

The model choice should follow this contract. A ConvNeXt that optimizes image
accuracy is the wrong first model if the metric is patient-level weighted
logloss after consistency constraints.

## 4. Validation Geometry

### 4.1 Group-Aware Split

Use group-aware folds when one real entity appears multiple times. This is
mandatory for medical imaging, WSI, keypoints by signer, writer ID, pairwise
identity, camera/source tasks, and any near-duplicate data.

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

def make_group_folds(df, label_col, group_col, n_splits=5, seed=42):
    df = df.copy()
    df["fold"] = -1
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for f, (_, va) in enumerate(cv.split(df, df[label_col], groups=df[group_col])):
        df.loc[df.index[va], "fold"] = f
    assert (df["fold"] >= 0).all()
    for f in range(n_splits):
        tr_g = set(df.loc[df.fold != f, group_col])
        va_g = set(df.loc[df.fold == f, group_col])
        assert not (tr_g & va_g)
    return df
```

If `StratifiedGroupKFold` creates rare-class gaps, repair folds manually or use
a repeated split and select the one with the best fold-level label support.

### 4.2 Duplicate And Near-Duplicate Split

```python
from PIL import Image
import imagehash
import numpy as np

HASH_FUNCS = [imagehash.average_hash, imagehash.phash,
              imagehash.dhash, imagehash.whash]

def multi_hash(path):
    img = Image.open(path).convert("RGB")
    return np.concatenate([np.asarray(fn(img).hash).reshape(-1)
                           for fn in HASH_FUNCS])

def duplicate_pairs(paths, threshold=0.90):
    hashes = np.stack([multi_hash(p) for p in paths])
    sims = (hashes[:, None, :] == hashes[None, :, :]).mean(-1)
    mask = (sims > threshold) & (~np.eye(len(paths), dtype=bool))
    return np.argwhere(mask)
```

For resized/cropped/augmented duplicates, use a pretrained embedding model and
cosine similarity. Put duplicate clusters into the same fold. If duplicate
labels conflict, inspect and decide whether to remove, relabel, or group them.

### 4.3 Pairwise Split

For kinship, face verification, duplicate matching, or product pairs, random
pair splits leak through shared identities.

```python
def assert_pair_groups_clean(train_pairs, val_pairs, left_id, right_id):
    train_ids = set(train_pairs[left_id]) | set(train_pairs[right_id])
    val_ids = set(val_pairs[left_id]) | set(val_pairs[right_id])
    overlap = train_ids & val_ids
    assert not overlap, f"identity leakage: {list(overlap)[:10]}"
```

If a complete identity split is impossible, keep a stricter holdout for model
selection and report the residual overlap.

### 4.4 Fold Stability

For tiny or noisy public tests, select models by:

- mean OOF metric
- fold standard deviation
- correlation across repeated seeds
- metric on a stricter group or out-of-domain holdout

## 5. Data Audit

Run these audits before architecture search:

- label distribution by fold and group
- missing/invalid image files
- image size and aspect ratio distribution
- corrupt files, all-white/all-black images, low-quality JPEGs
- duplicate and cross-label duplicate clusters
- train/test source or resolution shifts
- metadata missingness and source correlation
- per-class confusion from a fast baseline
- metric row coverage in the submission writer

For medical and WSI:

- patient/study/slide ID multiplicity
- DICOM slice order, spacing, orientation, and windowing
- WSI dimensions, magnification proxies, tile counts, tissue fraction
- scanner/site/provider labels if available

For keypoints/strokes:

- sequence length distribution
- NaN/missing point distribution
- left/right handedness and mirroring semantics
- deployment format and runtime

## 6. Input Construction

### 6.1 Ordinary RGB Images

Default:

- decode RGB
- preserve aspect ratio unless fixed square is standard for the backbone
- train at one practical resolution, then raise resolution for fine-grained
  classes
- use ImageNet mean/std unless domain-specific normalization is proven

Good first augmentations:

- `RandomResizedCrop`
- horizontal flip if class-invariant
- mild shift/scale/rotate
- brightness/contrast/color jitter for natural images
- `CoarseDropout` or random erasing
- mixup/cutmix only when label semantics permit mixed labels

Avoid semantic-destroying transforms: flips for asymmetric digits/medical
laterality, heavy compression for forensic tasks, and aggressive rotations for
directional symbols.

### 6.2 Fine-Grained Object Images

Use higher resolution and crop strategies:

- 384/448/512 for many plant/product tasks
- 640/768/896 only when object detail justifies it
- object detector or saliency crop if the object is small
- aspect-ratio preserving resize when shape matters
- hard-example mining from confusion matrix

Fine-grained tasks often benefit from retrieval-style heads even when the final
output is a class ID.

### 6.3 Small Grayscale Digits And Symbols

Digits and scripts often do not need pretrained ImageNet models.

- compact CNN, SE-CNN, ResNet18/34, or EfficientNet-lite style models
- geometry-only augmentation: small rotation, shift, scale, shear
- no horizontal flip unless the script is invariant
- ensemble seeds/folds; small datasets are high variance
- pseudo-label cautiously and keep no-pseudo models

For multi-component scripts, model both the components and the joint grapheme
when the metric rewards components but the image is a single symbol.

### 6.4 DICOM And Medical Volumes

Medical classification is usually not raw image classification. It is often:

- DICOM normalization and windowing
- localizing anatomy
- building slice or crop sequences
- aggregating to study/patient labels
- enforcing label consistency

Windowing pattern:

```python
import numpy as np

def dicom_pixels(ds):
    x = ds.pixel_array.astype("float32")
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    x = x * slope + intercept
    if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
        x = x.max() - x
    return x

def window_image(x, center, width):
    lo = center - width / 2
    hi = center + width / 2
    x = np.clip(x, lo, hi)
    x = (x - lo) / max(hi - lo, 1e-6)
    return x.astype("float32")

def three_window_channels(pixel_array):
    # Example CT brain windows from RSNA hemorrhage solutions.
    brain = window_image(pixel_array, 40, 80)
    subdural = window_image(pixel_array, 80, 200)
    bone = window_image(pixel_array, 600, 2800)
    return np.stack([brain, subdural, bone], axis=-1)
```

Slice sorting pattern:

```python
def slice_position(ds):
    ipp = getattr(ds, "ImagePositionPatient", None)
    if ipp is not None and len(ipp) >= 3:
        return float(ipp[2])
    if hasattr(ds, "SliceLocation"):
        return float(ds.SliceLocation)
    if hasattr(ds, "InstanceNumber"):
        return float(ds.InstanceNumber)
    return 0.0

def sort_slices(datasets):
    return sorted(datasets, key=slice_position)
```

Adjacent-slice channels are repeatedly strong:

```python
def adjacent_slice_stack(volume, i):
    j0 = max(i - 1, 0)
    j1 = i
    j2 = min(i + 1, len(volume) - 1)
    return np.stack([volume[j0], volume[j1], volume[j2]], axis=-1)
```

For CT/MRI studies, compare:

- 2D slice classifier
- 2.5D current+neighbor channels
- 2D encoder embeddings plus GRU/LSTM/Transformer
- 3D CNN only if compute/data support it
- anatomy segmentation or coordinate model before classification

Robust DICOM details from radiology sources:

- prefer fixed task windows over noisy per-file DICOM windows
- handle multi-valued `WindowCenter`/`WindowWidth`
- detect reversed z order and missing/corrupt slices
- resample spacing only when the route is truly 3D
- cache PNG/JPEG/NPY slices if raw DICOM decoding dominates runtime

### 6.5 Whole-Slide Images

Do not resize a whole slide to one square as the main route. Use:

- streaming reader: `pyvips`, `OpenSlide`, `skimage.io.MultiImage`, libpng-like
  row streaming for giant PNGs
- thumbnail tissue mask
- tile selection
- tile feature extraction
- slide-level MIL/top-k pooling

Simple tile ranking pattern:

```python
import numpy as np

def pad_to_multiple(img, tile):
    h, w = img.shape[:2]
    ph = (-h) % tile
    pw = (-w) % tile
    return np.pad(img, ((0, ph), (0, pw), (0, 0)), constant_values=255)

def top_dark_tiles(img, tile=256, n_tiles=64):
    img = pad_to_multiple(img, tile)
    h, w = img.shape[:2]
    tiles = img.reshape(h // tile, tile, w // tile, tile, 3)
    tiles = tiles.transpose(0, 2, 1, 3, 4).reshape(-1, tile, tile, 3)
    scores = tiles.reshape(len(tiles), -1, 3).sum(axis=(1, 2))
    keep = np.argsort(scores)[:n_tiles]
    return tiles[keep]
```

For H&E pathology:

- standardize tissue scale/magnification when possible
- treat TMA and WSI scale differences explicitly
- test stain augmentation before stain normalization
- keep tile count under runtime limits
- compare Chowder/ABMIL/DSMIL/TransMIL/CLAM/top-k pooling

### 6.6 Multi-Channel Microscopy

Do not force multi-channel biological images into ordinary RGB without testing.

Human Protein Atlas-style RGBy:

- read `[red, green, blue, yellow]` in fixed order
- use sigmoid multilabel head, not softmax
- adapt pretrained 3-channel conv to 4 channels

plate-shifted cellular imaging-style cell data:

- handle 6 channels or selected subsets
- normalize per image/channel or experiment/plate
- respect plate/site/domain effects

First-conv adaptation:

```python
import torch
import torch.nn as nn

def replace_first_conv(conv, in_chans):
    new = nn.Conv2d(in_chans, conv.out_channels, conv.kernel_size,
                    conv.stride, conv.padding, bias=(conv.bias is not None))
    with torch.no_grad():
        old_w = conv.weight
        new.weight[:, :old_w.shape[1]] = old_w
        if in_chans > old_w.shape[1]:
            extra = old_w.mean(dim=1, keepdim=True).repeat(1, in_chans - old_w.shape[1], 1, 1)
            new.weight[:, old_w.shape[1]:] = extra
        if conv.bias is not None:
            new.bias.copy_(conv.bias)
    return new
```

### 6.7 Keypoints And Strokes

ASL/keypoint competitions are sequence classification, not RGB image
classification.

High-value features:

- selected hand/lip/pose landmarks
- NaN handling
- normalization around a stable body landmark
- motion deltas
- distance/angle features for fingers and lips
- participant split
- deployment graph/runtime validation

Compact preprocessing pattern:

```python
import numpy as np

def keypoint_features(x, point_idx, max_len=128):
    # x: T x P x C with NaNs.
    x = x[:, point_idx, :2].astype("float32")
    center = np.nanmean(x[:, :1, :], axis=(0, 1), keepdims=True)
    scale = np.nanstd(x, axis=(0, 1), keepdims=True) + 1e-6
    x = (x - center) / scale
    x = np.nan_to_num(x)
    x = x[:max_len]
    d1 = np.pad(np.diff(x, axis=0), ((0, 1), (0, 0), (0, 0)))
    return np.concatenate([x.reshape(len(x), -1),
                           d1.reshape(len(x), -1)], axis=1)
```

For strokes/doodles, train both:

- raster CNN from rendered strokes
- sequence RNN/Transformer/1D CNN from raw stroke order

Then blend or rerank top-k predictions.

For ASL/TFLite competitions, model code is part of the submission. Keep
preprocessing inside the exported graph and validate the exact signature:

```python
def validate_tflite(interpreter, sample):
    runner = interpreter.get_signature_runner("serving_default")
    out = runner(inputs=sample)
    assert "outputs" in out
    assert out["outputs"].ndim == 2
    return out["outputs"]
```

### 6.8 Giant Images With Tiny Objects

If the final target is a count, sum, or class derived from small objects in a
large image, use decomposition:

- synthetic or weakly supervised detector
- high-recall detection threshold
- crop classifier
- false-positive filter
- final aggregation

- train detector on synthetic large images
- train crop classifier on synthetic and original small images
- keep detector recall high
- remove contained boxes and low-confidence crop classifications
- validate by visualizing wrong aggregates

If final output is boxes/points rather than class/count, route to image
detection as primary.

## 7. Model Families

### 7.1 Timm Baselines

Use installed names, but keep concrete candidates:

```python
import timm

def first_available(candidates):
    installed = set(timm.list_models())
    for name in candidates:
        if name in installed:
            return name
    raise RuntimeError("none installed: " + ", ".join(candidates))

RGB_BASELINES = [
    "convnext_tiny.fb_in22k_ft_in1k",
    "convnext_small.fb_in22k_ft_in1k",
    "tf_efficientnetv2_s.in21k_ft_in1k",
    "resnet50.a1_in1k",
]
```

### 7.2 Ordinary And Fine-Grained Classification

Reliable families:

- ResNet/ResNet-D/ResNet-RS
- SE-ResNeXt, SENet154
- EfficientNet B0-B7, EfficientNetV2 S/M/L
- ConvNeXt Tiny/Small/Base/Large
- Swin Tiny/Base/Large
- BEiT/EVA/ViT when enough data and compute exist
- ResNeSt and RegNetY for product/fine-grained tasks
- CoaT/MaxViT for medical or high-resolution settings

### 7.3 GeM Pooling

GeM is useful for retrieval, product, landmark, and some WSI/tile/image tasks.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class GeM(nn.Module):
    def __init__(self, p=3.0, eps=1e-6, trainable=True):
        super().__init__()
        value = torch.ones(1) * p
        self.p = nn.Parameter(value) if trainable else value
        self.eps = eps

    def forward(self, x):
        x = x.clamp(min=self.eps).pow(self.p)
        x = F.avg_pool2d(x, (x.size(-2), x.size(-1)))
        return x.pow(1.0 / self.p).flatten(1)
```

For product/landmark, add BN between pooling and margin head if OOF improves.

### 7.4 Metric Learning Heads

Use margin heads when:

- many classes
- classes are visually similar
- top-k/GAP/MAP rewards ranking
- train labels are identities/products/landmarks

Good heads:

- ArcFace
- sub-center ArcFace
- CosFace
- CircleSoftmax
- Triplet/contrastive losses as auxiliary

At inference, use both classifier logits and normalized embeddings.

### 7.5 Medical Sequence Models

Strong medical pattern:

1. 2D/2.5D CNN extracts slice embeddings or logits.
2. Resize or pad feature sequence.
3. GRU/LSTM/Transformer/1D CNN aggregates to image-level and study-level labels.
4. Postprocess consistency constraints if the metric expects them.

```python
import torch
import torch.nn as nn

class StudyGRU(nn.Module):
    def __init__(self, in_dim, hidden=512, out_dim=6):
        super().__init__()
        self.rnn = nn.GRU(in_dim, hidden, batch_first=True,
                          bidirectional=True)
        self.attn = nn.Sequential(nn.Linear(hidden * 2, 1))
        self.head = nn.Linear(hidden * 4, out_dim)

    def forward(self, x, mask=None):
        h, _ = self.rnn(x)
        scores = self.attn(h).squeeze(-1)
        if mask is not None:
            scores = scores.masked_fill(~mask, -1e9)
        w = torch.softmax(scores, dim=1).unsqueeze(-1)
        attn_pool = (h * w).sum(dim=1)
        max_pool = h.max(dim=1).values
        return self.head(torch.cat([attn_pool, max_pool], dim=1))
```

### 7.6 WSI MIL Models

Use tile embeddings from:

- Phikon or pathology foundation encoders when available locally
- CTransPath
- LUNIT/DINO pathology encoders
- DINOv2/ViT/ConvNeXt/EfficientNet tile encoders

Aggregators:

- mean/top-k/max pooling
- Chowder
- ABMIL
- CLAM
- DSMIL
- TransMIL
- DTFD-MIL

Chowder-style seed ensembling can be more stable than a single complex MIL
model.

### 7.7 Metadata Fusion Models

If tabular metadata exists, build:

- image-only CNN/ViT OOF predictions
- tabular GBDT using metadata
- fused GBDT/MLP using metadata + image OOF features

Strong GBDT families:

- LightGBM
- CatBoost
- XGBoost

Use grouped OOF features only. Never train a stacker on in-fold predictions.

## 8. Training Recipes

### 8.1 First Fold

For ordinary RGB:

- pretrained `convnext_tiny` or `tf_efficientnetv2_s`
- `AdamW`
- cosine or OneCycle schedule
- 5 to 20 epochs depending data size
- label smoothing only after checking metric
- mixed precision
- save best by exact validation metric

For fine-grained:

- higher resolution
- stronger crop policy
- possibly GeM/margin head
- hard-example review

For medical:

- start with a 2.5D baseline
- patient split
- simple windowing
- localize or crop next

For WSI:

- tile/embedding cache first
- train slide-level MIL quickly
- inspect tile coverage and runtime

### 8.2 Optimizers And Schedules

Good defaults:

- `AdamW(lr=1e-4 to 5e-4, weight_decay=1e-4 to 1e-2)`
- `SGD` with 1cycle for some fastai/fine-grained pipelines
- `RAdam + Lookahead` for ASL/TFLite-style keypoint tasks
- cosine annealing with warmup for long training
- ReduceLROnPlateau for older Keras digit/symbol baselines

### 8.3 Losses

Single-label:

- cross entropy
- focal loss for imbalance/hard examples
- label smoothing only if CV-positive

Multilabel:

- `BCEWithLogitsLoss`
- weighted BCE
- focal or asymmetric loss
- Lovasz/F1-like fine-tuning only after BCE convergence, if OOF improves

Ordinal:

- CE plus regression/ordinal/cumulative heads
- threshold optimization for QWK

Medical segmentation auxiliary:

- BCE/CE classification plus small Dice/BCE mask loss

```python
class AuxSegLoss(nn.Module):
    def __init__(self, dice_loss, seg_weight=0.125):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = dice_loss
        self.seg_weight = seg_weight

    def forward(self, logits, targets, mask_logits, masks):
        cls_loss = self.bce(logits, targets.float())
        seg_loss = self.dice(mask_logits.float(), masks.float())
        return cls_loss + self.seg_weight * seg_loss
```

### 8.4 Imbalance

Use:

- class-weighted loss
- weighted sampler
- balanced batches
- rare-class oversampling
- focal/asymmetric loss
- per-class thresholds
- specialist models only after broad baseline

For extremely imbalanced AUC/pAUC tasks, training batches can be balanced 1:1
positive/negative while validation remains naturally distributed.

### 8.5 Pseudo-Labeling

Use only if rules allow test/unlabeled/external labels.

Safe pattern:

1. Train strong folds on original data.
2. Predict unlabeled/test/external data with fold/model ensemble.
3. Keep only high confidence examples.
4. Train a new model with pseudo labels.
5. Compare OOF where possible and keep original model in ensemble.

Do not use pseudo-labeling when the rules ban it or when CV cannot detect
overfitting to the test distribution.

### 8.6 External Data

Before using external data:

- confirm rules
- audit test/train overlap
- check class mapping
- check source/domain shift
- run an external-vs-train classifier
- keep a no-external model in the ensemble

External data can help, hurt, or become disallowed. Treat flower, dermoscopy, medical image classification, and
camera-model external-data notes as governance warnings unless the new
competition explicitly permits analogous data.

## 9. Metric-Specific Playbooks

### 9.1 Logloss

Optimize calibrated probabilities. Do not threshold. Do not overconfidently set
0/1 unless labels are certain.

Use:

- OOF calibration
- temperature scaling
- class-prior adjustment only with validation support
- weighted logloss if official metric weights classes/rows

### 9.2 AUC And pAUC

AUC cares about ranking. pAUC cares about ranking in a high-sensitivity or
low-FPR region. dermoscopy pAUC needs exact implementation.

```python
from sklearn.metrics import roc_auc_score

def pauc_above_tpr(y_true, y_score, min_tpr=0.80):
    # Pattern used for dermoscopy pAUC: score the high-TPR region by inverting
    # labels/scores and using max_fpr = 1 - min_tpr.
    v_gt = abs(y_true - 1)
    v_pred = -1.0 * y_score
    max_fpr = abs(1 - min_tpr)
    return roc_auc_score(v_gt, v_pred, max_fpr=max_fpr)
```

For pAUC/rare positive:

- hard-negative mining matters
- patient-relative features can help
- image OOF predictions can be powerful tabular features
- calibration is secondary to ranking unless downstream stackers use the score

### 9.3 F1/F2 Multilabel

Never use `0.5` thresholds by default. Tune thresholds on OOF.

```python
import numpy as np
from sklearn.metrics import fbeta_score

def best_global_threshold(y_true, y_prob, beta=2):
    grid = np.arange(0.01, 0.99, 0.01)
    scores = [fbeta_score(y_true, y_prob > t, beta=beta, average="samples")
              for t in grid]
    i = int(np.argmax(scores))
    return float(grid[i]), float(scores[i])

def best_per_class_thresholds(y_true, y_prob, beta=2):
    th = []
    for c in range(y_true.shape[1]):
        grid = np.arange(0.01, 0.99, 0.01)
        scores = [fbeta_score(y_true[:, c], y_prob[:, c] > t,
                              beta=beta, zero_division=0)
                  for t in grid]
        th.append(float(grid[int(np.argmax(scores))]))
    return np.array(th)
```

Useful constraints:

- minimum one label for F2 when empty rows are bad
- per-label thresholds for rare labels
- separate thresholds for label families if semantics differ
- top-k fallback only if legal and OOF-positive

### 9.4 QWK And Ordinal Metrics

QWK rewards ordered classes. Train both:

- class probabilities
- ordinal/regression expectation

Then tune thresholds on OOF.

```python
import numpy as np
from sklearn.metrics import cohen_kappa_score

def apply_thresholds(x, thresholds):
    return np.digitize(x, np.asarray(thresholds))

def qwk_score(y_true, continuous_pred, thresholds):
    y_hat = apply_thresholds(continuous_pred, thresholds)
    return cohen_kappa_score(y_true, y_hat, weights="quadratic")
```

For retinal/ordinal whole-slide pathology data:

- clean noisy labels with OOF disagreement, not in-fold predictions
- group duplicate images before denoising
- tune thresholds by provider/source if validation supports it

### 9.5 Balanced Accuracy And Outliers

Balanced accuracy makes rare/unknown classes valuable. If an `Other` or unknown
class is absent or underrepresented:

- use entropy or ensemble variance to flag outliers
- train one-vs-rest or OOD detector if data exists
- calibrate threshold on OOF or internal validation
- avoid public-LB-only thresholding

### 9.6 GAP/MAP/Top-K

Use exact top-k formatting. For retrieval-like tasks, the score is often driven
by ranking rather than top-1 classification.

```python
def topk_submission_strings(prob, classes, k=3):
    idx = np.argsort(-prob, axis=1)[:, :k]
    return [" ".join(str(classes[j]) for j in row) for row in idx]
```

For landmarks/products:

- combine classifier and kNN candidates
- aggregate neighbor scores by class
- include non-landmark/distractor penalty if relevant
- tune top-k candidate formatting on OOF

### 9.7 Custom Medical Consistency Metrics

Some medical competitions score multiple related labels. Enforce consistency
only when it matches the label ontology and metric.

Pattern:

1. Generate raw probabilities.
2. Build candidate consistent states.
3. Compute metric-weighted loss from raw probabilities to each state.
4. Choose the lower-loss consistent state.

This was useful for PE-style labels. It should not be blindly applied to
ordinary multilabel tasks with noisy co-occurrence.

## 10. Metadata Fusion

### 10.1 When Metadata Is First-Class

Metadata is first-class when the task description provides:

- patient attributes
- lesion measurements
- scanner/site/acquisition fields
- image coordinates
- anatomical location
- study/series metadata
- weather/location/time fields
- tabular engineered features

Use tabular methods as a secondary route when metadata can independently predict labels.

### 10.2 OOF Image Features Into GBDT

Strong dermoscopy pattern:

```python
def add_oof_image_features(tab, oof_pred, test_pred, id_col="image_id"):
    train = tab["train"].copy()
    test = tab["test"].copy()
    train["img_pred"] = train[id_col].map(oof_pred)
    test["img_pred"] = test[id_col].map(test_pred)
    # Add patient-relative score if patient_id exists.
    if "patient_id" in train:
        train["img_pred_patient_mean"] = train.groupby("patient_id")["img_pred"].transform("mean")
        train["img_pred_ratio_patient"] = train["img_pred"] / (train["img_pred_patient_mean"] + 1e-6)
    return train, test
```

Use raw metadata plus:

- image model OOF scores
- standardized model scores
- ratios to patient/site mean
- local outlier factor / ugly-duckling features
- patient lesion counts
- anatomical-location aggregates

Use group folds by patient/source.

### 10.3 Stacking Rules

- Generate OOF predictions for every base model.
- Train stacker only on OOF predictions.
- Apply stacker to averaged fold test predictions.
- Store all prediction arrays.
- Use simple averaging if OOF stacker does not improve.

For small datasets, GBDT stackers can overfit. Use repeated folds and fold
stability.

## 11. Subtype Cookbooks

### 11.1 Ordinary RGB Classification

Baseline:

- stratified or group-stratified 5-fold split
- ConvNeXt/EfficientNetV2
- 224/384 first pass
- CE loss
- fold average

Upgrade:

- duplicate grouping
- 384/448/512 resolution
- stronger augmentation
- architecture diversity
- TTA
- pseudo-label if legal and high-confidence

### 11.2 Fine-Grained Plants, Animals, Food, Products

Baseline:

- high-res ConvNeXt/EfficientNet/ResNeSt/RegNet
- group/duplicate-aware folds
- confusion matrix review

Upgrade:

- GeM pooling
- ArcFace/CosFace/CircleSoftmax
- object crop or detector if object small
- hard-example mining
- OOF calibration
- ensemble across resolution and data-cleaning variants

### 11.3 Multilabel Tags And Attributes

Baseline:

- sigmoid head
- BCEWithLogits
- iterative multilabel or grouped folds
- global threshold search

Upgrade:

- per-class thresholds
- rare-class sampling/loss weights
- asymmetric/focal loss
- second-layer GBDT over model probabilities
- family-specific thresholds
- min/max label constraints if OOF-positive

### 11.4 Dermoscopy And Rare Lesion Detection

Baseline:

- patient-grouped folds
- tabular GBDT using official metadata
- image CNN/ViT OOF prediction
- pAUC metric parity

Upgrade:

- image predictions as GBDT features
- patient-relative lesion features
- location/anatomical aggregates
- EVA02/EdgeNeXt/ConvNeXt/EfficientNetV2 image models
- class-balanced sampling
- external/synthetic data only if legal and validation-positive

### 11.5 DICOM CT/MRI/X-Ray

Baseline:

- patient/study folds
- DICOM sort/windowing
- 2.5D CNN
- study aggregation by max/mean/attention

Upgrade:

- anatomy segmentation/coordinate model
- crop around organs/levels
- slice encoder + GRU/LSTM/Transformer
- multi-window or neighboring slices
- auxiliary segmentation loss
- consistency postprocess
- fold/model ensemble

Radiology route patterns:

- abdominal trauma: segment/crop organs first, use soft-tissue/liver/angio
  windows, train 2.5D CNN plus GRU/attention, aggregate slice probabilities by
  max, and use organ-visibility masks as soft labels when available
- lumbar spine: predict instance numbers and coordinates before severity;
  crop per level/side/condition, add random coordinate and slice shifts during
  training, then use MIL/BiLSTM over nearby slices
- intracranial aneurysm: localize vessels or Circle-of-Willis ROI first; use
  masked pooling, 3D/2.5D classifiers, LR flip with label swaps, and fallback
  logic for bad DICOM tags
- intracranial hemorrhage: use fixed brain/subdural/bone windows plus adjacent
  slices; feed CNN logits/features into LSTM/1D-CNN/GBDT stackers
- pulmonary embolism: lung crop, PE window, neighboring slices, sequence model
  over embeddings, then enforce exam-label consistency
- broad X-ray/body-part tasks: may stay closer to plain image classification,
  but still use study/source grouping and task-specific pretrained backbones

### 11.6 Retinal/Ordinal Severity

Baseline:

- crop black borders
- illumination normalization
- high-resolution CNN
- patient/eye split
- CE plus regression/ordinal output

Upgrade:

- threshold search for QWK
- ensemble classification and regression predictions
- label-noise review
- TTA

### 11.7 WSI And Histopathology

Baseline:

- tissue thumbnail mask
- tile selection
- tile encoder
- top-k mean/max slide pooling

Upgrade:

- Phikon/CTransPath/LUNIT/DINO features
- Chowder/ABMIL/DSMIL/TransMIL/CLAM
- magnification standardization
- TMA/WSI-specific tile size
- entropy/variance outlier detection when `Other` exists
- seed/fold ensembles of MIL models
- slide-level calibration

### 11.8 Multi-Channel Microscopy

Baseline:

- correct channel order
- adapted first conv
- per-channel normalization
- sigmoid for multilabel

Upgrade:

- class-wise thresholds
- label-correlation model
- plate/site-aware validation
- domain-specific normalization or AdaBN
- constraint-aware postprocess only if competition structure supports it

### 11.9 Plate-Dominated Cellular Classification

This is narrow but important.

Use:

- experiment/plate-aware splits
- six-channel inputs or selected channel subsets
- per-image/channel standardization
- domain-aware batches
- AdaBN or site-specific normalization if OOF-positive
- constraint-aware assignment only when the competition defines per-plate class
  uniqueness

Assignment pattern:

```python
from scipy.optimize import linear_sum_assignment

def plate_assignment(prob):
    # prob: rows on one plate x candidate classes.
    rows, cols = linear_sum_assignment(1.0 - prob)
    labels = cols[np.argsort(rows)]
    return labels
```

Do not transfer plate assignment to ordinary biology tasks.

### 11.10 Landmark And Product Retrieval

Baseline:

- train classifier with margin head
- extract L2-normalized embeddings
- aggregate kNN neighbors by class
- blend classifier and retrieval scores

Upgrade:

- DOLG/DELG or local+global descriptor models
- Hybrid Swin+EffNet
- multi-stage training: small clean image size, larger noisy dataset, high-res
  fine-tune
- distractor/non-landmark classifier
- top-k formatting tuned on validation

Chunked kNN pattern:

```python
import torch

def chunked_topk(test_emb, train_emb, k=20, chunk=2048):
    out_scores, out_idx = [], []
    train_t = train_emb.T.contiguous()
    for s in range(0, len(test_emb), chunk):
        sim = test_emb[s:s+chunk] @ train_t
        score, idx = torch.topk(sim, k=k, dim=1)
        out_scores.append(score.cpu())
        out_idx.append(idx.cpu())
    return torch.cat(out_scores), torch.cat(out_idx)
```

### 11.11 Pairwise Face/Kinship/Verification

Use:

- pretrained face/person descriptors when legal
- Siamese or pair MLP over descriptor algebra
- family/person split
- AUC/logloss calibration

Pair features:

```python
def pair_features(a, b):
    return np.concatenate([
        np.abs(a - b),
        (a - b) ** 2,
        a * b,
        a ** 2 - b ** 2,
    ], axis=-1)
```

### 11.12 Camera/AI-Generated/Forensics

Preserve signal:

- avoid heavy JPEG/recompression unless augmenting official manipulations
- use patches/crops at native-ish resolution
- split by device/source/original image
- filter external images by EXIF, camera model, editing software, quality
- use model/crop/checkpoint ensembles

Do not apply standard natural-image color augmentation blindly. It can erase the
very artifacts being classified.

### 11.13 ASL/Keypoint Sequence

Baseline:

- participant split
- selected hand/lip/pose points
- normalize, fill NaNs, add deltas
- 1D CNN plus light Transformer
- CE with label smoothing if OOF-positive

Upgrade:

- temporal resampling/masking
- coordinate affine/cutout
- AWP/drop path/dropout for long training
- EfficientNet over point-time tensors
- BERT/DeBERTa sequence helpers
- TFLite conversion/runtime validation

### 11.14 Multi-Component Grapheme Symbols

This route is image classification, but the output contract is not ordinary
single-label argmax. For multi-component grapheme grapheme competitions:

- targets can be component heads such as root/vowel/consonant
- the metric can weight components differently
- submission may require multiple rows per image
- unseen component combinations can dominate generalization

Baseline:

- crop/threshold grayscale symbols
- one shared CNN encoder with three heads
- weighted component loss matching the metric
- component-wise recall callback
- fold validation by rare or unseen component combinations when feasible

Upgrade:

- joint grapheme-combination classifier plus component heads
- SE-ResNeXt/EfficientNet/ResNet ensembles
- OHEM or class-balanced loss for rare components
- synthetic font data only if allowed and domain-validated
- CycleGAN/domain transfer only after a strong component baseline

Submission row pattern:

```python
def multi_component_grapheme_rows(image_ids, root, vowel, consonant):
    rows = []
    for img_id, r, v, c in zip(image_ids, root, vowel, consonant):
        rows.append((f"{img_id}_grapheme_root", int(r)))
        rows.append((f"{img_id}_vowel_diacritic", int(v)))
        rows.append((f"{img_id}_consonant_diacritic", int(c)))
    return rows
```

### 11.15 Stroke-Sequence Drawing

Baseline:

- raster CNN on rendered strokes
- LSTM/GRU on raw strokes
- MAP@3 top-k formatting

Upgrade:

- many CNN architectures/seeds
- LightGBM stacker over top-k model probabilities and stroke features
- raw stroke timestamp/length features
- weighted top-k voting

Equal public-test class-count postprocessing is not a general recipe. Treat it
as a forbidden count-fit unless the competition explicitly states the test class
counts and allows using them.

### 11.16 Multi-View Body Scanner/Zones

Use:

- subject/scan grouped folds
- zone-specific crops or shared encoder with zone embedding
- per-zone metrics and submission-row validation
- multi-view aggregation if rows map to one subject
- task-specific scan loaders for `.aps`, `.a3daps`, `.a3d`, or `.ahi`
- zone visibility masks; some zones are not visible from every angle
- per-zone crops or plates assembled from multiple views

If the task is actually object localization in scans, route to detection.

### 11.17 Open-Set Writer Or Identity Classification

Use when the final label space includes an unknown class such as `-1`.

Baseline:

- embedding model with classification head
- grouped writer/person split
- calibrated confidence threshold for unknown
- separate reporting for known-class accuracy and unknown detection

Upgrade:

- ArcFace/CosFace embeddings
- class prototypes and nearest-neighbor distance
- entropy or margin thresholds
- source-shift validation if test writing/scanning conditions differ

Do not submit closed-set argmax when the task allows unknown identities.

### 11.18 Satellite/Geospatial/Site-Shifted Images

Use:

- site/location/time-aware validation
- metadata fusion for location/date/source
- high-res crops if labels are local
- source-specific calibration only if OOF-positive

Avoid random splits when neighboring geographies or repeated sites leak.

### 11.19 Few-Shot Transfer Classification

For extremely small binary or multiclass image folders, a frozen backbone can be
the right first baseline.

Use:

- ImageFolder or explicit CSV dataset
- pretrained ResNet18/34/EfficientNet-B0
- freeze encoder first, train head
- repeated stratified splits or leave-one-group-out if groups exist

Do not overstate CV from a tiny random split.

### 11.20 Special Robustness/Unlearning

Use only if the task explicitly asks for unlearning, defense, or robustness.

Transferable ideas:

- ensemble diversity improves robustness
- adversarial/fool-detector logic can be useful in defense contests
- forget/retain objectives are specialized to unlearning metrics
- runtime/memory isolation of model predictions can make large ensembles
  feasible

Do not import unlearning/adversarial objectives into ordinary classification
without a matching metric.

### 11.21 Grid-Transformation Reasoning

grid-transformation JSON grids are not image classification even if stored in the local
folder. Use reasoning/program-synthesis/text-sequence guidance. The only
transferable lesson is to respect the real output format and validation loop.

## 12. Inference And Ensembling

### 12.1 Store Predictions

Save:

- OOF predictions with IDs/folds
- fold test predictions
- TTA predictions if useful
- model configs
- class mapping
- metric results
- submission validation output

Never ensemble only by interactive-session state.

### 12.2 Averaging

Default:

- average probabilities/logits across folds
- use geometric mean only if probabilities are calibrated and OOF improves
- for retrieval, average embeddings or aggregate ranks carefully

For top-k:

- combine candidate lists with rank-aware weights
- verify exact string formatting

### 12.3 TTA

Use TTA when transforms are label-preserving:

- flips and multi-crop for natural/product/plant
- multi-size for fine-grained
- D4 only when orientation invariant
- no destructive TTA for forensic/camera
- cautious TTA for medical laterality

Confidence-gated TTA can save runtime:

```python
def maybe_tta(p, image, infer_tta, min_conf=0.90):
    if float(p.max()) >= min_conf:
        return p
    return infer_tta(image)
```

### 12.4 Calibration

Use OOF:

- temperature scaling
- isotonic/logistic calibration
- class-prior adjustment
- rank normalization for GBDT blending

For dermoscopy GBDT ensembles, rank-averaging model outputs can be more stable
than raw score averaging.

### 12.5 Submission Validator

Validate:

- exact row count
- exact IDs/order if required
- no missing/extra columns
- probabilities finite and in range
- rows sum to one for softmax tasks
- multilabel strings nonempty if metric punishes empty rows
- top-k length and legal class IDs
- one row per image-label/study-label as specified
- TFLite/model file existence and runtime if applicable

## 13. Narrow High-ROI Tricks

Use only under the stated conditions:

- **BN between pooling and margin head**: product/large-class retrieval.
- **OOF disagreement denoising**: ordinal pathology with noisy labels and
  duplicate-safe folds.
- **Entropy/variance outlier detection**: balanced-accuracy tasks with `Other`
  class missing or scarce in train.
- **Auxiliary segmentation loss**: medical classification where masks/crops are
  available or generated.
- **Neighbor-slice channels**: CT/MRI slice tasks.
- **Patient-relative image scores**: multiple lesions/images per patient.
- **Magnification standardization**: WSI/TMA mixed pathology.
- **AdaBN/domain batches**: plate/site-dominated cell images.
- **High-recall detector plus crop classifier**: giant images with tiny simple
  objects.
- **Artifact-preserving preprocessing**: camera/generator forensics.
- **TFLite graph rewrites**: deployment-constrained keypoint tasks where
  conversion/runtime changes leaderboard feasibility.

## 14. Avoid Or Delay

Delay until after a reliable baseline:

- huge ViTs
- full 3D CNNs
- complex MIL without top-k baseline
- per-label specialist models
- public-LB-tuned ensemble weights
- hard-coded class-count balancing
- external data
- pseudo-labels
- stain normalization
- synthetic data
- complex differentiable F1 losses

Avoid entirely unless explicitly allowed:

- public-test label inference
- hidden-label manual annotation
- sample-submission leakage
- old-rule datasets
- leaderboard score algebra
- metadata shortcuts not available at test time
