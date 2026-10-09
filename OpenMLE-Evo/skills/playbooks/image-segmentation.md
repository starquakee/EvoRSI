# Image Segmentation Category Playbook

This playbook is the full method library for Kaggle-style `image_segmentation`
tasks. It is intended as direct reading material for a skill-generating agent:
after reading a task description, the agent should route the task, choose the
right validation geometry, then extract only the applicable recipes.

Do not use this playbook to justify leaderboard probing, public-test label
inference, hidden labels, manual test labeling, sample-submission label updates,
or disallowed external data.

## 1. Task Routing

### 1.1 Binary semantic segmentation

Use when each image has one foreground/background mask and the score is Dice,
IoU, F-score over IoU thresholds, or an RLE mask metric.

Default plan:

- implement RLE decode/encode before modeling
- stratify folds by positive/empty status, mask area, source, and group ids
- train a strong U-Net/FPN/Unet++/DeepLabV3+ model with a pretrained encoder
- use BCE/CE plus Dice/Focal/Lovasz/Tversky depending on imbalance and metric
- tune pixel threshold and minimum component area on OOF predictions
- add TTA and model ensembles after thresholded OOF score is stable

### 1.2 Class-channel semantic segmentation

Use when every image has multiple independent mask channels or rows, such as
`ImageId_ClassId` or `Image_Label`, and many class rows are empty.

Default plan:

- train a multi-channel segmenter or per-class specialists
- keep empty image-class rows in validation and submission
- tune per-class label thresholds, pixel thresholds, and min-area filters
- consider a classifier gate, segmentation classification head, or max/top-k
  pixel probability as the existence score
- validate mean Dice/F1 at the image-class row level, not only batch pixel Dice

### 1.3 Instance segmentation

Use when the submission has one row per object instance and the metric matches
predicted masks to ground-truth masks with IoU thresholds.

Default plan:

- preserve instance identity through preprocessing
- train Mask R-CNN/HTC/Cascade/Detectron/MMDetection/RTMDet/YOLO-seg, or U-Net
  with a robust instance recovery stage
- validate mask AP after score thresholds, mask NMS, overlap removal, and RLE
  encoding
- tune max detections, per-class thresholds, duplicate suppression, and mask
  binary threshold
- enforce submission constraints: no invalid RLE, no duplicate pixels, and no
  forbidden overlaps

For biomedical instance segmentation/cell-instance segmentation/cell tasks, a semantic foreground mask is not enough unless
it is split into instances by connected components, watershed, borders,
distance maps, centers, or learned proposals.

### 1.4 Medical and CXR segmentation

Use when the task mentions DICOM, X-ray, CT/MRI, radiology, patient/study ids,
pneumothorax, lesion masks, or many negative studies.

Default plan:

- split by patient/study before augmentation, tiling, or slice extraction
- keep negatives as first-class scoring rows
- handle DICOM windowing, orientation, spacing, and image normalization
- use a segmentation model plus high-threshold existence decision, or a separate
  classifier gate if OOF supports it
- tune thresholds under the official empty-mask behavior

medical binary segmentation sources are especially useful for triplet thresholding: a high threshold
and min-area gate decide whether the mask exists, then a lower threshold recovers
mask extent.

### 1.5 Tissue, histology, and organ segmentation

Use when the task includes histology, cell-level multi-label imaging/large medical-tissue segmentation, organ, tissue, stain shift,
large TIFF/WSI-style images, donor/patient/source split, or pixel-size metadata.

Default plan:

- split by donor/patient/organ/source, not random tiles
- inspect stain/source/pixel-size differences before choosing folds
- train SegFormer/UPerNet/Swin/CoaT/UNet-family models
- use stain augmentation/normalization and source diagnostics
- tune thresholds by organ or tissue class
- use group norm or small-batch-friendly normalization when batch size is tiny

### 1.6 3D and 2.5D volumetric segmentation

Use when inputs are tomograms, kidney/organ volumes, CT chunks, z-stacks, or
surface masks, and when the output is a full volume or per-slice volume RLE.

Default plan:

- hold out full volumes/specimens/organs
- score decoded full volumes, not random slices
- choose full 3D, 2.5D multi-axis, or hybrid based on memory and metric
- train nnU-Net/MONAI/3D U-Net/SegResNet or 2.5D U-Net/ConvNeXt over x/y/z
  slices
- run full-volume thresholding, connected components, hole filling, and surface
  or topology scoring

### 1.7 Geospatial, satellite, and polygon segmentation

Use when the task has satellite images, multispectral bands, WKT polygons,
geographic tiles, or mask-to-polygon submission.

Default plan:

- split by scene/tile/geography, not random patches
- rasterize polygons to masks for training and vectorize masks back for
  submission
- preserve multispectral channels and align bands before modeling
- use U-Net/FPN/DeepLab-style models, possibly per-class specialists for rare
  classes
- tune `epsilon`, `min_area`, and contour validity repair on validation

### 1.8 Segmentation-assisted cell classification

Use when masks are used to crop cells, but the final score is cell/image
classification rather than mask quality.

Default plan:

- use the segmentation route only to produce cell masks, boxes, and crops
- route final label modeling to image classification
- handle border cells, cells without nuclei, and weak image-level labels
- combine image-level and cell-level predictions if the final score requires
  cell label probabilities

cell-level multi-label imaging sources are strong evidence for this route: do not over-invest in replacing
cell-level multi-label imagingCellSeg unless segmentation quality is the actual metric.

### 1.9 Forensic or defect-mask segmentation

Use when the image is either authentic/empty or contains a forged/defect region
to segment.

Default plan:

- train a mask model on forged/defect masks and authentic zero masks
- preserve source/manipulation groups when possible
- use an authentic fallback when predicted mask area or confidence is too low
- use edge/gradient or morphology postprocess only after OOF validation
- validate both forged-mask F1 and false positives on authentic images

### 1.10 Route out

Handle separately when:

Exception: use this playbook for a mask helper stage if the target task requires a
segmenter before final detection or classification.

## 2. Validation And Leakage

### 2.1 Define the scored unit before splitting

Ask what one row of the metric represents:

- image mask
- image-class mask
- object instance
- cell
- tile
- slide
- patient/study
- volume/slice
- polygon
- API step

Then split by the largest natural unit that could leak into validation:

- patient, study, series
- donor, organ, slide, WSI, tissue block
- volume, specimen, kidney, scroll chunk
- image source, camera, scanner, document/manipulation source
- geography, satellite tile, scene
- steel strip/coil or acquisition run
- experiment, plate, cell type
- video/frame sequence if segmentation is frame-based

Make fold assignment before creating tiles, cells, crops, slices, pseudo masks,
or augmented variants.

### 2.2 Empty-heavy validation

For large-object satellite segmentation, multi-class defect segmentation, medical binary segmentation, and multi-class cloud segmentation-like tasks, empty masks are part of the
score. Validation must keep:

- empty image prevalence
- class-wise empty row prevalence
- mask-size distribution
- rare positive classes
- image-level positive/negative structure

Do not train only on positives and then report positive-only Dice. If negatives
are undersampled for training speed, still validate and threshold on all scoring
rows.

### 2.3 Class-channel validation

For one row per image-class:

- score each image-class row
- tune thresholds per class on OOF
- inspect false-positive rows separately from false-negative rows
- store OOF logits/probabilities before and after postprocess
- check that the submission has every required image-class row exactly once

Batch-level Dice can hide a model that predicts tiny false masks on thousands of
empty rows.

### 2.4 Instance validation

For instance masks:

- evaluate after instance decoding, score thresholds, NMS/overlap suppression,
  and RLE/COCO encoding
- compute AP across IoU thresholds when the official metric does
- validate object counts, duplicate masks, wrong classes, and same-class overlaps
- store per-instance score, mask area, bbox, class, and source image
- tune max detections and per-class score thresholds

For biomedical instance segmentation/cell-instance segmentation-like tasks, validate the instance recovery algorithm, not only
the semantic mask probability.

### 2.5 Medical validation

For medical segmentation:

- split by patient/study/series
- handle multiple images per study as one group when possible
- preserve negative studies
- tune empty-mask decisions on OOF
- inspect anatomy-specific preprocessing, orientation, and spacing

If metadata can reveal source, scanner, or site, use it for validation
diagnostics even if it is not used as a feature.

### 2.6 Histology and organ validation

For histology/tissue/organ:

- group by patient/donor/slide/source
- stratify by organ, stain, pixel size, and positive area
- validate with the same full-scale or tile-stitch inference as test
- inspect source-specific performance before ensembling
- treat external/pseudo data as a separate source and check whether it helps
  target-source validation

### 2.7 3D volume validation

For 3D tasks:

- hold out entire volumes/specimens
- run full inference and full postprocess before scoring
- score surface dice/topology/VOI or official volume metric, not only slice Dice
- tune thresholds and connected components on full-volume OOF
- preserve voxel spacing and ignore labels

### 2.8 Geospatial validation

For satellite/geospatial:

- split by scene/tile/geography
- avoid train/validation patches from the same large image
- preserve rare-class coverage and object area distribution
- validate raster-to-polygon conversion if submission is vectorized
- inspect edge artifacts from tiling

### 2.9 Forensic validation

For authentic/forged masks:

- split by source/manipulation/document family when available
- validate both mask overlap and authentic false positives
- tune authentic fallback gates on a validation set that includes authentic
  images
- check that the empty sentinel exactly matches competition format

## 3. Metrics And Submission Contracts

### 3.1 Dice and mean Dice

Dice is sensitive to threshold and empty masks.

For empty-empty rows, many Kaggle tasks define Dice as 1. Verify the specific
task. In code, avoid division-by-zero behavior that silently treats empty rows
as 0.

```python
import numpy as np

def dice_score(y_true, y_pred, eps=1e-7):
    y_true = np.asarray(y_true).astype(bool)
    y_pred = np.asarray(y_pred).astype(bool)
    denom = y_true.sum() + y_pred.sum()
    if denom == 0:
        return 1.0
    return float(2.0 * np.logical_and(y_true, y_pred).sum() / (denom + eps))
```

### 3.2 IoU/F2/AP over instance masks

Instance segmentation metrics match predicted objects to ground-truth objects.
Pixel foreground Dice is not a substitute.

```python
import numpy as np

def precision_at_iou_threshold(true_labels, pred_labels, threshold):
    true_labels = np.asarray(true_labels, dtype=np.int32)
    pred_labels = np.asarray(pred_labels, dtype=np.int32)

    true_ids = np.unique(true_labels)
    pred_ids = np.unique(pred_labels)
    true_ids = true_ids[true_ids != 0]
    pred_ids = pred_ids[pred_ids != 0]

    if len(true_ids) == 0 and len(pred_ids) == 0:
        return 1.0
    if len(true_ids) == 0 or len(pred_ids) == 0:
        return 0.0

    intersections = np.zeros((len(true_ids), len(pred_ids)), dtype=np.float32)
    for i, tid in enumerate(true_ids):
        t = true_labels == tid
        for j, pid in enumerate(pred_ids):
            p = pred_labels == pid
            inter = np.logical_and(t, p).sum()
            union = np.logical_or(t, p).sum()
            intersections[i, j] = inter / max(union, 1)

    matches = intersections > threshold
    tp = 0
    used_true = set()
    used_pred = set()
    pairs = np.argwhere(matches)
    pairs = sorted(pairs, key=lambda ij: intersections[ij[0], ij[1]], reverse=True)
    for i, j in pairs:
        if i not in used_true and j not in used_pred:
            used_true.add(i)
            used_pred.add(j)
            tp += 1
    fp = len(pred_ids) - len(used_pred)
    fn = len(true_ids) - len(used_true)
    return tp / max(tp + fp + fn, 1)

def mean_instance_ap(true_labels, pred_labels, thresholds=None):
    if thresholds is None:
        thresholds = np.arange(0.50, 1.00, 0.05)
    return np.mean([
        precision_at_iou_threshold(true_labels, pred_labels, t)
        for t in thresholds
    ])
```

Use score-sorted matching if the official metric uses ranked predictions.

### 3.3 Surface dice, topology, and VOI

Surface/volume tasks score boundary and topology properties, not only foreground
overlap.

For blood-vessel tasks:

- boundary-weighted loss can improve surface dice
- full-volume connected components matter
- orthogonal-axis ensemble is a modeling primitive
- empty slice RLE can have task-specific sentinel such as `1 0`

For large-volume material segmentation-like tasks:

- preserve ignore labels in training/validation
- threshold sweeps matter
- topology postprocess can dominate the final score
- validate thresholds on full volumes when possible

Do not optimize a 2D slice metric and assume it transfers to surface dice.

### 3.4 RLE contracts

Kaggle mask RLE often uses column-major order: flatten `mask.T` or reshape with
Fortran order. Some tasks use row-major or JSON list RLE. Always inspect the
competition description and sample submission.

```python
import numpy as np

def rle_encode_col_major(mask):
    """Encode a binary HxW mask using 1-indexed column-major runs."""
    pixels = np.asarray(mask, dtype=np.uint8).T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    changes = np.where(pixels[1:] != pixels[:-1])[0] + 1
    changes[1::2] -= changes[::2]
    return " ".join(str(x) for x in changes)

def rle_decode_col_major(rle, shape):
    """Decode 1-indexed column-major runs to an HxW mask."""
    h, w = shape
    mask = np.zeros(h * w, dtype=np.uint8)
    if rle is None or str(rle).strip() == "":
        return mask.reshape((w, h)).T
    values = [int(x) for x in str(rle).split()]
    starts = np.asarray(values[0::2], dtype=np.int64) - 1
    lengths = np.asarray(values[1::2], dtype=np.int64)
    for start, length in zip(starts, lengths):
        mask[start:start + length] = 1
    return mask.reshape((w, h)).T
```

Roundtrip test every submission writer:

```python
def assert_rle_roundtrip(mask):
    rle = rle_encode_col_major(mask)
    decoded = rle_decode_col_major(rle, mask.shape)
    assert np.array_equal(decoded.astype(bool), mask.astype(bool))
```

### 3.5 COCO compressed mask contracts

large-scale multi-class detection and some cell-level multi-label imaging/MMDetection-style submissions use compressed COCO RLE.

```python
import base64
import zlib
import numpy as np
from pycocotools import _mask as coco_mask

def encode_binary_mask_coco(mask):
    mask = np.asarray(mask)
    if mask.dtype != np.bool_:
        mask = mask.astype(bool)
    if mask.ndim != 2:
        raise ValueError("mask must be a 2D array")
    encoded = coco_mask.encode(np.asfortranarray(mask[:, :, None].astype(np.uint8)))[0]
    counts = encoded["counts"]
    compressed = zlib.compress(counts, zlib.Z_BEST_COMPRESSION)
    return base64.b64encode(compressed).decode("ascii")
```

If the target task uses plain RLE, do not use this format.

### 3.6 Polygon contracts

multispectral polygon segmentation-style submissions require polygon geometries. Validate raster-to-vector and
vector-to-raster conversion.

```python
import cv2
import numpy as np
from shapely.geometry import MultiPolygon, Polygon

def mask_to_polygons(mask, epsilon=1.0, min_area=10.0):
    mask = np.asarray(mask, dtype=np.uint8)
    contours, hierarchy = cv2.findContours(
        mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_KCOS
    )
    if hierarchy is None:
        return MultiPolygon()
    hierarchy = hierarchy[0]
    child_contours = set()
    polygons = []
    for idx, contour in enumerate(contours):
        if idx in child_contours:
            continue
        if cv2.contourArea(contour) < min_area:
            continue
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) < 3:
            continue
        shell = approx[:, 0, :]
        holes = []
        child = hierarchy[idx][2]
        while child != -1:
            child_contours.add(child)
            hole = cv2.approxPolyDP(contours[child], epsilon, True)
            if len(hole) >= 3 and cv2.contourArea(hole) >= min_area:
                holes.append(hole[:, 0, :])
            child = hierarchy[child][0]
        poly = Polygon(shell, holes)
        if poly.is_valid and poly.area >= min_area:
            polygons.append(poly)
    return MultiPolygon(polygons).buffer(0)
```

Tune `epsilon` and `min_area` with the official metric.

## 4. Input Construction And Preprocessing

### 4.1 Decode masks once, cache safely

For RLE tasks, cache decoded masks only after validating:

- height/width
- axis order
- empty mask representation
- class-channel layout
- instance-vs-semantic distinction

Do not mix tasks where the same string means row-major, column-major, COCO RLE,
relative offsets, or JSON list.

### 4.2 Tiling and stitching

Use tiling when images are too large for full inference. Assign folds before
tiling.

```python
import numpy as np

def sliding_windows(h, w, tile, stride):
    ys = list(range(0, max(h - tile + 1, 1), stride))
    xs = list(range(0, max(w - tile + 1, 1), stride))
    if ys[-1] != h - tile:
        ys.append(max(h - tile, 0))
    if xs[-1] != w - tile:
        xs.append(max(w - tile, 0))
    for y in ys:
        for x in xs:
            yield y, x

def stitch_probabilities(image, predict_tile, tile=512, stride=384):
    h, w = image.shape[:2]
    acc = np.zeros((h, w), dtype=np.float32)
    cnt = np.zeros((h, w), dtype=np.float32)
    for y, x in sliding_windows(h, w, tile, stride):
        patch = image[y:y + tile, x:x + tile]
        pred = predict_tile(patch)
        ph, pw = pred.shape[:2]
        acc[y:y + ph, x:x + pw] += pred
        cnt[y:y + ph, x:x + pw] += 1
    return acc / np.maximum(cnt, 1)
```

For central-crop tile prediction, only stitch the valid center region and discard
tile borders to reduce edge artifacts.

### 4.3 Medical image preprocessing

For DICOM/CXR/CT/MRI:

- preserve patient/study grouping
- apply slope/intercept and windowing when relevant
- preserve orientation and laterality
- normalize consistently across train/test
- keep negative studies in validation

For plain PNG/JPEG CXR segmentation, still inspect whether original DICOM
metadata or resized versions change performance.

### 4.4 Stain and tissue preprocessing

For histology:

- inspect stain/source differences
- use stain augmentation before aggressive normalization
- use normalization/color transfer only if target-source validation supports it
- preserve pixel-size metadata when organ masks depend on physical scale
- avoid random tile folds from the same slide

### 4.5 Multispectral and geospatial preprocessing

For satellite/multispectral polygon segmentation-style tasks:

- preserve all relevant bands
- align band resolutions and coordinate systems
- normalize per-band robustly
- rasterize polygons at the model resolution
- validate vectorized polygons after scaling back

Rare classes may require class-specific models, smaller patches, or false
positive penalties.

### 4.6 2.5D volume construction

2.5D uses neighboring slices as channels. Multi-axis inference restores 3D
continuity without full 3D memory cost.

```python
import numpy as np

def slice_stack(volume, axis, index, offsets=(-1, 0, 1)):
    vol = np.asarray(volume)
    if axis == 0:
        n = vol.shape[0]
        get = lambda i: vol[np.clip(i, 0, n - 1), :, :]
    elif axis == 1:
        n = vol.shape[1]
        get = lambda i: vol[:, np.clip(i, 0, n - 1), :]
    elif axis == 2:
        n = vol.shape[2]
        get = lambda i: vol[:, :, np.clip(i, 0, n - 1)]
    else:
        raise ValueError("axis must be 0, 1, or 2")
    return np.stack([get(index + off) for off in offsets], axis=0)

def place_axis_prediction(out, pred2d, axis, index):
    if axis == 0:
        out[index, :, :] += pred2d
    elif axis == 1:
        out[:, index, :] += pred2d
    elif axis == 2:
        out[:, :, index] += pred2d
```

For anisotropic voxels, axis averaging may need weights or resampling.

### 4.7 Cell crops

For cell-level multi-label imaging/cell-instance segmentation-like tasks:

- segment cells/nuclei first
- remove border cells if they are unreliable
- crop with context margin
- keep cell id, image id, bbox, mask area, and source metadata
- resize masks with nearest neighbor for labels, but use bilinear when matching
  Mask R-CNN mask-head training geometry if validated

Cell-level labels may be weak if inherited from image labels. Use lower loss
weights or soft labels when the source evidence is uncertain.

## 5. Utility Patterns

### 5.1 Class-channel postprocess

Use per-class thresholds and minimum areas for class-channel tasks.

```python
import cv2
import numpy as np

def post_process_channel(prob, pixel_threshold, min_area):
    mask = (prob > pixel_threshold).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    clean = np.zeros_like(mask, dtype=np.uint8)
    for component_id in range(1, n):
        area = stats[component_id, cv2.CC_STAT_AREA]
        if area >= min_area:
            clean[labels == component_id] = 1
    return clean

def post_process_multichannel(probs, pixel_thresholds, min_areas):
    masks = []
    for c in range(probs.shape[0]):
        masks.append(post_process_channel(probs[c], pixel_thresholds[c], min_areas[c]))
    return np.stack(masks, axis=0)
```

Tune thresholds on OOF predictions. Do not copy multi-class defect segmentation or multi-class cloud segmentation constants
unless the new task has the same metric, resolution, and class priors.

### 5.2 Classifier-gated mask suppression

Use when many image-class rows are empty.

```python
def apply_class_gate(mask_probs, class_probs, class_thresholds):
    gated = mask_probs.copy()
    for c, threshold in enumerate(class_thresholds):
        if class_probs[c] < threshold:
            gated[c] = 0.0
    return gated
```

Good gates improve empty-row Dice. Bad gates turn small true positives into
false negatives. Tune with OOF and inspect per-class recall.

### 5.3 Triplet threshold for medical empty masks

```python
import numpy as np

def triplet_threshold_mask(prob, top_threshold, min_area, bottom_threshold):
    high_mask = prob > top_threshold
    if int(high_mask.sum()) < min_area:
        return np.zeros_like(prob, dtype=np.uint8)
    return (prob > bottom_threshold).astype(np.uint8)
```

The high threshold decides existence; the lower threshold recovers extent. Tune
all three values on OOF predictions.

### 5.4 Convex hull and component cleanup

Convex hull and component filtering can serve as late-stage postprocessing. Use only when shape priors support it.

```python
import cv2
import numpy as np

def convex_hull_fill(mask):
    mask = mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros_like(mask)
    for contour in contours:
        hull = cv2.convexHull(contour)
        cv2.drawContours(out, [hull], -1, 1, thickness=-1)
    return out
```

Convex hull can improve broad cloud-like masks but can overfill thin structures,
ships, medical lesions, and holes.

### 5.5 Tile-to-full RLE memory safety

For huge images, avoid holding unnecessary copies. A less-memory RLE encoder
mutates only a flattened view/copy and detects runs.

```python
def rle_encode_less_memory(mask):
    pixels = np.asarray(mask, dtype=np.uint8).T.flatten()
    if len(pixels) == 0:
        return ""
    pixels[0] = 0
    pixels[-1] = 0
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 2
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)
```

Use only after confirming that forcing first/last pixels to zero matches the
task. Otherwise use the safer standard encoder.

### 5.6 Semantic mask to instances

For U-Net baselines on instance tasks, connected components can produce one RLE
per object. This is a baseline, not a complete solution for touching objects.

```python
from skimage import measure, morphology

def prob_to_instance_labels(prob, threshold=0.5, min_size=20):
    mask = prob > threshold
    mask = morphology.remove_small_objects(mask, min_size=min_size)
    labels = measure.label(mask)
    return labels.astype("int32")

def labels_to_instance_masks(labels):
    for instance_id in sorted(set(labels.ravel()) - {0}):
        yield labels == instance_id
```

If touching objects are common, add border/distance/center targets and watershed.

### 5.7 Score-ordered overlap removal

Some instance submissions forbid overlapping masks. Process high-confidence masks
first and remove already-used pixels.

```python
import numpy as np

def remove_instance_overlaps(masks, scores, min_remaining_pixels=1):
    order = np.argsort(scores)[::-1]
    occupied = np.zeros_like(masks[0], dtype=bool)
    kept_masks = []
    kept_scores = []
    for idx in order:
        m = masks[idx].astype(bool) & (~occupied)
        if int(m.sum()) < min_remaining_pixels:
            continue
        kept_masks.append(m)
        kept_scores.append(scores[idx])
        occupied |= m
    return kept_masks, kept_scores
```

Tune overlap policy with AP. Aggressive clipping can improve validity but hurt
mask IoU.

### 5.8 Bbox and mask rescoring

```python
import numpy as np

def rescore_instance(box_score, mask_prob, binary_mask):
    if binary_mask.sum() == 0:
        return 0.0
    mask_conf = float(mask_prob[binary_mask.astype(bool)].mean())
    return float(box_score) * mask_conf
```

Use this as an OOF-tuned feature, not a universal rule.

### 5.9 Boundary-weighted loss

Surface and small-boundary tasks often need boundary emphasis.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class EdgeWeightedBCE(nn.Module):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, logits, targets, boundary_weight):
        loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        weight = 1.0 + self.alpha * boundary_weight
        return (loss * weight).mean()
```

Combine with Dice/Focal/Tversky when the metric balances overlap and boundary.

### 5.10 3D connected component cleanup

Use for volumes and surface masks after thresholding.

```python
import numpy as np
from scipy import ndimage as ndi

def remove_small_components_3d(mask, min_voxels):
    labels, n = ndi.label(mask.astype(bool))
    if n == 0:
        return mask.astype(bool)
    counts = np.bincount(labels.ravel())
    keep = np.zeros(n + 1, dtype=bool)
    keep[np.where(counts >= min_voxels)[0]] = True
    keep[0] = False
    return keep[labels]

def close_and_fill_3d(mask, radius=1):
    structure = ndi.generate_binary_structure(3, 1)
    closed = ndi.binary_closing(mask.astype(bool), structure=structure, iterations=radius)
    return ndi.binary_fill_holes(closed)
```

For thin sheets, ordinary fill/dilation can overgrow. Validate surface and
topology metrics, not just visual smoothness.

### 5.11 Authentic-Sample Fallback

```python
import cv2
import numpy as np

def edge_enhanced_forgery_mask(prob, original_size, alpha_grad=0.35):
    gx = cv2.Sobel(prob, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(prob, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx * gx + gy * gy)
    grad = grad / (grad.max() + 1e-6)
    enhanced = (1.0 - alpha_grad) * prob + alpha_grad * grad
    enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0)
    threshold = float(enhanced.mean() + 0.3 * enhanced.std())
    mask = (enhanced > threshold).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return cv2.resize(mask, original_size, interpolation=cv2.INTER_NEAREST)

def authentic_or_mask(mask, prob_small, min_area=400, min_mean_conf=0.30):
    area = int(mask.sum())
    if area == 0:
        return "authentic"
    resized = cv2.resize(mask, prob_small.shape[::-1], interpolation=cv2.INTER_NEAREST)
    mean_conf = float(prob_small[resized.astype(bool)].mean()) if area else 0.0
    if area < min_area or mean_conf < min_mean_conf:
        return "authentic"
    return mask
```

The exact sentinel and RLE format are task-specific. document-authenticity segmentation used `"authentic"`
for empty predictions and JSON-list RLE for forged masks.

## 6. Model Families

### 6.1 U-Net family

Default choices:

- U-Net with ResNet34/50, SE-ResNeXt50, EfficientNet-B3/B5, ConvNeXt-T/S
- FPN for large-scale context and multi-class masks
- Unet++/Nested U-Net for medical/defect tasks
- LinkNet/PAN/DeepLabV3+ when local code supports them
- SCSE/attention/hypercolumns when small structures or channel attention matter

### 6.2 Transformer and modern decoder routes

Use when the dataset has enough scale or the local stack already supports them:

- SegFormer `mit-b3/b4/b5`
- UPerNet with Swin/ConvNeXt/CoaT
- Swin Transformer decoders
- MaxViT/ConvNeXt encoders in U-Net/FPN decoders

SegFormer is a strong large-data option. Transformer routes are not
automatically better for small data or strict runtime.

### 6.3 Instance segmentation frameworks

Use for per-object mask AP:

- Mask R-CNN
- Cascade Mask R-CNN
- HTC
- Detectron2/MMDetection
- RTMDet/YOLOX with mask supervision
- YOLOv7/YOLOv8 segmentation for fast baselines
- PointRend/mask-head upgrades when boundaries matter

For large medical-tissue segmentation/cell-instance segmentation-like tasks, optimize bbox quality, score calibration, and
mask overlap constraints before spending time on exotic mask heads.

### 6.4 Cell-specific methods

Use when objects are dense, small, touching, and biologically shaped:

- Cellpose with diameter tuning
- U-Net with border/center/distance/vector-field heads
- Mask R-CNN/Detectron with smaller anchors and high max detections
- crop-level UPerNet/U-Net inside detected boxes
- cell-level multi-label imagingCellSeg for cell-level multi-label imaging cell masks

Neuronal/SH-SY5Y-like cells can be harder because of irregular concave shapes;
use class/cell-type diagnostics.

### 6.5 3D model families

Use for full volumes or surface metrics:

- nnU-Net 3D fullres/lowres/cascade
- MONAI U-Net/FlexibleUNet/SegResNet/DynUNet
- 3D Attention U-Net, 3D SEResNeXt/ResNet encoders
- 2.5D ConvNeXt/U-Net over orthogonal axes
- SDF regression U-Net for topology-sensitive surfaces

If full 3D is too expensive, use 2.5D multi-axis and validate full-volume
aggregation.

### 6.6 Foundation-feature segmenters

Use only when local weights are allowed and validation supports them:

- frozen DINOv2/ViT encoder plus lightweight decoder
- SAM-like mask generators only if the competition permits the weights and the
  task has a reliable prompt/proposal route
- image embeddings plus classical postprocess for authentic/forged or defect
  masks

Foundation encoders still need task-specific thresholds, RLE, and validation.

## 7. Training, Losses, And Augmentation

### 7.1 Loss selection

Default stable choices:

- binary: BCEWithLogits plus Dice
- multi-class: CE plus Dice or Lovasz
- class-channel multi-label: BCEWithLogits plus Dice/Focal
- extreme imbalance: Focal/Tversky/Focal-Tversky plus Dice
- AP/IoU-sensitive masks: Lovasz or Dice fine-tune after BCE
- surface metrics: boundary-weighted CE plus Dice/Focal/Tversky
- topology/SDF: signed-distance L1 or smooth L1 plus mass/Dice-like term

### 7.2 Focal plus log Dice for extreme imbalance

For rare foreground pixels, combine focal loss with `-log(soft dice)`.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

def soft_dice_prob(prob, target, eps=1.0):
    prob = prob.reshape(-1)
    target = target.reshape(-1)
    inter = (prob * target).sum()
    return (2.0 * inter + eps) / (prob.sum() + target.sum() + eps)

class FocalLogDiceLoss(nn.Module):
    def __init__(self, focal_weight=10.0, gamma=2.0):
        super().__init__()
        self.focal_weight = focal_weight
        self.gamma = gamma

    def forward(self, logits, target):
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        pt = torch.exp(-bce)
        focal = ((1.0 - pt) ** self.gamma * bce).mean()
        dice = soft_dice_prob(torch.sigmoid(logits), target)
        return self.focal_weight * focal - torch.log(dice.clamp_min(1e-6))
```

Use this when missing small positives is worse than a small increase in false
positives, and validate empty-mask behavior.

### 7.3 Border and distance targets

For touching instances:

- full foreground channel
- border/contour channel
- center or distance transform
- empty-border softmax target
- vector field or flow target

For touching instances, border-aware targets plus watershed can matter more than swapping architectures.

### 7.4 Sampling schedules

For empty-heavy medical/defect tasks:

- start with a high positive sampling rate to learn the target
- gradually restore natural negative prevalence
- keep validation at natural prevalence

For rare classes:

- oversample positives
- stratify folds by class and area
- use per-class loss weights only when they improve metric

For 3D:

- sample non-empty crops
- include hard negative/background crops
- sample across volumes and axes
- avoid training only on easy dense-label regions if test has sparse labels

### 7.5 Augmentation

Common safe augmentations:

- horizontal/vertical flip when label-safe
- rotate90 for microscopy/satellite/volume if orientation is arbitrary
- scale/shift/crop
- brightness/contrast/gamma
- blur/noise
- elastic/grid distortion for cells/nuclei/tissue
- stain/color augmentation for histology
- 3D flips/rotations for volume tasks

High-risk augmentations:

- vertical flip in anatomy or driving scenes if orientation matters
- heavy color jitter in forensic/camera tasks
- elastic distortion in geospatial/man-made structures if geometry matters
- nearest-neighbor resizing of soft/probability masks before thresholding
- applying augmentations inconsistently between image and mask

### 7.6 Pseudo labels and external data

Use pseudo labels only after the base pipeline is stable.

Good patterns:

- soft pseudo masks rather than hard thresholded masks
- classifier and segmentation agreement for empty-heavy tasks
- target-domain unlabeled data allowed by rules
- separate source tags or validation reports for pseudo/external sources
- sparse-to-dense refinement in 3D when dense labels are scarce

Risky patterns:

- public-LB-derived pseudo labels
- hidden/manual labels
- external data with incompatible annotation policy
- pseudo labels that improve public but hurt private source validation

## 8. Inference, Postprocessing, And Ensembling

### 8.1 Threshold tuning

Tune thresholds on OOF predictions after all preprocessing and resizing.

Parameters to tune:

- pixel threshold
- label/existence threshold
- min area/min component area
- high/low triplet thresholds
- per-class thresholds
- max detections per image
- mask binary threshold for instance masks
- 3D volume threshold
- hysteresis high/low thresholds

Use coarse grids first. Avoid overfitting hundreds of thresholds unless the
validation set supports it.

### 8.2 TTA

Common TTA:

- horizontal flip
- vertical flip
- rotate90/transpose for orientation-free images
- scale TTA for organ/tissue and instance masks
- 3D axis flips for volumes

Average probabilities/logits before thresholding. For SDF models, average SDF
outputs or logits consistently.

Do not use TTA that changes semantics, such as vertical flips in some anatomy or
driving tasks.

### 8.3 Instance ensembling

Instance ensembling options:

- WBF on boxes, then mask head/crop model for masks
- mask IoU NMS across model predictions
- score averaging or weighted score fusion
- per-class thresholds after ensemble
- rescoring by mask mean probability
- remove overlaps in score order

For large class hierarchies, expand or calibrate parent/child classes only if
the task metric and label hierarchy require it.

### 8.4 3D volume postprocess

large-volume material segmentation/3D vessel segmentation style postprocess:

- remove small components
- binary closing or anisotropic closing
- hole/cavity filling
- high/low hysteresis thresholding
- axis-consistency voting
- cc3d dust removal
- height-map or PCA-based local hole patching
- SDF threshold sweep
- ignore-region cleanup

Validate each step cumulatively. Some postprocess can improve visual continuity
while degrading surface dice or topology.

### 8.5 Submission validators

Semantic RLE checks:

- required row count
- required ids
- empty-mask sentinel
- decoded mask shape
- sorted positive runs
- no duplicate decoded pixels
- roundtrip encode/decode

Instance checks:

- one row per instance
- class id valid
- confidence in expected range
- no forbidden same-class overlaps
- max detections if specified
- masks resized to required submission size

Volume checks:

- volume shape matches original
- ignore labels handled
- slice ids complete
- empty slice sentinel correct
- TIFF/RLE/NPY format exact

Forensics checks:

- authentic sentinel exact
- forged mask RLE format exact
- sample submission order preserved

## 9. Route-Specific Playbooks

### 9.1 Ship / sparse object semantic masks

Use for large-object satellite segmentation-like tasks.

First implementation:

- RLE decode/encode roundtrip
- stratify by ship count/area and empty images
- train U-Net/FPN with ResNet/EfficientNet/ConvNeXt
- oversample positives but validate on all images
- focal/Dice or BCE/Dice loss
- connected components to split ships if metric expects objects
- ship/no-ship classifier gate if OOF supports it

Try next:

- hard-mine false positives from empty images
- higher resolution or full-res fine-tune
- Mask R-CNN or hybrid U-Net plus Mask R-CNN for crowded ships
- weak-connection separation by erosion/watershed

Avoid:

- random split when scenes/near-duplicates leak
- one semantic blob for multiple ships when metric scores objects
- public-LB threshold tuning

### 9.2 Steel / cloud class-channel masks

First implementation:

- one multi-channel model or per-class models
- class prevalence/empty row folds
- BCE/Dice/Lovasz/Focal loss
- per-class label threshold, pixel threshold, and min area
- classifier gate or segmentation classification head
- H/V flip TTA if label-safe

Try next:

- separate rare-class specialists
- pseudo labels selected by classifier/segmenter agreement
- defect/cloud-specific augmentations
- checkpoint/model ensemble

Avoid:

- global threshold for all classes without OOF evidence
- dropping rare classes as a default
- cloud train-only constraints applied to test unless guaranteed

### 9.3 Pneumothorax / medical CXR

First implementation:

- patient/study folds
- keep negative studies
- U-Net/FPN/SCSEUnet/Nested U-Net with ResNet/EfficientNet/SE-ResNeXt encoder
- BCE/Dice/Focal or Combo loss
- triplet threshold `(top, min_area, bottom)`
- checkpoint averaging and flip TTA

Try next:

- lower-to-higher-resolution uptraining
- sliding positive sampling schedule
- external/pretrain data only if rules and validation allow it
- classifier gate only if threshold is stable

Avoid:

- positive-only validation
- random slice/image split across patients
- public-tuned triplet thresholds as final truth

### 9.4 Nuclei and cell instance segmentation

First implementation:

- instance AP scorer
- Detectron/Mask R-CNN/Cellpose or U-Net with connected components
- small-object anchors or higher max detections
- border/center/distance targets if using U-Net
- watershed or component splitting
- score-ordered overlap removal if submission forbids overlaps

Try next:

- LiveCell/external pretraining if allowed
- bbox-first detector plus crop mask model
- UPerNet/Mask R-CNN mask ensemble
- candidate-level GBDT scorer using morphology features

Avoid:

- treating foreground mask Dice as final metric
- letting touching cells merge
- using image-level weak labels as hard cell labels without lower weight or
  denoising

### 9.5 Large Medical-Tissue Segmentation

First implementation:

- donor/organ/source-aware folds
- SegFormer/UPerNet/Swin/CoaT or strong U-Net/FPN
- stain/color augmentation
- organ-specific thresholds
- full-scale inference if memory allows, otherwise validated tile stitching
- group norm when batch size is one

Try next:

- stain normalization/color transfer
- pixel-size-aware resizing
- external/pseudo organ data with source audit
- organ-specific specialists, especially lung/prostate-like failure modes

Avoid:

- cell-level multi-label imaging-only CV as final validation for large medical-tissue segmentation-like test
- random tile folds
- CutMix across incompatible organs unless validated

### 9.6 Vasculature 2D Instance/Semantic Hybrid

First implementation:

- source/WSI/spatial holdout
- train RTMDet/Mask R-CNN/Cascade/HTC or U-Net++ depending on final output
- focus on bbox AP if final metric is instance mask AP
- use EMA/SWA/checkpoint averaging
- WBF boxes and mask mean/ensemble

Try next:

- mask supervision to improve bbox predictions
- glomerulus/false-positive exclusion if annotations support it
- no-dilation and dilation variants as separate submissions

Avoid:

- trusting public LB on dilation
- random tile folds across WSI/source

### 9.7 3D Vessel Segmentation

First implementation:

- hold out full kidney/specimen
- train 2.5D U-Net/ConvNeXt with 3 neighboring slices
- infer x/y/z axes and average
- use boundary-weighted CE plus Dice/Focal/Tversky
- threshold full volume and remove tiny components
- output exact per-slice RLE and empty sentinel

Try next:

- random 3D rotation for non-axis-aligned slices
- soft pseudo labels for sparse-to-dense refinement
- full 3D model ensemble if memory allows
- private-resolution emulation/interpolation if task geometry supports it

Avoid:

- over-trusting public/private with few volumes
- more neighboring slices just because compute allows it
- hard pseudo masks when soft labels are safer

### 9.8 Large-Volume Surface Detection

First implementation:

- full-volume/specimen validation
- nnU-Net 3D with patch sizes around 128/160/192 depending on memory
- probability/logit ensemble or SDF ensemble
- threshold sweep
- remove small components
- close/fill holes with conservative topology checks
- preserve original `(D,H,W)` geometry at submission

Try next:

- patch-size diversity ensemble
- SDF regression and SDF threshold sweeps
- hysteresis thresholding
- height-map/PCA local hole repair
- anisotropic morphology reflecting sheet geometry

Avoid:

- slice Dice as the only selector
- ordinary dilation that overgrows sheets
- threshold overfit to public LB
- ignoring label/ignore-border artifacts

### 9.9 Multispectral Geospatial Polygons

First implementation:

- scene/tile folds
- band alignment and per-band normalization
- polygon rasterization for train masks
- U-Net/FPN/DeepLab with multispectral channels
- mask-to-polygon conversion with `min_area`, simplification, and validity repair
- class-specific thresholds

Try next:

- class specialists for tiny/rare classes
- spectral indices and domain features if validated
- central-crop tile training to avoid edge artifacts

Avoid:

- treating WKT/polygon conversion as formatting only
- random patch split from same scene
- applying waterway/spectral tricks to non-geospatial tasks

### 9.10 Single-Cell Segmentation-Assisted Classification

First implementation:

- use cell-level multi-label imagingCellSeg or provided cell masks
- crop/pad/resize cells
- remove unreliable border/no-nucleus cells if validation supports it
- route final labels to image classification
- blend image-level and cell-level predictions

Try next:

- dual-head image/cell model
- CAM overlap to assign soft cell labels
- confidence scaling for border cells
- antibody/image similarity features if allowed and private-stable

Avoid:

- optimizing segmentation quality when final score is classification
- hard image labels for every cell without noise handling
- duplicate public/test sample exploitation

### 9.11 Fashion And Driving Instance Masks

First implementation:

- Mask R-CNN/Cascade/HTC/Detectron/MMDetection baseline
- COCO pretraining if allowed
- exact submission format: plain RLE, COCO compressed RLE, attributes, or custom
  per-instance rows
- multi-scale training and high-resolution inference
- score threshold/NMS/mask-threshold tuning
- class hierarchy or attribute head only when metric requires it

Try next:

- deformable/strong backbones
- per-class/attribute thresholds
- parent/child hierarchy expansion for large-scale multi-class detection-like tasks
- mask-level NMS and TTA
- AP-predicting reranker for duplicate predictions

Avoid:

- pure boxes as final output
- semantic U-Net without instance separation
- wrong RLE compression or mask resize requirement

### 9.12 Scientific forgery / authentic masks

First implementation:

- source/manipulation split if available
- DINOv2 frozen encoder plus light decoder, or U-Net/SegFormer baseline
- authentic images as zero masks
- BCEWithLogits for mask head
- H/V flip TTA
- adaptive edge/gradient postprocess
- area and mean-confidence gate to return authentic sentinel
- JSON-list or task-specific RLE validator

Try next:

- progressive decoder
- rotation TTA
- two-model ensemble
- grid search over area/confidence gates with authentic validation images

Avoid:

- row-major educational RLE snippets when competition uses column-major JSON RLE
- forged-only validation that misses authentic false positives

## 10. Ensembling Strategy

Useful diversity axes:

- folds and seeds
- backbones: ResNet/EfficientNet/SE-ResNeXt/ConvNeXt/Swin/SegFormer/CoaT
- decoders: U-Net/FPN/Unet++/UPerNet/DeepLab/Mask R-CNN
- image sizes and tile sizes
- loss functions: BCE/Dice/Lovasz/Focal/Tversky/boundary/SDF
- postprocess thresholds
- 2D vs 2.5D vs 3D
- instance detector vs semantic segmenter

Blend probabilities/logits before thresholding for semantic masks. For instance
masks, combine boxes/masks with WBF, mask NMS, or AP-aware reranking. For 3D
surface tasks, validate probability vs logit vs SDF fusion because public and
private behavior can differ.

Keep OOF predictions for every base model. Do not tune ensemble weights on
public LB.

## 11. Failure Modes

### 11.1 Wrong prediction unit

Symptoms:

- high pixel Dice but poor AP
- merged objects in instance tasks
- valid boxes but invalid mask submission
- slice score improves but full-volume score fails

Fix:

- reroute by metric and submission row
- implement exact metric and validator
- add instance recovery or volume aggregation

### 11.2 RLE or geometry failure

Symptoms:

- blank submission despite visible masks
- masks appear transposed/rotated
- submission rejected for duplicate pixels or invalid runs
- correct local mask but wrong leaderboard

Fix:

- decode sample submission and one train mask
- roundtrip encode/decode
- visualize overlays at submission resolution
- assert row count/order and empty sentinel

### 11.3 Empty-mask failure

Symptoms:

- many tiny false masks
- good positive Dice but poor mean Dice
- classifier gate removes true positives

Fix:

- tune label threshold, pixel threshold, and min area on OOF
- validate false positives on empty rows
- add a classifier gate only if stable

### 11.4 Domain shift failure

Symptoms:

- random CV high, group CV low
- external/pseudo data hurts
- one organ/stain/source dominates errors

Fix:

- rebuild group/source folds
- report per-source/per-class metrics
- use source-specific augmentation or thresholding only if OOF supports it

### 11.5 3D topology failure

Symptoms:

- visually smooth slices but broken surfaces
- holes/cavities hurt topology
- public/private threshold mismatch

Fix:

- score full volumes
- threshold sweep
- connected-component and hole repair ablations
- keep conservative/no-postprocess fallback
