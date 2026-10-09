# Image Detection Category Playbook

This playbook is the full method library for Kaggle-style `image_detection`
tasks. It is intended as direct reading material for a skill-generating agent:
after reading a task description, the agent should route the task, choose the
right validation geometry, then extract only the applicable recipes.

Do not use this playbook to justify leaderboard probing, public-test label
inference, hidden labels, manual test labeling, sample-submission label updates,
or disallowed external data.

## 1. Task Routing

### 1.1 Ordinary 2D object detection

Use when the submission is a set of 2D bounding boxes per image or row, usually
with confidence scores and an IoU-based metric.

Default plan:

- leakage-safe image folds, stratified by class, source, and box count when
  available
- one serious detector first: YOLOv8/YOLO11, YOLOv5, EfficientDet D3-D7,
  Faster R-CNN/FPN, Cascade R-CNN, FCOS, RetinaNet, or Detectron2/MMDetection
  equivalent
- exact annotation conversion between COCO `xywh`, Pascal/VOC `xyxy`, YOLO
  normalized center format, and competition submission strings
- official metric on OOF predictions after NMS/WBF/thresholding
- WBF, multi-scale inference, flips, and second detector family only after a
  stable OOF baseline

### 1.2 Dense small-object detection

Use when boxes are small, numerous, or close together: wheat heads, helmets,
coral/starfish, geospatial objects, aerial imagery, microscopy objects, weeds,
and similar domains.

Default plan:

- preserve resolution early; do not downsample until small objects vanish
- stratify folds by source and bbox density
- consider tiling/patch training with careful boundary filtering
- use mosaic/mixup, scale/shift/crop, and rotate90/transpose when geometry
  allows
- tune max detections, score threshold, and NMS/WBF IoU with the official metric
- inspect edge boxes and clipped boxes visually

High-value model families:

- YOLOv5/YOLOv8/YOLO11 large-resolution variants
- EfficientDet D3-D7 with multi-scale inference
- Cascade R-CNN, Faster R-CNN/FPN, FCOS, RetinaNet, CenterNet-style detectors
- ensemble diversity from at least two detector families when runtime allows

### 1.3 Medical detection with positives and negatives

Use for CXR, DICOM, radiology, pathology tiles, medical images, and other
domains with many negative studies or subjective annotations.

Default plan:

- patient/study grouped folds before augmentation, tiling, or crop generation
- keep negative images as first-class training rows when the metric penalizes
  false positives on empty images
- train a positive/negative image classifier gate when the dataset has many
  negatives or detector scores are not well calibrated
- evaluate both positive-only detection quality and all-image metric impact
- tune thresholds on OOF predictions, not on detector loss
- consider box-size calibration only if annotation policy supports it and OOF
  metric improves

Useful model families:

- RetinaNet, Faster R-CNN/FPN, Deformable R-FCN, Deformable Relation Networks,
  Cascade R-CNN, YOLO/FCOS variants
- classifier gate with EfficientNetV2, ConvNeXt, DenseNet, Xception,
  InceptionResNetV2, NFNet, or similar strong image classifiers

### 1.4 Video or temporal detection

Use when the test interface serves frames in order, the metric has temporal
tolerance, or detections form events over frames.

Default plan:

- split by `video_id`, sequence, game, or source; never random frames
- train a detector candidate stage on frames
- add tracking, optical flow, or temporal crop classification only after frame
  detection has high recall
- score at the official event/frame unit, including temporal matching rules
- use video NMS, duplicate suppression, top-k per video, or score propagation
  only after OOF validation

High-value model families:

- candidate detector: YOLOv5/YOLOv8/YOLO11, EfficientDet, Faster R-CNN,
  Detectron2, CenterNet, FCOS
- temporal classifier: TSM EfficientNet/ResNet, 3D CNN, SlowFast-style ROI
  classifier, EfficientNet B3/B5 over stacked frames or crops
- tracking: IoU tracker, OpenCV optical flow, RAFT-like flow when available and
  allowed

### 1.5 Volumetric point localization

Use when the output is a 3D point or centroid, often `x,y,z`, in tomograms,
microscopy volumes, cryo-ET, or biomedical/scientific volumes.

Default plan:

- route as 3D point localization, not ordinary 2D object detection
- split by experiment, tomogram, run, source dataset, scanner, or biological
  sample
- train 3D U-Net/nnU-Net/MONAI models on heatmaps, blobs, masks, or distance
  transforms
- decode dense outputs into centroids with connected components, local maxima,
  or max-pool NMS
- calibrate presence/point thresholds with the official F-beta or distance
  metric
- use 2D-slice detectors only if they aggregate detections back into 3D

High-value model families:

- 3D U-Net, nnU-Net, MONAI UNet/FlexibleUNet/SegResNet/DynUNet
- 3D ResNet/ResNeXt/DenseNet/X3D encoders inside U-Net or FPN-style heads
- Gaussian/EDT/blob regression heads for sparse points
- 2.5D slice YOLO only as a compute-saving route with 3D clustering/NMS

### 1.6 Single-object 3D presence and localization

Use when each volume has zero or one target point and no-target rows must use
sentinel coordinates.

Default plan:

- train a heatmap/blob or small-object segmentation target around the point
- include no-object volumes in validation and threshold calibration
- decode the strongest peak or connected component
- tune the presence threshold or quantile threshold from OOF predictions
- emit sentinel coordinates such as `-1` only when calibrated confidence is
  below threshold

This route is common in microscopy and tomogram localization. It is not a
classification task because coordinate error is scored.

### 1.7 3D pose from image

Use when each object has 2D image evidence but the submission requires
`pitch yaw roll x y z confidence` or similar 6DoF pose strings.

Default plan:

- CenterNet-style heatmap for object centers
- regression heads for depth/translation, size or width/height, and angles
- represent periodic angles with sine/cosine or wrapped residuals where useful
- use camera geometry, masks, perspective transforms, and duplicate removal
- validate with the official pose metric, not only 2D center heatmap loss

### 1.8 Large-scale multi-class detection

Use for large-scale multi-class detection-style tasks with hundreds of classes, class hierarchy,
per-class AP, and severe class imbalance.

Default plan:

- preserve rare class support in folds
- sample or weight classes so frequent classes do not dominate
- tune class-aware thresholds and max detections
- apply valid hierarchy expansion after prediction when the metric rewards
  parent labels
- keep per-class AP diagnostics, not only global loss
- use pseudo labels only when allowed and confidence-filtered

Useful model families:

- Faster R-CNN/FPN, Cascade R-CNN, RetinaNet, FCOS, YOLO, Deformable DETR/DINO
  variants when implementation/runtime permit
- backbone diversity from ResNeXt, Swin, ConvNeXt, EfficientNet, NFNet, or
  task-provided pretrained detectors

### 1.9 Domain-specific 2D detection

Use for geospatial, aerial, agriculture, underwater, industrial, retail, or
domain-specific boxes.

Default plan:

- identify the leakage axis: site, camera, flight, video, geography, farm,
  scanner, machine, batch, or source dataset
- audit train/test image sizes, aspect ratios, object density, and background
  prevalence
- preserve domain-specific resolution and color channels
- use metadata only through fold-safe fusion or grouping
- build robust postprocess before chasing exotic backbones

This route usually borrows from ordinary 2D detection plus a stricter split.

## 2. Validation And Leakage

### 2.1 Define the scored unit before training

Detection training often happens on crops, tiles, frames, patches, or volumes,
but validation must score the official prediction unit.

Typical scored units:

- image: boxes or empty prediction string
- video frame: frame-level boxes
- event: temporally matched impact/object event
- study/patient image: medical boxes and negative studies
- object instance: class-labeled box with per-class AP
- tomogram/run: 3D point list or one 3D point
- pose string: one or more 6DoF objects per image
- API step: ordered frame or time step

Save OOF predictions in the exact scored geometry after postprocess. If training
uses tiles, also reconstruct full-image predictions before scoring. If training
uses 2D slices, aggregate to 3D before scoring.

### 2.2 Leakage axes to audit

Before making folds, inspect:

- repeated frames from the same video or sequence
- patient/study/scan identity
- camera, site, farm, source, geography, flight, run, tomogram, or scanner id
- game/play/team identity for sports video
- synthetic vs real source
- annotation source or reader policy
- duplicate and near-duplicate images
- crops/tiles generated from one original image
- negative images or no-target volumes
- class hierarchy and rare-class coverage

When in doubt, group by the largest capture unit that could share target
appearance or annotation policy.

### 2.3 Stratified image folds by source and box density

For ordinary image boxes, stratify by class/source and object density where
possible. This catches failures where every fold has images but only some folds
have dense objects or a specific source.

`Implementation Pattern`:

```python
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

df_folds = marking[["image_id"]].copy()
df_folds.loc[:, "bbox_count"] = 1
df_folds = df_folds.groupby("image_id").count()
df_folds.loc[:, "source"] = (
    marking[["image_id", "source"]].groupby("image_id").min()["source"]
)
df_folds.loc[:, "stratify_group"] = np.char.add(
    df_folds["source"].values.astype(str),
    df_folds["bbox_count"].apply(lambda x: f"_{x // 15}").values.astype(str),
)
df_folds.loc[:, "fold"] = 0

for fold_number, (_, val_index) in enumerate(
    skf.split(X=df_folds.index, y=df_folds["stratify_group"])
):
    df_folds.loc[df_folds.iloc[val_index].index, "fold"] = fold_number
```

Adaptation notes:

- replace `source` with site/camera/farm/scanner/dataset id when present
- for multi-class tasks, add rare-class indicators or dominant class to the
  stratification group
- validate that every important class and density bucket appears in each fold
- if a source is a hard leakage unit, use grouped folds instead of stratifying
  across it

### 2.4 Group folds for video and sequence tasks

For video detection, folds must hold out whole videos, sequences, games, or
capture sessions.

`Implementation Pattern`:

```python
from sklearn.model_selection import GroupKFold

kf = GroupKFold(n_splits=3)
df = df.reset_index(drop=True)
df["fold"] = -1

for fold, (_, val_idx) in enumerate(kf.split(df, groups=df.video_id.tolist())):
    df.loc[val_idx, "fold"] = fold
```

Adaptation notes:

- use `game_id`, `sequence`, `study_id`, `run_id`, or `tomogram_id` when that is
  the true capture unit
- do not let adjacent frames from the same event appear in train and validation
- if the metric has temporal tolerance, score after temporal matching and
  duplicate suppression

### 2.5 Medical negative validation

For medical detection with empty images:

- include negative studies in every validation fold
- compute all-image metric and positive-only metric separately
- track false positives per negative image
- tune classifier-gate and box thresholds on OOF predictions
- inspect images where the detector emits confident false positives
- do not train only positives unless the metric ignores negatives, which is
  rare

### 2.6 3D/tomogram validation

For volumetric detection:

- split by run/tomogram/experiment/source, not by patches
- hold out entire volumes before patch extraction
- score decoded `x,y,z` points, not voxel loss
- calibrate point threshold on OOF volumes
- report recall and false positives per class or volume
- preserve voxel spacing and coordinate convention in OOF files

For sparse single-point tasks, validate both:

- presence classification: should the volume emit a point?
- localization: is the emitted point within the official distance threshold?

### 2.7 What to trust when CV and leaderboard disagree

First check:

- coordinate conversion and image scaling
- empty-row formatting and max detections
- fold leakage or source mismatch
- metric implementation, especially IoU thresholds and temporal windows
- postprocess tuned on train loss rather than OOF metric
- train/test resolution or annotation policy shift

Then try:

- stricter group/source validation
- source-specific diagnostics
- OOF visual inspection after NMS/WBF
- calibration plots of confidence vs true positives
- model diversity only after metric code is trusted

Do not respond to mismatch by adding fragile leaderboard heuristics.

## 3. Annotation Conversion And Submission Safety

### 3.1 Coordinate conventions

Always name the format in variable names:

- `xyxy`: `x1, y1, x2, y2`, usually Pascal/VOC pixel coordinates
- `xywh`: `x, y, width, height`, often COCO or submission format
- `cxcywh`: center `x, y, width, height`, used by YOLO after normalization
- normalized YOLO: `class cx cy w h` in `[0, 1]`
- 3D points: be explicit whether order is `x,y,z` or `z,y,x`
- pose: preserve exact order, angle units, and confidence sorting

Visualize at least 20 training images after conversion, including empty images,
large boxes, small boxes, edge boxes, and dense images.

### 3.2 COCO/VOC/YOLO conversion

`Implementation Pattern`:

```python
def voc2yolo(bboxes, image_height, image_width):
    bboxes = bboxes.copy().astype(float)
    bboxes[..., [0, 2]] /= image_width
    bboxes[..., [1, 3]] /= image_height
    wh = bboxes[..., [2, 3]] - bboxes[..., [0, 1]]
    bboxes[..., [0, 1]] += wh / 2
    bboxes[..., [2, 3]] = wh
    return bboxes

def yolo2voc(bboxes, image_height, image_width):
    bboxes = bboxes.copy().astype(float)
    bboxes[..., [0, 2]] *= image_width
    bboxes[..., [1, 3]] *= image_height
    bboxes[..., [0, 1]] -= bboxes[..., [2, 3]] / 2
    bboxes[..., [2, 3]] += bboxes[..., [0, 1]]
    return bboxes

def coco2yolo(bboxes, image_height, image_width):
    bboxes = bboxes.copy().astype(float)
    bboxes[..., [0, 2]] /= image_width
    bboxes[..., [1, 3]] /= image_height
    bboxes[..., [0, 1]] += bboxes[..., [2, 3]] / 2
    return bboxes

def yolo2coco(bboxes, image_height, image_width):
    bboxes = bboxes.copy().astype(float)
    bboxes[..., [0, 2]] *= image_width
    bboxes[..., [1, 3]] *= image_height
    bboxes[..., [0, 1]] -= bboxes[..., [2, 3]] / 2
    return bboxes
```

Common mistakes:

### 3.3 YOLO label export

When exporting YOLO labels, normalize only the bbox columns.

`Implementation Pattern`:

```python
df["x_center"] = df.x + df.w / 2
df["y_center"] = df.y + df.h / 2

label_values = mini[["class", "x_center", "y_center", "w", "h"]].astype(float).values
label_values[:, 1:] /= image_size
```

Adaptation notes:

- use per-image width/height when images are not square or not uniformly resized
- if training on tiles, subtract tile origin before normalizing
- keep empty label files for negative images when the training framework expects
  them

### 3.4 TorchVision/Faster R-CNN target dict

`Implementation Pattern`:

```python
boxes = records[["x", "y", "w", "h"]].values.astype("float32")
boxes[:, 2] += boxes[:, 0]
boxes[:, 3] += boxes[:, 1]

target = {
    "boxes": torch.as_tensor(boxes, dtype=torch.float32),
    "labels": torch.ones((len(records),), dtype=torch.int64),
    "image_id": torch.tensor([index]),
    "area": torch.as_tensor(
        (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]),
        dtype=torch.float32,
    ),
    "iscrowd": torch.zeros((len(records),), dtype=torch.int64),
}
```

Adaptation notes:

- for multi-class tasks, map labels to contiguous positive class ids expected by
  the framework
- for empty images, confirm whether the detector supports zero boxes in the
  target dict
- after Albumentations, rebuild `target["boxes"]` from transformed bboxes
- ensure `area` is positive after clipping

### 3.5 Bbox-safe augmentation

Albumentations can be useful only if `bbox_params` matches the current format
and label fields. Do not use ordinary image transforms that ignore boxes.

`Implementation Pattern`:

```python
def get_train_transforms():
    return A.Compose(
        [
            A.RandomSizedCrop(
                min_max_height=(800, 800), height=1024, width=1024, p=0.5
            ),
            A.OneOf(
                [
                    A.HueSaturationValue(
                        hue_shift_limit=0.2,
                        sat_shift_limit=0.2,
                        val_shift_limit=0.2,
                        p=0.9,
                    ),
                    A.RandomBrightnessContrast(
                        brightness_limit=0.2, contrast_limit=0.2, p=0.9
                    ),
                ],
                p=0.9,
            ),
            A.ToGray(p=0.01),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Resize(height=512, width=512, p=1.0),
            A.CoarseDropout(max_holes=8, max_height=64, max_width=64, p=0.5),
            ToTensorV2(p=1.0),
        ],
        p=1.0,
        bbox_params=A.BboxParams(
            format="pascal_voc",
            min_area=0,
            min_visibility=0,
            label_fields=["labels"],
        ),
    )
```

Adaptation notes:

- disable vertical flips when object orientation is physically meaningful
- rotate90/transpose can help aerial, underwater, microscopy, and some
  agriculture tasks, but can hurt driving and medical anatomy tasks
- set `min_visibility` carefully for tiny objects; aggressive filtering can
  delete positives
- for tiling, filter boxes near patch boundaries only if validation confirms it

### 3.6 Prediction string construction

For common single-class `PredictionString` formats, emit confidence and box
coordinates in the exact order required by the competition.

General pattern:

```python
def format_prediction_string(boxes_xywh, scores):
    parts = []
    for score, (x, y, w, h) in zip(scores, boxes_xywh):
        parts.extend([f"{score:.4f}", f"{x:.0f}", f"{y:.0f}", f"{w:.0f}", f"{h:.0f}"])
    return " ".join(parts)
```

Submission checks:

- one row per required id, including empty images
- empty prediction string is legal if no detections are emitted
- confidence sorting matches metric expectations
- coordinates use original test image scale
- boxes are clipped inside image bounds
- no NaN, negative width/height, duplicate ids, or missing sentinel rows

## 4. Model Families And When To Prefer Them

### 4.1 YOLO family

Use YOLOv5/YOLOv8/YOLO11/YOLO12-style detectors when:

- a strong one-script baseline is needed quickly
- objects are small but visible at high resolution
- submission is ordinary boxes
- framework export and inference speed matter
- video frames require fast candidate generation

High-value knobs:

- image size and rectangular inference
- mosaic/mixup probability
- class weights or sampling for imbalance
- confidence and IoU thresholds
- max detections
- multi-scale and flip TTA
- patch/tiling strategy for high-resolution dense images

### 4.2 EfficientDet

Use EfficientDet D3-D7 when:

High-value knobs:

- input size: 512/768/1024/1280 depending on object size and memory
- EMA and stable augmentation
- WBF over folds/scales/models
- pseudo labels when train/test source shift and rules permit
- box and score thresholds tuned by OOF metric

### 4.3 Faster R-CNN, Cascade R-CNN, FCOS, RetinaNet

Use these families when:

- box localization precision matters
- negatives must be handled carefully
- YOLO confidence calibration is weak
- a second detector family is needed for ensemble diversity
- Detectron2/MMDetection is already feasible in the environment

Model choices:

- Faster R-CNN/FPN with ResNet50/101, ResNeXt, ConvNeXt, Swin, or ResNet152
- Cascade R-CNN for stricter localization and multi-IoU refinement
- RetinaNet for one-stage dense detection and negative-image tradeoffs
- FCOS/CenterNet for anchor-free diversity
- Deformable R-FCN or Deformable Relation Networks when local code supports
  them and the task resembles historical RSNA/large-scale multi-class detection-style routes

High-value knobs:

- anchors and strides for object size
- max detections per image
- class-agnostic vs class-specific NMS
- positive/negative image sampling
- score calibration and classifier gates
- TTA with scale and flips

### 4.4 DETR-like detectors

- framework support is already installed
- objects are medium/large and categories are many
- long training or pretrained weights are allowed
- a high-quality YOLO/Faster R-CNN baseline already exists

Delay them when:

- compute is tight
- objects are extremely tiny
- the task is dominated by thresholding, empty images, or temporal postprocess
- submission code and validation are not yet stable

### 4.5 Classifier gates and crop re-scorers

Use a classifier gate when:

- many images are negative
- detector false positives dominate the metric
- medical or rare-event detection has a separate image-level signal
- frame-level detector proposals need event confirmation

Use a crop re-scorer when:

- detector recall is high but precision is poor
- false positives are visually separable in crops
- the metric rewards lower thresholds but penalizes noisy detections
- OOF detector predictions can create supervised crop labels

Useful classifiers:

- EfficientNet B0-B7/EfficientNetV2
- ConvNeXt-T/S/B, NFNet, RegNet, ResNet/ResNeXt
- DenseNet169/201, Xception, InceptionResNetV2 for medical-style legacy routes
- TSM/3D CNN/SlowFast for temporal crops

### 4.6 3D and volumetric model families

Use these for tomograms and sparse 3D points:

- nnU-Net as a strong automated baseline if package/runtime allow
- MONAI UNet, FlexibleUNet, SegResNet, DynUNet
- custom 3D U-Net with ResNet/EfficientNet-like encoders
- blob/EDT/Gaussian heatmap regression
- 2D/2.5D MaxViT/CoaT/ConvNeXt slice classifiers or detectors only with 3D
  reconstruction

High-value knobs:

- voxel spacing normalization
- patch size and output crop size
- overlap and Gaussian weighting at inference
- class-specific radius for synthetic masks/blobs
- positive-centered crop sampling
- connected-component size thresholds
- max-pool NMS radius
- quantile or OOF threshold calibration

## 5. Training Recipes

### 5.1 Strong first 2D detection loop

Build this before complex upgrades:

1. Parse annotations and build one canonical dataframe with `image_id`,
   `class`, `x1`, `y1`, `x2`, `y2`, `width`, `height`, and source/group ids.
2. Create folds that protect source, video, patient, study, geography, or box
   density.
3. Export labels for one framework only.
4. Train one strong detector at a resolution that preserves the object.
5. Run OOF inference and score the official metric after postprocess.
6. Save OOF boxes, test boxes, folds, and model configs.
7. Visualize OOF true positives, false positives, and false negatives.
8. Tune thresholds/NMS/WBF using OOF.
9. Add model diversity or pseudo labels only after these artifacts are stable.

### 5.2 Input resolution and tiling

Resolution is often a bigger lever than backbone name.

Use full-image training when:

- objects are medium or large
- memory allows official aspect ratio
- context matters
- submission boxes are full-image boxes and tiling adds boundary risk

Use patch/tiling training when:

- full image downsampling hides tiny objects
- dense objects are concentrated in small regions
- high-res images exceed memory
- negative background patches are useful

Tiling rules:

- store tile origin and scale
- clip boxes to tile boundaries
- decide whether to remove boundary boxes or keep them with visibility filters
- reconstruct full-image coordinates before NMS/WBF
- validate on full images, not independent tiles

### 5.3 Augmentation by domain

Generally useful:

- horizontal flip when geometry permits
- scale/shift/crop with bbox-safe transforms
- brightness/contrast/hue/saturation/noise/blur
- coarse dropout/cutout
- mosaic/mixup for dense objects
- multi-scale training

Domain-specific:

- rotate90/transpose: aerial, microscopy, underwater, some agriculture
- vertical flip: only if orientation is not meaningful
- color inversion: some radiology pipelines used it, but validate carefully
- mild augmentation: medical anatomy often needs less aggressive geometry
- background mix: can help video/frame detectors with sparse positives

Avoid:

- crop policies that delete most small boxes
- HSV shifts that break domain-specific color cues
- arbitrary rotations when gravity/camera perspective is meaningful
- augmentation that creates impossible anatomy or vehicle orientation

### 5.4 Label cleaning and annotation policy

Useful checks:

- tiny boxes below plausible object size
- giant boxes or boxes outside image
- missing labels in obvious dense regions
- duplicate boxes with high IoU
- class hierarchy inconsistencies
- train/test annotation policy differences
- box size distribution by source/reader

Safe repairs:

- remove or clip impossible boxes
- fix obvious parse errors
- keep an audit file of repaired rows
- validate before and after repair on OOF

Risky repairs:

- manual relabeling of test images
- inferring hidden labels from public leaderboard
- shrinking/expanding boxes solely because one public split improved
- relabeling subjective medical data without domain rationale

### 5.5 Pseudo labels

Pseudo labels can help under source shift or incomplete annotations, but they
are late-round tools.

Use when:

- target-domain unlabeled data is allowed
- teacher OOF/test predictions are strong
- confidence thresholds are class-aware
- visual inspection confirms reasonable labels
- OOF metric detects noise amplification

Avoid when:

- test pseudo-labeling is forbidden
- pseudo labels come from public/hidden labels
- the model is not yet stable
- rare classes would be drowned by noisy frequent classes
- validation cannot represent the pseudo-label domain

Practical patterns:

- train teacher on folds
- generate pseudo labels with WBF/TTA ensemble
- filter by score, size, class, and source
- mix pseudo labels with lower weight or later epochs
- repeat at most a few rounds and stop when OOF stops improving

### 5.6 Multi-class imbalance and hierarchy

For large-scale multi-class detection-like tasks:

- measure positives per class and per fold
- keep rare classes in every validation fold when possible
- oversample images containing rare classes
- use class-aware thresholds
- evaluate per-class AP and macro behavior
- expand parent labels only if the metric and label hierarchy support it
- check that hierarchy expansion does not create invalid duplicate predictions

Do not optimize only global loss. A frequent class can hide catastrophic
rare-class AP loss.

## 6. Inference, Ensembling, And Postprocessing

### 6.1 Weighted Boxes Fusion

WBF is one of the highest-ROI ensemble steps for 2D boxes. Use normalized boxes,
consistent labels, and OOF-tuned thresholds.

`Implementation Pattern`:

```python
def run_wbf(predictions, image_index, image_size=512, iou_thr=0.44, skip_box_thr=0.43):
    boxes = [
        (prediction[image_index]["boxes"] / (image_size - 1)).tolist()
        for prediction in predictions
    ]
    scores = [
        prediction[image_index]["scores"].tolist()
        for prediction in predictions
    ]
    labels = [
        np.ones(prediction[image_index]["scores"].shape[0]).tolist()
        for prediction in predictions
    ]
    boxes, scores, labels = weighted_boxes_fusion(
        boxes,
        scores,
        labels,
        weights=None,
        iou_thr=iou_thr,
        skip_box_thr=skip_box_thr,
    )
    boxes = boxes * (image_size - 1)
    return boxes, scores, labels
```

Adaptation notes:

- for non-square or original-size boxes, normalize by per-image width and height
  rather than one `image_size`
- for multi-class detection, pass real class labels
- tune `iou_thr` and `skip_box_thr` with OOF metric
- calibrate scores across models before weighted fusion if model score scales
  differ strongly
- compare against NMS and Soft-NMS; WBF is not universally best

### 6.2 TTA

Useful TTA:

- horizontal flip
- vertical flip or transpose only if valid for the domain
- multi-scale inference around training size
- tile overlap at inference
- rotation TTA for orientation-invariant microscopy/aerial tasks

TTA combination:

- invert transforms back to original coordinate system
- clip boxes after inverse transform
- WBF or NMS transformed predictions
- optionally multiply score by agreement count across TTAs
- keep runtime constraints in mind early

### 6.3 Thresholding and max detections

Tune with the official OOF metric:

- global score threshold
- class-specific threshold
- image-level classifier threshold
- NMS/WBF IoU
- max detections per image
- minimum box size
- per-video top-k
- 3D presence threshold
- connected-component voxel count threshold

Metric behavior:

- mAP over IoU thresholds: over-threshold false positives can hurt; tune
  confidence sorting and duplicate suppression
- F2/recall-heavy metrics: lower detector thresholds and crop re-scorers can
  help
- empty-image penalties: classifier gates and higher thresholds can help
- per-class AP: tune thresholds per class
- temporal F1: suppress duplicates within the accepted time window
- 3D F-beta: prioritize recall but control duplicate points

### 6.4 NMS, Soft-NMS, WBF, NMW

Postprocess options:

### 6.5 Classifier gates and crop rescoring

Implementation pattern:

1. Use OOF detector predictions to crop candidate boxes.
2. Assign crop labels by IoU to ground truth or event match.
3. Train a classifier on OOF-like proposals, not only perfect GT crops.
4. At inference, multiply or blend detector score with classifier score.
5. Tune the blend and thresholds on OOF.

For medical negatives:

- image-level classifier score can multiply box scores or gate the whole image
- tune for all-image metric, not only classifier AUC
- keep some detector-only boxes if classifier false negatives are costly

For recall-heavy video:

- crop re-scorer can run at low detector confidence threshold
- temporal classifier can use stacked frames around the candidate

### 6.6 Video postprocess

For frame sequences:

- track detections across adjacent frames with IoU, optical flow, or learned
  tracking
- propagate score from high-confidence frame to nearby low-confidence frames
  only if validation improves
- suppress duplicate detections for the same event/player/object within the
  official temporal tolerance window
- limit top predictions per video if the metric and class prevalence support it
- keep API ordering constraints and runtime state separate from training code

General temporal matching sketch:

```python
for gt in ground_truth_events:
    for pred in predictions:
        if abs(gt.frame - pred.frame) <= frame_tolerance and iou(gt.box, pred.box) >= iou_thr:
            cost[gt.index, pred.index] = -pred.score

rows, cols = linear_sum_assignment(cost)
```

Adaptation notes:

- use the official rule for one prediction per ground truth
- suppress already matched or nearby duplicate predictions
- validate on whole videos, not sampled frames

### 6.7 Empty images and no-target rows

Detection submissions often fail on empty cases.

Checklist:

- include negative images in folds and train files
- ensure inference emits empty strings/rows correctly
- tune thresholds for false positives per empty image
- if the metric gives zero for empty images with any false positive, be
  conservative or use a classifier gate
- for 3D single-point tasks, emit sentinel values exactly when no object is
  predicted
- never drop required rows from submission

## 7. Volumetric Point Localization

### 7.1 Target construction

Common target choices:

- binary mask spheres around points
- Gaussian heatmaps centered at points
- Euclidean distance transform blobs
- class-specific radius masks for multi-particle tomograms
- downsampled heatmaps for memory-constrained U-Nets

Tradeoffs:

- mask spheres work well with segmentation losses and connected components
- Gaussian/EDT blobs work well for local maxima and sparse points
- smaller radii reduce merged components but can lower recall
- larger radii improve recall but merge nearby particles
- class-specific radii matter when object sizes differ

### 7.2 3D model training

Strong defaults:

- normalize each tomogram or run before patch extraction
- use patch sizes such as `96^3`, `(64, 128, 128)`, `(128, 256, 256)`, or
  task-fit variants based on memory and object size
- oversample patches centered near positives
- train 3D U-Net/nnU-Net/MONAI models with BCE, Dice, CE, focal, or blob
  regression losses
- weight positive voxels heavily when targets are sparse
- use EMA, flips, axis swaps, mixup, and TTA if validated

Inference:

- sliding-window with overlap
- Gaussian weighting for overlap blending
- output full-volume probability maps
- decode to points with connected components or local maxima
- convert from `z,y,x` array order to `x,y,z` submission order

### 7.3 Connected components to centroids

`Implementation Pattern`:

```python
cc = cc3d.connected_components(mask == class_id)
stats = cc3d.statistics(cc)

centroids_zyx = stats["centroids"][1:] * voxel_spacing
keep = stats["voxel_counts"][1:] > min_voxels
points_xyz = np.ascontiguousarray(centroids_zyx[keep, ::-1])
```

Adaptation notes:

- `stats["centroids"][0]` is background
- use class-specific `min_voxels` for multi-class particles
- preserve voxel spacing and coordinate units
- if output probability maps are available, threshold before components and
  optionally score components by mean or max probability

### 7.4 2D slices to 3D clustering

When compute forces a slice detector, aggregate slice detections into 3D points.

`Implementation Pattern`:

```python
coords = np.vstack((particle_xs, particle_ys, particle_zs)).T

tree = cKDTree(coords)
pairs = tree.query_pairs(r=max_distance, p=2)

uf = UnionFind(len(coords))
coords_xy = coords[:, :2]
coords_z = coords[:, 2]

for u, v in pairs:
    if abs(coords_z[u] - coords_z[v]) > z_distance:
        continue
    if np.linalg.norm(coords_xy[u] - coords_xy[v]) > xy_distance:
        continue
    uf.union(u, v)

roots = np.array([uf.find(i) for i in range(len(coords))])
unique_roots, inverse_indices, counts = np.unique(
    roots, return_inverse=True, return_counts=True
)
conf_sums = np.bincount(inverse_indices, weights=particle_confidences)
cluster_scores = conf_sums / np.maximum(counts, 1)
```

Adaptation notes:

- tune `z_distance`, `xy_distance`, minimum cluster size, and score threshold
  per class
- sort by cluster score before applying max detections or sentinel decisions
- evaluate final 3D points, not slice-level boxes

### 7.5 Single-point presence thresholding

For zero-or-one target volumes:

- decode candidate peaks/components
- choose the top candidate by confidence
- calibrate a global, source-specific, or quantile threshold on OOF
- emit sentinel coordinates for no-object volumes below threshold
- audit false positives and false negatives separately

Useful threshold families:

- fixed probability threshold
- OOF quantile threshold based on expected positive prevalence
- threshold from precision/recall sweep under the official distance metric
- per-source thresholds only if source is available at test and validation
  supports it

### 7.6 3D NMS for slice detections

Slice detectors can emit several nearby points around the same object.

General pattern:

```python
final_detections = perform_3d_nms(all_detections, nms_threshold)
final_detections.sort(key=lambda x: x["confidence"], reverse=True)

if not final_detections:
    return {
        "tomo_id": tomo_id,
        "Motor axis 0": -1,
        "Motor axis 1": -1,
        "Motor axis 2": -1,
    }
```

Adaptation notes:

- implement 3D NMS in the official coordinate units
- never use 2D IoU alone for final tomogram points
- for single-object tasks, choose the best post-NMS point or sentinel

## 8. Video And Temporal Detection

### 8.1 Detector plus temporal classifier

Strong route:

1. Train a high-recall frame detector.
2. Generate OOF candidate boxes at low threshold.
3. Track or align candidates across nearby frames.
4. Crop a temporal stack around each candidate.
5. Train a classifier on event/no-event or IoU-bin labels.
6. Blend detector and classifier scores.
7. Apply temporal duplicate suppression and score official metric.

### 8.2 Temporal crop design

Crop design knobs:

- number of frames: 9, 16, or metric-window-sized stack
- crop size: 128x128 or larger if object context matters
- recenter using tracking velocity or optical flow
- scale crop so object median size is consistent
- include grayscale stacked channels or RGB frame stack
- sample positives around the official temporal tolerance
- add hard negatives from false positives of undertrained detectors

Useful model families:

- TSM on EfficientNet B0-B3 or ResNet18/34
- 3D CNN over frame stacks
- SlowFast-style ROI classifier
- EfficientNet B3/B5 on concatenated or stacked frames

### 8.3 Video NMS and top-k

For event metrics:

- sort predictions by confidence
- greedily keep one prediction per object/event window
- suppress nearby frames for the same tracked object
- use class/type-specific suppression if events can recur with different types
- tune per-video top-k and threshold on video-level OOF

Do not use frame-wise AP alone when the metric matches events over time.

### 8.4 Stateful test APIs

Some video competitions use an API that serves frames in order.

Implementation priorities:

- maintain lightweight state per video/sequence
- store recent detections for score propagation or suppression
- avoid expensive future-frame logic unless the API allows buffering
- validate by replaying frames in the same order as inference
- keep batch size and model count within API runtime limits

## 9. 3D Pose From Image

### 9.1 Center heatmap plus regression heads

For car pose and related tasks, treat each object as a heatmap center with
regression heads.

Typical heads:

- center heatmap
- local offset within heatmap cell
- depth or `z`
- translation `x,y,z`
- yaw/pitch/roll or sine/cosine variants
- width/height or size proxy
- confidence

Losses:

- focal or BCE-like loss for heatmap
- L1/SmoothL1 for regression heads
- angle-aware loss or sine/cosine representation for periodic angles
- metric-aligned validation after decoding pose strings

### 9.2 Pose target preprocessing

`Implementation Pattern`:

```python
IMG_WIDTH = 2048
IMG_HEIGHT = IMG_WIDTH // 4
MODEL_SCALE = 8

def _regr_preprocess(regr_dict):
    for name in ["x", "y", "z"]:
        regr_dict[name] = regr_dict[name] / 100
    regr_dict["roll"] = rotate(regr_dict["roll"], np.pi)
    regr_dict["pitch_sin"] = sin(regr_dict["pitch"])
    regr_dict["pitch_cos"] = cos(regr_dict["pitch"])
    regr_dict.pop("pitch")
    regr_dict.pop("id")
    return regr_dict

def preprocess_image(img):
    img = img[img.shape[0] // 2 :]
    bg = np.ones_like(img) * img.mean(1, keepdims=True).astype(img.dtype)
    bg = bg[:, : img.shape[1] // 4]
    img = np.concatenate([bg, img, bg], 1)
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    return (img / 255).astype("float32")

def get_mask_and_regr(img, labels):
    mask = np.zeros(
        [IMG_HEIGHT // MODEL_SCALE, IMG_WIDTH // MODEL_SCALE],
        dtype="float32",
    )
    regr = np.zeros(
        [IMG_HEIGHT // MODEL_SCALE, IMG_WIDTH // MODEL_SCALE, 7],
        dtype="float32",
    )
    coords = str2coords(labels)
    xs, ys = get_img_coords(labels)

    for x, y, regr_dict in zip(xs, ys, coords):
        x, y = y, x
        x = (x - img.shape[0] // 2) * IMG_HEIGHT / (img.shape[0] // 2) / MODEL_SCALE
        y = (y + img.shape[1] // 4) * IMG_WIDTH / (img.shape[1] * 1.5) / MODEL_SCALE
        x = np.round(x).astype("int")
        y = np.round(y).astype("int")

        if 0 <= x < IMG_HEIGHT // MODEL_SCALE and 0 <= y < IMG_WIDTH // MODEL_SCALE:
            mask[x, y] = 1
            regr_dict = _regr_preprocess(regr_dict)
            regr[x, y] = [regr_dict[n] for n in sorted(regr_dict)]
    return mask, regr
```

Adaptation notes:

- this is a pose-pattern snippet, not a universal image preprocessing recipe
- preserve exact angle order and units from the target competition
- handle roll/yaw/pitch wrap carefully before scoring
- decode and validate full pose strings, not only heatmap centers

### 9.3 Geometry and duplicate removal

High-value pose postprocess:

- perspective transform or input cropping that reduces near/far scale mismatch
- road/semantic masks to suppress impossible object centers
- duplicate removal in projected image space and 3D translation space
- global replacement or calibration for weak angle components only when OOF
  validates it
- confidence sorting under official pose mAP thresholds

Use pose-specific methods if the task is keypoint-only, human pose, or
metric-specific keypoint detection with no object box/centroid/6DoF structure.

## 10. Ensembling Strategy

### 10.1 First useful ensemble

The first ensemble should usually be:

- same detector across folds
- TTA inverted to original coordinates
- WBF or NMS over fold/TTA predictions
- OOF-tuned score threshold

This is often more useful than adding a weak second architecture.

### 10.2 Diverse detector ensemble

Add diversity when:

- single-family OOF has stable metric and submission code
- false negatives differ by model family
- boxes are close but not identical across models
- runtime allows multiple inference passes

Good diversity pairs:

- YOLO + EfficientDet
- YOLO + Faster/Cascade R-CNN
- EfficientDet + Faster R-CNN/FPN
- RetinaNet + classifier gate for medical negatives
- 3D U-Net + 2D slice detector for tomogram points

Combine with:

- WBF for 2D boxes
- score averaging or geometric mean for classifier gates
- connected-component union/local maxima for 3D maps
- temporal NMS for video

### 10.3 Late-round pseudo-label and retraining

Late-round sequence:

1. Freeze validation and metric code.
2. Train fold ensemble.
3. Generate pseudo labels with TTA/WBF.
4. Filter labels by confidence, class, size, and source.
5. Retrain with pseudo labels at lower weight or in later epochs.
6. Compare OOF and visual diagnostics.
7. Stop if gains are not robust across folds.

Do not use public leaderboard response as pseudo-label validation.

### 10.4 Calibration and blending

Calibration targets:

- detector confidence vs true positive probability
- image-level negative probability
- class-specific thresholds
- event confidence after temporal classifier
- 3D point confidence after component size/score

Useful blend forms:

- average probabilities across folds/models
- rank average when score scales differ and metric is rank-like
- multiply detector score by classifier score when gate false negatives are
  acceptable
- weighted average with OOF search when model count is small
- hard gate only if OOF shows minimal recall loss

## 11. Route-Specific Playbooks

### 11.1 Ordinary 2D boxes

Strong first implementation:

- parse boxes to canonical `xyxy`
- group/stratified folds by source/class/box count
- train YOLOv8/YOLO11 or YOLOv5 at task-fit resolution
- run OOF inference
- implement official metric and submission validator
- tune confidence, NMS IoU, max detections
- add WBF across folds/TTA

Try next:

- EfficientDet D5/D7 or Faster/Cascade R-CNN for diversity
- multi-scale TTA
- pseudo labels if allowed
- classifier gate if many false positives on empty images

Avoid:

- random folds when source exists
- unvisualized conversion
- downsampling small objects to invisibility

### 11.2 Dense small objects

Strong first implementation:

- preserve high resolution or train tiled detector
- stratify by source and bbox count
- use YOLO/EfficientDet with mosaic/mixup and scale augmentations
- tune max detections and low confidence threshold
- WBF full-image reconstructed boxes

Try next:

- patch model plus full-frame model ensemble
- boundary-box policy search
- crop re-scorer for false positives
- multi-scale inference

Avoid:

- validating tiles as if independent images
- visibility filters that delete tiny objects
- NMS IoU too low for crowded scenes

### 11.3 Medical detection

Strong first implementation:

- patient/study folds
- include negative images
- train detector plus image classifier gate
- score all-image mAP/F1 after gate and thresholds
- inspect false positives on negatives

Try next:

- RetinaNet vs Faster/Cascade R-CNN diversity
- Deformable/Relation-style detectors if local code exists
- multi-scale TTA
- OOF box-size calibration if annotation policy supports it
- classifier score multiplication rather than hard gate

Avoid:

- training only positives when negatives are scored
- tuning thresholds on public prevalence
- shrinking boxes without OOF evidence

### 11.4 Video detection

Strong first implementation:

- GroupKFold by video/game/sequence
- train frame detector
- low-threshold candidate generation
- video-level OOF metric with temporal tolerance
- temporal NMS or duplicate suppression

Try next:

- crop re-scorer with 9-16 frames
- optical-flow/IoU tracking
- score propagation to adjacent frames
- top-k per video if validated

Avoid:

- random frame folds
- scoring frame AP instead of event metric
- postprocess that suppresses valid repeated events

### 11.5 Volumetric point localization

Strong first implementation:

- split by tomogram/run/experiment
- train 3D U-Net/nnU-Net heatmap or blob model
- sliding-window inference with overlap
- decode connected components/local maxima
- tune threshold with official distance/F-beta metric

Try next:

- MONAI SegResNet/DynUNet diversity
- class-specific radii and thresholds
- 2.5D slice detector plus 3D clustering
- quantile thresholds for single-point presence
- TTA flips/axis swaps

Avoid:

- scoring voxel loss only
- outputting `z,y,x` when submission needs `x,y,z`
- patch-level random folds

### 11.6 3D pose from image

Strong first implementation:

- CenterNet-style center heatmap
- regression heads for pose/translation/size
- image preprocessing that preserves camera geometry
- decode pose strings and official metric
- duplicate suppression

Try next:

- perspective transform or masks
- sine/cosine angle heads
- larger input resolution
- ensemble heatmap models
- angle-specific calibration if OOF supports it

Avoid:

- treating pose as ordinary 2D bbox output
- ignoring angle periodicity
- optimizing center heatmap loss without pose metric validation

### 11.7 Large-scale multi-class detection

Strong first implementation:

- class-frequency audit
- rare-class-aware folds or sampling
- train robust detector with class-aware threshold diagnostics
- per-class AP report
- valid hierarchy expansion if supported

Try next:

- Cascade/Faster R-CNN + YOLO/RetinaNet diversity
- class-specific confidence thresholds
- pseudo labels for rare classes if allowed
- ensemble boxes by class

Avoid:

- optimizing only frequent-class loss
- invalid parent/child label expansion
- dropping rare labels during fold construction

### 11.8 Domain-specific 2D detection

Strong first implementation:

- identify source/site/camera/geography/scanner leakage before modeling
- preserve native aspect ratio and object scale
- train YOLOv8/YOLO11, EfficientDet, or Faster/Cascade R-CNN at a resolution
  that keeps the object visible
- use domain-valid augmentation only
- reconstruct full-image boxes if training on tiles
- tune threshold/NMS/WBF on source-grouped OOF predictions

Try next:

- metadata-aware folds and post-hoc diagnostics
- source-specific error reports
- patch model plus full-image model ensemble
- crop re-scorer for recurring background false positives
- pseudo labels only when target-domain unlabeled data is allowed

Avoid:

- random folds across sites or repeated capture units
- color/geometry augmentation that breaks domain physics
- source-specific thresholds unless the source is available at test and OOF
  supports them
