# Video Classification Category Playbook

This playbook is direct reading material for a skill-generating agent. It is not a
final task skill. A final task-specific skill should select only the relevant
route, metric, validation plan, model stack, and failure modes.

## 1. Task Routing

### 1.1 Video-level clip classification

Use this route when the submission has one row per video/clip and the target is
a class label or probability vector.

Typical signals:

- one file path per video
- metric is logloss, AUC, AP, F1, accuracy, or top-k accuracy
- output columns are video id plus class probabilities or labels
- no timestamped event rows, boxes, masks, or object identities

Strong first approach:

- grouped folds by video/source/subject/device
- fixed-count frame sampler or fixed-duration clip sampler
- `VideoMAE`/`VideoMAEv2`, `TimeSformer`, `Swin3D`, `SlowFast`, `X3D`, or
  `R(2+1)D` if raw video and compute allow
- 2D CNN plus temporal pooling if compute is constrained:
  `tf_efficientnet_b0/b3`, `efficientnetv2_s`, `convnext_tiny`,
  `resnext50_32x4d`, `seresnext50`, `xception`
- OOF prediction saved after video-level pooling
- probability/rank averaging across folds and windows

Avoid:

- random frame splits
- scoring per-frame accuracy only
- using the first frame as the whole video
- pushing every task into a huge video transformer before verifying the metric

### 1.2 Deepfake face-crop binary classification

Use this route when the task is manipulated media, fake/real videos, or face
forensics.

- detect faces with MTCNN, BlazeFace, dlib, or RetinaFace
- crop faces with context margin, often 30% to 50% beyond the face box
- train frame/crop classifiers on faces, not only full frames
- aggregate crop probabilities to one video-level fake probability
- optimize video-level logloss

Model priorities:

- EfficientNet B3/B5/B7, especially Noisy Student-style weights where
  available
- EfficientNetV2, ConvNeXt, SE-ResNeXt, Xception, ResNeXt as diversity
- 3D ResNet, I3D, MC3, R(2+1)D, or EfficientNet with late 3D convolutions for
  face tracks
- simple frame classifier first; temporal model later if OOF proves value

High-value details:

- group folds by original video/person/manipulation source
- track real/fake paired originals if metadata exposes `original`
- uniform frame sampling across the whole video
- compression/downscale/noise/blur/color augmentations
- aligned real/fake mixup when paired originals are known
- confidence-weighted video pooling or conservative mean pooling
- neutral fallback when decoding or face detection fails
- light probability clipping for logloss; label smoothing can reduce
  overconfident errors

Avoid:

- external face/person data without explicit permission and rights
- public-test manual labels or public-LB overfitting
- max-only pooling of the fakest frame unless OOF shows it is robust
- overly strong spatial augmentations that destroy face identity or temporal
  coherence

### 1.3 Temporal event localization

Use this route when the output is event timestamp rows such as:

- `video_id,time,event,score`
- `game,frame,event,confidence`
- temporal AP/mAP over tolerance windows

- train dense per-frame or per-stride event scores
- hold out whole games/videos
- optimize event AP after smoothing and peak selection
- suppress duplicate peaks near the same event
- submit ranked timestamped detections

This is a route inside the video playbook, but it is not ordinary clip
classification. Combine temporal localization, action localization, or video-detection methods as required.

Model priorities:

- 2.5D EfficientNetV2 B0/B1, ConvNeXt, EfficientNet, ResNet, RegNet
- neighboring grayscale/RGB frames as channels
- late 3D convolution or TSM
- 1D CNN/U-Net/Transformer over per-frame features
- CSN, SlowFast, X3D, R(2+1)D, VideoMAE for ROI/action clips
- object/ROI detector only if full-frame event model misses the event source

Labeling:

- use hard labels in a small window around event for BCE
- or Gaussian/soft labels centered on event time
- include background frames, but do not let easy background dominate
- class-specific windows can be useful when tolerances differ

Postprocess:

- rolling mean or Gaussian smoothing
- local peak detection or max-pooling peak extraction
- class-specific suppression width
- convert frame id to seconds with the correct FPS
- use score values that preserve AP ranking

Avoid:

- selecting models by frame accuracy instead of event AP
- random frame split from the same game
- duplicate detections around one ground truth
- public-test game overlap exploitation

### 1.4 Feature-based multi-label ranking and localization

Use this route for large-scale video-feature ranking datasets:

- precomputed video-level features such as mean RGB/audio embeddings
- precomputed frame-level embeddings
- many sparse labels
- GAP@k, MAP@k, or class-wise top segment lists

- read TFRecord/video feature shards
- hold out file shards or videos
- train temporal aggregators over frame/audio embeddings
- calibrate/rank labels across classes
- emit top-k `label score` pairs per video or top segments per class

Model priorities:

- NetVLAD, NextVLAD, NetFV/Fisher Vector, DBoF, GatedDBoF
- GRU/LSTM, fast-forward LSTM, BiGRU
- Transformer or attention/convolution temporal encoders
- Mixture-of-Experts output head
- class-aware rerankers or class-as-input models
- teacher-student distillation under model-size limits

High-value details:

- audit rare labels and label co-occurrence
- tune cross-class calibration because GAP sorts predictions globally
- for segment MAP, maintain per-class candidate heaps
- use video-level model as high-recall candidate generator for segment models
- temporal filtering can improve segment consistency
- distill ensembles into compact students when upload/runtime is constrained

Avoid:

- treating BCE validation loss as the final metric
- outputting uncalibrated logits for GAP/MAP
- keeping every segment score in memory when a heap is enough
- assuming all unrated segments are negatives

### 1.5 Event anticipation and crash prediction

Use this route when the task asks whether an event will happen soon in a video,
often with event time or alert time in training.

Signals:

- dashcam/vehicle/collision/near miss
- one risk score per video
- metric is AP/AUC or AP over time-to-event thresholds
- positive videos have `event_time` or `alert_time`

Strong first route:

- event-relative window labeling: only frames before event are allowed
- sample positive clips ending near official time-to-event thresholds
- sample negatives from normal videos and hard-negative near-miss windows
- train `VideoMAE`/`VideoMAEv2`, `TimeSformer`, `Swin3D`, or 2D CNN+GRU
- infer over the final clip if test videos are trimmed near events; otherwise
  infer sliding windows and aggregate by max/mean/rank
- validate official AP/time-to-event metric, not accuracy

Useful features:

- motion: optical flow, frame difference, camera motion compensation
- vehicles: YOLO/segmentation car crops, object counts, near-car boxes
- depth/proximity: depth-estimation maps or relative scale cues
- road context and ego-motion if available

Avoid:

- using frames after event time for positive labels
- selecting by validation accuracy when official score is AP over TTA
- naive undersampling that removes all hard negatives
- threshold-only thinking; AP is ranking-first

### 1.6 Detection, tracking, and constrained assignment

Use this route only when the task has a real video-temporal subproblem but the
output is boxes, tracks, or per-object identities. It is not ordinary clip
classification.

Signals:

- per-frame bounding boxes
- player/object labels per detection
- external tracking coordinates
- one-to-one assignment constraints
- duplicate label/box restrictions
- score combines IoU and identity correctness

Strong first route:

- prioritize image detection and tracking methods
- train or use a strong detector first: YOLOv5/v7/v8, RT-DETR, EfficientDet
- validate box quality separately from identity quality
- synchronize external tracks to video frames
- project image detections to field/map/track coordinates
- use Hungarian assignment, ICP/similarity transforms, or camera registration
- use DeepSORT/ByteTrack/IoU tracking to stabilize identities over time
- enforce submission constraints before saving

Useful auxiliary signals:

- team/color classifier
- jersey/number classifier with high-precision overrides
- ReID embeddings
- camera/view classifier
- field line or homography features
- assignment matrix ensembling

Avoid:

- route as video classification if the score is per-object identity
- box-only WBF without checking identity assignment
- assuming fixed camera orientation
- violating max boxes per frame or duplicate label constraints

### 1.7 Video plus audio or metadata

Use this route when the task has raw video and meaningful audio or metadata.

Guidance:

- primary route follows the scored output unit
- use audio methods for raw waveform/mel/SED/audio embeddings
- use tabular methods for source/camera/device/game metadata
- OOF fusion must be at the official video/segment/event unit
- avoid metadata that leaks split identity unless group validation proves value

### 1.8 Runtime-constrained video

Video tasks are often won by throughput decisions.

Prioritize:

- frame count and clip length before backbone size
- cached frames, crops, or embeddings
- batched decoding/detection
- mixed precision
- lower resolution with robust augmentations
- infer every N frames and interpolate only when metric allows it
- small backbones with diverse folds before one huge model
- per-class heaps for top-k segment output

Always compute a per-video or per-segment runtime budget from the official
kernel time and test size.

### 1.9 Route out

Route out when the main output is:

- generated text: ASR, OCR, captioning, subtitle restoration, WER/CER
- masks only: semantic/instance segmentation
- boxes/points/poses only: image detection/keypoint/pose
- object identities only with detections already provided: tracking playbook
- recommender/retrieval candidate generation
- game policy, reinforcement learning, or simulation
- judged EDA with no predictive target

Use this playbook only for a real scored video-classifier/event-reranker substage.

Examples of guardrailed in-category routes:

- deepfake classification-like sports event rows are temporal localization, not clip labels.
- sports tracking-like helmet assignment is detection/tracking/geometry first.
- large-scale video-feature ranking-like segment lists are ranking/localization first.
- Nexar-like crash prediction is event anticipation; exact test trimming and
  time-to-event assumptions must be revalidated for each task.

## 2. Validation And Leakage

### 2.1 Define the scored unit

Before building folds, write down:

- what each submission row represents
- which train examples can share visual content
- whether the model sees frames before or after an event
- whether a raw video produces many crops/windows
- whether OOF predictions must be pooled before scoring

Common scored units:

- video probability
- clip/window probability
- timestamped event detection
- class-wise ranked segment
- top-k label string
- box plus object identity

### 2.2 Video/source group validation

Default for raw videos:

- one group per original video/source clip
- group all derived crops/windows/frames with the source video
- stratify at video level, not frame level
- save fold assignments before preprocessing creates many rows

Use group split when:

- multiple crops per video
- multiple segments from one video
- the same subject/person/team/game appears repeatedly
- metadata reveals `original`, `source`, `game`, `play`, `video_id`, `camera`,
  `device`, `track`, `session`, or `subject`

### 2.3 Deepfake validation

Deepfake data leaks through:

- real original and generated fakes
- same person across multiple manipulated clips
- train/test generation method differences
- face crop duplicates
- public-test artifacts

Use:

- grouped folds by original/person when known
- chunk/source holdout when official train is chunked
- video-level logloss after crop aggregation
- separate diagnostics for real and fake videos
- calibration audit on video probabilities

Do not choose a validation split only because it tracks public LB. Private data
can contain unseen manipulation methods.

### 2.4 Event detection validation

For event localization:

- hold out full videos/games/matches
- run the complete inference postprocess on validation videos
- compute event AP over official tolerances
- tune smoothing, thresholds, and suppression on OOF only
- audit AP per event class

Random frame validation is misleading because:

- adjacent frames are nearly duplicates
- the metric matches sparse timestamp peaks
- duplicate nearby detections are false positives
- public test may share game context

### 2.5 Feature-based ranking validation

For large-scale video-feature ranking data:

- split by TFRecord/file shard or video id
- track label prevalence by fold
- evaluate GAP/MAP/top-k exactly
- check cross-label calibration
- save per-video or per-class ranked OOF outputs

If segment labels are sparse or unrated segments are ignored by the metric, do
not mark every unrated segment as a negative without a route-specific reason.

### 2.6 Event anticipation validation

For event anticipation:

- positive training windows must end before event time or alert time
- sample several time-to-event offsets if the metric evaluates several offsets
- validation should compute AP over the same offsets
- negative windows should include hard near-event-like negatives
- if test videos are trimmed near the event, validate an end-of-video inference
  policy; otherwise validate sliding-window inference

### 2.7 Tracking and assignment validation

For detection-tracking-assignment:

- validate detector boxes by IoU/recall/precision
- validate identity assignment with ground-truth boxes if available
- validate full submission score after detector, tracker, and assignment
- debug per camera/view/game
- enforce max boxes, duplicate labels, and duplicate boxes in validation output

### 2.8 OOF artifacts

Save OOF at the official unit:

- video: one row per video and class/probability
- deepfake: per-frame/crop predictions plus video-pooled OOF
- event: dense frame scores plus postprocessed event rows
- feature ranking: video logits plus top-k labels; segment/class heaps if used
- tracking: detector predictions, track ids, assignment matrix, final rows

OOF artifacts are required for safe ensembling. Do not train a meta-model on
in-sample frame/crop predictions.

## 3. Metrics And Target Shapes

### 3.1 Logloss

Used by deepfake and many binary video classifiers.

Implications:

- output calibrated probability, not just rank
- clip only lightly, e.g. `[1e-6, 1 - 1e-6]`, unless validation supports
  stronger clipping
- no-face/decode failure should return a neutral probability, often `0.5`
- label smoothing can reduce overconfident mistakes
- average at video level before scoring

### 3.2 AUC/AP/F1

For video-level binary or multi-label tasks:

- AUC/AP reward ranking; raw calibration matters less
- F1 needs threshold tuning on OOF
- AP often benefits from hard-negative mining
- rank averaging can be competitive when models differ in calibration

### 3.3 Accuracy/top-k

Use CE/softmax for true single-label tasks. For top-k metrics:

- optimize probability ranks
- preserve all class probabilities until top-k construction
- do not tune only top-1 accuracy if MAP@k/top-k is scored

### 3.4 Event AP over timestamp tolerances

Event AP scores ranked timestamp detections.

Key properties:

- matching is per event class, tolerance, and video
- detections are sorted by score
- at most one detection can match one ground truth
- duplicate nearby peaks become false positives
- predictions outside scoring intervals may be dropped
- AP is averaged across tolerances and event classes

`Implementation Pattern`:

```python
tolerances = {
    "challenge": [0.30, 0.40, 0.50, 0.60, 0.70],
    "play": [0.15, 0.20, 0.25, 0.30, 0.35],
    "throwin": [0.15, 0.20, 0.25, 0.30, 0.35],
}

detections = detections.sort_values("score", ascending=False).dropna()
```

### 3.5 GAP@k and MAP@k

For GAP@k:

- keep top-k label-score pairs per video
- metric globally sorts all pairs
- cross-class calibration matters

For MAP@k segment localization:

- predictions may be class-wise ranked lists of segments
- memory is dominated by class x segment scores
- heaps and pruning are practical
- unrated segments may be ignored by scoring

### 3.6 Time-to-event AP

For event anticipation:

- metric may evaluate several time-to-accident thresholds
- train windows should align to those offsets
- validation accuracy is insufficient
- output should preserve ranking, not only thresholded labels

### 3.7 IoU plus identity assignment

For tracking/assignment:

- detector box quality affects identity score
- each ground-truth object may match the submitted box with highest IoU
- label equality and impact weights can dominate the final score
- duplicate labels per frame are invalid in some tasks
- constraints can matter as much as model output

## 4. Video Loading, Sampling, And Windows

### 4.1 Fixed-count frame sampler

Use for video-level inference when every video needs the same number of frames.

`Implementation Pattern`:

```python
def sample_frame_indices(num_frames, n_samples):
    if num_frames <= 0:
        return np.array([], dtype=np.int64)
    return np.linspace(0, num_frames - 1, n_samples).astype(np.int64)

def read_sampled_frames(path, n_samples):
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = set(sample_frame_indices(total, n_samples))
    frames = []

    frame_id = 0
    ok = True
    while ok:
        ok = cap.grab()
        if not ok:
            break
        if frame_id in indices:
            ok, frame = cap.retrieve()
            if ok:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        frame_id += 1

    cap.release()
    return frames
```

Use `grab/retrieve` to skip frames without decoding every image. For training,
also consider random temporal offsets so the same video contributes different
views across epochs.

### 4.2 FPS-aware event sampling

For timestamp labels, convert seconds to frame indices with the actual FPS.

`Implementation Pattern`:

```python
def seconds_to_frame(time_s, fps):
    return int(round(time_s * fps))

def make_event_labels(num_frames, events, fps, radius_frames):
    y = np.zeros((num_frames, len(event_names)), dtype=np.float32)
    for event_time, event_name in events:
        center = seconds_to_frame(event_time, fps)
        lo = max(0, center - radius_frames)
        hi = min(num_frames, center + radius_frames + 1)
        y[lo:hi, event_to_idx[event_name]] = 1.0
    return y
```

For deepfake classification-style 25 fps data, timestamp conversion can be as simple as
`time = frame_id / 25`, but do not hard-code FPS unless the competition
defines it.

### 4.3 2.5D frame stacks

A low-cost temporal representation is to stack adjacent frames as channels.

Use cases:

- event detection on small datasets
- grayscale sports videos
- frame-difference or optical-flow-like cues
- cheap alternatives to full 3D backbones

Pattern:

```python
def make_stack(frames, center, offsets=(-1, 0, 1), grayscale=True):
    xs = []
    for off in offsets:
        idx = min(max(center + off, 0), len(frames) - 1)
        img = frames[idx]
        if grayscale:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img = img[..., None]
        xs.append(img)
    return np.concatenate(xs, axis=-1)
```

Try time strides `1`, `2`, and `3` for ensemble diversity in event detection.

### 4.4 Event-relative windows

For anticipation tasks, labels depend on the last timestamp inside a clip.

`Implementation Pattern`:

```python
def label_window(last_time, is_positive_video, event_time, horizon=1.5):
    if not is_positive_video:
        return 0
    if last_time >= event_time:
        return None  # do not train on frames after the event
    return int(event_time - horizon <= last_time < event_time)
```

For metrics with several time-to-event thresholds, sample multiple horizons and
score each horizon separately.

### 4.5 Precomputed feature records

large-scale video-feature ranking tasks provide embeddings instead of raw frames.

`Implementation Pattern`:

```python
for raw in tf.python_io.tf_record_iterator(video_tfrecord):
    ex = tf.train.Example.FromString(raw)
    video_id = ex.features.feature["id"].bytes_list.value[0].decode("utf-8")
    labels = list(ex.features.feature["labels"].int64_list.value)
    rgb = np.asarray(ex.features.feature["mean_rgb"].float_list.value)
    audio = np.asarray(ex.features.feature["mean_audio"].float_list.value)
    x_video = np.concatenate([rgb, audio])
```

Frame-level records:

```python
seq = tf.train.SequenceExample.FromString(raw)
n = len(seq.feature_lists.feature_list["rgb"].feature)
rgb_seq = [
    tf.decode_raw(
        seq.feature_lists.feature_list["rgb"].feature[i].bytes_list.value[0],
        tf.uint8,
    )
    for i in range(n)
]
audio_seq = [
    tf.decode_raw(
        seq.feature_lists.feature_list["audio"].feature[i].bytes_list.value[0],
        tf.uint8,
    )
    for i in range(n)
]
```

Modern code should adapt this to the installed TensorFlow version, but the
contract remains: video id, sparse labels, frame RGB embeddings, and frame audio
embeddings.

### 4.6 Runtime budget

Compute budget early:

```python
seconds_per_video = total_kernel_seconds / max(1, num_test_videos)
```

Then decide:

- frames per video
- crops per frame
- detector batch size
- clip length and stride
- number of folds/models
- whether to cache face/object crops
- whether to use a small backbone or distilled model

## 5. Face, Object, And ROI Crops

### 5.1 Face detection route

Useful face detectors include MTCNN, BlazeFace, or dlib. The route:

1. sample frames
2. detect faces in batches
3. optionally keep all faces, not only the largest
4. expand boxes by a margin
5. crop, resize, normalize
6. predict per crop
7. aggregate to video probability

Avoid Haar cascades except for EDA or very weak baselines.

### 5.2 Aspect-preserving crop resize

`Implementation Pattern`:

```python
def isotropic_resize(img, size, interpolation=cv2.INTER_AREA):
    h, w = img.shape[:2]
    if w >= h:
        new_w = size
        new_h = max(1, h * size // w)
    else:
        new_h = size
        new_w = max(1, w * size // h)
    return cv2.resize(img, (new_w, new_h), interpolation=interpolation)

def pad_to_square(img, value=0):
    h, w = img.shape[:2]
    side = max(h, w)
    bottom = side - h
    right = side - w
    return cv2.copyMakeBorder(
        img, 0, bottom, 0, right, cv2.BORDER_CONSTANT, value=value
    )
```

### 5.3 Margin crop

Use margin to keep blending, boundary, body, or object-context artifacts.

```python
def expand_box(x1, y1, x2, y2, image_w, image_h, margin=0.30):
    w = x2 - x1
    h = y2 - y1
    dx = margin * w
    dy = margin * h
    return (
        int(max(0, x1 - dx)),
        int(max(0, y1 - dy)),
        int(min(image_w, x2 + dx)),
        int(min(image_h, y2 + dy)),
    )
```

### 5.4 Detection and tracking crops

For object-centric tasks:

- crop around detections for secondary classifiers
- keep frame id and video id with every crop
- maintain detector confidence and box coordinates for assignment
- use class/person/team constraints to reject impossible labels
- preserve original image coordinates for final submission

For sports tracking-like tasks, detector output is not enough; it must feed tracking,
camera mapping, and constrained assignment.

## 6. Model Families

### 6.1 2D CNN plus temporal pooling

Use when raw video transformers are too expensive or object crops are central.

Backbones:

- `tf_efficientnet_b0_ns`, `tf_efficientnet_b3_ns`, `tf_efficientnet_b5_ns`,
  `tf_efficientnet_b7_ns`
- `efficientnetv2_s`, `efficientnetv2_b0`, `efficientnetv2_b1`
- `convnext_tiny`, `convnext_small`
- `resnext50_32x4d`, `seresnext50`, `xception`, `resnet18/34/50`

Temporal heads:

- mean/max/geometric pooling
- attention pooling
- 1D CNN over frame embeddings
- GRU/LSTM/BiGRU
- Transformer encoder

`Implementation Pattern`:

```python
class CNNGRU(nn.Module):
    def __init__(self, cnn, feature_dim=512, hidden=128):
        super().__init__()
        self.cnn = cnn
        self.gru = nn.GRU(feature_dim, hidden, batch_first=True, bidirectional=True)
        self.head = nn.Linear(hidden * 2, 1)

    def forward(self, x):
        # x: [batch, time, channels, height, width]
        b, t, c, h, w = x.shape
        feats = self.cnn(x.reshape(b * t, c, h, w)).reshape(b, t, -1)
        seq, _ = self.gru(feats)
        return self.head(seq[:, -1])
```

### 6.2 Deepfake backbones

Strong choices:

- EfficientNet B5/B7
- EfficientNet B7 Noisy Student
- SE-ResNeXt, ResNeXt50, Xception
- I3D, 3D ResNet34, MC3, R(2+1)D
- EfficientNet with 3D convolutions added to later blocks

Start with one or two strong 2D crop classifiers. Add 3D/sequence diversity
only after crop extraction and video-level logloss are stable.

### 6.3 Temporal event models

Strong routes:

- 2.5D EfficientNetV2 B0/B1 over grayscale frame stacks
- EfficientNet + TSM + 1D U-Net
- 11-channel frame difference stacks
- CSN/SlowFast/X3D/VideoMAE over ROI clips
- ball/ROI detector followed by action classifier

Use small backbones when data is small; larger backbones can overfit quickly.

### 6.4 Feature-video aggregators

For precomputed frame/audio features:

- DBoF / GatedDBoF
- NetVLAD / NextVLAD
- NetFV / Fisher Vector
- GRU / LSTM / fast-forward LSTM
- Transformer / attention
- MoE heads
- distillation students

Choose several diverse aggregation families rather than many copies of the same
pooling operator.

### 6.5 Event anticipation models

Priorities:

- `VideoMAE`/`VideoMAEv2` for pretrained spatiotemporal understanding
- `TimeSformer`, `Swin3D`, `SlowFast`, `X3D`
- `ResNet18/34/50` plus GRU as a lighter baseline
- depth/flow/car-crop branches if raw frames support them

Use smaller variants when GPU memory is limited, but preserve event-relative
sampling and official AP validation.

### 6.6 Audio-video fusion

Use only when audio is a real input and likely predictive.

Patterns:

- train audio model separately using audio playbook
- train video model separately
- fuse OOF probabilities/logits at official unit
- include metadata only with group-safe validation

Do not add audio playbook simply because a video file has an audio track; inspect
whether the task and metric make audio useful.

### 6.7 Detector and tracker models

Detection:

- YOLOv5/YOLOv7/YOLOv8
- RT-DETR
- EfficientDet
- Faster/Mask R-CNN if local package support exists

Tracking/assignment:

- DeepSORT, ByteTrack, OC-SORT
- simple IoU tracker for speed
- ReID/ArcFace embeddings for same-object links
- Hungarian assignment for one-to-one matching
- ICP/similarity transform for camera-to-map alignment

## 7. Training, Augmentation, And Targets

### 7.1 Generic video augmentations

Use:

- random resized crop
- horizontal flip when labels are symmetric
- color jitter
- blur/noise
- CutMix/MixUp when label semantics allow
- temporal jitter
- frame dropping or FPS jitter
- compression/downscale for web/video tasks

Be careful with:

- vertical flip for real-world video
- 90-degree rotations for orientation-sensitive tasks
- aggressive spatial cutout on face/identity tasks
- augmenting frames in a clip inconsistently when temporal coherence matters

### 7.2 Deepfake augmentations

High-value:

- JPEG/video compression
- random downscale/upscale
- blur and Gaussian noise
- brightness/contrast/hue
- horizontal flip
- random crop/erase, mild scale/rotate
- face landmark or half-face masking only if OOF supports it

`Implementation Pattern`:

```python
def aligned_real_fake_mixup(real_x, fake_x, alpha=0.5):
    lam = np.random.beta(alpha, alpha)
    x = (1.0 - lam) * real_x + lam * fake_x
    y = np.asarray(lam, dtype=np.float32)
    return x, y
```

This is useful only when real/fake frames are aligned by original video, frame
number, and box coordinates.

### 7.3 Event detection targets

Options:

- hard labels inside a small event window
- Gaussian labels centered on event timestamp
- class-specific windows matching metric tolerances
- background sampling to control imbalance
- focal loss when background dominates

Validate the label shape with event AP, not only BCE.

### 7.4 Multi-label feature targets

Use:

- BCE or sampled BCE for sparse labels
- MoE/gated output heads
- label-frequency weighting where OOF supports it
- weak negatives from video-level labels for segment models
- distillation soft targets from ensembles

Avoid assuming all missing labels are true negatives unless the task contract
says so.

### 7.5 Event anticipation targets

Use:

- positive clips ending before event time
- windows at official time-to-event offsets
- random negative windows
- hard-negative mining after first model
- focal loss or class-balanced sampling if negative windows dominate

Do not include frames at or after the collision/event if the prediction is meant
to be early.

## 8. Inference And Postprocessing

### 8.1 Video probability pooling

For deepfake and video-level classification:

```python
def confidence_weighted_mean(probs, eps=1e-6):
    probs = np.asarray(probs, dtype=np.float64)
    if len(probs) == 0:
        return 0.5
    weights = np.abs(probs - 0.5) + eps
    return float(np.sum(probs * weights) / np.sum(weights))
```

Use plain mean as the baseline. Confidence weighting can help when bad frames
produce near-0.5 outputs, but it can amplify overconfidence, so validate it.

### 8.2 Temporal smoothing and peak extraction

`Implementation Pattern`:

```python
def make_event_submission(scores, fps, window_size=10, ignore_width=10):
    rows = []
    event_names = ["challenge", "throwin", "play"]

    for video_id, gdf in scores.groupby("video_id"):
        for event in event_names:
            prob = (
                gdf[event]
                .rolling(window=window_size, center=True)
                .mean()
                .fillna(-100)
                .to_numpy()
            )
            order = np.argsort(-prob)
            used = np.zeros(len(prob), dtype=bool)
            rank = 0

            for idx in order:
                if used[idx] or prob[idx] <= 0:
                    continue
                rows.append(
                    {
                        "video_id": video_id,
                        "time": float(gdf["frame_id"].iloc[idx] / fps),
                        "event": event,
                        "score": float(1.0 / (rank + 1)),
                    }
                )
                lo = max(0, idx - ignore_width)
                hi = min(len(prob), idx + ignore_width + 1)
                used[lo:hi] = True
                rank += 1

    return pd.DataFrame(rows, columns=["video_id", "time", "event", "score"])
```

Replace inverse-rank score with smoothed probability when validation AP
improves. Tune `window_size` and `ignore_width` per class.

### 8.3 Horizontal flip TTA

`Implementation Pattern`:

```python
with torch.no_grad():
    logits = model(x)
    logits_flip = model(torch.flip(x, dims=[-1]))
    logits = 0.5 * (logits + logits_flip)
```

Use only when horizontal flip preserves label semantics. For driving direction,
field side, or text/OCR tasks, validate carefully.

### 8.4 Top-k video and segment ranking

GAP@k video row:

```python
def format_gap_row(video_id, scores, k=20):
    scores = np.asarray(scores)
    idx = np.argpartition(scores, -k)[-k:]
    idx = idx[np.argsort(scores[idx])[::-1]]
    labels = " ".join(f"{i} {scores[i]:.6g}" for i in idx)
    return {"VideoId": video_id, "LabelConfidencePairs": labels}
```

Class-wise MAP heap:

```python
def update_class_heaps(heaps, segment_id, scores, max_items=100_000):
    for cls, score in enumerate(scores):
        heapq.heappush(heaps[cls], (float(score), segment_id))
        if len(heaps[cls]) > max_items:
            heapq.heappop(heaps[cls])
```

Use heaps when class x segment scores are too large to materialize.

### 8.5 Tracking and assignment postprocess

Frame synchronization:

```python
tracks["est_frame"] = (
    (tracks["snap_offset"] * video_fps) + frame_offset
).round().astype("int")
```

Cluster majority relabel:

```python
def add_track_majority_label(dets, track_col="track_id", label_col="label"):
    label_map = (
        dets.groupby(track_col)[label_col]
        .agg(lambda s: s.value_counts().idxmax())
        .to_dict()
    )
    dets[f"{label_col}_track"] = dets[track_col].map(label_map)
    return dets
```

Hungarian assignment:

```python
def assign_frame(pred_xy, track_xy, team_penalty=None, track_bonus=None):
    cost = pairwise_distances(pred_xy, track_xy)
    if team_penalty is not None:
        cost = cost + team_penalty
    if track_bonus is not None:
        cost = cost - track_bonus
    det_idx, track_idx = linear_sum_assignment(cost)
    return det_idx, track_idx
```

Submission guards:

```python
assert pred.groupby("video_frame").size().le(22).all()
assert not pred[["video_frame", "label"]].duplicated().any()
assert not pred[["video_frame", "left", "width", "top", "height"]].duplicated().any()
```

### 8.6 Multimodal fusion

Fuse at the official unit:

```python
stack = np.column_stack([
    video_oof,
    audio_oof,
    metadata_oof,
])
meta = LogisticRegression(max_iter=1000)
meta.fit(stack[train_idx], y[train_idx])
```

Use only OOF base predictions. For logloss, meta-model calibration can help; for
ranking metrics, constrained weighted averages or rank averages may be safer.

### 8.7 Failure fallbacks

Use fallback by metric:

- logloss: neutral probability such as `0.5`
- AP/ranking: low confidence but valid row, or skip if schema allows
- event detection: no rows for a video can be valid but may hurt recall
- assignment: fallback nearest mapping or prior labels after constraints

Never let a decode failure crash the whole inference kernel.

## 9. Ensembling And Calibration

### 9.1 OOF-first ensemble

Save:

- fold id
- model name
- seed
- video/window/frame id
- raw logits
- pooled official prediction
- postprocessed rows if event/assignment

Then blend using OOF metric:

- simple mean
- weighted mean
- rank average
- positive linear/Ridge blend
- per-class/per-event weights
- assignment-matrix average for identity tasks

### 9.2 Calibration

For logloss:

- label smoothing during training
- temperature scaling on OOF logits
- light clipping
- mean pooling vs confidence pooling audit

For AP/MAP/GAP:

- monotonic transformations preserve within-class ranks but not cross-class
  calibration
- rank averaging can help when model scales differ
- per-class thresholds are for F1/constraints, not AP itself

### 9.3 Distillation and compression

Use distillation when:

- competition limits model upload size
- inference runtime blocks ensembles
- large-scale video-feature ranking feature models produce large ensembles
- teacher OOF predictions are stable

Distill with a mixture of hard labels and teacher soft labels. Preserve the
official metric in validation, not just distillation loss.

### 9.4 Pseudo labels

Use only if rules allow and validation is trustworthy.

Better candidates:

- unlabeled target-domain videos for clip classification
- high-confidence event frames after full video holdout
- video-level weak labels for segment models

Risky candidates:

- public-test labels inferred from LB
- pseudo labels that include future event frames
- pseudo labels on data already used to tune postprocess without OOF discipline

## 10. Route-Specific Playbooks

### 10.1 Ordinary video-level classification

Strong first implementation:

- grouped stratified folds by video/source/subject
- fixed frame count or fixed clip length
- VideoMAE/TimeSformer/Swin3D if feasible; otherwise 2D CNN+GRU/mean pooling
- exact metric and video-level OOF
- multi-window inference and probability/rank averaging

Try next:

- frame-rate jitter and temporal TTA
- crop/ROI branch if objects drive label
- audio branch if task includes predictive sound
- calibration and weighted blends

Avoid:

- random frame split
- first-frame baseline as final route
- overlarge backbone before runtime audit

### 10.2 Deepfake face-crop binary

Strong first implementation:

- group folds by original/person/video chunk
- MTCNN/BlazeFace/RetinaFace face detection
- margin face crops, 224 to 380 input size
- EfficientNet B3/B5/B7 or ConvNeXt crop classifier
- compression/downscale/noise/blur/color augmentations
- video-level mean pooling, neutral no-face fallback
- logloss with light clipping or label smoothing

Try next:

- all-face crops instead of largest face
- aligned real/fake mixup
- 3D ResNet/I3D/R(2+1)D over face tracks
- confidence-weighted pooling
- model diversity by crop margin, frame count, backbone, seed

Avoid:

- external face data with unclear rights
- max-only fakest-frame pooling without OOF proof
- public-LB tuned heuristics as default

### 10.3 Sports temporal event detection

Strong first implementation:

- hold out full videos/games
- label event windows around timestamps
- 2.5D EfficientNetV2/ConvNeXt over adjacent grayscale/RGB frames
- infer dense frame scores or every N frames plus interpolation
- rolling smoothing, peak selection, local suppression
- submit `video_id,time,event,score`
- optimize official event AP

Try next:

- 1D U-Net/TSM over frame features
- VideoMAE/SlowFast/CSN ROI clips
- ball/player/event-area detector branch
- class-specific smoothing/NMS
- horizontal flip TTA if label symmetric
- ensemble time strides and resolutions

Avoid:

- frame accuracy as model selector
- duplicate peaks near same event
- test game overlap tricks

### 10.4 Large-Scale Video-Feature Ranking

Strong first implementation:

- parse video/frame/audio embeddings
- shard/video-level validation
- DBoF/NetVLAD/NextVLAD/GRU baseline with MoE head
- exact GAP/MAP scorer
- top-k label row or class-wise heap submission

Try next:

- model family ensemble: NetVLAD, NetFV, DBoF, GRU, Transformer
- distillation into compact student
- class-wise score calibration
- video-level candidate generation plus segment reranking
- temporal smoothing over neighboring segments

Avoid:

- raw BCE loss as final model selector
- uncalibrated top-k labels
- assuming unrated segments are negatives

### 10.5 Crash/event anticipation

Strong first implementation:

- group split by video/source
- sample positive windows ending before event time, near official TTA offsets
- sample negatives plus hard negatives
- VideoMAE/VideoMAEv2 or ResNet18/34+BiGRU
- final-window inference if test videos are trimmed near event
- AP/time-to-event validation

Try next:

- sliding-window inference with max/rank aggregation
- focal loss or smarter negative mining
- optical flow/frame difference branch
- depth estimation/proximity branch
- YOLO vehicle crop/segmentation branch
- ensemble clip lengths and frame rates

Avoid:

- frames at/after event as positive evidence
- validation accuracy as primary selection metric
- arbitrary final-2-second rule without validation

### 10.6 Detection-tracking-assignment

Strong first implementation:

- prioritize image detection when boxes are core
- train YOLO/RT-DETR detector, validate box IoU/recall
- synchronize external tracking to video frame ids
- map detections to track coordinates with simple geometry
- Hungarian assignment with distance cost
- track/cluster majority vote for temporal consistency
- enforce submission constraints

Try next:

- WBF and flip TTA for boxes
- DeepSORT/ByteTrack/ReID
- ICP/similarity transform with camera orientation sweep
- team/color/jersey auxiliary classifiers
- assignment-matrix ensembling

Avoid:

- treating it as clip classification
- ignoring camera/view orientation
- duplicate labels or too many boxes per frame
