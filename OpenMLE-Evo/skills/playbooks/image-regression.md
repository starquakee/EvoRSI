# Image Regression Category Playbook

This playbook is the full method library for Kaggle-style `image_regression`
tasks. It is intended as direct reading material for a later task-specific
skill generator: read the task description first, route to the right sections,
and extract a compact task-specific skill. Do not copy the whole playbook into a
final skill.

Do not use this playbook to justify leaderboard probing, public-test label
inference, hidden labels, manual test labeling, sample-submission label updates,
or disallowed external data.

## 1. What Counts As Image Regression

Use this playbook when the final target is numeric and the input is an image or an
image-like array. The numeric target may be:

- a scalar, such as age, distance, score, severity, biomass, count, or route
  length
- a multi-output vector, such as plant traits, biomass components, abundance
  vectors, spectra, or per-class counts
- an image-shaped continuous map, such as subsurface velocity
- a time-indexed signal extracted from a chart or ECG image
- a geometric object, such as a fundamental matrix, rotation matrix, translation
  vector, or camera center
- a mean plus uncertainty vector

Route out when the final score is on class probabilities, boxes, masks,
polygons, generated text, retrieval lists, or judged narrative methodology.
Use this playbook only for any numeric image-derived substage.

## 2. Safety Boundary

- public/private leaderboard probing
- public score inversion or target-root solving
- using hidden test labels, manual test labels, or sample submission targets
- fitting constants to public leaderboard changes
- using external pretrained weights, detectors, simulator code, geographic
  metadata, or public datasets when rules do not allow them
- treating visible test folders as representative when the challenge states
  hidden test data appears only during evaluation scoring
- train/test identity matching by filenames, row order, or site leakage

Transferable lesson: build validation and preprocessing that mimic hidden
deployment.

## 3. Output Contract First

Before modeling, write down:

- prediction unit: image, crop, left/right image pair, sequence, burst, slide
  spot, scene, image pair, planet, ECG record, waveform file, or `oid_ypos` row
- target shape: scalar, `K` targets, count vector, spectrum, sigma vector,
  signal series, map, fundamental matrix, `R/t`, or camera center
- metric: RMSE, MAE, R2, weighted R2, Spearman, Pearson, MCRMSE, GLL, SNR,
  mAA, count MAE/RMSE, or custom
- row schema: one row per image, one row per target, one row per lead/timestep,
  one row per y-position, pose strings, or sequence rows
- grouped leakage units: subject, slide, plot, state, date, species, camera,
  scene, dataset, star, planet, simulation family, record, or image source
- output constraints: nonnegative, integer-like, target identities, physical
  range, sigma positive, valid signal length, valid pose format

The model is downstream of the contract. A larger backbone does not rescue a
wrong split or a wrong submission shape.

## 4. Validation Geometry

### 4.1 Natural Groups

Use group validation when any repeated entity can leak:

- patient/study/scanner for medical image regression
- slide for spatial pathology
- plot/state/date/species for field or biomass data
- site/camera/location/sequence for camera-trap counts
- colony/scene/source for aerial counts
- scene/dataset for image matching and SfM
- image id/type/lead for ECG digitization
- star/planet for exoplanet spectra
- simulation family/file/event for waveform inversion
- geography/time for remote sensing regression

Random image folds are only a fallback when the task description and EDA show
no repeated entity, source, or generated family.

### 4.2 Stratify Numeric Targets

Regression folds should preserve target shape:

- target bins for scalar targets
- per-target quantile bins for multi-output regression
- zero/nonzero flags for sparse biomass/count components
- source or species plus target bins for biological tasks
- target magnitude, image size, or difficulty clusters for synthetic tasks

Use fold diagnostics:

- overall official metric
- per-target score
- per-source or per-group score
- worst fold and fold variance
- score before/after constraints and clipping

### 4.3 Implementation Pattern: Grouped Regression Split

The exact groups differ by task. The pattern below is appropriate when
image-level targets repeat by source/date/site and target distribution is
imbalanced. It is a generic adaptation of local biomass and plant-trait fold
ideas, including state/date grouping, species/target bins, visual clusters, and
rare zero-target stratification.

```python
from sklearn.model_selection import StratifiedGroupKFold
import pandas as pd

def make_regression_bins(df, target_cols, source_cols, n_bins=5):
    parts = []
    for c in target_cols:
        y = df[c].fillna(df[c].median())
        parts.append(pd.qcut(y.rank(method="first"), n_bins,
                             labels=False, duplicates="drop").astype(str))
        parts.append((df[c].fillna(0) <= 0).astype(int).astype(str))
    for c in source_cols:
        if c in df:
            parts.append(df[c].astype(str).fillna("NA"))
    return pd.Series(["|".join(x) for x in zip(*parts)], index=df.index)

df["strata"] = make_regression_bins(
    df,
    target_cols=["Dry_Green_g", "Dry_Clover_g", "Dry_Dead_g"],
    source_cols=["State", "Sampling_Date", "Species"],
)
splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
df["fold"] = -1
for fold, (_, valid_idx) in enumerate(splitter.split(df, df["strata"], df["group_id"])):
    df.loc[valid_idx, "fold"] = fold
```

## 5. Metrics And Target Handling

### 5.1 RMSE, MSE, MAE, And SmoothL1

Target transforms:

- standardize targets for optimization, but score in original units
- use `log1p` as model diversity when target is positive and skewed, not as a
  default
- keep raw-space models in the blend when the metric is raw RMSE/MAE/R2
- for count-like targets, compare raw, square-root, and log transforms with OOF

### 5.2 R2 And Weighted R2

For mean R2 over traits or weighted R2 over biomass components:

- compute the exact official metric locally
- track per-target R2 because one bad trait can dominate model selection
- use target standardization for training only if the metric callback inverts
  it correctly
- match loss weights to target weights when weights are explicit
- avoid optimizing a simple averaged MSE when targets have different score
  weights and variances

### 5.3 Spearman And Pearson

For rank/correlation metrics:

- preserve ordering across samples or spots
- consider rank-transforming targets for a model variant
- use MSE/MAE as a training proxy only after confirming OOF correlation
- blend models by correlation or rank stability, not raw error alone
- avoid public-LB overfitting when public/private distributions are weakly
  correlated

### 5.4 GLL And Uncertainty Columns

Use:

- validation residuals by wavelength/channel/planet
- ensemble variance
- GP/Bayesian posterior uncertainty
- spectrum smoothness or dynamics-based uncertainty
- sigma clipping and calibration on OOF

### 5.5 SNR And Signal Metrics

For ECG/chart digitization:

### 5.6 Pose Metrics

For image matching/SfM:

- pair match quality is only a proxy
- final metric may evaluate relative pose mAA or registered camera centers
- validate by scene, registered-image count, pose thresholds, and reconstruction
  quality
- handle unregistered images according to the submission specification

## 6. General Image-To-Scalar Or Image-To-Vector Regression

### 6.1 Strong First Implementation

For ordinary image regression:

1. Parse metadata and build the official submission schema.
2. Inspect target distributions, outliers, source groups, image sizes, and
   duplicate/near-duplicate images.
3. Implement exact metric.
4. Create grouped/stratified folds.
5. Train a pretrained backbone with a regression head.
6. Save OOF predictions and test predictions at the official row level.
7. Clip/transform predictions only according to OOF.
8. Add metadata or embedding stacks after the image-only baseline is measured.

Backbone priorities:

- DINOv2/DINOv3 ViT variants for small data and patch-token features
- ConvNeXt/ConvNeXtV2, EfficientNetV2, Swin, ViT, EVA/EVA02, SigLIP, MaxViT,
  CoaT, CaFormer, InceptionNeXt for diversity
- frozen embeddings plus Ridge/LightGBM/CatBoost when data is tiny or runtime is
  tight
- high resolution or tiling when target depends on small visual structures

Loss priorities:

- `SmoothL1Loss`/Huber for noisy labels
- `MSELoss` for clean RMSE tasks
- `L1Loss` for MAE or physical-map tasks
- weighted multi-target losses for explicit target weights
- auxiliary classification/bin heads when magnitude bins are easier than raw
  regression

### 6.2 Implementation Pattern: Weighted Multi-Target Loss

multi-target biomass uses five target rows with explicit metric weights. The pattern
generalizes to any multi-output image regression task with weighted targets.

```python
import torch
import torch.nn.functional as F

target_cols = ["Dry_Green_g", "Dry_Dead_g", "Dry_Clover_g", "GDM_g", "Dry_Total_g"]
weights = torch.tensor([0.1, 0.1, 0.1, 0.2, 0.5], device=device)
weights = weights / weights.sum()

pred = model(images)                         # shape: [batch, 5]
target = batch["target"].to(device).float()   # shape: [batch, 5]
per_target = F.smooth_l1_loss(pred, target, beta=5.0, reduction="none").mean(0)
loss = (weights * per_target).sum()
```

### 6.3 Embeddings Plus Tabular Regressors

Frozen embedding stacks are strong when:

- training data is small
- metadata has high signal
- image labels are noisy
- competition runtime is limited
- multiple target columns behave differently

Useful regressors:

- Ridge / ElasticNet
- CatBoostRegressor
- LightGBM / XGBoost
- SVR or KNN as small diversity models
- target-specific regressors with OOF stacking

Feature families:

- global CLS/pooled embeddings
- patch-token means, stds, quantiles, PCA/PLS components
- metadata and coordinates
- predicted class/species probabilities
- image quality, size, brightness, color histograms
- detector/segmenter count summaries when relevant

### 6.4 Implementation Pattern: Embedding Late Fusion

```python
# Implementation Pattern: image embedding + metadata/semantic features.
emb = image_encoder(batch_of_tiles).mean(axis=0)
features = [
    pca.transform(emb[None, :]),
    pls.transform(emb[None, :]),
    semantic_scores[image_id][None, :],
    tabular_meta.loc[image_id].values[None, :],
]
X = np.concatenate(features, axis=1)

for k, target_name in enumerate(target_cols):
    reg = make_regressor(target_name)   # Ridge, CatBoost, LightGBM, etc.
    reg.fit(X_train, transform_y(y_train[:, k], target_name))
    pred[:, k] = inverse_transform_y(reg.predict(X_test), target_name)
```

### 6.5 Tiling And Scale Preservation

- pasture texture or vegetation density
- small animals or dot labels
- ECG waveform traces
- histology spots
- route length where image scale is meaningful

Avoid arbitrary resizing when scale encodes the target. Resizing can destroy route-length information.

### 6.6 Implementation Pattern: Reflect-Padded Tiling

```python
def split_image(image, patch_size=520, overlap=16):
    h, w, c = image.shape
    stride = patch_size - overlap
    patches, coords = [], []
    for y in range(0, h, stride):
        for x in range(0, w, stride):
            patch = image[y:y + patch_size, x:x + patch_size, :]
            if patch.shape[0] < patch_size or patch.shape[1] < patch_size:
                pad_h = patch_size - patch.shape[0]
                pad_w = patch_size - patch.shape[1]
                patch = np.pad(patch, ((0, pad_h), (0, pad_w), (0, 0)),
                               mode="reflect")
            patches.append(patch)
            coords.append((y, x, y + patch_size, x + patch_size))
    return patches, coords

patches, coords = split_image(image)
tile_features = encoder(preprocess_batch(patches))
image_feature = tile_features.mean(axis=0)
```

## 7. Plant Traits, Biomass, And Biological Numeric Targets

### 7.1 Plant-Trait Regression

Plant trait tasks often combine:

- plant images
- species identity or species-proxy visual evidence
- climate, soil, satellite, and environmental metadata
- multiple continuous traits with different noise and scale

High-value moves:

- DINOv2/DINOv3 or PlantCLEF/Pl@ntNet-style pretrained backbones
- metadata fusion, especially for traits tied to environment
- species-proxy classification heads or soft species-to-trait lookup
- trait-vector losses such as cosine similarity in addition to per-trait error
- CatBoost over image embeddings plus tabular features
- OOF stacking of image, metadata, and species-proxy predictions

### 7.2 Multi-Target Biomass

multi-target biomass is a prototypical multi-target image regression problem with
biological identities:

- `GDM = Dry_Green_g + Dry_Clover_g`
- `Dry_Total_g = GDM + Dry_Dead_g`
- equivalently `Dry_Total_g = Dry_Green_g + Dry_Clover_g + Dry_Dead_g`

The metric is weighted R2 over five target rows. `Dry_Total_g` and `GDM_g` are
more important than the three components.

Strong routes:

- predict three primary components and derive the rest
- predict all five targets but add consistency loss or post-hoc projection
- use DINOv3 dense/local features instead of only CLS/global features
- train at higher resolution if compute permits
- use paired left/right field images with shared backbone fusion
- try interval/bin auxiliary heads because biomass behaves like count/density
- use frozen DINOv3 embeddings plus Ridge/CatBoost as robust diversity

### 7.3 Implementation Pattern: Constrained Biomass Heads

```python
# Implementation Pattern: nonnegative component heads and derived targets.
green = F.softplus(head_green(feat)).squeeze(1)
clover = F.softplus(head_clover(feat)).squeeze(1)
dead = F.softplus(head_dead(feat)).squeeze(1)

gdm = green + clover
total = gdm + dead

pred = torch.stack([green, dead, clover, gdm, total], dim=1)
loss = weighted_smooth_l1(pred, target, weights=[0.1, 0.1, 0.1, 0.2, 0.5])
```

```python
pred_total = preds_direct[:, 0]
pred_gdm = preds_direct[:, 1]
pred_green = preds_direct[:, 2]

pred_clover = np.maximum(0, pred_gdm - pred_green)
pred_dead = np.maximum(0, pred_total - pred_gdm)
pred5 = np.stack([pred_green, pred_dead, pred_clover, pred_gdm, pred_total], axis=1)
```

### 7.4 Implementation Pattern: Linear Constraint Projection

```python
def reconcile_biomass(df_preds):
    ordered = ["Dry_Green_g", "Dry_Clover_g", "Dry_Dead_g", "GDM_g", "Dry_Total_g"]
    Y = df_preds[ordered].values.T
    # Constraints:
    # Dry_Green + Dry_Clover - GDM = 0
    # Dry_Dead + GDM - Dry_Total = 0
    C = np.array([[1, 1, 0, -1, 0],
                  [0, 0, 1,  1, -1]], dtype=float)
    P = np.eye(5) - C.T @ np.linalg.inv(C @ C.T) @ C
    out = df_preds.copy()
    out[ordered] = (P @ Y).T.clip(min=0)
    return out
```

Use projection as an OOF-validated postprocess. Hard constraints can hurt when
labels are noisy.

### 7.5 Implementation Pattern: Two-View Field Image Fusion

multi-target biomass data may provide left/right views of the same sample. A shared
backbone with concatenated features is a strong first model.

```python
class TwoViewRegressor(nn.Module):
    def __init__(self, model_name, n_targets=3):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=True,
                                          num_classes=0, global_pool="avg")
        nf = self.backbone.num_features
        self.head_total = nn.Sequential(nn.Linear(nf * 2, nf), nn.ReLU(),
                                        nn.Dropout(0.3), nn.Linear(nf, 1))
        self.head_gdm = nn.Sequential(nn.Linear(nf * 2, nf), nn.ReLU(),
                                      nn.Dropout(0.3), nn.Linear(nf, 1))
        self.head_green = nn.Sequential(nn.Linear(nf * 2, nf), nn.ReLU(),
                                        nn.Dropout(0.3), nn.Linear(nf, 1))

    def forward(self, left, right):
        fl = self.backbone(left)
        fr = self.backbone(right)
        feat = torch.cat([fl, fr], dim=1)
        return torch.cat([self.head_total(feat),
                          self.head_gdm(feat),
                          self.head_green(feat)], dim=1)
```

### 7.6 Dense Patch Tokens And Weak Density

- keep ViT patch tokens by setting `global_pool=""`
- fuse tokens from multiple views or crop scales
- pool local density estimates, not only global features
- use high input resolution if the target depends on texture and small plant
  components
- compare global, local, and global+local heads

This is the image-regression analog of counting: the target is scalar, but the
evidence is distributed spatially.

## 8. Image-Derived Counting

### 8.1 Decide Whether Counting Is Really Regression

If the final row is image-level counts, `image_regression` can be primary.
If boxes, masks, centroids, or instance AP are scored, route to detection or
segmentation.

Counting tasks often benefit from:

- object detector or segmenter first
- crop classifier for object subtype
- count features and scale features
- patch count regression fallback
- sequence/burst aggregation
- nonnegative clipping and integer/threshold ablations

### 8.2 aerial wildlife-count Aerial Counts

- dotted training images can provide approximate coordinates
- close objects and scale variation make pure CNN count regression fragile
- U-Net/star masks plus connected components can handle clustered animals
- per-class regressors over mask sums/blob counts are strong
- downscaling or scale estimation can matter
- public-LB fitted postprocess is not transferable and should be excluded

### 8.3 Camera-Trap Sequence Counts

camera-trap wildlife counting count tasks are not ordinary per-image regression:

- train and test cameras can be different locations
- sequence-level count is scored
- MegaDetector or similar detector outputs can dominate
- full-image classification and crop classification complement each other
- max-per-sequence count heuristics can beat summing noisy detections
- location/camera grouped validation is mandatory

Use image classification for species probabilities, detection for animal boxes,
and image regression only for the final count aggregation if counts are the
scored output.

### 8.4 Implementation Pattern: Patch Count Regression

```python
# Implementation Pattern: patch image -> per-class count vector.
model = Sequential()
model.add(Conv2D(32, (3, 3), activation="relu", padding="same",
                 input_shape=(width, width, 3)))
model.add(Conv2D(32, (3, 3), activation="relu", padding="same"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation="relu", padding="same"))
model.add(Conv2D(64, (3, 3), activation="relu", padding="same"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3), activation="relu", padding="same"))
model.add(Conv2D(128, (3, 3), activation="relu", padding="same"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(256, activation="relu"))
model.add(Dense(5, activation="linear"))
```

Modernize this with EfficientNet/ConvNeXt/DINO features if the task allows.

### 8.5 Detector/Segmenter Count Features

For a stronger count system, build a feature table:

- detector count by class and confidence threshold
- mask area by class
- connected components by area range
- average box size and image scale
- crop classifier probabilities
- full-image species/source probabilities
- burst max, mean, and top-k counts
- metadata such as camera, site, time, and source when allowed

Then fit LightGBM/CatBoost/Ridge/Poisson/Huber regressors on OOF detector
features. Validate at the official image or sequence unit.

## 9. Image Matching, Relative Pose, Camera Pose, And SfM

This is a separate geometric route, not normal image regression. Use it when
the output is a fundamental matrix, relative pose, camera pose, or registered
camera centers.

### 9.1 Pairwise Fundamental Matrix

A robust multi-view image matching route:

- pairs are given
- run robust local feature matching per pair
- estimate fundamental matrix with MAGSAC/USAC/RANSAC
- submit flattened fundamental matrix
- score by relative pose mAA derived from the predicted F matrix

Strong sources concatenate high-quality matches from LoFTR, SuperGlue/
SuperPoint, DKM, crops, and multiple resolutions, then estimate one robust
geometry. Match-level ensembling is more useful than averaging matrices.

### 9.2 Multi-View Reconstruction

The route:

- shortlist image pairs or use exhaustive pairs if scene size/runtime permits
- extract keypoints/descriptors
- match pairs with LightGlue/SuperGlue/LoFTR/DKM or complementary matchers
- import keypoints and matches into COLMAP
- run geometric verification and incremental mapping
- choose reconstruction by registered images/points
- format per-image `R/t` or camera centers

Validate by scene. Random pair validation is misleading because final score
depends on the full reconstruction graph.

### 9.3 Implementation Pattern: Feature Matching To COLMAP

```python
# Implementation Pattern: geometry pipeline skeleton.
pairs = shortlist_pairs(global_descriptors, min_pairs=50)

detect_keypoints(
    images=image_paths,
    extractor="ALIKED",
    out_keypoints="keypoints.h5",
    out_descriptors="descriptors.h5",
)

match_pairs(
    pairs=pairs,
    matcher="LightGlue",
    keypoints="keypoints.h5",
    descriptors="descriptors.h5",
    out_matches="matches.h5",
)

database_path = import_into_colmap(
    image_paths=image_paths,
    keypoints_h5="keypoints.h5",
    matches_h5="matches.h5",
)
pycolmap.match_exhaustive(database_path)
maps = pycolmap.incremental_mapping(database_path, image_dir, output_dir)
best = max(maps.values(), key=lambda rec: rec.num_reg_images())
```

### 9.4 Pair Selection

Choose pair selection by scene size and overlap:

- given pairs: match every official pair
- small/medium scenes: exhaustive all-pairs can outperform descriptor
  retrieval because retrieval misses hard pairs
- large scenes: global descriptor shortlist with minimum neighbor count
- use match-count thresholds to remove weak edges
- add kNN completion so every image has enough neighbors

### 9.5 Matchers And Merge Strategy

Useful matchers:

- ALIKED + LightGlue: strong sparse baseline for multi-view image matching-style tasks
- SuperPoint + SuperGlue: stable sparse route
- LoFTR and DKM: dense/detector-free diversity
- DISK, SIFT, RootSIFT, HardNet/AffNet variants: useful depending on packages

Merge with care:

- concatenate complementary matches before robust geometry
- quantize/merge detector-free points to avoid exploding track counts
- use confidence-guided NMS or top-k keypoint pruning
- run geometric verification after merging

### 9.6 Crops, Rotation, And Transparent Scenes

High-ROI tricks:

- crop covisible regions using match-keypoint clusters, but keep original-image
  matches as well
- test rotations for 0/90/180/270 orientation issues
- use multi-resolution matching rather than generic flip TTA
- for transparent/reflective objects, route to foreground/object masks,
  sequence ordering, circular/circumferential camera placement, or MST-aided
  SfM
- avoid blind CLAHE/mask/dense-match pileups unless scene-level CV improves

### 9.7 Implementation Pattern: Pose Submission Formatting

Serialize row-major `R` and `t`; unregistered images may
need at least one `nan` depending on the competition version.

```python
def arr_to_str(x):
    return ";".join("nan" if np.isnan(v) else f"{v:.09f}" for v in np.ravel(x))

rows = []
for image_name in image_names:
    if image_name in registered:
        R = registered[image_name].cam_from_world.rotation.matrix()
        t = registered[image_name].cam_from_world.translation
    else:
        R = np.full((3, 3), np.nan)
        t = np.full(3, np.nan)
    rows.append({
        "image_path": image_name,
        "rotation_matrix": arr_to_str(R),
        "translation_vector": arr_to_str(t),
    })
submission = pd.DataFrame(rows)
```

## 10. ECG, Chart, And Image-To-Signal Digitization

### 10.1 High-Performance Pipeline Shape

Strong route:

1. Detect and rectify the ECG paper or grid.
2. Preserve high resolution after rectification.
3. Crop away header/personal-info regions.
4. Split into four ECG row/segment regions.
5. Predict waveform with heatmaps, coordinate distributions, or direct
   regression.
6. Convert pixels to mV/time using grid scale or learned calibration.
7. Split row predictions into 12 leads.
8. Resample to required row counts.
9. Apply robust ensembling and gated physiologic corrections.
10. Validate exact SNR per lead/type.

### 10.2 Rectification And Grid Strategy

- paper corner detection
- grid intersection heatmaps
- line index prediction
- template registration
- RANSAC/local homographies
- iterative interpolation for missed grid points
- synthetic ECG image kit annotations for lead boxes/masks when legal

High-resolution preservation matters: estimate geometry at lower resolution if
needed, but apply the final warp to the original or high-resolution image.

### 10.3 Signal Heads

Model choices:

- segmentation heatmaps where each column predicts waveform y-location
- coordinate distributions with softmax over y and Gaussian labels
- soft-argmax heads converting logits to y coordinates
- direct regression of lead signals from cropped images
- hybrid auxiliary segmentation plus waveform regression

### 10.4 Implementation Pattern: Column-Wise Signal Extraction

```python
def trace_dark_signal(binary_roi):
    ys = []
    h = binary_roi.shape[0]
    for x in range(binary_roi.shape[1]):
        dark = np.where(binary_roi[:, x] > 0)[0]
        if len(dark) == 0:
            ys.append(np.nan)
        else:
            ys.append(h - np.median(dark))
    y = pd.Series(ys).interpolate(limit_direction="both").values
    y = (y - np.nanmedian(y)) / pixels_per_mv
    return scipy.signal.resample(y, required_length)
```

### 10.5 Losses And Labels

Useful labels:

- one-pixel or two-pixel column masks
- fractional y labels distributed between adjacent pixels
- Gaussian y-distribution labels
- coordinate targets after rectification
- waveform regression labels in mV

Column-wise JSD/CE over y distributions can beat BCE for thin waveforms. Direct
regression avoids cumulative segmentation postprocess error, but requires
excellent rectification and cropping.

### 10.6 Postprocess

Use:

- Fourier-domain resampling when it improves validation
- weighted median or robust average ensembles to suppress spikes
- edge artifact suppression
- confidence-based interpolation/replacement
- horizontal flip and crop/brightness TTA if validated
- physiological identities such as `II = I + III` only when OOF improves

## 11. Calibrated Sensor Regression

calibrated exoplanet-sensor regression predicts 283 spectrum means and 283 uncertainties per planet from
sequential sensor frames. It is best treated as calibrated scientific
time/image regression.

### 11.1 Calibration Before Modeling

Start from domain calibration:

- invert analog-to-digital conversion with gain/offset
- mask hot/dead pixels
- optionally correct nonlinearity
- subtract dark frames with integration time
- correlated double sampling
- time binning
- flat-field correction
- crop wavelength axis to target wavelengths
- handle foreground/background rows

Do not replace this with generic RGB normalization.

### 11.2 Implementation Pattern: Sensor Calibration Pipeline

```python
def ADC_convert(signal, gain, offset):
    signal = signal.astype(np.float64)
    signal /= gain
    signal += offset
    return signal

def mask_hot_dead(signal, dead, dark):
    hot = sigma_clip(dark, sigma=5, maxiters=5).mask
    hot = np.tile(hot, (signal.shape[0], 1, 1))
    dead = np.tile(dead, (signal.shape[0], 1, 1))
    signal = np.ma.masked_where(dead, signal)
    signal = np.ma.masked_where(hot, signal)
    return signal

def get_cds(signal):
    return signal[:, 1::2, :, :] - signal[:, ::2, :, :]

def bin_obs(cds_signal, binning):
    x = cds_signal.transpose(0, 1, 3, 2)
    out = np.zeros((x.shape[0], x.shape[1] // binning, x.shape[2], x.shape[3]))
    for i in range(x.shape[1] // binning):
        out[:, i] = np.sum(x[:, i * binning:(i + 1) * binning], axis=1)
    return out

def correct_flat_field(flat, dead, signal):
    flat = flat.transpose(1, 0)
    dead = dead.transpose(1, 0)
    flat = np.ma.masked_where(dead, flat)
    flat = np.tile(flat, (signal.shape[0], 1, 1))
    return signal / flat
```

### 11.3 Physics And Signal Extraction

Top calibrated exoplanet-sensor regression routes emphasize:

- transit boundary detection
- polynomial or separable drift modeling over time and wavelength
- separating mean transit depth from wavelength-dependent atmospheric features
- foreground/background subtraction
- inpainting/interpolating invalid pixels when jitter moves signal
- smoothing/binning spectra based on dynamics
- PCA per star or other artifact removal when validated

Strong non-DL models include Gaussian processes, Bayesian fitting, AutoEncoder,
NMF, polynomial regression, and Ridge over engineered features. Neural baselines
include 1D CNNs for white curves/mean depth and 2D CNNs for wavelength
variation.

### 11.4 Implementation Pattern: GLL Metric

Use the exact likelihood when sigma columns are scored.

```python
from scipy.stats import norm

def gaussian_log_likelihood(y_true, mu, sigma, sigma_floor=1e-6):
    sigma = np.maximum(np.asarray(sigma), sigma_floor)
    return norm.logpdf(y_true, loc=mu, scale=sigma).mean()

def build_calibrated_exoplanet_sensor_regression_submission(planet_ids, pred_mu, pred_sigma):
    pred_sigma = np.maximum(pred_sigma, 1e-6)
    rows = []
    for i, planet_id in enumerate(planet_ids):
        row = {"planet_id": planet_id}
        for j in range(pred_mu.shape[1]):
            row[f"spectrum_{j}"] = pred_mu[i, j]
        for j in range(pred_sigma.shape[1]):
            row[f"sigma_{j}"] = pred_sigma[i, j]
        rows.append(row)
    return pd.DataFrame(rows)
```

### 11.5 Sigma Strategies

Useful sigma sources:

- residual standard deviation by wavelength
- per-planet prediction standard deviation
- ensemble variance
- GP posterior uncertainty
- ingress/egress disagreement
- constant baseline sigma blended with spectrum-dependent sigma

Calibrate sigma on validation; overconfident wrong sigmas are heavily punished.

## 12. Waveform Inversion And Image-Shaped Physical Maps

Waveform inversion predicts a 70x70 subsurface velocity map from seismic
waveforms. It is not segmentation: input `p(g,t)` and output `c(x,z)` are not
spatially aligned.

### 12.1 Data Shape And Submission

Common local shapes:

- input: 5 sources/channels, 1000 time steps, 70 geophones
- output: 70x70 velocity map
- submission: all 70 y rows and odd x columns only: `x_1, x_3, ..., x_69`

### 12.2 Models

Baseline:

- InversionNet with Conv/ConvTranspose blocks
- L1/MAE loss

- ConvNeXt with U-Net decoder
- HGNetV2
- CAFormer-B36 full-resolution U-Net-style models

- ViT/EVA/DINOv2-style backbones
- sigmoid output scaled to physical velocity range
- AdamW, cosine/constant-cosine LR, EMA, horizontal flip/TTA
- long training and checkpoint ensembles

### 12.3 Implementation Pattern: Physical Output Scaling And CSV

```python
class TestDataset(torch.utils.data.Dataset):
    def __init__(self, test_files):
        self.test_files = test_files

    def __len__(self):
        return len(self.test_files)

    def __getitem__(self, i):
        test_file = self.test_files[i]
        test_stem = test_file.split("/")[-1].split(".")[0]
        return np.load(test_file), test_stem

def denormalize_velocity(head_output):
    # Map normalized outputs to the legal physical range.
    # or sigmoid scaled to [1500, 4500].
    return head_output * 1500.0 + 3000.0

x_cols = [f"x_{i}" for i in range(1, 70, 2)]
fieldnames = ["oid_ypos"] + x_cols

with open("submission.csv", "wt", newline="") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for vel_map, oid in predictions:          # vel_map shape: [70, 70]
        vel_map = np.clip(vel_map, 1500, 4500)
        for y in range(70):
            row = {"oid_ypos": f"{oid}_y_{y}"}
            row.update({f"x_{x}": vel_map[y, x] for x in range(1, 70, 2)})
            writer.writerow(row)
```

### 12.4 Physics And Synthetic Data

- forward solver `velocity -> seismic waveform`
- legal synthetic paired data generation
- family-specific priors
- differentiable refinement by matching simulated waveforms to observed
  waveforms
- total variation, p-norm, edge, GP, or SVD priors by family
- BFGS/Gauss-Newton or gradient-based FWI after a DL model

Use solver refinement only if:

- rules allow the solver/code/data
- runtime permits
- validation can measure family-wise improvement
- the first DL baseline and submission writer already work

### 12.5 Family-Aware Validation And Postprocess

Use family/file/event folds. Track hardest families separately. Ensemble by:

- architecture
- reshape strategy
- checkpoint
- seed
- family-specialist model
- stage output

Postprocess:

- clip to physical velocity range
- round or restore discrete layers for families where the physics supports it
- apply family-specific blends only if validation supports the family detector

## 13. Spatial Pathology And Spot-Level Abundance Regression

EL Hackathon-style tasks predict continuous abundance vectors from H&E patches
and spot coordinates. This is image+tabular multi-output regression with a
spatial validation risk.

Use:

- slide-level folds, often leave-one-slide-out when there are few slides
- patch encoders from TIMM, DINO, ConvNeXt, EVA, or histology-pretrained models
- coordinate features, local spot neighborhoods, and slide normalization
- multi-output regression heads or embedding-to-tabular regressors
- rank-aware target transforms for Spearman metrics
- OOF stacking across image-only, coordinate-only, and fused models

Avoid spot-level random splits because neighboring spots and slide style leak
heavily. If public/private spots are weakly correlated, trust slide-level CV
more than public score movement.

## 14. Synthetic Geometry/Route-Length Image Regression

synthetic route-length tasks predict a numeric route length from synthetic images.
The important lesson is scale preservation:

- image size and pixel geometry encode the target
- resizing can destroy the target scale
- shallow CNNs and even channel/geometry features can be competitive
- validate original pixel geometry
- use early stopping to avoid overfitting synthetic patterns

Useful models:

- small CNN/ResNet/EfficientNet regression
- hand-engineered path/edge/line/color features
- segmentation or graph extraction if the route is visually parseable
- classical regressors over geometry features

Route out to optimization/graph search if the input contains an explicit graph
and the task asks for a route rather than route length prediction.

## 15. Geospatial And Environmental Methodology Boundary

Do not treat judged-methodology tasks as supervised image regression. For similar tasks:

- if no train/test numeric target exists, route to data analysis/scientific
  methodology rather than modeling
- if remote sensing images produce supervised numeric targets, use this playbook
  plus time-series/tabular/geospatial validation
- if the deliverable is a narrative methodology deliverable, focus on reproducibility,
  assumptions, citations, visual evidence, and robustness, not model ensembling

## 16. Ensembling And TTA

### 16.1 General Ensembling

Good ensemble axes:

- backbone family
- input resolution
- crop/tiling strategy
- target transform: raw, standardized, log, rank
- image-only vs metadata-only vs fused
- global vs dense/local heads
- folds and seeds
- regression head vs embedding GBDT/Ridge
- physical/statistical vs neural models

Blend by OOF score, per-target weights, and fold stability. Keep row-level OOF
predictions for every model.

### 16.2 Implementation Pattern: TTA Averaging

```python
views = [
    make_transform(resize=img_size, hflip=False, vflip=False),
    make_transform(resize=img_size, hflip=True,  vflip=False),
    make_transform(resize=img_size, hflip=False, vflip=True),
]

view_preds = []
for transform in views:
    loader = make_test_loader(test_df, transform=transform)
    model_preds = [predict_model(model, loader) for model in fold_models]
    view_preds.append(np.mean(model_preds, axis=0))

final_pred = np.mean(view_preds, axis=0)
```

### 16.3 When TTA Is Not Worth It

Delay TTA when:

For ECG, flips must be transformed back correctly. For geometry, rotation checks
are more meaningful than generic TTA. For biomass, flip TTA is an ablation, not
a default.

## 17. Submission Validation

Always implement a validator before modeling becomes complex.

Check:

- exact row count
- exact column names and order
- no missing required ids
- no duplicate ids
- finite numeric values unless `nan` is explicitly required for unregistered
  geometry
- nonnegative targets when required
- sigma positive
- physical range for maps/signals
- correct long/wide reshaping
- exact lead/time/target naming
- row-level aggregation matches sample submission

### 17.1 Implementation Pattern: Long Biomass Submission

multi-target biomass tasks often require one row per image-target pair.

```python
def make_long_submission(df_pred, image_col="image_path"):
    target_cols = ["Dry_Green_g", "Dry_Dead_g", "Dry_Clover_g", "GDM_g", "Dry_Total_g"]
    melted = df_pred.melt(
        id_vars=[image_col],
        value_vars=target_cols,
        var_name="target_name",
        value_name="target",
    )
    image_id = (melted[image_col].str.replace(r"^.*/", "", regex=True)
                              .str.replace(".jpg", "", regex=False))
    melted["sample_id"] = image_id + "__" + melted["target_name"]
    return melted[["sample_id", "target"]]
```

### 17.2 Implementation Pattern: ECG Lead Submission Skeleton

ECG sources submit one row per `image_id_timestep_lead`.

```python
def write_ecg_submission(record_id, lead_to_signal, number_of_rows):
    rows = []
    for lead, signal in lead_to_signal.items():
        n = number_of_rows[lead]
        signal = scipy.signal.resample(np.asarray(signal, dtype=float), n)
        signal = pd.Series(signal).interpolate(limit_direction="both").fillna(0).values
        for t, value in enumerate(signal):
            rows.append({"id": f"{record_id}_{t}_{lead}", "value": float(value)})
    return pd.DataFrame(rows)
```

## 18. Common Failure Modes

Wrong family:

- using image classification for count regression
- using image regression for detection AP
- using segmentation for waveform inversion because output is image-shaped
- using ordinary regression for image matching/SfM
- using RGB normalization for calibrated sensor data

Wrong validation:

- random spots within the same slide
- random images from the same camera/site/scene/planet/simulation family
- random image pairs for full-scene reconstruction
- image-level validation for sequence counts
- public-score selection when public/private distributions differ

Wrong target handling:

- optimizing unweighted MSE for weighted R2 targets
- ignoring sigma in likelihood metrics
- ignoring target identities such as `Total = components`
- clipping or rounding counts before measuring OOF
- hard constraints that conflict with noisy labels

Wrong preprocessing:

- resizing away scale
- cropping away small target evidence
- removing ECG grids before using them for rectification
- throwing away sensor calibration frames
- treating dead/hot pixels uniformly across tasks when they may contain or
  destroy information

Wrong ensembling:

- model pileups without OOF row alignment
- averaging poses instead of merging matches
- TTA that changes label semantics
- per-target blends that violate constraints

## 19. Route-Specific First Plans

### 19.1 Ordinary Image Regression

First round:

- grouped/stratified folds
- `timm` ConvNeXt/EfficientNet/DINO/EVA backbone
- SmoothL1 or MSE
- target standardization with exact inverse
- OOF metric and submission validator
- simple clipping

Round 2:

- higher resolution or tiling
- frozen embeddings + Ridge/CatBoost/LightGBM
- metadata fusion
- per-target heads and weights

Late round:

- OOF stacking
- target transform diversity
- pseudo-labels only if safe
- source-specific calibration if OOF proves it

### 19.2 Biomass/Plant Traits

First round:

- DINOv2/DINOv3 or strong TIMM backbone
- grouped folds by site/date/species/slide where available
- weighted multi-target loss
- target constraints only as OOF-tested postprocess
- metadata fusion if task provides real covariates

Round 2:

- dense patch-token aggregation
- species/proxy classification heads
- interval target bins
- embedding-to-GBDT stacks

Late round:

- constraint projection
- per-target blend search
- raw/log/rank model diversity
- cautious source/state/date-specific postprocess

### 19.3 Counts

First round:

- detector/segmenter or patch-count baseline depending on labels
- image/sequence-level OOF metric
- nonnegative clipping
- threshold tuning on OOF

Round 2:

- crop classifier and count-feature regressors
- scale features and connected components
- sequence/burst aggregation

Late round:

- detector ensemble
- per-class thresholds
- location/camera diagnostics

### 19.4 Geometry/SfM

First round:

- ALIKED/LightGlue or SuperPoint/SuperGlue baseline
- pair shortlist or exhaustive pairs
- robust geometry or COLMAP reconstruction
- scene-level validation and pose submission validator

Round 2:

- add LoFTR/DKM dense matches
- multi-resolution/crop matching
- multiple COLMAP thresholds

Late round:

- transparent/reflective special route
- relocalize unregistered images
- MST/coarse-to-fine graph construction

### 19.5 ECG/Chart Digitization

First round:

- rectification/cropping pipeline
- heatmap or coordinate model
- exact SNR validation
- signal resampling and submission writer

Round 2:

- grid/keypoint template mapping
- higher resolution
- direct regression or coordinate distribution heads

Late round:

- robust ensembles
- gated physiological corrections
- OOD/layout specialist models

### 19.6 Scientific Inverse Problems

First round:

- exact calibration and metric
- physical/statistical baseline plus neural baseline
- natural-unit validation
- physical output constraints

Round 2:

- stronger backbone/reshape strategy
- uncertainty calibration or family specialists
- synthetic data if legal

Late round:

- physical solver refinement
- family-specific priors
- model/physics blend search
