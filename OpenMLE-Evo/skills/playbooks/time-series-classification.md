# Time-Series Classification Category Playbook

This playbook is direct reading material for an agent that must generate a
task-specific Kaggle skill from one task description. It is intentionally broad
and practical: it covers sequence labels, dense per-step labels, sparse event
timestamps, action segments, scientific waveforms, irregular light curves,
longitudinal medical rows, market states, and route-out tasks.

## 0. Safety Boundary

Do not turn these into recipes:

- public/private leaderboard probing
- hidden test labels
- manual private test inspection or corrections
- disallowed external data
- sample-submission inversion
- row-order/cumcount artifacts that do not exist at deployment
- public-agent copying for non-agent tasks
- task-specific leakage formulas such as irregular astronomical light-curve classification class-99 probing
- dense ion-state classification ion private max10 leakage exploitation
- grouped inertial-sensor classification test chaining as a practical robotics method

## 1. Output Taxonomy

Start from the submission, not from the raw input.

### 1.1 Sequence-Level Closed-Set Classification

One row per sequence/window/object, one class or probability vector per row.

- fixed-window biological sensor classification: 60-second biological sensor sequences, binary state, AUC.
- grouped inertial-sensor classification: 128-step IMU series, floor-surface class.
- wearable activity recognition: 1-second inertial windows, activity label, macro F1.
- irregular astronomical light-curve classification: one probability vector per astronomical object, weighted logloss.

Default first stack:

- engineered-feature GBDT: LightGBM, CatBoost, XGBoost, ExtraTrees
- sequence model: 1D CNN + BiGRU/LSTM, Conv2D over sensor x time + GRU,
  DeepConvLSTM, TinyHAR, Attend-and-Discriminate
- optional DAE or self-supervised features if transductive feature learning is
  permitted

### 1.2 Dense Per-Step Classification

One probability/class per time step.

Examples:

- dense ion-state classification: open channel class for each timestamp.
- freezing-of-gait event classification: confidence per timestep for StartHesitation, Turn, Walking.
- some EEG/power-line pipelines after subwindow expansion.

Default first stack:

- WaveNet/TCN, Conv1D + BiGRU/LSTM, U-Net-1D, Transformer encoder,
  Squeezeformer
- mask loss by valid/annotated/time regions
- overlap inference and boundary trimming

### 1.3 Sparse Event Timestamp Extraction

Model may output dense probabilities, but submission contains discrete events.

Examples:

- sleep-event detection onset/wakeup events.

Default first stack:

- dense model over anglez/enmo with soft/decayed targets
- candidate peak detection
- second-stage candidate rescoring with LightGBM/CatBoost or small sequence
  models
- event-level greedy selection or NMS under metric tolerances

### 1.4 Segment/Interval Action Recognition

Submission rows are intervals: action, agent, target, start, stop.

Examples:

- multi-agent keypoint behavior mouse behavior detection.

Default first stack:

- per-action binary classifiers with action masks
- keypoint geometry features + XGBoost/LightGBM/ExtraTrees
- Squeezeformer/Transformer/GNN over windows
- per-lab/action thresholds, segment merge, overlap removal

### 1.5 Scientific Waveform Detection

Raw signals are long and domain structure matters.

Examples:

- power-line fault detection power-line partial discharge.
- gravitational-wave detection gravitational waves.
- continuous gravitational waves.
- EEG seizure prediction prediction.

Default first stack:

- physical/signal features: peaks, PSD, FFT, power bands, wavelets,
  coherence, Riemannian covariance, matched filters
- raw Conv1D for waveform tasks
- spectrogram/CQT/CWT CNN only after comparing with raw Conv1D and engineered
  features
- subject/detector/measurement-aware validation

### 1.6 Irregular Scientific Light Curves

Irregular time series with passbands, errors, and metadata.

Example:

- irregular astronomical light-curve classification astronomical classification.

Default first stack:

- GP or parametric light-curve features + LightGBM/CatBoost
- metadata/redshift-aware features
- RNN/GRU with passband, flux, flux_err, time deltas, detected flag
- train-to-test cadence/noise augmentation while grouping by original object

### 1.7 Longitudinal Tabular Classification

Rows over time but GBDT features dominate.

Examples:

- clinical early-warning sepsis.
- distracted driving risk.
- app user next-step prediction.
- market reversal states.

Default first stack:

- tabular classification primary
- group/time-safe rolling and recency features
- CatBoost/LightGBM/XGBoost with class weights
- OOF probability stacking

### 1.8 Route-Out Families

Route out unless a genuine temporal classifier is a scored subproblem:

- Brain-to-text: sequence-to-sequence / ASR, WER/CER, decoding and language
  models.
- interactive game-policy tasks: interactive agent/policy ladder, not supervised TSC.
- Pure image/video/audio labels: use modality-specific methods first.
- Continuous forecasts, coordinates, quantiles: use time-series regression.
- Boxes/masks/keypoints as final output: use detection/segmentation/keypoint
  playbook.

## 2. Validation Geometry

Validation must mirror deployment. Random row folds are the default failure
mode in this category.

### 2.1 Group Selection

Choose the strongest leakage axis:

| Task signal | Fold group |
| --- | --- |
| wearable/HAR/gesture | subject, participant, session, device |
| FOG/sleep | subject, series, patient, recording |
| animal behavior | video, lab, tracking pipeline, recording |
| seizure/sepsis | patient, admission, one-hour segment, hospital unit |
| power-line | measurement id, phase triplet, acquisition batch |
| gravitational waves | simulated seed/noise family if available; otherwise stratified and detector-aware audits |
| irregular astronomical light-curve classification | object id; keep augmented siblings in one fold |
| market | ticker plus time period; purged/chronological split when possible |
| user-event | user/session/app/time group |

Code pattern:

```python
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
import numpy as np

def make_group_folds(df, y, group_col, n_splits=5, stratify=True, seed=42):
    groups = df[group_col].values
    if stratify:
        splitter = StratifiedGroupKFold(
            n_splits=n_splits, shuffle=True, random_state=seed
        )
        return list(splitter.split(df, y, groups))
    splitter = GroupKFold(n_splits=n_splits)
    return list(splitter.split(df, y, groups))

def assert_no_group_overlap(df, folds, group_col):
    for fold, (tr, va) in enumerate(folds):
        a = set(df.iloc[tr][group_col])
        b = set(df.iloc[va][group_col])
        overlap = a.intersection(b)
        assert not overlap, (fold, list(overlap)[:5])
```

### 2.2 Time-Aware and Purged Splits

Use chronological or purged folds when labels depend on future periods,
markets, app sessions, patient trajectories, or event lead times.

```python
def purged_group_time_split(df, time_col, group_col, n_splits=5, embargo=0):
    order = df[[time_col]].drop_duplicates().sort_values(time_col)
    times = order[time_col].to_numpy()
    chunks = np.array_split(times, n_splits)
    for valid_times in chunks:
        lo, hi = valid_times.min(), valid_times.max()
        valid = df[time_col].between(lo, hi).to_numpy()
        train = ((df[time_col] < lo - embargo) | (df[time_col] > hi + embargo)).to_numpy()
        # Optional: remove groups seen in validation if deployment uses unseen groups.
        va_groups = set(df.loc[valid, group_col])
        train &= ~df[group_col].isin(va_groups).to_numpy()
        yield np.flatnonzero(train), np.flatnonzero(valid)
```

Do not remove time columns by habit. Removing them can help if they only encode
row order and leak split identity, but can hurt if time-of-day, minute, or
session phase is a legitimate predictor.

### 2.3 Validation at the Official Unit

For dense/event tasks, row-level loss can improve while event score worsens.
Always transform OOF predictions into the official submission unit before
comparing experiments.

```python
def oof_event_eval(series_ids, steps, y_events, pred_prob, event_extractor, metric):
    rows = []
    for sid in np.unique(series_ids):
        m = series_ids == sid
        rows.append(event_extractor(sid, steps[m], pred_prob[m]))
    pred_events = concat_event_rows(rows)
    return metric(y_events, pred_events)
```

## 3. Windowing, Resampling, and Masks

### 3.1 Fixed Windows

Use fixed windows when the sequence is short or the submission unit is already
a window.

```python
import numpy as np

def make_windows(x, y=None, window=512, stride=256, pad_value=0.0):
    xs, ys, spans = [], [], []
    n = len(x)
    for start in range(0, max(1, n - window + 1), stride):
        stop = start + window
        chunk = x[start:stop]
        if len(chunk) < window:
            pad = np.full((window - len(chunk),) + x.shape[1:], pad_value, dtype=x.dtype)
            chunk = np.concatenate([chunk, pad], axis=0)
        xs.append(chunk)
        spans.append((start, min(stop, n)))
        if y is not None:
            ys.append(y[start:min(stop, n)])
    return np.asarray(xs), ys, spans
```

### 3.2 Overlap Inference with Boundary Trimming

```python
def overlap_average(n_steps, spans, preds, trim=32):
    out = np.zeros((n_steps, preds[0].shape[-1]), dtype=np.float32)
    cnt = np.zeros((n_steps, 1), dtype=np.float32)
    for (start, stop), p in zip(spans, preds):
        left = min(trim, max(0, stop - start) // 2)
        right = max(0, stop - start - left)
        s = start + left
        e = start + right
        if e <= s:
            s, e = start, stop
            left = 0
        out[s:e] += p[left:left + (e - s)]
        cnt[s:e] += 1
    return out / np.maximum(cnt, 1)
```

### 3.3 Masked Loss

Use masks for unannotated time, padded chunks, invalid regions, and task-valid
regions. freezing-of-gait event classification and multi-agent keypoint behavior tasks are particularly sensitive.

```python
import torch
import torch.nn.functional as F

def masked_bce_with_logits(logits, targets, valid_mask):
    loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    mask = valid_mask.float()
    while mask.ndim < loss.ndim:
        mask = mask.unsqueeze(-1)
    loss = loss * mask
    return loss.sum() / mask.sum().clamp_min(1.0)
```

For multiclass frame labels with invalid actions, mask logits before softmax:

```python
def masked_softmax_logits(logits, allowed_mask):
    # allowed_mask: True for legal classes for this row/video/action set.
    return logits.masked_fill(~allowed_mask, -1e9)
```

### 3.4 Patch-Level Target Reduction

FOG sources reduced target resolution by grouping time steps into patches and
using max target within the patch, then tiled predictions back.

```python
def reduce_targets_by_patch(y, patch):
    n = (len(y) // patch) * patch
    y = y[:n]
    return y.reshape(n // patch, patch, y.shape[-1]).max(axis=1)

def expand_patch_predictions(p, patch, n_original):
    dense = np.repeat(p, patch, axis=0)
    return dense[:n_original]
```

Use this when the metric tolerates short localization error or when the event
duration is longer than the patch. Avoid it when exact boundary timing matters.

## 4. Metrics and Postprocessing

### 4.1 Threshold Tuning for MCC, F1, and AP-like Metrics

power-line fault detection sources tuned MCC thresholds; multi-agent keypoint behavior sources tuned lab/action F1 thresholds;
many macro-F1 tasks need class-specific thresholds.

```python
import numpy as np
from sklearn.metrics import matthews_corrcoef, f1_score

def tune_binary_threshold(y_true, y_prob, metric="mcc", grid=None):
    if grid is None:
        grid = np.linspace(0.01, 0.99, 99)
    best_score, best_thr = -1.0, 0.5
    for thr in grid:
        pred = y_prob >= thr
        if metric == "mcc":
            score = matthews_corrcoef(y_true, pred)
        elif metric == "f1":
            score = f1_score(y_true, pred)
        else:
            raise ValueError(metric)
        if score > best_score:
            best_score, best_thr = score, float(thr)
    return best_thr, best_score

def tune_thresholds_by_group_action(oof, y, group_key, action_key):
    out = {}
    for key in sorted(set(zip(group_key, action_key))):
        m = np.array([(g, a) == key for g, a in zip(group_key, action_key)])
        if m.sum() == 0:
            continue
        out[key] = tune_binary_threshold(y[m], oof[m], metric="f1")[0]
    return out
```

### 4.2 Peak Extraction for Timestamp Events

```python
import numpy as np
from scipy.signal import find_peaks

def extract_event_peaks(series_id, steps, prob, min_distance, height, top_k=None):
    peaks, props = find_peaks(prob, distance=min_distance, height=height)
    scores = props.get("peak_heights", prob[peaks])
    order = np.argsort(scores)[::-1]
    if top_k is not None:
        order = order[:top_k]
    rows = []
    for rank, idx in enumerate(order):
        p = peaks[idx]
        rows.append({
            "series_id": series_id,
            "step": int(steps[p]),
            "score": float(scores[idx]),
            "rank": rank,
        })
    return rows
```

### 4.3 Greedy Event Selection with Tolerance Discount

When the metric matches only one prediction to one truth event, predictions too
close to a selected event compete with each other. Use OOF to tune the
tolerance windows.

```python
def greedy_tolerance_events(steps, scores, tolerances, max_events):
    # scores: dense event probability or second-stage score per step.
    remaining = scores.astype(np.float64).copy()
    chosen = []
    for _ in range(max_events):
        i = int(np.argmax(remaining))
        if remaining[i] <= 0:
            break
        chosen.append((int(steps[i]), float(remaining[i])))
        for tol in tolerances:
            lo = np.searchsorted(steps, steps[i] - tol)
            hi = np.searchsorted(steps, steps[i] + tol, side="right")
            remaining[lo:hi] *= 0.0
    return chosen
```

### 4.4 Frame Probabilities to Segments

multi-agent keypoint behavior submissions need valid intervals. Convert dense classes to segments,
drop invalid short spans, merge optional gaps, and remove overlaps.

```python
import pandas as pd
import numpy as np

def probabilities_to_segments(video_id, frames, proba, class_names, thresholds,
                              min_len=1, merge_gap=0):
    best = proba.argmax(axis=1)
    best_score = proba[np.arange(len(proba)), best]
    labels = np.array([
        class_names[c] if best_score[i] >= thresholds.get(class_names[c], 0.5) else "none"
        for i, c in enumerate(best)
    ])
    rows = []
    start = 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[start]:
            label = labels[start]
            if label != "none":
                s, e = int(frames[start]), int(frames[i - 1]) + 1
                if e - s >= min_len:
                    rows.append([video_id, label, s, e, float(best_score[start:i].max())])
            start = i

    # Optional merge: same label separated by small gaps.
    merged = []
    for row in rows:
        if merged and row[1] == merged[-1][1] and row[2] - merged[-1][3] <= merge_gap:
            merged[-1][3] = row[3]
            merged[-1][4] = max(merged[-1][4], row[4])
        else:
            merged.append(row)
    return pd.DataFrame(merged, columns=["video_id", "action", "start_frame", "stop_frame", "score"])
```

### 4.5 Weighted Multiclass Logloss

irregular astronomical light-curve classification and aviation-style tasks require probability hygiene.

```python
import numpy as np

def normalize_probs(p, eps=1e-15):
    p = np.asarray(p, dtype=np.float64)
    p = np.clip(p, eps, 1.0 - eps)
    return p / p.sum(axis=1, keepdims=True)

def weighted_multiclass_logloss(y_true, p, classes, class_weights):
    p = normalize_probs(p)
    y_true = np.asarray(y_true)
    losses = []
    weights = []
    for j, cls in enumerate(classes):
        m = y_true == cls
        if not np.any(m):
            continue
        w = class_weights.get(cls, 1.0)
        losses.append(-w * np.log(p[m, j]).mean())
        weights.append(w)
    return float(np.sum(losses) / np.sum(weights))
```

### 4.6 Submission Validators

Build validators before final training.

```python
def validate_sample_id_order(sample, sub, id_col):
    assert list(sample[id_col]) == list(sub[id_col]), "submission ids do not match sample order"

def validate_probability_rows(df, prob_cols, atol=1e-4):
    p = df[prob_cols].to_numpy()
    assert np.isfinite(p).all()
    assert (p >= 0).all()
    s = p.sum(axis=1)
    assert np.allclose(s, 1.0, atol=atol), (s.min(), s.max())

def validate_segments(df):
    assert (df["start_frame"] < df["stop_frame"]).all()
    key_cols = [c for c in ["video_id", "agent_id", "target_id"] if c in df.columns]
    for _, g in df.sort_values(key_cols + ["start_frame"]).groupby(key_cols):
        prev_stop = None
        for _, r in g.iterrows():
            if prev_stop is not None:
                assert r.start_frame >= prev_stop
            prev_stop = r.stop_frame
```

## 5. Feature Engineering Library

### 5.1 Generic Sensor Features

Use raw channels plus derivative, magnitude, rolling, and frequency summaries.

```python
def add_imu_features(df, group_col="sequence_id"):
    out = df.copy()
    out["acc_mag"] = np.sqrt(out.acc_x ** 2 + out.acc_y ** 2 + out.acc_z ** 2)
    if "rot_w" in out:
        out["rot_angle"] = 2 * np.arccos(out.rot_w.clip(-1, 1))
    for col in ["acc_x", "acc_y", "acc_z", "acc_mag"]:
        out[f"{col}_diff"] = out.groupby(group_col)[col].diff().fillna(0)
        out[f"{col}_absdiff"] = out[f"{col}_diff"].abs()
        out[f"{col}_roll_mean_8"] = (
            out.groupby(group_col)[col]
            .rolling(8, min_periods=1).mean()
            .reset_index(level=0, drop=True)
        )
        out[f"{col}_roll_std_8"] = (
            out.groupby(group_col)[col]
            .rolling(8, min_periods=2).std()
            .reset_index(level=0, drop=True)
            .fillna(0)
        )
    return out
```

Sequence-level aggregation:

```python
def aggregate_sequence_features(df, group_col, value_cols):
    aggs = ["mean", "std", "min", "max", "median", "skew"]
    feat = df.groupby(group_col)[value_cols].agg(aggs)
    feat.columns = [f"{c}_{a}" for c, a in feat.columns]
    return feat.reset_index()
```

### 5.2 tsflex / seglearn-style Window Features

```python
def calculate_tsflex_like_features(df, group_col, time_col, cols, window=5000):
    rows = []
    for gid, g in df.sort_values([group_col, time_col]).groupby(group_col):
        x = g[cols].to_numpy(np.float32)
        for start in range(0, len(x), window):
            w = x[start:start + window]
            if len(w) == 0:
                continue
            row = {group_col: gid, "window_start": int(start)}
            for j, col in enumerate(cols):
                v = w[:, j]
                row[f"{col}_mean"] = float(np.mean(v))
                row[f"{col}_std"] = float(np.std(v))
                row[f"{col}_min"] = float(np.min(v))
                row[f"{col}_max"] = float(np.max(v))
                row[f"{col}_rms"] = float(np.sqrt(np.mean(v * v)))
                row[f"{col}_zcr"] = float(np.mean(np.diff(np.sign(v)) != 0)) if len(v) > 1 else 0.0
            rows.append(row)
    return pd.DataFrame(rows)
```

### 5.3 Keypoint and Multi-Agent Behavior Features

```python
def pairwise_dist(a, b):
    return np.sqrt(((a - b) ** 2).sum(axis=-1))

def add_mouse_pair_features(df, prefix_a="m1", prefix_b="m2", fps=30.0):
    out = df.copy()
    bodyparts = ["nose", "ear_left", "ear_right", "body_center", "tail_base"]
    for bp in bodyparts:
        ax, ay = f"{prefix_a}_{bp}_x", f"{prefix_a}_{bp}_y"
        bx, by = f"{prefix_b}_{bp}_x", f"{prefix_b}_{bp}_y"
        if ax in out and bx in out:
            out[f"dist_{bp}_{bp}"] = np.sqrt((out[ax] - out[bx]) ** 2 + (out[ay] - out[by]) ** 2)
    for prefix in [prefix_a, prefix_b]:
        cx, cy = f"{prefix}_body_center_x", f"{prefix}_body_center_y"
        if cx in out:
            out[f"{prefix}_speed"] = np.sqrt(out[cx].diff().fillna(0) ** 2 + out[cy].diff().fillna(0) ** 2) * fps
            out[f"{prefix}_accel"] = out[f"{prefix}_speed"].diff().fillna(0) * fps
    if f"{prefix_a}_body_center_x" in out and f"{prefix_b}_body_center_x" in out:
        out["center_dist"] = np.sqrt(
            (out[f"{prefix_a}_body_center_x"] - out[f"{prefix_b}_body_center_x"]) ** 2
            + (out[f"{prefix_a}_body_center_y"] - out[f"{prefix_b}_body_center_y"]) ** 2
        )
        out["relative_speed"] = (out[f"{prefix_a}_speed"] - out[f"{prefix_b}_speed"]).abs()
    return out
```

### 5.4 Waveform Peak Features

```python
from scipy.signal import find_peaks, peak_widths, peak_prominences
import numpy as np

def extract_peak_features(signal, height=5.0):
    pos, _ = find_peaks(signal, height=height)
    neg, _ = find_peaks(-signal, height=height)
    peaks = np.concatenate([pos, neg])
    if len(peaks) == 0:
        return {
            "n_peaks": 0, "height_mean": 0.0, "height_max": 0.0,
            "width_mean": 0.0, "prom_mean": 0.0,
        }
    signed = np.concatenate([signal[pos], -signal[neg]])
    widths = np.concatenate([
        peak_widths(signal, pos)[0] if len(pos) else np.array([]),
        peak_widths(-signal, neg)[0] if len(neg) else np.array([]),
    ])
    prom = np.concatenate([
        peak_prominences(signal, pos)[0] if len(pos) else np.array([]),
        peak_prominences(-signal, neg)[0] if len(neg) else np.array([]),
    ])
    return {
        "n_peaks": int(len(peaks)),
        "n_pos_peaks": int(len(pos)),
        "n_neg_peaks": int(len(neg)),
        "height_mean": float(np.mean(np.abs(signed))),
        "height_max": float(np.max(np.abs(signed))),
        "height_std": float(np.std(np.abs(signed))),
        "width_mean": float(widths.mean()) if len(widths) else 0.0,
        "width_max": float(widths.max()) if len(widths) else 0.0,
        "prom_mean": float(prom.mean()) if len(prom) else 0.0,
    }
```

Add phase/measurement aggregation:

```python
def aggregate_measurement_phase_features(meta, signal_features):
    # signal_features has one row per signal_id and includes id_measurement.
    agg = signal_features.groupby("id_measurement").agg(["mean", "max", "std", "sum"])
    agg.columns = [f"{c}_{a}" for c, a in agg.columns]
    return meta[["id_measurement", "target"]].drop_duplicates().merge(
        agg.reset_index(), on="id_measurement", how="left"
    )
```

### 5.5 Wavelet Denoising

```python
import pywt
import numpy as np

def wavelet_denoise(x, wavelet="db4", level=1):
    coeff = pywt.wavedec(x, wavelet, mode="per")
    detail = coeff[-level]
    sigma = np.median(np.abs(detail - np.median(detail))) / 0.6745
    uthresh = sigma * np.sqrt(2 * np.log(len(x)))
    coeff[1:] = [pywt.threshold(c, value=uthresh, mode="hard") for c in coeff[1:]]
    return pywt.waverec(coeff, wavelet, mode="per")[:len(x)]
```

### 5.6 Seizure and EEG Features

```python
from scipy.signal import welch

def bandpower_features(x, fs, bands):
    # x: [time, channels]
    feats = {}
    for ch in range(x.shape[1]):
        f, pxx = welch(x[:, ch], fs=fs, nperseg=min(512, len(x)))
        total = np.trapz(pxx, f) + 1e-12
        for name, lo, hi in bands:
            m = (f >= lo) & (f < hi)
            bp = np.trapz(pxx[m], f[m]) if np.any(m) else 0.0
            feats[f"ch{ch}_{name}_logrel"] = float(np.log1p(bp / total))
        feats[f"ch{ch}_std"] = float(np.std(x[:, ch]))
        feats[f"ch{ch}_skew_proxy"] = float(np.mean(((x[:, ch] - x[:, ch].mean()) / (x[:, ch].std() + 1e-6)) ** 3))
    corr = np.corrcoef(x.T)
    iu = np.triu_indices_from(corr, k=1)
    for i, val in enumerate(corr[iu]):
        feats[f"corr_{i}"] = float(np.nan_to_num(val))
    return feats
```

Aggregate subwindows by max, mean, and std; max pooling helps when predictive patterns may not occupy the full
10-minute clip.

### 5.7 Irregular Light-Curve Features

Irregular light curves need passband-aware aggregation and measurement-error
features.

```python
def make_lightcurve_features(obs, meta):
    df = obs.copy()
    df["flux_ratio_sq"] = (df["flux"] / df["flux_err"].clip(lower=1e-6)) ** 2
    df["flux_by_flux_ratio_sq"] = df["flux"] * df["flux_ratio_sq"]
    aggs = {
        "flux": ["min", "max", "mean", "median", "std", "skew"],
        "flux_err": ["mean", "std", "max"],
        "detected": ["mean", "sum"],
        "flux_ratio_sq": ["sum", "mean"],
        "flux_by_flux_ratio_sq": ["sum", "mean"],
        "mjd": ["min", "max"],
    }
    feat = df.groupby("object_id").agg(aggs)
    feat.columns = [f"{c}_{a}" for c, a in feat.columns]
    feat["mjd_span"] = feat["mjd_max"] - feat["mjd_min"]

    pb = df.groupby(["object_id", "passband"])["flux"].agg(["min", "max", "mean", "std"])
    pb = pb.unstack("passband")
    pb.columns = [f"pb{p}_flux_{a}" for a, p in pb.columns]
    feat = feat.join(pb, how="left")
    feat = feat.reset_index().merge(meta, on="object_id", how="left")
    return feat
```

Add advanced features when compute allows:

- GP predictions and GP fit-quality features
- Bazin/SALT-like curve parameters
- feets/FATS/cesium features
- Lomb-Scargle periodogram features
- peak width, rise/fall, curve angle
- detected-only summaries and SNR windows

### 5.8 User/App Sequence Features

For event-log classification, flatten recent sequences carefully.

```python
def parse_recent_steps(step_list, k=10, sep=" "):
    if isinstance(step_list, str):
        steps = [s for s in step_list.replace(",", " ").split(sep) if s != ""]
    else:
        steps = list(step_list)
    tail = steps[-k:]
    out = {f"last_step_{i}": tail[-1 - i] if i < len(tail) else "__none__" for i in range(k)}
    for n in [1, 2, 3]:
        grams = ["|".join(tail[i:i + n]) for i in range(max(0, len(tail) - n + 1))]
        out[f"n_unique_{n}gram"] = len(set(grams))
    out["n_steps"] = len(steps)
    return out
```

Use fold-safe target encoding only if the validation split groups by user/time.

## 6. Model Families

### 6.1 Engineered-Feature GBDT

Use for:

- HAR sequence baselines
- waveform peak/stat features
- irregular astronomical light-curve classification feature tables
- sepsis/driver/user longitudinal tables
- multi-agent keypoint behavior XGB per-action models

Concrete models:

- `lightgbm.LGBMClassifier`
- `lightgbm.LGBMRegressor` for AP-style multilabel probabilities when source
  code uses regression wrappers
- `xgboost.XGBClassifier`
- `catboost.CatBoostClassifier`
- `sklearn.ensemble.ExtraTreesClassifier`
- `RandomForestClassifier` as a robust baseline

GBDT checklist:

- pass class/sample weights for imbalance
- use early stopping on grouped OOF folds
- save OOF probabilities
- tune thresholds on OOF
- test feature alignment between train/test

### 6.2 WaveNet / TCN for Dense Labels

```python
import torch
from torch import nn

class WaveBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, n_layers=8):
        super().__init__()
        self.proj = nn.Conv1d(in_ch, out_ch, 1)
        self.tanh = nn.ModuleList()
        self.sigmoid = nn.ModuleList()
        self.res = nn.ModuleList()
        for i in range(n_layers):
            d = 2 ** i
            pad = d * (kernel_size - 1) // 2
            self.tanh.append(nn.Conv1d(out_ch, out_ch, kernel_size, padding=pad, dilation=d))
            self.sigmoid.append(nn.Conv1d(out_ch, out_ch, kernel_size, padding=pad, dilation=d))
            self.res.append(nn.Conv1d(out_ch, out_ch, 1))

    def forward(self, x):
        x = self.proj(x)
        out = x
        for t, s, r in zip(self.tanh, self.sigmoid, self.res):
            z = torch.tanh(t(out)) * torch.sigmoid(s(out))
            out = out + r(z)
        return out

class TCNClassifier(nn.Module):
    def __init__(self, n_features, n_classes):
        super().__init__()
        self.net = nn.Sequential(
            WaveBlock(n_features, 32, n_layers=8),
            WaveBlock(32, 64, n_layers=6),
            nn.Conv1d(64, n_classes, 1),
        )

    def forward(self, x):
        # x: [B, T, C]
        return self.net(x.transpose(1, 2)).transpose(1, 2)
```

### 6.3 Conv1D + GRU/LSTM

Use for FOG, sleep, sensor gestures, wearable activity recognition, and dense labels.

```python
class ConvGRUHead(nn.Module):
    def __init__(self, n_features, n_classes, hidden=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_features, hidden, 5, padding=2),
            nn.BatchNorm1d(hidden),
            nn.SiLU(),
            nn.Conv1d(hidden, hidden, 5, padding=2),
            nn.BatchNorm1d(hidden),
            nn.SiLU(),
        )
        self.gru = nn.GRU(hidden, hidden, batch_first=True, bidirectional=True)
        self.out = nn.Linear(hidden * 2, n_classes)

    def forward(self, x):
        h = self.conv(x.transpose(1, 2)).transpose(1, 2)
        h, _ = self.gru(h)
        return self.out(h)
```

For sequence-level classification, pool over time:

```python
def temporal_pool(h, mask=None):
    if mask is None:
        return torch.cat([h.mean(1), h.max(1).values], dim=1)
    m = mask.float().unsqueeze(-1)
    mean = (h * m).sum(1) / m.sum(1).clamp_min(1.0)
    h_masked = h.masked_fill(~mask.unsqueeze(-1), -1e9)
    return torch.cat([mean, h_masked.max(1).values], dim=1)
```

### 6.4 Transformer / Squeezeformer / GNN

Use when:

- actions span medium/long context
- different agents interact
- local Conv1D is not enough
- model diversity matters in an ensemble

Concrete choices:

- Transformer encoder with relative position or learned position
- Squeezeformer for efficient temporal modeling
- GNN/TransformerConv over agents/body parts, then temporal transformer
- phase-aware attention for structured gesture phases
- modality-specific stems before fusion

### 6.5 Multi-Kernel Conv1D for Waveforms

```python
class MultiKernelConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, kernels=(16, 32, 64, 128, 256)):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_ch, out_ch, k, padding=k // 2),
                nn.BatchNorm1d(out_ch),
                nn.SiLU(),
            )
            for k in kernels
        ])
        self.fuse = nn.Sequential(
            nn.Conv1d(out_ch * len(kernels) + in_ch, out_ch, 1),
            nn.BatchNorm1d(out_ch),
            nn.SiLU(),
        )

    def forward(self, x):
        parts = [b(x)[..., :x.shape[-1]] for b in self.branches]
        return self.fuse(torch.cat(parts + [x], dim=1))
```

Use high-pass/band-pass filtering only when source data supports it. For
gravitational-wave detection-like black-hole signals, raw higcell-level multi-label imagingss-filtered Conv1D was stronger than
early CQT/spectrogram attempts; for continuous-wave signals, template/power
summation can beat generic deep learning.

### 6.6 Spectrogram, CQT, CWT, and Images

Use image-like transforms when:

- signal physics is frequency-localized
- pretrained image backbones are feasible
- raw Conv1D and engineered features have been compared

Recipe:

- normalize per detector/channel
- avoid leaking labels through transform parameters fitted on all labels
- try multiple frontends: spectrogram, CQT, CWT, trainable frontend
- use EfficientNet/ResNet/ConvNeXt/Swin only if runtime permits
- use TTA such as time shift, amplitude flip, or window choice only if OOF
  improves

## 7. Subtype Recipes

### 7.1 Fixed Sensor Sequence Classification

Use for fixed-window biological sensor classification, wearable activity recognition, grouped inertial-sensor classification-like tasks.

Pipeline:

1. Parse one sequence id into `[T, C]`.
2. Build subject/group folds.
3. Normalize channels fold-locally or per sequence depending on deployment.
4. Train engineered-feature GBDT.
5. Train 1D CNN + BiGRU or Conv2D + GRU.
6. Save OOF predictions and blend.
7. Tune probability calibration or hard-label thresholds if metric demands.

Models to try:

- DeepConvLSTM
- TinyHAR
- Attend-and-Discriminate
- Conv1D + BiGRU/LSTM
- Conv2D over `[time, sensors]` + GRU
- LightGBM/CatBoost/XGBoost/ExtraTrees
- DAE features + MLP/GRU when legal

Narrow tricks:

- subject-group folds are often more reliable than sequence folds
- swap-noise DAE features worked in fixed-window biological sensor classification with train+test inputs
- high spatial dropout helped some fixed-window sensor neural models
- wearable activity recognition needs participant generalization and sensor-condition augmentation
- fixed-window sensor tabular features: sequence-local lag1, first differences, rolling means
  over short windows, then per-sensor mean/std/skew/kurt/min/max
- grouped inertial-sensor classification should use `group_id` or recording-session-aware validation when
  available; features include Euler-angle differences, total angular velocity,
  total linear acceleration, ratios, FFT power, quantiles, Hilbert/Hann/STA-LTA
  style signal features, and per-series summary dynamics
- grouped inertial-sensor classification orientation-derived or chunk-neighbor features can leak acquisition
  adjacency; treat chaining as a warning

### 7.2 Dense Sensor Event Detection

Use for FOG, ion switching, and dense frame labels.

Pipeline:

1. Identify valid/annotated mask.
2. Split by subject/series/group.
3. Train dense model with masked BCE/CE.
4. Infer with overlapping windows and central-region averaging.
5. Tune thresholds per class on OOF.
6. Convert to required probabilities or event rows.

FOG-specific:

- separate tdcsfog and defog models if source distributions differ
- use `Valid & Task` mask
- normalize per Id for some accelerometer models
- train short windows and infer longer windows when OOF supports it
- use pseudo-label notype data only when allowed and validated
- consider patch target reduction if exact boundaries are not required

dense ion-state classification-specific:

- treat data as discontinuous batches
- remove drift/50Hz noise only from source-supported batches
- use shifted local features and RFC probabilities as stack features
- WaveNet/TCN with macro F1
- beware known private leak and corrupted synthetic CV

### 7.3 Sleep-Style Timestamp Detection

Use when submission contains event timestamps with confidence.

Pipeline:

1. Create soft targets around event times using tolerance windows or Gaussian
   kernels.
2. Train dense model on raw channels, rolling stats, and time features.
3. Clean labels before modeling: remove null events, audit incomplete
   onset/wakeup pairs, and explicitly handle no-event series instead of letting
   paired-interval code silently break.
4. Filter impossible/unannotated/device-off periods if rules and OOF support it.
5. Extract candidate peaks.
6. Train second-stage candidate scorer using first-stage OOF predictions and
   local signal features around peaks.
7. Greedily select events under tolerance constraints.
8. Blend event lists with WBF-like merge or average dense predictions before
   extraction.

Implementation features:

- hour/minute/weekday
- periodicity/device-off flag
- anglez/enmo normalization
- rolling mean/std/max
- long centered rolling stats over 5 min, 30 min, 2 h, and 8 h when the
  sampling rate matches sleep-source assumptions
- absolute angle diff rolling median
- daily chunks with offset
- trim chunk edges by about 30 minutes when OOF supports it

Do not use public-test-specific event counts unless the task rules imply the
same event frequency at deployment.

### 7.4 Multi-Agent Keypoint Action Segments

Use when action intervals are submitted for agent/target pairs.

Pipeline:

1. Normalize FPS and align labels to the same frame base.
2. Do not assume constant FPS; scale rolling windows, shifts, minimum
   durations, and speed thresholds by each video's `frames_per_second` unless
   files prove a fixed frame rate.
3. Build a master skeleton or fallback bodypart mapping.
4. Add missingness flags; interpolate only when OOF improves.
5. Create solo and pair features separately from `behaviors_labeled`.
6. Build lab/video grouped folds and score at frame level before segment
   conversion.
7. Train per-action binary XGB/LightGBM/ExtraTrees and/or temporal NN.
8. Mask invalid actions for each video/lab/agent/target.
9. Tune thresholds per lab/action on OOF.
10. Convert probabilities to segments, smooth if validated, apply minimum
    duration, remove overlaps, and validate schema.

Models:

- XGBoost per action
- LightGBM/CatBoost/ExtraTrees
- LSTM/Squeezeformer/Transformer
- GNN over mice/bodyparts + temporal Squeezeformer
- ST-GCN-like models

Narrow tricks:

- action-rich window sampling for imbalanced actions
- focal loss with per-class weights
- egocentric inter-mouse features
- lab embeddings and action labels as inputs
- flip TTA
- discard 32 frames at window boundaries before averaging
- threshold by lab-action rather than one global threshold
- fallback intervals only for videos with zero predictions, and only if local
  metric improves

Manual correction of private tracking or visual ID swaps is not a general
recipe. Include only a warning if encountered.

### 7.5 Power-Line Fault Detection

Pipeline:

1. Read long signals efficiently by signal id.
2. Extract local maxima/peak features.
3. Aggregate to measurement id and phase-aware features.
4. Train LightGBM with repeated seeds or folds.
5. Tune MCC threshold on OOF predictions.
6. Validate target distribution and phase-triplet consistency.

First model:

- LightGBM on about 10-100 engineered peak features.

Second route:

- DWT denoise or high-pass, BiLSTM attention, CNN/LSTM hybrids.

Do:

- compare denoised and raw features
- threshold by OOF MCC
- aggregate across the three phases
- inspect adversarial train/test separability

Do not:

- assume denoising always helps
- train on signal id when labels are measurement-level inconsistent without
  auditing

### 7.6 Gravitational-Wave Detection

Black-hole style:

- high-pass/band-pass filtered raw waveform
- stacked detector channels
- Conv1D with multi-kernel blocks
- channel shuffle for compatible detectors
- small inter-channel time shifts
- SGD/Nesterov can outperform AdamW for simple no-synthetic Conv1D
- synthetic pretraining only if simulated noise and signal match enough
- separate encoders per detector can help when noise distributions differ

Continuous-wave style:

- do not default to CNN
- template/power summation, StackSlide, matched filters, random search over
  signal parameters, noise normalization, and sinc-kernel refinement are
  first-class routes
- generated data and robust noise simulation are required for CNNs
- validation may be weak; keep several random-seed audits

### 7.7 EEG Seizure Prediction

Pipeline:

1. Respect patient and one-hour segment integrity.
2. Split 10-minute clips into 20s/30s subwindows.
3. Extract frequency-domain and cross-channel features.
4. Train subject-specific and pooled models; compare both.
5. Aggregate subwindow predictions by max, mean, and std; select by OOF.
6. Blend ranked predictions to reduce calibration overfit.

Feature families:

- Welch PSD band powers
- log relative power
- entropy and spectral edge
- AR error coefficients
- fractal/Hurst/Hjorth
- wavelet energy
- channel correlation/coherence/eigenvalues
- Riemannian tangent-space covariance features

Models:

- XGBoost bags
- KNN
- logistic regression / GLM / elastic net
- linear SVM
- RUS boosted trees
- shallow autoencoder features

### 7.8 Clinical Early Warning

Route often combines This playbook with tabular classification.

Pipeline:

1. Define prediction timestamp and label lead time.
2. Use patient/admission group folds.
3. Build only features available at or before the timestamp.
4. Add rolling 3h/6h stats over vitals/labs.
5. Add medication/route TF-IDF or counts per person-time.
6. Use CatBoost/LightGBM/XGBoost with class weights.
7. Optimize PR-AUC or official metric with OOF probabilities.

Do not:

- use future labs, medications, or post-sepsis rows
- downsample/upsample blindly
- trust public LB without patient-group CV

Useful models:

- CatBoost with `auto_class_weights="Balanced"` or manual class weights
- LightGBM DART/GBDT ensemble
- XGBoost
- HistGradientBoosting
- logistic/Ridge baselines

### 7.9 Wearable Sensor Gesture and Sensor Gesture Classification

Use when the task is one label per sequence from IMU, thermal, ToF, or similar
wearable sensors, especially when the score distinguishes target/non-target and
gesture class quality.

Pipeline:

1. Implement the exact hierarchical metric or a faithful local proxy. Do not
   optimize plain accuracy if the official score averages binary
   target-vs-non-target F1 with macro gesture F1.
2. Build subject-group folds.
3. Group by `sequence_id`; preserve one label per sequence and pad/truncate to
   a fixed sequence length.
4. Create IMU features: acceleration magnitude, gravity-removed linear
   acceleration, jerk, quaternion 6D or quaternion-derived angular velocity,
   angular distance, and quaternion diffs.
5. Create THM/TOF features: per-sensor mean/std/min/max, cross-sensor range and
   std, missingness flags, and gated branches for missing modalities.
6. Mirror left-handed sequences to the right-handed coordinate convention when
   metadata supports it.
7. Train modality-stem models: IMU residual-SE CNN, TOF/thermal Conv1D or 2D
   ToF grid CNN, then BiGRU/BiLSTM/attention or Transformer/BERT-style head.
8. Add robustness by masking THM/TOF during training so IMU-only hidden tests
   are not catastrophic.
9. Blend IMU-only and all-feature models with OOF-tuned weights.
10. Apply only structure-backed postprocess, such as per-subject no-repeat
    assignment, when the data design explicitly implies it.

Concrete models:

- Residual SE-CNN + attention
- CNN-GRU / CNN-LSTM
- Gated GRU
- BERT/Transformer encoder over CNN tokens
- 3D CNN branch for ToF grids
- LightGBM/CatBoost on sequence summary features as diversity

Augmentation:

- mixup, preferably phase-aligned if phases are known
- time shift and time stretch
- random feature masking
- Gaussian noise and scaling
- handedness mirroring

### 7.10 Irregular Light Curves

Pipeline:

1. Parse metadata and observations.
2. Preserve negative flux and measurement errors.
3. Split galactic/extragalactic using redshift zero or task-specific metadata.
4. Build passband-wise features, detected-only features, and SNR features.
5. Add GP/parametric curve features when feasible.
6. Augment/degrade training objects to match test cadence/noise; keep siblings
   in the same fold.
7. Train LightGBM/CatBoost and RNN/NN branches.
8. Optimize weighted logloss with clipping and row normalization.
9. Treat class-99 as unknown/open-set calibration, not LB probing.

Models:

- LightGBM on GP/raw features
- CatBoost with large feature sets
- BiGRU/attention with metadata branch
- shallow stackers or class-wise weighted averages

Narrow but valuable:

- GP with Matern kernel for uncertainty-aware light-curve features
- hostgal_specz pseudo-feature model if spec-z exists for some test objects and
  rules permit using metadata from test
- log averaging of probabilities
- class-wise ensemble weights optimized on OOF
- sample weighting by train/test similarity only if OOF supports it

### 7.11 Market Reversal and User-Event Boundaries

Market reversal:

User next-step:

- extract last-N steps, n-grams, transition counts, time gaps
- do not treat anonymized version numbers as ordinal unless documentation says
  so
- AutoGluon/tree ensembles can be strong after flattening
- use user/time splits if repeated users or chronological logs exist

Distracted driving:

- latent driver/session identity may be very high signal but can be borderline
  if recovered indirectly
- adversarial train/test probabilities are a drift diagnostic first and a
  feature only after rules and CV support it

## 8. Augmentation, Pseudo Labels, and Transductive Learning

Use augmentation when it matches real invariances:

- wearable: jitter, scaling, time warp, 3D rotation, sensor dropout
- wearable gestures: handedness mirroring, time shift, time stretch, mixup aligned
  by behavior phase
- multi-agent keypoint behavior: flip, Gaussian noise, keypoint dropout, scale noise
- gravitational-wave detection: channel shuffle, small time shifts, amplitude flip, generated
  noise+signal pretraining
- irregular astronomical light-curve classification: sample flux from flux_err, drop observations, degrade cadence,
  simulate redshift/time dilation/seasonal gaps

Pseudo labels:

- Use soft pseudo labels when probabilities are uncertain.
- Use hard labels only for very high-confidence, well-validated cases.
- Keep pseudo-label models out of the fold that generated the labels when
  evaluating.
- In online/API settings, test-time adaptation must be rule-legal and robust to
  early wrong pseudo labels.

Transductive DAE/self-supervised:

- Legal only if the competition permits using unlabeled test inputs.
- Use reconstruction/noise prediction, not target leakage.
- Validate downstream OOF and avoid overfitting to public test quirks.

## 9. Ensembling

Strong time-series classification systems often combine diversity plus OOF
postprocess, not just one bigger model.

Useful diversity axes:

- engineered GBDT vs sequence NN
- raw waveform Conv1D vs spectrogram/CQT/CWT image CNN
- per-source models vs pooled model
- different window lengths and strides
- different seeds and folds
- per-class/per-target loss weights
- per-lab/per-action models
- separate galactic/extragalactic or tdcsfog/defog branches
- rank averaging when calibration differs
- logistic/Ridge/LGBM stackers on OOF probabilities

Rules:

- Save OOF and test predictions for every model.
- Tune ensemble weights on OOF only.
- For logloss, average logits or log probabilities may beat arithmetic
  probability averaging; verify locally.
- For F1/MCC/AP, tune thresholds after ensembling, not before.
- Avoid meta-models that train on OOF labels and early-stop on the same OOF
  rows without a nested validation.

## 10. Anti-Patterns and Route-Out Details

### 10.1 Public-LB Probing

Better substitute:

- open-set score from max known-class confidence
- entropy/margin features
- validation-calibrated unknown prior
- OOF class-wise calibration

### 10.2 Hidden or Manual Test Corrections

### 10.3 Random Row Splits

Random row splits inflate CV for:

- overlapping windows
- repeated subjects
- video frames
- adjacent app events
- patient timelines
- market ticks
- phase-triplet waveforms

Always ask: "Could the same underlying entity appear in both train and
validation?"

### 10.4 interactive game-policy tasks

Route to game-agent policy. Transfer only:

- online adaptation
- multi-armed bandit over candidate policies
- randomized action selection
- recent and long-history features
- self-play as execution check

Do not use interactive game-policy tasks public-agent copying or ladder resubmission strategy for
ordinary supervised TSC.

## 11. Practical Starting Templates

### 11.1 Sequence-Level Classifier

```python
def train_sequence_stack(train_seq, y, groups, test_seq, make_features, train_nn):
    X = make_features(train_seq)
    X_test = make_features(test_seq)
    folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=42).split(X, y, groups))

    oof_gbdt = np.zeros((len(X), len(np.unique(y))))
    test_gbdt = np.zeros((len(X_test), len(np.unique(y))))
    for tr, va in folds:
        model = make_lgbm_classifier()
        model.fit(X.iloc[tr], y[tr], eval_set=[(X.iloc[va], y[va])])
        oof_gbdt[va] = model.predict_proba(X.iloc[va])
        test_gbdt += model.predict_proba(X_test) / len(folds)

    oof_nn, test_nn = train_nn(train_seq, y, groups, test_seq, folds)
    return 0.5 * oof_gbdt + 0.5 * oof_nn, 0.5 * test_gbdt + 0.5 * test_nn
```

### 11.2 Dense Event Model

```python
def dense_event_pipeline(train_df, labels, test_df, group_col):
    folds = make_group_folds(train_df, labels["any_event"].values, group_col)
    oof = np.zeros((len(train_df), labels.shape[1]), dtype=np.float32)
    test_pred_parts = []
    for tr, va in folds:
        model = train_dense_model(train_df.iloc[tr], labels.iloc[tr],
                                  valid_df=train_df.iloc[va],
                                  valid_y=labels.iloc[va])
        oof[va] = predict_dense(model, train_df.iloc[va])
        test_pred_parts.append(predict_dense(model, test_df))

    event_params = tune_event_extractor_on_oof(train_df, labels, oof)
    test_dense = np.mean(test_pred_parts, axis=0)
    submission = dense_to_submission(test_df, test_dense, event_params)
    return oof, test_dense, submission
```

### 11.3 Event Segment Model

```python
def segment_pipeline(train_frames, labels, test_frames, groups, actions):
    folds = make_group_folds(train_frames, labels["has_action"].values, "video_id")
    oof = {a: np.zeros(len(train_frames), dtype=np.float32) for a in actions}
    test = {a: [] for a in actions}
    for action in actions:
        y = make_binary_action_target(labels, action)
        for tr, va in folds:
            model = train_action_model(train_frames.iloc[tr], y[tr])
            oof[action][va] = model.predict_proba(train_frames.iloc[va])[:, 1]
            test[action].append(model.predict_proba(test_frames)[:, 1])
    thresholds = tune_action_thresholds(oof, labels, group_key=train_frames["lab_id"])
    test_prob = np.column_stack([np.mean(test[a], axis=0) for a in actions])
    return probabilities_to_segments_for_all_videos(test_frames, test_prob, actions, thresholds)
```
