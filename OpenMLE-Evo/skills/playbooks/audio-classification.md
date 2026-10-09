# Audio Classification Category Playbook

This playbook is the full method library for Kaggle-style `audio_classification`
tasks. It is intended as direct reading material for a skill-generating agent:
after reading a task description, the agent should route the task, select the
right validation strategy, then extract only the applicable recipes.

Do not use this playbook to justify leaderboard probing, public-test label
inference, hidden labels, manual test labeling, disallowed external data, or ASR
decoding for a classification task.

## 1. Task Routing

### 1.1 Clip-level single-label audio tagging

Use when each audio clip has exactly one class, such as sound type, instrument,
spoken keyword, animal call category, machine state, or environmental event.

Default plan:

- stratified folds unless source/user/device/group leakage exists
- log-mel spectrogram frontend
- timm CNN classifier or simple SED head
- `CrossEntropyLoss` for mutually exclusive labels
- probability averaging across folds
- top-k construction if metric is MAP@k

### 1.2 Multi-label clip or recording tagging

Use when one clip can contain several simultaneous labels. This includes
multi-species audio, overlapping environmental events, tags, and pathologies.

Default plan:

- iterative multilabel folds when available; otherwise stratify on coarse label
  count and repair rare classes manually
- `BCEWithLogitsLoss`, focal BCE, or asymmetric loss
- per-class prevalence audit
- OOF threshold tuning only for metrics that require hard labels
- probability or rank averaging for AUC/LRAP/mAP-style metrics

Do not force softmax if more than one class can be true.

### 1.3 Long soundscape window classification

Use when the submission rows are file/time-window ids and the test audio is a
long recording split into fixed windows.

Default plan:

- validation must reflect the soundscape domain, not only random training clips
- train on longer random clips if labels are weak
- infer on official fixed windows, optionally with overlap or shifted-window TTA
- use SED models when framewise outputs are useful
- smooth adjacent time windows only after validation
- keep runtime in mind early because soundscape inference multiplies model cost

### 1.4 Bioacoustic species classification

Use for birds, frogs, insects, mammals, or other species calls, especially when
training clips are from curated repositories while test is passive acoustic
monitoring soundscapes.

Default plan:

- log-mel CNN/SED with EfficientNet/EfficientNetV2/NFNet/RegNet/ConvNeXt
- source/site/date/location-aware validation whenever available
- long-train short-infer strategy
- soundscape background mixing
- soft secondary labels when labels include primary/secondary species
- target-domain pseudo labels if allowed unlabeled soundscapes exist

Common risk: a random clip fold can look good while failing on test soundscapes.

### 1.5 Sparse TP/FP sound event detection

Use when the training set provides event time/frequency bounds, true positives,
false positives, or incomplete annotations.

Default plan:

- split by recording id before expanding annotations into crops
- crop around known TP/FP events for supervised windows
- use SED or temporal CNN outputs
- ignore unknown class/time regions in the loss
- infer with overlapping windows and aggregate by max or top-k mean
- use pseudo labels to fill ignored regions only after a good teacher exists

### 1.6 Top-k audio tagging

Use when the metric asks for ranked labels such as MAP@3.

Default plan:

- train with the natural target objective, usually CE for single-label or BCE
  for multi-label
- average probabilities across folds, durations, and models before top-k
  selection
- optimize ranks, not thresholds
- construct the exact submission string/order required by the metric

### 1.7 Speech classification vs ASR route-out

Use this playbook for speech labels such as emotion, language, command, accent,
speaker class, or pathology class.

## 2. Validation And Leakage

### 2.1 First define the validation unit

The validation unit must match the scored unit:

- clip tasks: validate clips, grouped by source if needed
- recording tasks: validate recordings, not crops
- soundscape tasks: validate windows and aggregate by soundscape file
- event tasks: split recordings, then generate event windows within each fold
- top-k tasks: validate final top-k ranking from probabilities
- speech classification: group by speaker/session/source whenever possible

Save OOF predictions at the same unit as the metric. If you train on crops,
also save an OOF aggregation at the official submission unit.

### 2.2 Leakage axes to audit

Before training, inspect:

- duplicate or near-duplicate audio files
- same recording split into multiple rows
- same speaker/user/device/site/date/source in train and validation
- same soundscape file split across folds
- generated crops from one original clip crossing folds
- species taxonomy or location metadata that encodes target distribution
- external labels/pretrained systems that may not be allowed

For event annotations, never split `train_tp.csv` rows directly if multiple
annotations belong to the same recording.

### 2.3 When random CV is suspect

Random folds are suspect when:

- train clips are curated but test is noisy soundscape audio
- positives are weak clip labels rather than localized events
- label quality differs by source
- the metric scores time windows but labels are file-level
- unlabeled classes are common
- audio duration distribution differs between train and test

Use source/site/date holdout, soundscape-like validation, synthetic target-domain
validation, or past-year holdout when available. If no reliable split exists,
keep model choices conservative and prefer techniques that improve robustness
across plausible splits.

### 2.4 Multi-label fold audit

For multi-label tasks, every fold should have positive support for important
classes. If not, treat per-class thresholding and rare-class model selection as
unreliable.

Useful fold report:

```python
def multilabel_fold_report(y, folds):
    report = []
    for fold in sorted(set(folds)):
        yf = y[folds == fold]
        pos = yf.sum(axis=0)
        report.append({
            "fold": int(fold),
            "rows": int(len(yf)),
            "empty_classes": int((pos == 0).sum()),
            "min_pos": int(pos.min()),
            "median_pos": float(np.median(pos)),
        })
    return report
```

Audio
multi-label failures are often fold-support failures.

### 2.5 Sparse TP/FP validation

For sparse event labels:

- validate by recording id
- evaluate known TP/FP windows separately from recording-level predictions
- keep an OOF prediction table for every known labeled segment
- keep a separate recording/window aggregate if the submission is recording-level
- do not punish the model for unlabeled classes unless the dataset explicitly
  confirms absence

### 2.6 Metric alignment

Match output behavior to the metric:

- Logloss: calibrated probabilities, light clipping, probability averaging.
- Accuracy: top-1 class, but still use probabilities for ensembling.
- F1/F2: tune thresholds on OOF; per-class thresholds only with enough support.
- AUC/LRAP/cMAP/mAP: optimize rank quality; probability monotonic transforms can
  help or hurt, so validate.
- MAP@k: average probabilities first, then choose top-k labels.
- Soundscape window metrics: align row ids, window end times, and smoothing with
  the official scoring unit.

## 3. Data Audit And Label Handling

### 3.1 Audio inventory

Build a small audit table before modeling:

- path, duration, sample rate, channels
- RMS/energy/silence fraction
- label count and rare classes
- source/site/date/device/speaker metadata
- train/test duration distributions
- corrupt files, NaNs, all-zero clips
- duplicate hashes or highly similar waveforms

### 3.2 Weak primary and secondary labels

For weak labels:

- use longer training crops to increase the chance that the labeled sound is
  present
- use secondary labels as soft positives when task semantics support them
- do not label every crop from a weak file as strongly positive without checking
  whether the event is likely present
- mine missing labels from OOF predictions only after the first model is stable

For bioacoustics, primary/secondary labels are often incomplete. Soft targets and
label cleanup can beat larger backbones.

### 3.3 Noisy verified/unverified labels

If the dataset has verified and unverified labels:

- train a first model on verified labels or with higher weights for verified
  examples
- add unverified/noisy labels with lower weight or later rounds
- inspect high-confidence OOF disagreements
- use label smoothing or soft targets instead of hard relabeling when uncertain

Treat verified and unverified labels differently in weak-label audio tasks.

### 3.4 Denoising and enhancement

Denoising is not a default improvement.

- Test enhancement per domain and per class.
- Keep original-audio fallback models.
- Denoisers can remove target events, especially weak or distant calls.
- Enhancement can help when train/test share a known noise mechanism, but it can
  also create domain shift.

### 3.5 Metadata and taxonomy

Metadata can help when it is available at inference:

- site/location/date/time: species priors and seasonality
- device/source: domain shift and calibration
- duration/RMS: quality and silence
- taxonomy: class hierarchy or similar species groups
- speaker/session: speech classification grouping

Use tabular methods for GBDT metadata models and OOF stacking. In this audio
playbook, metadata is secondary unless the task description makes it central.

Do not use metadata that only exists for training or that leaks the target
through collection procedure.

## 4. Audio Frontend And Windowing

### 4.1 Sample rate defaults

Common strong defaults:

- `32000` Hz for bird/environmental/bioacoustic soundscape tasks.
- `16000` Hz for speech classification or keyword spotting.
- Keep native high sample rates only when target events need high frequencies
  and compute allows.

Always resample consistently across train, validation, and test.

### 4.2 Crop, pad, and duration policy

Use fixed-length waveform segments before spectrogram extraction. Random crop in
training, deterministic crop/window at inference.

```python
def random_crop_or_pad(audio, input_length):
    if len(audio) > input_length:
        max_offset = len(audio) - input_length
        offset = np.random.randint(max_offset)
        audio = audio[offset:offset + input_length]
    else:
        max_offset = max(input_length - len(audio), 1)
        offset = np.random.randint(max_offset)
        audio = np.pad(audio, (offset, input_length - len(audio) - offset),
                       mode="constant")
    return audio
```

Adaptations:

- For evaluation, replace random offset with official windows or center crop.
- For very short bioacoustic clips, cyclic padding can preserve signal better
  than silence padding.
- For weak clip labels, train on longer windows than inference if it improves
  label presence.

### 4.3 Log-mel spectrogram baseline

```python
def audio2melspec(audio_data, cfg):
    if np.isnan(audio_data).any():
        mean_signal = np.nanmean(audio_data)
        audio_data = np.nan_to_num(audio_data, nan=mean_signal)

    mel_spec = librosa.feature.melspectrogram(
        y=audio_data,
        sr=cfg.FS,
        n_fft=cfg.N_FFT,
        hop_length=cfg.HOP_LENGTH,
        n_mels=cfg.N_MELS,
        fmin=cfg.FMIN,
        fmax=cfg.FMAX,
        power=2.0,
        pad_mode="reflect",
        norm="slaney",
        htk=True,
        center=True,
    )
    mel_db = librosa.power_to_db(mel_spec, ref=np.max)
    mel_norm = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8)
    return mel_norm.astype(np.float32)

def process_audio_segment(audio_data, cfg):
    target_len = cfg.FS * cfg.WINDOW_SIZE
    if len(audio_data) < target_len:
        audio_data = np.pad(audio_data, (0, target_len - len(audio_data)))
    mel = audio2melspec(audio_data, cfg)
    if mel.shape != cfg.TARGET_SHAPE:
        mel = cv2.resize(mel, cfg.TARGET_SHAPE, interpolation=cv2.INTER_LINEAR)
    return mel.astype(np.float32)
```

### 4.4 Parameter atlas

Useful starting ranges:

| Task family | Sample rate | Duration | Mel bins | FFT/hop | Notes |
| --- | --- | --- | --- | --- | --- |
| Generic clip tagging | 16-32 kHz | 2-10s | 64-128 | 1024/320 or 2048/512 | resize to 224-384 width |
| bioacoustic soundscape | 32 kHz | train 10-30s, infer 5s | 128-224 | 1024-4096, hop 500-1252 | fmax 15-16 kHz |
| sparse sound-event detection/event SED | 32 kHz | 3-6s crops, 60s recordings | 128-384 | 2048/512 common | crop around TP/FP |
| environmental audio tagging-like | 16-44.1 kHz | variable; sample several lengths | 64-128 | task-dependent | top-k from averaged probabilities |
| Speech classification | 16 kHz | 1-20s | 64-128 | 400/160 or 1024/320 | consider speech encoders |

Vary mel parameters across ensemble members only after a stable baseline exists.
Using one shared mel configuration can be better under strict CPU inference.

### 4.5 Long-recording inference

For soundscapes, load a file once, slice official windows, and preserve row ids.

```python
audio_data, _ = librosa.load(audio_path, sr=cfg.FS)
total_segments = int(len(audio_data) / (cfg.FS * cfg.WINDOW_SIZE))

for segment_idx in range(total_segments):
    start = segment_idx * cfg.FS * cfg.WINDOW_SIZE
    end = start + cfg.FS * cfg.WINDOW_SIZE
    segment_audio = audio_data[start:end]

    end_time_sec = (segment_idx + 1) * cfg.WINDOW_SIZE
    row_id = f"{soundscape_id}_{end_time_sec}"

    mel = process_audio_segment(segment_audio, cfg)
    x = torch.tensor(mel, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(cfg.device)

    segment_preds = []
    for model in models:
        with torch.no_grad():
            probs = torch.sigmoid(model(x)).cpu().numpy().squeeze()
            segment_preds.append(probs)
    final_preds = np.mean(segment_preds, axis=0)
```

Adaptations:

- Add overlap or shifted-window TTA when runtime permits.
- Use framewise SED outputs when the model emits time-resolved predictions.
- Check whether the official row id uses window start or window end time.

### 4.6 Tensor shapes

Common shapes:

- waveform model input: `[batch, samples]` or `[batch, 1, samples]`
- mel image input: `[batch, 1, mel_bins, time]`
- ImageNet CNN input: expand to `[batch, 3, mel_bins, time]`
- SED temporal features: `[batch, channels, time]`
- framewise predictions: `[batch, frames, classes]`

Normalize mel spectrograms per segment or with train-set statistics. For
ImageNet-pretrained CNNs, either use one-channel input with modified first conv
or replicate to three channels and apply appropriate normalization.

## 5. Model Families

### 5.1 Plain log-mel CNN classifier

Use for clip-level single-label/multi-label tasks and as a fast baseline for
soundscapes.

```python
class AudioClassifier(nn.Module):
    def __init__(self, model_name, num_classes, in_channels=1, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            in_chans=in_channels,
            num_classes=0,
        )
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(self.backbone.num_features, num_classes)

    def forward(self, x):
        features = self.backbone.forward_features(x)
        if len(features.shape) == 4:
            features = self.pooling(features).flatten(1)
        return self.classifier(features)
```

First backbones:

- `tf_efficientnet_b0_ns`: fastest strong default.
- `tf_efficientnet_b3_ns`: stronger single model if compute allows.
- `tf_efficientnetv2_s`: strong modern EfficientNetV2 option.
- `regnety_008`/`regnety_016`: efficient diversity.
- `efficientvit_b0` or `mnasnet_100`: runtime-constrained soundscape inference.

### 5.2 SED CNN with temporal pooling

Use when labels are weak, events are localized in time, or inference needs
window/frame aggregation.

Core idea:

1. Convert waveform to spectrogram/log-mel.
2. Run a 2D CNN encoder over mel image.
3. Average or pool over frequency.
4. Keep time dimension.
5. Apply max+average temporal pooling or attention.
6. Emit clipwise and framewise/segmentwise outputs.

```python
class AudioSEDModel(nn.Module):
    def __init__(self, encoder, sample_rate, window_size, hop_size,
                 mel_bins, fmin, fmax, classes_num):
        super().__init__()
        self.interpolate_ratio = 30
        self.spectrogram_extractor = Spectrogram(
            n_fft=window_size, hop_length=hop_size, win_length=window_size,
            window="hann", center=True, pad_mode="reflect",
            freeze_parameters=True,
        )
        self.logmel_extractor = LogmelFilterBank(
            sr=sample_rate, n_fft=window_size, n_mels=mel_bins,
            fmin=fmin, fmax=fmax, ref=1.0, amin=1e-10, top_db=None,
            freeze_parameters=True,
        )
        self.spec_augmenter = SpecAugmentation(
            time_drop_width=64, time_stripes_num=2,
            freq_drop_width=8, freq_stripes_num=2,
        )
        self.encoder = encoder_params[encoder]["init_op"]()
        self.fc1 = nn.Linear(encoder_params[encoder]["features"], 1024)
        self.att_block = AttBlock(1024, classes_num, activation="sigmoid")
        self.bn0 = nn.BatchNorm2d(mel_bins)

    def forward(self, input, mixup_lambda=None):
        x = self.spectrogram_extractor(input)
        x = self.logmel_extractor(x)
        frames_num = x.shape[2]

        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)
        if self.training:
            x = self.spec_augmenter(x)
        if self.training and mixup_lambda is not None:
            x = do_mixup(x, mixup_lambda)

        x = x.expand(x.shape[0], 3, x.shape[2], x.shape[3])
        x = self.encoder.forward_features(x)
        x = torch.mean(x, dim=3)
        x = F.max_pool1d(x, 3, stride=1, padding=1) + F.avg_pool1d(x, 3, stride=1, padding=1)

        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.fc1(x.transpose(1, 2))).transpose(1, 2)
        clipwise, norm_att, segmentwise = self.att_block(x)
        segmentwise = segmentwise.transpose(1, 2)
        framewise = interpolate(segmentwise, self.interpolate_ratio)
        framewise = pad_framewise_output(framewise, frames_num)
        return {"clipwise_output": clipwise, "framewise_output": framewise}
```

Use a TimmSED-style implementation when using timm backbones. The important part
is not this exact class; it is preserving temporal outputs.

### 5.3 Framewise interpolation and padding

```python
def interpolate(x: torch.Tensor, ratio: int):
    batch_size, time_steps, classes_num = x.shape
    upsampled = x[:, :, None, :].repeat(1, 1, ratio, 1)
    return upsampled.reshape(batch_size, time_steps * ratio, classes_num)

def pad_framewise_output(framewise_output: torch.Tensor, frames_num: int):
    pad = framewise_output[:, -1:, :].repeat(
        1, frames_num - framewise_output.shape[1], 1
    )
    return torch.cat((framewise_output, pad), dim=1)
```

Use this when the CNN temporal stride shortens the time axis and you need
predictions aligned back to input frames.

### 5.4 Model atlas

Recommended model families by route:

| Model family | Concrete names | Best use |
| --- | --- | --- |
| Fast EfficientNet | `tf_efficientnet_b0_ns`, `tf_efficientnet_b3_ns`, `tf_efficientnet_b4_ns` | default CNN/SED, strong speed-quality |
| EfficientNetV2 | `tf_efficientnetv2_s`, `tf_efficientnetv2_b3`, ImageNet-21K variants | strong single models and ensembles |
| NFNet | `eca_nfnet_l0` | high-quality bioacoustic soundscape/sparse sound-event detection-style SED component |
| RegNet | `regnety_008`, `regnety_016` | efficient soundscape ensemble diversity |
| ConvNeXt | `convnext_tiny.fb_in22k`, `convnextv2_tiny` | modern ConvNet diversity |
| Lightweight CPU | `efficientvit_b0`, `mnasnet_100`, `spnasnet_100`, `mobilenetv3_large_100` | strict inference budgets |
| ResNet family | `resnet18/34/50`, `resnext50`, `seresnext26d_32x4d`, `resnest50` | reliable diverse baselines |
| Dense/PANN | `densenet121/161`, PANN `CNN14`/`CNN10` | AudioSet/general environmental audio |
| Audio transformers | `CED-small/base`, `BEATs`, `HTS-AT`, `PaSST`, `AST` | general audio, speech labels, late ensemble diversity |
| Speech encoders | `WavLM`, `Whisper` encoder | speech classification only; route out transcription |

Practical default for a first solution:

1. `tf_efficientnet_b0_ns` or `tf_efficientnetv2_s` plain CNN/SED.
2. Add `tf_efficientnet_b3_ns` and `eca_nfnet_l0`.
3. Add one speed-diverse model such as `regnety_008`, `efficientvit_b0`, or
   `mnasnet_100`.
4. Add foundation/audio transformer only after CNN/SED OOF is stable.

### 5.5 Foundation models and pretrained audio encoders

Use foundation models when:

- the task is general audio with limited labels
- speech labels may benefit from pretrained speech representations
- the compute budget allows fine-tuning or embedding extraction
- CNN/SED OOF plateaus and diversity is needed

Use them cautiously when:

- inference is CPU-only or strict
- sample rate/duration mismatches are painful
- test domain is narrow bioacoustic soundscape where log-mel CNNs are already
  proven
- external pretrained weights are disallowed

Strong candidates from general model knowledge:

- `CED-small`, `CED-base`: strong AudioSet-style audio classifiers.
- `BEATs`: strong self-supervised audio representation.
- `HTS-AT`, `PaSST`, `AST`: transformer-style spectrogram/audio models.
- `WavLM`, `Whisper` encoder: speech classification features.

### 5.6 Runtime and export

Audio inference can be dominated by frontend computation. Under strict runtime:

- use one shared mel configuration across models
- cache mels in RAM when allowed
- prefer `tf_efficientnet_b0_ns`, `regnety_008`, `efficientvit_b0`, `mnasnet_100`
- export to ONNX/OpenVINO only after numerical checks
- reduce folds before reducing window coverage if the metric depends on all
  windows
- batch windows from the same file

## 6. Objectives, Targets, And Losses

### 6.1 Single-label losses

Use `CrossEntropyLoss` when labels are mutually exclusive. For MAP@k tasks,
train CE, average class probabilities, then select the top labels.

Use label smoothing when labels are noisy, but be careful with rare classes and
class-balanced metrics.

### 6.2 Multi-label losses

Use:

- `BCEWithLogitsLoss` as the robust baseline
- focal BCE for rare positive events
- asymmetric loss when negative classes dominate
- per-class weights only if validation supports them

Do not use softmax when multiple labels can be true.

### 6.3 Masked or agnostic loss for sparse labels

Implementation Pattern, rewritten into a safer PyTorch form:

```python
def masked_bce_with_logits(logits, targets, ignore_value=0.5):
    known = targets != ignore_value
    if known.sum() == 0:
        return logits.sum() * 0.0
    return F.binary_cross_entropy_with_logits(logits[known], targets[known])
```

```python
def masked_bce_from_concat(logits, y):
    # y[:, :num_classes] contains 0/1 targets.
    # y[:, num_classes:] contains 0/1 mask values.
    num_classes = logits.shape[1]
    targets = y[:, :num_classes].to(logits.dtype)
    mask = y[:, num_classes:].to(logits.dtype)
    loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    return (loss * mask).sum() / mask.sum().clamp_min(1.0)
```

Do not set every unknown class to zero. That teaches false negatives.

### 6.4 SED two-way supervision

For SED models, combine clipwise and framewise/segmentwise losses when available:

- clipwise loss on the official label
- framewise max or attention output loss for localization
- lower weight on pseudo/framewise labels if noisy

This is useful for weak labels because framewise predictions expose where the
model thinks the event occurs.

### 6.5 Class imbalance and sampling

Useful tools:

- class-balanced sampling for rare species/classes
- per-class positive weights with careful OOF checks
- focal/asymmetric loss
- oversample rare classes but avoid repeating identical noisy clips too often
- for pseudo labels, sample by confidence or max-label sum

Do not let rare-class fixes destroy calibration for logloss or rank metrics.

## 7. Augmentation And Pseudo Labels

### 7.1 Mixup

Mixup is one of the most reusable audio augmentations because it matches real
overlapping sounds and supports soft labels.

```python
def mixup(data, targets, alpha):
    indices = torch.randperm(data.size(0))
    data2 = data[indices]
    targets2 = targets[indices]
    lam = torch.FloatTensor([np.random.beta(alpha, alpha)]).to(data.device)
    data = data * lam + data2 * (1 - lam)
    targets = targets * lam + targets2 * (1 - lam)
    return data, targets
```

```python
def do_mixup(x: torch.Tensor, mixup_lambda: torch.Tensor):
    out = (x[0::2].transpose(0, -1) * mixup_lambda[0::2] +
           x[1::2].transpose(0, -1) * mixup_lambda[1::2]).transpose(0, -1)
    return out

class Mixup:
    def __init__(self, mixup_alpha, random_seed=1234):
        self.mixup_alpha = mixup_alpha
        self.random_state = np.random.RandomState(random_seed)

    def get_lambda(self, batch_size):
        lambdas = []
        for _ in range(0, batch_size, 2):
            lam = self.random_state.beta(self.mixup_alpha, self.mixup_alpha, 1)[0]
            lambdas.extend([lam, 1.0 - lam])
        return torch.from_numpy(np.array(lambdas, dtype=np.float32))
```

Use mixup for:

- multi-label audio
- weak labels
- pseudo-labeled soundscapes
- background mixing

Avoid mixup when the metric requires exact hard labels and validation shows
calibration damage.

### 7.2 SpecAugment and spectrogram erasing

SpecAugment masks time/frequency bands. It is usually useful, but can erase
narrow target events. Tune strength:

- smaller frequency masks for narrow bird/frog calls
- smaller time masks for short keywords
- stronger masks for broad environmental scenes
- disable or reduce for rare classes if OOF recall drops

Do not use image augmentations that ignore audio axes, such as arbitrary
rotation or vertical flip, unless the transformed spectrogram still represents a
valid audio signal for the task.

### 7.3 Waveform and domain augmentations

High-value augmentations:

- random gain/volume
- Gaussian or pink noise by SNR
- background noise from plausible soundscapes
- time shift
- light pitch shift and time stretch
- reverb/room impulse response for distant soundscapes
- high-frequency attenuation for distance effects
- cyclic padding for short clips

```python
augments = Compose([
    TimeStretch(min_rate=0.8, max_rate=2.0, p=0.5,
                leave_length_unchanged=False),
    RoomSimulator(p=0.3),
    OneOf([
        AddBackgroundNoise(sounds_path=["/path_to_noise"],
                           min_snr_in_db=5.0, max_snr_in_db=30.0, p=1.0),
        AddGaussianNoise(min_amplitude=0.005, max_amplitude=0.015, p=1.0),
    ], p=0.7),
    Gain(min_gain_in_db=-6, max_gain_in_db=6, p=0.2),
])
```

Adapt paths and transforms to the competition data. Do not assume speech
augmentation strengths are right for bird or industrial sounds.

### 7.4 Target-domain pseudo labels

1. Train a clean first-stage model or ensemble on labeled data.
2. Predict unlabeled target-domain soundscapes.
3. Keep soft probability vectors.
4. Mix pseudo windows into supervised batches.
5. Combine true and pseudo targets by max or convex mixing when labels can
   overlap.
6. Use OOF pseudo prediction if the pseudo audio has already influenced the
   teacher.
7. Apply probability powers greater than `1` to suppress noisy low-confidence
   pseudo labels when later iterations become too noisy.

Do not start with pseudo labels. First stabilize validation, metric code, and a
base model.

### 7.5 Label cleanup and OOF mining

Useful late-round tools:

- flag examples where OOF confidence strongly contradicts labels
- mine missing secondary labels only with high precision
- remove duplicates or corrupt audio
- soft-correct labels instead of hard overwriting when uncertain
- retrain with downweighted suspected-noisy examples

Be conservative: label cleanup can overfit public score if validation is weak.

## 8. Inference, Ensembling, And Postprocessing

### 8.1 Framewise to clipwise prediction

SED outputs can be reduced by max/mean over time depending on metric.

```python
output = model(input)
framewise = output["framewise_output"]
clip_probs = torch.sigmoid(torch.max(framewise, dim=1)[0])
```

Use max when any event occurrence should trigger the class. Use mean or smoothed
frame aggregation when the metric rewards stable window probabilities.

### 8.2 Soundscape smoothing and shifted windows

Useful long-recording postprocess:

- infer official non-overlapping windows
- optionally infer shifted windows, such as a half-window shift
- average neighboring predictions into official bins
- apply a small smoothing kernel over time

### 8.3 Fold and model averaging

For probability metrics, average probabilities or logits consistently. For rank
metrics, rank averaging can be useful. For diverse models, constrained weighted
blends are a late-stage upgrade.

```python
result = single_df.set_index("row_id").multiply(weight[0]).add(
    openvino_df.set_index("row_id").multiply(weight[1]), fill_value=0
).add(
    fold_df.set_index("row_id").multiply(weight[2]), fill_value=0
).reset_index()
```

Use OOF scores to choose weights. Do not tune weights only on public LB.

### 8.4 Top-k construction

For MAP@k tasks, select top labels after all averaging.

```python
pred_list = [np.load(path) for path in prediction_paths]
prediction = np.ones_like(pred_list[0])
for pred in pred_list:
    prediction *= pred
prediction = prediction ** (1.0 / len(pred_list))
top_3 = np.array(LABELS)[np.argsort(-prediction, axis=1)[:, :3]]
```

Geometric mean can be strong when models are calibrated similarly and all
probabilities are positive. Otherwise use arithmetic probability averaging.

### 8.5 Tail-column power transforms

Late-stage rank metrics sometimes benefit from shrinking low-ranked noisy
classes.

```python
def apply_power_to_low_ranked_cols(p, top_k=30, exponent=2, inplace=True):
    if not inplace:
        p = p.copy()
    tail_cols = np.argsort(-p.max(axis=0))[top_k:]
    p[:, tail_cols] = p[:, tail_cols] ** exponent
    return p
```

Use this only when OOF or a trusted validation proxy shows improvement. It is
easy to overfit class-prior quirks.

### 8.6 Metadata fusion

If metadata is available at inference:

- train audio models and metadata GBDT/MLP models on the same folds
- save OOF audio probabilities
- fit metadata and fusion models only on OOF predictions
- include class priors by site/date only if validation uses the same grouping
- keep a pure audio fallback

### 8.7 Submission checks

Before submission:

- row count exactly matches sample submission
- row ids match official window start/end convention
- class columns in exact order
- probabilities finite and within valid range
- top-k strings contain valid labels and no duplicates
- no missing windows from short/long files
- inference covers all audio, not only first segment
- OOF metric helper and submission format agree

## 9. Subtype Recipes

### 9.1 Top-K Environmental Audio Tagging

Use when clips are short/variable length and the metric is MAP@3 or top-k.

First solution:

- stratified 5-10 fold CV, grouped by source if available
- log-mel CNN with `tf_efficientnet_b0_ns` or ResNet/ResNeXt
- CE loss for single-label classes
- random crop/pad during training
- infer multiple clip lengths or multiple crops per clip if runtime allows
- average probabilities, then choose top 3 labels

Upgrades:

- use verified labels at higher weight
- add noisy/unverified examples later
- ensemble CNN families
- geometric mean across folds/crops
- add shallow audio statistics only as ensemble diversity, not as main route

Avoid:

- optimizing hard top-k directly before probability models are stable
- using MFCC-only features as the main route

### 9.2 Sparse Sound-Event Classification

Use when labels include TP/FP event time/frequency bounds and missing classes are
common.

First solution:

- split by recording id
- generate fixed windows centered on TP/FP events
- create targets with explicit known-mask
- train SED or CNN temporal model with masked BCE/focal
- infer overlapping windows across full recording
- aggregate max over windows for recording-level labels
- evaluate LRAP/LWLRAP or official rank metric at recording level

Upgrades:

- use framewise outputs for pseudo labels
- fill ignored labels with downweighted teacher predictions
- add frequency-aware masks if event frequency ranges are reliable
- ensemble mel bins/frontends/backbones
- calibrate class priors only with validation support

Avoid:

- splitting annotation rows across folds
- treating unknown labels as negatives
- manual labels or public-test prior hacks unless explicitly allowed

### 9.3 Bioacoustic Soundscapes

Use when training audio is curated species clips and test is long passive
soundscape audio.

First solution:

- define official window length, usually 5s
- train log-mel CNN/SED on random 10-30s crops or adjacent 5s chunks
- use soft primary/secondary labels when available
- use `tf_efficientnet_b0_ns` or `tf_efficientnetv2_s`, then add `eca_nfnet_l0`
  and `regnety_008`
- infer all soundscape windows with fold averaging
- smooth adjacent windows only after OOF or validation check

High-ROI upgrades:

- filter corrupt/duplicate/no-bird clips
- background mix with target-domain soundscapes
- pseudo-label unlabeled soundscapes with a first-stage ensemble
- keep pseudo labels soft; use power transforms in later iterations
- OOF-predict pseudo labels if reusing pseudo-trained soundscapes
- add fast CPU models for ensemble diversity
- export to ONNX/OpenVINO after numerical checks

Avoid:

- trusting random Xeno-Canto CV as the only selection signal
- hard thresholding pseudo labels too early
- public-LB-tuned species priors as a general recipe
- expensive transformers before the CNN/SED route is strong

### 9.4 Ordinary environmental audio multi-label

Use when audio clips may contain multiple broad sound classes.

First solution:

- log-mel CNN with BCEWithLogits
- moderate mixup and SpecAugment
- iterative multilabel or stratified fold by label count
- average folds and tune thresholds if metric requires hard labels

Upgrades:

- PANN CNN14 or CED/BEATs as pretrained audio models
- multi-resolution mel ensemble
- class-balanced sampling
- per-class calibration
- background augmentation from training negatives

### 9.5 Speech classification

Use when the label is a class, not text.

First solution:

- 16 kHz waveform or log-mel
- speaker/session grouped folds
- CNN/SED or speech encoder features
- CE or BCE depending on target

Upgrades:

- `WavLM`, `Whisper` encoder, BEATs, or CED embeddings
- duration-aware batching
- light room/noise/gain augmentation
- no text decoder unless the task is actually ASR

Route out:

- WER/CER
- transcript generation
- CTC/seq2seq decoding
- language model rescoring
- punctuation restoration

### 9.6 Audio plus metadata

Use when the task has real metadata at inference:

- site, latitude/longitude, date/time, device
- species taxonomy or source
- user/speaker/session
- clip duration or quality statistics

First solution:

- audio model on raw audio/mel
- simple metadata features added to final classifier only if fold-safe
- grouped validation that protects both audio and metadata leakage

Upgrades:

- GBDT on metadata + OOF audio predictions
- calibrated per-class priors by allowed metadata
- OOF stacking with tabular playbook patterns

Avoid:

- using metadata unavailable at test
- allowing site/date to leak fold identity
- replacing the audio model with metadata priors unless the task is explicitly a
  prior-estimation problem

## 10. Common Failure Modes

### 10.1 Wrong prediction unit

Training on clips and validating on clips can be misleading when the submission
is soundscape windows or recording-level labels. Always aggregate to the official
unit for OOF evaluation.

### 10.2 Wrong negatives

Sparse audio labels often mean "unknown", not "absent". Marking unknown classes
as zero is one of the most damaging mistakes in TP/FP event tasks.

### 10.3 Over-strong augmentation

SpecAugment, pitch shift, time stretch, denoising, and background mixing can
erase or distort the target event. Tune strength per domain and rare class.

### 10.4 Public-LB postprocess overfit

Class-prior scaling, threshold probing, tail-power transforms, and smoothing can
look strong on public score and fail private. Keep them late, small, and
validation-backed.

### 10.5 Runtime failure

Soundscape submissions can require thousands of windows. A model that trains
well but cannot infer within limits is not a viable route. Choose model size,
window overlap, TTA, and folds under the final inference budget.

### 10.6 ASR contamination

Do not import ASR decoding machinery into classification tasks. Speech encoders
can be feature extractors for speech labels, but transcripts, WER/CER, CTC, and
language models belong to a different playbook.
