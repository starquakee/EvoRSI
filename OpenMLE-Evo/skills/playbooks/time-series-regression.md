# Time-Series Regression Category Playbook

This playbook is the full method library for Kaggle-style
`time-series_regression` tasks. It is intended as direct reading material for a
later skill-generation agent: read the task description first, route to the
right sections, and extract a compact task-specific skill. Do not copy the
entire playbook into a final skill.

## 1. What Counts As Time-Series Regression

Use this playbook when the final target is a future continuous value, count,
quantile, interval, coordinate sequence, time-indexed vector, rank/correlation
score, or physical signal parameter derived from ordered observations.

Covered task forms include:

- store-item-date and warehouse demand forecasting
- hierarchical retail forecasting and probabilistic quantiles
- web traffic and sparse page/count forecasts
- county/month microbusiness or economic panels
- crypto/commodity/market streams with online API inference
- player engagement and repeated-entity sequence forecasting
- trajectory coordinates and multimodal future paths
- flood/weather/hydrology spatiotemporal regression
- acoustic/seismic/Raman/sensor segment regression
- exoplanet transit-depth and sigma prediction
- epidemic/location forecasts with public covariates

Route out if the final output is a categorical label, symbolic integer puzzle,
generated text, object boxes/masks, route/search solution, or judged EDA. Use
This playbook only for any temporal regressor substage.

## 2. Safety Boundary

- public/private leaderboard probing
- copying known public-period labels
- public-test label inference
- sample-submission labels or hidden labels
- row-order or cumcount leakage
- test-position reconstruction
- public-LB multipliers without validation
- third-party data that includes future target dates
- features unavailable at the forecast timestamp
- random folds for future deployment

The safe default is to build validation and
inference simulation that match the hidden test contract.

## 3. Output Contract First

Before modeling, write down:

- prediction unit: series-date, store-item-date, county-month, asset-timestamp,
  target column, player-day, agent-frame, node-timestep, planet-channel, sensor
  segment, or API batch
- horizon: one-step, fixed `H`, direct `T+1...T+H`, 28-day block, variable
  frames, rolling API, or full autoregressive rollout
- target shape: scalar, count, cumulative count, daily increment, quantiles,
  lower/upper intervals, multioutput vector, trajectory, or sigma/uncertainty
- metric: RMSE, MAE, RMSLE, SMAPE/MAPE, WRMSSE, weighted pinball, Pearson,
  Spearman Sharpe, trajectory NLL, standardized RMSE, or custom
- legal feature availability: known-future covariates, revealed lags, static
  metadata, test timestamps, public data cutoff, online batches
- leakage groups: series id, item, store, warehouse, county, asset, timestamp,
  game/play/player, event/model, planet, sensor/device, quake cycle, location
- submission constraints: nonnegative, integer, clipped range, monotone
  quantiles, hierarchy aggregation, confidence sums, field bounds, valid masks

The model choice follows this contract. A powerful sequence model is wrong if
the real failure is a lag feature that accidentally uses the prediction window.

## 4. Validation Geometry

### 4.1 Rolling-Origin And Horizon Blocks

For panel forecasting, validation must mirror the forecast horizon.

Use multiple rolling blocks when possible:

- hierarchical retail forecasting: 28-day blocks such as `d_1830..d_1857`,
  `d_1858..d_1885`, `d_1886..d_1913`
- store-item forecasting: 15 or 16-day tail blocks because test is short
- future-sales forecasting: month 33 as validation, month 34 as test
- warehouse demand forecasting: 60/61-day forward warehouse-aware splits
- Web Traffic: repeated truncation/walk-forward windows

```python
def rolling_origin_splits(dates, horizon, min_train, step=None, n_splits=3):
    dates = np.asarray(sorted(pd.unique(dates)))
    step = horizon if step is None else step
    max_start = len(dates) - horizon
    starts = [max_start - i * step for i in range(n_splits)][::-1]
    for start in starts:
        if start < min_train:
            continue
        train_dates = dates[:start]
        valid_dates = dates[start:start + horizon]
        yield train_dates, valid_dates
```

Select by mean and fold variance, not by one lucky public-like window.

### 4.2 Group-Time And Entity Splits

Split by the entity that can leak:

- store/item/warehouse/county/location
- asset and timestamp
- game/play/player/scene
- event/model for flood simulations
- planet id or sensor run
- quake cycle
- device/plate for Raman transfer
- article/customer/entity for longitudinal tabular tasks

If the same entity appears in both train and validation, ask whether that is
also true at inference. If not, use a stricter group-time holdout.

### 4.3 Purged And Embargoed Market Splits

Financial targets often overlap future windows. Adjacent train rows can leak
into validation through target construction or highly similar market states.

```python
def purged_time_splits(unique_times, train_len, valid_len, gap, step, n_splits):
    times = np.asarray(sorted(unique_times))
    for i in range(n_splits):
        train_end = train_len + i * step
        valid_start = train_end + gap
        valid_end = valid_start + valid_len
        if valid_end > len(times):
            break
        train_times = times[:train_end]
        valid_times = times[valid_start:valid_end]
        yield train_times, valid_times
```

For online multi-asset market targets, use a gap matching the target window or a practical
embargo such as one week. For multi-target commodity daily targets, use sorted `date_id`
folds and keep any public-90 overlap as a diagnostic only.

### 4.4 Online Replay Validation

If the competition uses `iter_test()` or a predict server, create a local replay:

- reveal only the same batch rows and label lags available at that step
- update state buffers after each prediction
- recompute features exactly as inference will
- enforce output column order and row count on every step
- time the replay under the official runtime limit

Do not validate with a batch feature table that uses future test rows if the API
would not expose them yet.

### 4.5 Natural-Unit Validation For Scientific Signals

Sensor/scientific tasks often require non-random splits:

- seismic event-time: split by earthquake cycle or cycle-balanced folds
- calibrated exoplanet-sensor regression: group repeated transits by `planet_id`
- Raman: device/plate/source-aware folds
- Acea/hydrology: forward-time splits by waterbody
- flood: hold out full events/models and roll out after warmup

Random chunks from a single long experiment can leak distribution and phase.

## 5. Metrics And Postprocessing

### 5.1 SMAPE

SMAPE is unstable around zero. Always implement the competition definition,
including the both-zero case if specified.

```python
def smape(y_true, y_pred, eps=1e-12):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.abs(y_true) + np.abs(y_pred)
    out = np.zeros_like(denom, dtype=float)
    mask = denom > eps
    out[mask] = 2.0 * np.abs(y_pred[mask] - y_true[mask]) / denom[mask]
    return out.mean()
```

For sparse counts, compare last-value, seasonal median, MAE-like objectives, and
integer reconstruction. MAPE-like training objectives can be less stable than
MAE even when the official metric is percentage-like.

### 5.2 RMSLE And Count Metrics

For RMSLE:

- train on `log1p(y)` or use Poisson/Tweedie/Gamma-like objectives
- inverse with `expm1`
- clip predictions to nonnegative before scoring/submission
- inspect high-count and zero-count groups separately

For future-sales forecasting-style RMSE with clipped target, clip predictions to `[0, 20]`
if the official target is clipped.

### 5.3 WRMSSE And Hierarchical Metrics

WRMSSE rewards all hierarchy levels, not just bottom-level item rows. Strong hierarchical retail forecasting
solutions used:

- 28-day horizon folds
- bottom-level LightGBM/Tweedie models
- recursive and non-recursive diversity
- group-specific models by store/category/department
- top-level alignment with independent N-BEATS or aggregate models
- no unvalidated scalar magic multipliers

If hierarchy scoring exists, implement aggregation and weighting before model
selection.

### 5.4 Pinball, Quantiles, And Intervals

For quantile tasks:

- train separate quantile models or a multi-quantile model
- enforce monotone quantiles after blending
- clip negative count quantiles
- calibrate widths by horizon and aggregation level
- validate weighted pinball/WSPL directly

```python
def pinball_loss(y, q, alpha):
    err = y - q
    return np.maximum(alpha * err, (alpha - 1.0) * err)

def enforce_quantile_order(q05, q50, q95):
    q50 = np.maximum(q50, q05)
    q95 = np.maximum(q95, q50)
    return q05, q50, q95
```

```python
def get_group_preds(pred, level, cols, qs, ratios):
    df = pred.groupby(level)[cols].sum()
    q = np.repeat(qs, len(df))
    df = pd.concat([df] * len(qs), axis=0, sort=False)
    df.reset_index(inplace=True)
    df[cols] *= ratios.loc[q].values[:, None]
    df["id"] = [f"{lev}_X_{quant:.3f}_validation"
                for lev, quant in zip(df[level].values, q)]
    return df[["id"] + list(cols)]
```

This pattern turns a point forecast into hierarchy-level quantile submissions.
Use it only after validating quantile multipliers.

### 5.5 Correlation And Spearman Sharpe

Market competitions can reward direction/rank and stability, not magnitude.

```python
def weighted_corr(x, y, w):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=float)
    mx = np.average(x, weights=w)
    my = np.average(y, weights=w)
    cov = np.average((x - mx) * (y - my), weights=w)
    vx = np.average((x - mx) ** 2, weights=w)
    vy = np.average((y - my) ** 2, weights=w)
    return cov / np.sqrt(max(vx * vy, 1e-30))
```

```python
from scipy.stats import rankdata

def daily_spearman_sharpe(pred, target):
    daily = []
    for p, y in zip(np.asarray(pred), np.asarray(target)):
        mask = np.isfinite(y)
        if mask.sum() < 2:
            continue
        rp = rankdata(p[mask])
        ry = rankdata(y[mask])
        daily.append(np.corrcoef(rp, ry)[0, 1])
    daily = np.asarray(daily)
    return daily.mean() / max(daily.std(ddof=0), 1e-12)
```

For these metrics, rank averaging, target rank transforms, directional loss, or
listwise/rank objectives can beat raw RMSE optimization.

### 5.6 Trajectory NLL And Coordinates

For multimodal motion prediction:

- output all required modes and confidences
- ensure confidences sum to 1
- compute in the coordinate frame expected by the scorer
- use ADE/FDE only as diagnostics if official metric is NLL
- calibrate mode confidences and do not naively average unmatched modes

### 5.7 Standardized Graph RMSE

Flood/graph metrics may standardize by model and node type. Train/evaluate with
matching weights so abundant node types do not dominate.

## 6. Feature Availability And Inference Simulation

### 6.1 Known-Future Covariates

Use future-known features when the task provides them:

- calendar, month, weekday, payday, workday, holiday transfer/bridge
- planned promotions
- SNAP or event calendars
- static store/item/location metadata
- forecast horizon index
- known player schedule or game metadata

Audit every column:

- available for all test timestamps?
- known before prediction time?
- derived from target?
- revised after the fact?
- public external data frozen before the cutoff?

### 6.2 Horizon-Safe Lags

```python
def ewm_features(df, group_cols, target_col, alphas, lags):
    df = df.sort_values(group_cols + ["date"]).copy()
    for alpha in alphas:
        for lag in lags:
            name = f"{target_col}_ewm_a{str(alpha).replace('.', '')}_lag_{lag}"
            df[name] = (
                df.groupby(group_cols)[target_col]
                  .transform(lambda x: x.shift(lag).ewm(alpha=alpha).mean())
            )
    return df
```

### 6.3 Rolling And Calendar Features

```python
def create_date_features(df, date_col="date"):
    df = df.copy()
    dt = pd.to_datetime(df[date_col])
    df["month"] = dt.dt.month.astype("int8")
    df["day_of_month"] = dt.dt.day.astype("int8")
    df["day_of_year"] = dt.dt.dayofyear.astype("int16")
    df["day_of_week"] = (dt.dt.dayofweek + 1).astype("int8")
    df["week_of_month"] = ((dt.dt.day - 1) // 7 + 1).astype("int8")
    df["quarter"] = dt.dt.quarter.astype("int8")
    df["is_month_start"] = dt.dt.is_month_start.astype("int8")
    df["is_month_end"] = dt.dt.is_month_end.astype("int8")
    df["is_wknd"] = (dt.dt.weekday >= 5).astype("int8")
    return df
```

```python
def shifted_rolling(df, group_cols, target_col, windows, lags):
    df = df.sort_values(group_cols + ["date"]).copy()
    for lag in lags:
        shifted = df.groupby(group_cols)[target_col].shift(lag)
        for w in windows:
            key = f"{target_col}_roll_mean_{w}_lag_{lag}"
            df[key] = shifted.groupby([df[c] for c in group_cols]).transform(
                lambda s: s.rolling(w, min_periods=1).mean()
            )
    return df
```

### 6.4 Recursive Forecast Update

For recursive models, recompute target-derived features after inserting each
predicted day. Do not precompute all forecast-window lags from true test target
values.

```python
def recursive_forecast(history, future_rows, model, feature_fn, id_cols, target):
    data = history.copy()
    preds = []
    for date, rows in future_rows.groupby("date", sort=True):
        frame = pd.concat([data, rows], ignore_index=True, sort=False)
        frame = feature_fn(frame)
        X = frame.loc[frame["date"].eq(date), model.feature_names_in_]
        yhat = model.predict(X)
        rows = rows.copy()
        rows[target] = yhat
        preds.append(rows[id_cols + ["date", target]])
        data = pd.concat([data, rows], ignore_index=True, sort=False)
    return pd.concat(preds, ignore_index=True)
```

Direct horizon models avoid recursive error accumulation but need one model or
target per horizon.

## 7. Panel Retail And Business Forecasting

### 7.1 Default GBDT Panel Route

Use for hierarchical retail forecasting, store-item forecasting, demand forecasting, warehouse demand forecasting, future-sales forecasting, temporal microbusiness forecasting-like
tasks when the final solution is a long table of engineered lags and covariates.

First model:

- last-value and seasonal-naive baseline
- long panel table
- horizon-safe lags and rolling stats
- calendar/holiday/known-future covariates
- categorical IDs: store, item, family, warehouse, state, county
- LightGBM/CatBoost/XGBoost
- rolling-origin validation matching horizon
- metric-specific clipping/postprocess

Concrete models:

- `lightgbm.LGBMRegressor(objective="tweedie")` for zero-inflated counts
- `objective="poisson"` for count-like targets
- `objective="mae"` or quantile for percentage/MAE-like stability
- `CatBoostRegressor(loss_function="MAE" or "Quantile")`
- `XGBRegressor(tree_method="hist" or "gpu_hist")`

### 7.2 Retail Feature Priorities

Build:

- `lag_1`, `lag_7`, `lag_14`, `lag_28`, `lag_56`, `lag_365` only if horizon-safe
- rolling mean/median/std/min/max over 7/14/28/56/90/365
- EWM with several alphas
- price momentum, price relative to item/store history, discount/promo
- item/store/category aggregates
- release/availability and out-of-stock proxies
- holidays, transferred holidays, bridge days, workdays, paydays
- known future promotions and oil/weather if provided
- same-weekday, same-month, same-day-last-year medians

Delay high-cardinality target encodings until folds and time cutoffs are clear.

### 7.3 hierarchical retail Hierarchical Accuracy

Useful patterns:

- train models by store, store-category, or store-department for stability
- blend recursive and non-recursive models
- align bottom-level predictions to top-level aggregate forecasts if validated
- choose submissions by mean and variance across multiple 28-day windows
- avoid blind public-LB magic multipliers

N-BEATS/N-BEATSx/DeepAR/TFT can add top-level or sequence diversity, but a
well-built LightGBM/Tweedie panel model is a serious baseline.

### 7.4 hierarchical retail Uncertainty

Start from a strong median forecast. Then model distribution shape:

- quantile LightGBM/CatBoost
- DeepAR negative binomial/Tweedie sampling
- Keras quantile loss
- historical quantiles by item and hierarchy level
- level-specific multipliers calibrated on OOF
- enforce monotone quantiles

Higher aggregation levels usually need narrower relative intervals than sparse
bottom-level items.

### 7.5 Web Traffic And Sparse Counts

- seasonal medians over recent windows
- median of medians over exponentially spaced windows
- weekday medians and holiday adjustments
- Kalman/local-level weekly models
- seq2seq GRU/LSTM with explicit yearly/quarterly lags

```python
def median_of_recent_windows(values, windows=(6, 12, 18, 30, 48, 78, 126, 203, 329)):
    values = pd.Series(values).fillna(0.0).values
    nz = np.flatnonzero(values)
    if len(nz) == 0:
        return 0.0
    start = nz[0]
    estimates = []
    for w in windows:
        if w > len(values) - start:
            break
        estimates.append(np.median(values[-w:]))
    return float(np.median(estimates)) if estimates else float(np.median(values[start:]))
```

For RNN seq2seq, include:

- `log1p` target
- per-series normalization
- page/category/country/site metadata
- explicit lags at year/quarter/month offsets
- checkpoint/seed averaging
- walk-forward tuning but final train on all usable history if that matches
  future deployment

### 7.6 Nonstationary Small Panels

temporal microbusiness forecasting-like tasks can favor simple linear models and last-window validation:

- last value is a strong baseline
- small groups can be noisy and zero-heavy
- use minimal lag/census/economic features
- early stop greedy feature selection
- avoid per-county models without enough data
- if the true target is an integer count divided by population, model count or
  round back to count before converting to density

```python
def density_to_count_roundtrip(mbd, adult_population):
    active = np.rint(mbd * adult_population / 100.0)
    return active / adult_population * 100.0
```

Public/revealed-month probing belongs only in warnings.

### 7.7 Operations Forecasting

warehouse demand forecasting-style order forecasting:

- forward-time warehouse splits
- holiday weights and custom holiday calendars
- log features can stabilize shakeup
- MAE objective can be more stable than MAPE
- ensure every feature exists or can be known in test
- quarantine shutdown/weather/user-activity columns if absent from test or not
  reconstructable

## 8. Market, Commodity, And Online API Forecasting

### 8.1 Market Route Principles

Treat these as low-SNR, regime-shifting ranking/correlation problems.

First steps:

- exact metric callback
- purged walk-forward validation
- API replay
- simple LightGBM/Ridge/MLP baseline
- fast stateful features
- no public-LB target copies

### 8.2 online multi-asset market Pattern

Target is residualized future return. Useful features:

- lagged returns
- log returns over multiple horizons
- EMA/Hull moving averages
- realized volatility
- cross-asset timestamp averages
- relative-to-market returns
- asset ID encoding
- missingness and API-observation flags

Models:

- LightGBM squared/correlation-selected regression
- per-asset LightGBM
- MLP/Keras NN
- ridge/linear for stable rank diversity

Use weighted correlation for selection, not ordinary RMSE.

### 8.3 multi-target commodity Pattern

`target_pairs.csv` is the design map:

- each target has an asset or asset pair
- each target has a lag
- target-specific features can reduce noise
- one-model-per-target is a valid robust route
- global RNN/MLP/Transformer routes can work when the validation and metric
  support cross-target learning

Use revealed label lag batches only as the API exposes them.

```python
def align_features(frame, feature_names):
    frame = frame.copy()
    for col in feature_names:
        if col not in frame:
            frame[col] = 0.0
    return frame.reindex(columns=feature_names).replace([np.inf, -np.inf], np.nan).fillna(0.0)
```

For rank metrics, consider:

- rank-transforming targets by date
- listwise/rank loss plus MSE
- per-target models with pair-relevant assets
- prediction clipping to stable ranges
- daily Spearman Sharpe callback

### 8.4 Online API Output Safety

```python
def check_prediction_frame(pred, expected_cols, expected_rows):
    assert list(pred.columns) == list(expected_cols)
    assert len(pred) == expected_rows
    vals = pred.to_numpy(dtype=float)
    assert np.isfinite(vals).all()
```

For every API task, save model feature order and test it on the first batch.

## 9. Trajectory And Spatiotemporal Regression

### 9.1 Route Boundary

vehicle-trajectory, sports tracking, and flood tasks are in this category folder, but they are not
ordinary panel sales forecasts. Their first design artifact is the output schema
and coordinate/state frame.

### 9.2 Trajectory Residuals

Most strong routes predict residual deltas in a normalized local frame, then
reconstruct.

```python
def residual_targets(input_df, output_df, keys):
    last_xy = (
        input_df.sort_values(keys + ["frame_id"])
                .groupby(keys)[["x", "y"]]
                .tail(1)
                .to_numpy()
    )
    y = output_df[["x", "y"]].to_numpy() - last_xy
    return y

def reconstruct_xy(last_xy, pred_delta, xlim=(0, 120), ylim=(0, 53.3)):
    pred = last_xy[:, None, :] + pred_delta
    pred[..., 0] = np.clip(pred[..., 0], *xlim)
    pred[..., 1] = np.clip(pred[..., 1], *ylim)
    return pred
```

### 9.3 vehicle-trajectory

Output:

- 3 trajectory modes
- 50 future `(x, y)` points per mode
- confidences sum to 1
- world-coordinate offsets in submission

Models:

- EfficientNetB3/B5/B6/B7 raster models
- ResNet/ResNeXt/Xception/MixNet diversity
- multi-mode head over `[B, K, 50, 2]` plus confidence logits
- ensemble over raster sizes, pixel sizes, history frames

Do not average unmatched modes naively. Use distance sorting, GMM/fixed-point
mode merging, Set Transformer/head stacking, or validate mode-wise ensembling.

```python
def trajectory_head(raw, batch_size, modes=3, steps=50):
    coords = raw[:, :modes * steps * 2].reshape(batch_size, modes, steps, 2)
    conf = raw[:, modes * steps * 2:]
    conf = np.exp(conf - conf.max(axis=1, keepdims=True))
    conf = conf / conf.sum(axis=1, keepdims=True)
    return coords, conf
```

### 9.4 player-trajectory

Useful route:

- normalize play direction and field coordinates
- raw `x,y,s,a,dir,o` plus role/side/ball features
- predict per-frame deltas, then cumulative sum
- GRU/attention, spatiotemporal Transformer, Conv1D + player Transformer
- CatBoost residual baseline for diversity
- horizontal/vertical flip and 180-degree rotation augmentation
- EMA for stable validation
- clip to field and invert coordinates before submission

```python
class TemporalHuber(nn.Module):
    def __init__(self, delta=0.5, time_decay=0.03):
        super().__init__()
        self.delta = delta
        self.time_decay = time_decay

    def forward(self, pred, target, mask):
        err = pred - target
        abs_err = torch.abs(err)
        huber = torch.where(
            abs_err <= self.delta,
            0.5 * err * err,
            self.delta * (abs_err - 0.5 * self.delta),
        )
        if self.time_decay > 0:
            length = pred.size(1)
            t = torch.arange(length, device=pred.device).float()
            weight = torch.exp(-self.time_decay * t).view(1, length, 1)
            huber = huber * weight
            mask = mask.unsqueeze(-1) * weight
        return (huber * mask).sum() / (mask.sum() + 1e-8)
```

### 9.5 Urban Flood And Graph Rollout

Use this route when the task has explicit nodes/edges/events:

- work in relative depth or water-level delta
- separate 1D/2D domains if physics differs
- encode node static attributes and edge static attributes
- include dynamic rainfall, flow, velocity, volume, depth slope
- use signed log transforms for skewed signed flows
- warm up on observed history, then full autoregressive rollout
- validate by event/model and node type

```python
def signed_log1p(x):
    x = np.asarray(x)
    return np.sign(x) * np.log1p(np.abs(x))
```

Models:

- node-centric LSTM/GRU
- EdgeAware GNN with node/edge GRUs
- heterogeneous GNN with 1D, 2D, and coupling links
- TransformerConv/GAT variants if validation supports them
- Ridge/LightGBM delta specialists as robust baselines

## 10. Scientific, Sensor, And Physical Signal Regression

### 10.1 General Pattern

Do not feed raw long signals into a generic model first. Build physically
plausible intermediate representations:

- calibrate and denoise sensor readings
- match train rows to test granularity
- split by natural experiment unit
- extract robust segment/channel features
- model residuals with simple models before large networks
- calibrate uncertainty if the metric includes sigma or intervals

### 10.2 seismic event-time

The row is a 150k-sample acoustic segment. Good features:

- mean/std/min/max, absolute stats
- quantiles and trimmed means
- peak counts and crossing counts
- rolling mean/std quantiles
- FFT/Welch bands, Hilbert envelope, Hann smoothing
- STA/LTA features
- MFCC means
- train/test feature distribution checks

```python
def basic_signal_features(x):
    x = pd.Series(x).astype(float)
    feats = {
        "mean": x.mean(),
        "std": x.std(),
        "max": x.max(),
        "min": x.min(),
        "iqr": np.subtract(*np.percentile(x, [75, 25])),
        "q001": np.quantile(x, 0.001),
        "q999": np.quantile(x, 0.999),
        "abs_q99": np.quantile(np.abs(x), 0.99),
    }
    for w in [10, 100, 1000]:
        roll_std = x.rolling(w).std().dropna().values
        roll_mean = x.rolling(w).mean().dropna().values
        feats[f"roll_std_mean_{w}"] = roll_std.mean()
        feats[f"roll_std_q99_{w}"] = np.quantile(roll_std, 0.99)
        feats[f"roll_mean_q01_{w}"] = np.quantile(roll_mean, 0.01)
        feats[f"roll_mean_q99_{w}"] = np.quantile(roll_mean, 0.99)
    return feats
```

Models:

- LightGBM with robust/fair/huber losses
- XGBoost/CatBoost
- SVR
- shallow MLP with auxiliary targets such as time-since-failure
- hillclimb/geometric blends only after trustworthy CV

### 10.3 calibrated exoplanet-sensor regression

This is physics/sensor inference more than generic forecasting.

Strong ingredients:

- ADC inverse and sensor calibration
- nonlinearity correction
- dark-current correction with exposure time
- hot/dead pixel handling
- flat field and correlated double sampling
- time binning and sensor synchronization
- cosmic-ray removal
- transit physics with priors/posteriors
- Gaussian process/PCA spectrum regularization
- sigma calibration and metric-aware post-hoc correction

Models:

- Bayesian/least-squares physical solver
- BFGS/Gauss-Newton/Levenberg-Marquardt/Minuit
- GP/PCA prior over wavelength
- small residual MLP/GBM/Ridge only after calibrated physical features

Post-hoc "fudging" can be useful in Kaggle metrics, but the playbook should frame
it as validation-calibrated residual correction, not as a substitute for
physical modeling.

### 10.4 Raman Transfer Learning

Useful route:

- interpolate missing wavelengths
- MSC/SNV normalization
- baseline correction such as airPLS
- Savitzky-Golay smoothing and derivatives
- peak statistics and PCA/linear views
- HGB/ExtraTrees/Ridge/LinearRegression/RandomForest stacks
- device/plate-aware validation
- nonnegative clipping and train-range clipping

### 10.5 Hydrology And Water Levels

Acea-style water prediction:

- model each waterbody/target separately if dynamics differ
- use delayed rainfall, temperature, hydrometry, and seasonal components
- resample carefully and preserve lag availability
- TimeSeriesSplit/forward validation only
- compare ARIMA/Prophet/LSTM with lag-table GBDT baselines

## 11. Epidemic, Geospatial, Weather, And Public Data

### 11.1 Epidemic Quantiles

COVID-style tasks require:

- daily vs cumulative target distinction
- location hierarchy and rollups
- population-normalized rates
- rolling 7/14/21-day features
- fatality or case ratios
- nearby-location summaries
- direct horizon quantile models
- nonnegative and monotone quantiles
- long-horizon damping when curves become unstable

Manual adjustments to top countries or public-LB phases are not general
recipes. The transferable idea is to inspect high-weight/high-volume locations.

### 11.2 Geospatial Climate And Crop

For future climate/crop tasks:

- use forward-year holdouts and spatial diagnostics
- summarize daily sequences by chunks and seasonal windows
- include static soil, nitrogen, CO2, planting date, lat/lon
- compute location priors from train years only
- avoid using simulation year IDs as shortcuts
- test OOD stability, not just random CV

### 11.3 World Weather Boundary

## 12. Ensembling And Stability

### 12.1 Save OOF Artifacts

For every model:

- validation rows and groups
- fold/horizon/date id
- OOF predictions after inverse transform and postprocess
- test predictions at official row level
- model feature order and config
- metric per fold, horizon, group, and high-weight segment

Without OOF artifacts, thresholding, hierarchy alignment, quantile calibration,
and blending become public-LB guessing.

### 12.2 Useful Diversity

Use diversity across:

- direct and recursive forecasts
- GBDT and statistical/linear baselines
- different lag windows and seasonal priors
- store/category/department target models
- sequence models and tabular models
- market rank/rmse/correlation objectives
- physical solver and residual model
- graph model seeds and rollout lengths
- quantile shape and point forecast source

### 12.3 Stability Diagnostics

Track:

- fold mean and standard deviation
- score by horizon
- score by series group/hierarchy level
- high-weight group errors
- public-like vs private-like windows
- prediction distribution vs OOF distribution
- model correlation across OOF predictions
- validation bootstrap std for trajectory tasks
- metric sensitivity to postprocess constants

### 12.4 Blending

Default:

- simple average or weighted average selected by OOF
- geometric mean for positive count forecasts if OOF supports it
- Ridge/ElasticNet on OOF predictions for regression stacks
- rank average for correlation/rank metrics
- calibrate quantile widths after point forecast blending

Delay hillclimbing until the base model pool is strong and validation is stable.

## 13. Anti-Patterns And Quarantine

Do not include as normal recipes:

- multi-target commodity public-90 label copy probes
- temporal microbusiness forecasting public-month SMAPE root solving
- World Weather third-party data covering test dates
- MLB cumcount/row-order leakage
- seismic event-time test-position reconstruction
- public/private row masking to hide score
- public-LB multipliers without stable CV
- side-by-side validation for future web traffic
- KFold for warehouse demand forecasting/hierarchical retail forecasting/store-item future periods
- target encodings fitted on train+validation
- rolling windows that include forecast horizon targets
- features absent from test or unknown at prediction time
- symbolic integer sequence solvers as time-series forecasting methods

Narrow lessons that do transfer:

- public LB can be actively misleading
- last-window CV can matter for nonstationary panels
- exact metric implementation is mandatory
- simple baselines can beat overfit complex models
- feature availability is a modeling constraint, not bookkeeping
