# Tabular Regression Category Playbook

This playbook is the full method library for Kaggle-style `tabular_regression`
tasks. It is intended as direct reading material for a skill-generating agent:
after reading a task description, the agent should route the task, select the
right validation strategy, then extract only the applicable recipes.

Do not use this playbook to justify leaderboard probing, public-test label
inference, hidden labels, manually inferred test labels, sample-submission label
updates, leakage-based recipes, or disallowed external data.

## 1. Task Routing

### 1.1 Flat row-wise regression

Use for ordinary CSV/parquet rows with one numeric target and no obvious repeat
entity, time, group, or online stream.

Default plan:

- KFold or target-binned KFold
- exact metric helper and submission validator
- LightGBM, XGBoost, CatBoost
- simple missing/categorical handling
- row statistics, count/frequency encodings, numeric interactions
- OOF predictions saved for every model
- simple average or positive Ridge blend late

### 1.2 Skewed positive regression

Use when the target is price, revenue, loss, sales, calories, duration, counts,
or any positive heavy-tailed quantity. Metrics often include RMSLE, MSLE, RMSE
on log revenue, or ordinary RMSE/MAE with huge outliers.

Default plan:

- inspect target distribution and zero prevalence
- choose target transform: `log1p`, `log(y + shift)`, Box-Cox, Yeo-Johnson,
  square root, or no transform
- validate official metric after inverse transform
- clip predictions to legal range, often nonnegative
- use robust objectives or sample weights if outliers dominate
- audit high-target rows and never delete many rows without OOF evidence

Model families:

- XGBoost/LightGBM/CatBoost on transformed target
- Tweedie/Gamma/Poisson-like objectives for nonnegative skewed quantities
- Huber/Fair/custom robust objectives for MAE-like insurance loss
- Ridge/ElasticNet on log target for sparse or linear-friendly data

### 1.3 Categorical-heavy regression

Use when many features are object/category/id columns, anonymized categories, or
high-cardinality categorical variables.

Default plan:

- CatBoost native categorical baseline
- XGBoost categorical support when available
- LightGBM with ordinal/count/frequency/target encodings
- one-hot or sparse text-style encoding for low-cardinality/linear models
- fold-safe target aggregates for high-cardinality IDs
- category combinations only when they are stable and validated

Common features:

- category frequency
- group target mean/median/std/quantiles
- count per unique and ratios of target statistics
- string splits, digit/letter parts, category hierarchy
- rare category bucketing for NNs

### 1.4 Entity and relational regression

Use when the target is per user/customer/account/item/property/visit/entity and
the raw data has multiple rows per entity or multiple tables.

Default plan:

- define the prediction entity before feature engineering
- aggregate child/session/transaction tables into one row per prediction entity
- group validation by entity if repeated entities leak
- time validation if target is future value
- build recency, frequency, monetary, count, unique-count, and status-specific
  aggregates
- save OOF predictions at entity level

Examples:

- Google Analytics revenue: sessions are rows, but submission is per user and
  target is future user revenue.
- Elo/Merchant-style or customer spend tasks: repeated transactions are child
  rows.
- Property sales: repeated property IDs or repeated sale events require entity
  awareness.

### 1.5 Panel forecasting with tabular models

Use for store-item-date, restaurant-date, building-meter-hour, energy, sales,
traffic, demand, and similar future-period predictions.

Default plan:

- forward-time or rolling-origin validation
- direct horizon models, recursive models, or one model with horizon features
- lags, rolling windows, decayed means, first/last active days
- calendar, holidays, promotion, price, open/closed, known-future covariates
- group aggregates by item/store/class/building/site and their interactions
- use only information available at prediction time
- score after reconstructing the official horizon/submission rows

Model families:

- LightGBM/XGBoost/CatBoost over lag features
- one model per horizon/day when horizon behavior differs
- recursive GBDT/NN when predictions feed future lags
- MLP/LSTM/WaveNet/Temporal CNN as diversity when sequence length and data
  volume justify them

### 1.6 Financial, market, and online regression

Use when the task has `time_id`, `date_id`, assets, investment IDs, market
weights, hidden future periods, API batches, correlation/R2/Sharpe-like metrics,
or rank portfolios.

Default plan:

- forward-time, purged group-time, or local replay validation
- preserve the API information flow exactly
- evaluate official group-level metric: per-time Pearson, weighted R2, daily
  spread, Sharpe-like utility, RMSPE, or weighted MAE
- build cross-sectional features inside the current time batch when allowed
- build lag/rolling features only from past observations
- clip predictions if the metric or API expects bounded responders/allocation
- blend robust GBDTs with simple NNs/TabNet/Ridge only when OOF improves

Route warnings:

- do not use future test rows or reconstructed hidden order
- do not trust random folds on non-stationary markets
- do not optimize RMSE if the metric is correlation, R2, rank spread, or Sharpe

### 1.7 Text and multimodal regression

Use when numeric target depends on title, description, essay/log text, product
images, image quality, or metadata. This includes product price/demand, revenue,
and writing score tasks.

Default plan:

- primary tabular regression for metric, OOF fusion, and metadata
- secondary text or image methods if raw modality modeling is central
- sparse TF-IDF/count char-word features plus Ridge/linear baseline
- SVD/FastText/Transformer/image embeddings as dense features
- GBDT over dense metadata plus text/image OOF predictions
- sparse MLP/CNN/FM when text matrices are large and target is dense enough
- group validation by user/item/source/time if repeated entities exist

### 1.8 Multioutput scientific or high-dimensional regression

Use when there are many target columns, vertical profiles, gene/protein vectors,
climate columns, fMRI/loadings, or sparse matrices.

Default plan:

- identify target families and per-target metric weights
- group by donor/day/batch/site/scanner/time when needed
- normalize inputs and targets carefully
- reduce dimensionality with PCA/SVD/IncrementalPCA when matrices are huge
- train multioutput MLP/1D CNN/U-Net/Transformer/TabNet/GBDT depending on input
  shape
- track per-target OOF scores and clipped/weighted metric
- ensemble per target or per target family

Route warnings:

- do not treat 18,000 targets as 18,000 independent Kaggle rows
- do not ignore batch/site/domain shift
- do not score only global RMSE if the metric is column-wise or weighted

### 1.9 Survival, risk-score, and censored outcomes

Use when the target includes event time plus event indicator and the submission
is a risk score evaluated by C-index or stratified C-index.

Default plan:

- use survival-analysis methods
- use this playbook only for tabular preprocessing, GBDT/NN baselines, and OOF
  ensembling
- do not regress censored time directly as if exact
- build risk labels from Cox partial hazard, Kaplan-Meier survival,
  Nelson-Aalen cumulative hazard, rank transforms, or pairwise ranking losses
- evaluate C-index stratified by the fairness group when applicable
- track event and censored cases separately

Good model families:

- CoxPH/CoxBoost/survival forests when available
- CatBoost/XGBoost/LightGBM on engineered risk labels
- pairwise ranking neural nets with invalid censored-pair masks
- rank ensembles

### 1.10 Prediction intervals and quantile regression

Use when the submission requires lower and upper bounds, coverage, prediction
intervals, or quantiles rather than a point estimate.

Default plan:

- use uncertainty and interval-estimation methods
- train quantile models for lower and upper bounds
- use split conformal or cross-conformal calibration
- validate coverage and interval width/Winkler/pinball score
- consider detrending or residual variance models
- ensure `lower <= upper` after all postprocessing

Model families:

- CatBoost/LightGBM/XGBoost quantile objectives
- QuantileRegressor/GradientBoostingRegressor for smaller data
- TabM/RealMLP/NN multiquantile heads
- mean model plus residual/variance model when direct quantiles are weak

### 1.11 Geolocation and trajectory correction

Use when the output is latitude/longitude or indoor coordinates and the metric
is distance percentiles or trajectory quality. This is regression-adjacent but
not ordinary row RMSE.

Default plan:

- group by phone/route/building/site/trajectory
- predict corrections to a baseline rather than raw coordinates when a strong
  physical baseline exists
- use sensor, route, floor, Wi-Fi, satellite, and time features
- apply outlier removal, interpolation, smoothing, Kalman/least-squares, or
  snap-to-grid only when validated
- score percentile distance, not ordinary RMSE

Use geospatial and trajectory-specific methods.

### 1.12 Runtime/config ranking and graph regression

Use when the task predicts runtimes or ranks configurations, but the scored
output is a top-k or full ordering per graph/model.

Default plan:

- group validation by graph/model/kernel
- train a scorer for configuration runtime
- output rankings, not isolated row predictions
- evaluate Kendall tau, top-k slowdown, or graph-level rank metric
- use graph/config features, GNNs, sequence models, or GBDT features depending
  on input format

Route out to graph/ranking/optimization playbook when the core solution is not
ordinary tabular regression.

## 2. Validation And Leakage

### 2.1 Define the scored unit

Before modeling, answer:

- Is the score per row, user, entity, store-item-date, time batch, graph, phone,
  patient, target column, or interval?
- Does submission require transformed target, original target, risk score,
  lower/upper bounds, ranking, allocation, or top-k string?
- Are multiple raw rows aggregated into one submission row?
- Does the test period occur after the training period?
- Are labels censored, weighted, grouped, or partially ignored?

Save OOF predictions at this unit. If training is row-level but scoring is
user-level, aggregate OOF to user-level before selecting models.

### 2.2 Ordinary KFold and target-binned KFold

Use only when rows are plausibly independent.

For skewed regression, stratify on target bins to avoid folds with very
different high-value tails.

```python
def make_target_binned_folds(df, target, n_splits=5, seed=42, n_bins=20):
    df = df.copy()
    y_bin = pd.qcut(df[target], q=n_bins, labels=False, duplicates="drop")
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    df["fold"] = -1
    for fold, (_, va) in enumerate(cv.split(df, y_bin)):
        df.loc[df.index[va], "fold"] = fold
    return df
```

Use it when target tails drive
metric variance and there is no stronger group/time split.

### 2.3 Group validation

Use group folds for repeated entities:

- user/customer/account/visitor
- item/store/product
- property/license plate/graph/model
- patient/donor/race group
- phone/trajectory/building/site
- source/scanner/batch

```python
for tr_idx, va_idx in GroupKFold(n_splits=5).split(X, y, groups=df[group_col]):
    X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
    y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]
```

If both target distribution and group leakage matter, use stratified group folds
or manually repair folds, but do not let the same group cross train/validation.

### 2.4 Time and rolling-origin validation

Use time validation whenever the test is later in time.

`Implementation Pattern`:

```python
dates = np.sort(df[date_col].unique())
valid_dates = dates[-n_valid_dates:]

train_idx = df[~df[date_col].isin(valid_dates)].index
valid_idx = df[df[date_col].isin(valid_dates)].index
```

For repeated model selection, use multiple rolling-origin splits:

- train through month M, validate M+1
- train through week W, validate the next official horizon
- leave one year/quarter out when year-specific shocks matter
- include a gap/embargo for markets or target leakage through adjacent time

Do not tune on a validation period that contains an unrepeatable shock unless
the test is expected to contain the same shock.

### 2.5 Forward-looking entity targets

When the target is future user/entity revenue or value:

- build feature windows before the prediction window
- do not use rows from the gap or target period
- aggregate sessions/transactions into entity rows
- validate using historical windows that mimic train/test chronology
- consider two-stage modeling: probability of positive future activity times
  amount conditional on activity

This pattern is essential for Google Analytics-style revenue tasks.

### 2.6 Purged group-time and online replay

For market streams:

- split by time/date ranges
- purge adjacent periods if features or labels overlap
- group by `time_id`, `date_id`, or asset/time batch
- implement the evaluator information flow locally
- maintain lag state exactly as the API reveals it

Do not compare random-fold RMSE to time-fold correlation and call the better
number the truth. Trust the split that resembles deployment.

### 2.7 Adversarial validation

Use adversarial validation when train/test distributions may differ.

`Implementation Pattern`:

```python
xtrain = train_features.copy()
xtest = test_features.copy()
xtrain["istrain"] = 1
xtest["istrain"] = 0
xdat = pd.concat([xtrain, xtest], axis=0)

# Train a binary classifier to predict istrain.
# High AUC means random CV is suspect and the validation split should mimic
# the detected shift by time, source, geography, or entity.
```

Use the result to improve validation, not to build fragile public-test hacks.

### 2.8 External data and original data

External/original data can help, especially in synthetic-tabular tasks where the
synthetic train is derived from a real dataset. But use it carefully:

- confirm the rules allow it
- check if it contains exact test rows or labels
- keep OOF scores computed on competition train folds
- avoid using external data to construct validation labels
- report when original data is used only for training and not for OOF scoring

If an external dataset contains test rows with labels or can be joined to test
labels, quarantine it as leakage unless the task explicitly permits it.

## 3. Metrics And Target Transforms

### 3.1 RMSE and R2

For ordinary RMSE:

- train with squared error or RMSE-like objectives
- audit outliers and high-leverage rows
- use target-binned folds when tails dominate
- clip only when the target has physical/legal bounds

For R2:

- implement the exact centered/weighted formula
- a constant shift or scaling can matter if the metric is not pure correlation
- use sample weights if specified

### 3.2 RMSLE and log-target training

Use `log1p` for nonnegative targets under RMSLE/MSLE or log-revenue RMSE.

`Implementation Pattern`:

```python
y_log = np.log1p(train[target])
model.fit(X_train, y_log.iloc[train_idx])

valid_pred_log = model.predict(X_valid)
valid_pred = np.clip(np.expm1(valid_pred_log), 0, None)

rmsle = np.sqrt(mean_squared_log_error(
    train[target].iloc[valid_idx],
    valid_pred,
))

test_pred = np.clip(np.expm1(model.predict(X_test)), 0, None)
```

Common mistakes:

- scoring RMSE on log predictions when the metric requires RMSLE after inverse
- forgetting to clip negative predictions before RMSLE
- using `log(y)` when zeros exist
- applying `expm1` twice in the submission
- rounding continuous predictions unless the target is integer and OOF improves

### 3.3 Shifted log and robust transforms

Use `log(y + shift)` when zeros are not the only issue or the target can have a
minimum offset. Insurance loss solutions used shifted logs, power transforms,
and robust custom objectives.

Practical choices:

- `log1p(y)` for nonnegative skew
- `np.log(y + shift)` when shift is validated
- `sqrt(y)` for moderate count skew
- Yeo-Johnson when targets can be zero/negative
- residual target after detrending for time and price inflation

Always inverse-transform OOF predictions and score the official metric.

### 3.4 MAE, MAPE, SMAPE, and weighted errors

For MAE:

- try MAE/quantile objectives in LightGBM/CatBoost
- robust median-like features are often useful
- outlier clipping can help but must be OOF-validated

For MAPE/SMAPE:

- handle zeros explicitly
- floor denominators or predictions only if the official metric does
- MAE-like objectives are often better proxies than MSE
- percentage errors reward relative accuracy, so log/ratio features matter

For weighted errors:

- pass sample weights to model and metric where appropriate
- audit high-weight segments
- avoid improving unweighted CV while degrading high-weight rows

### 3.5 Correlation and weighted R2

Market tasks often score correlation or weighted R2. Optimize exact group-level
metric, not plain RMSE.

`Implementation Pattern`:

```python
def weighted_r2(y_true, y_pred, weight):
    num = np.average((y_pred - y_true) ** 2, weights=weight)
    den = np.average(y_true ** 2, weights=weight) + 1e-38
    return 1 - num / den

def mean_pearson_by_group(df, group_col, y_col, pred_col):
    return df.groupby(group_col).apply(
        lambda g: scipy.stats.pearsonr(g[y_col], g[pred_col])[0]
    ).mean()
```

Metric behavior:

- Pearson is invariant to positive linear scaling within a group, but not to
  bad ranking or nonlinear distortion.
- Weighted R2 punishes squared errors on high-weight rows.
- Daily Sharpe-like metrics care about stable group performance, not just
  global loss.

### 3.6 Multioutput metrics

For MCRMSE, column-wise RMSE, weighted R2, or average Pearson:

- track per-target OOF
- normalize targets when their scales differ
- avoid one target dominating shared loss unless metric weights say so
- consider separate heads/models for target families
- clip or zero-out unscored targets according to the official weighting file
- build the final submission with exact column order

### 3.7 C-index and risk scores

For survival/risk tasks:

- output risk scores where higher means higher risk if the metric expects that
- do not treat censored time as exact ground truth
- validate comparable pairs and censoring masks
- stratified C-index may subtract race/group dispersion, so fairness-group
  stability matters
- rank-based blending can be more stable than raw-score averaging

### 3.8 Prediction intervals

For interval tasks:

- lower and upper bounds are the prediction, not a postscript
- optimize quantile/Winkler/pinball or a close proxy
- calibrate coverage on held-out data
- enforce lower <= upper
- validate coverage, width, and official score

`Implementation Pattern`:

```python
lower_model = CatBoostRegressor(loss_function="Quantile:alpha=0.05", **params)
upper_model = CatBoostRegressor(loss_function="Quantile:alpha=0.95", **params)

lower_model.fit(X_train, y_train)
upper_model.fit(X_train, y_train)

lo_cal = lower_model.predict(X_calib)
hi_cal = upper_model.predict(X_calib)

q_lo = np.quantile(lo_cal - y_calib, 1 - alpha / 2)
q_hi = np.quantile(y_calib - hi_cal, 1 - alpha / 2)

submission["pi_lower"] = lower_model.predict(X_test) - q_lo
submission["pi_upper"] = upper_model.predict(X_test) + q_hi
```

## 4. Feature Engineering

### 4.1 Missingness and row statistics

Useful across flat and scientific tabular:

- missing count per row
- zero count per row
- mean/std/min/max/skew/kurtosis across numeric features
- count of positive/negative values
- row quantiles
- duplicated-row or constant-column flags
- feature family statistics by prefix

Be careful: row statistics over target-like future columns can leak if the table
contains time-expanded target periods.

### 4.2 Basic categorical encoding

Default order:

1. CatBoost native categorical handling for high-cardinality categories.
2. XGBoost categorical support when stable in the environment.
3. Frequency/count encoding from train+test feature values when allowed.
4. One-hot encoding for low-cardinality and linear/sparse models.
5. Ordinal/label encoding for tree models when category order is irrelevant.
6. Rare-category buckets for NNs and embeddings.

Use domain order when it exists, such as quality grades, condition, cut, color,
clarity, ordinal education, or risk categories.

### 4.3 Fold-safe target aggregates

Target-derived group statistics are high-value and high-risk. They include
target mean, median, std, min/max, quantiles, histograms, and category
interaction statistics.

Use nested folds for target aggregates inside an outer validation fold.

`Implementation Pattern`:

```python
outer = KFold(n_splits=5, shuffle=True, random_state=42)
inner = KFold(n_splits=5, shuffle=True, random_state=42)

for fold, (tr_idx, va_idx) in enumerate(outer.split(train)):
    outer_train = train.iloc[tr_idx].copy()
    outer_valid = train.iloc[va_idx].copy()
    test_fold = test.copy()

    outer_train[f"{col}_target_mean"] = np.nan

    for in_tr_idx, in_va_idx in inner.split(outer_train):
        in_train = outer_train.iloc[in_tr_idx]
        in_valid = outer_train.iloc[in_va_idx]
        mapping = in_train.groupby(col)[target].mean()
        prior = in_train[target].mean()
        outer_train.loc[in_valid.index, f"{col}_target_mean"] = (
            in_valid[col].map(mapping).fillna(prior)
        )

    mapping = outer_train.groupby(col)[target].mean()
    prior = outer_train[target].mean()
    outer_valid[f"{col}_target_mean"] = outer_valid[col].map(mapping).fillna(prior)
    test_fold[f"{col}_target_mean"] = test_fold[col].map(mapping).fillna(prior)
```

Adaptation notes:

- replace mean with median/std/quantile/histogram when metric benefits
- use smoothing for small categories
- for temporal tasks, compute mappings only from past rows
- never fit these statistics on validation or test target values

### 4.4 Count, frequency, and unsupervised aggregates

Count/frequency encodings can use train+test feature values if rules permit,
because they do not use labels.

```python
def add_frequency_encoding(train, test, cols):
    for col in cols:
        vc = pd.concat([train[col], test[col]], axis=0).value_counts(dropna=False)
        train[f"{col}_freq"] = train[col].map(vc).astype("float32")
        test[f"{col}_freq"] = test[col].map(vc).astype("float32")
    return train, test
```

Useful extensions:

- counts of category pairs
- count per unique ratio
- `groupby(col1)[col2].nunique()`
- category frequency by time period
- interaction keys such as `store_item`, `city_category`, `brand_shipping`

### 4.5 Relational and entity aggregates

For child tables:

- aggregate by prediction entity
- compute counts, sums, means, medians, stds, min/max, nunique
- build recency windows: last 7/30/90 days
- status-specific aggregates: active/closed/refused/approved
- ratio features: recent mean / long mean, amount per count
- trend features: last value, difference, slope

For user revenue:

- one row per `fullVisitorId`
- session counts and recency
- channel/device/geo/source aggregates
- previous revenue and transaction count
- future-window validation

### 4.6 Time-series lag and rolling features

Use only past values.

`Implementation Pattern`:

```python
df = df.sort_values(group_cols + [time_col])

for col in value_cols:
    grouped = df.groupby(group_cols)[col]
    df[f"{col}_lag1"] = grouped.shift(1)

    for window in windows:
        df[f"{col}_roll{window}_mean"] = (
            df.groupby(group_cols)[col]
              .transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
        )
        df[f"{col}_roll{window}_std"] = (
            df.groupby(group_cols)[col]
              .transform(lambda s: s.shift(1).rolling(window, min_periods=2).std())
        )
```

High-value lag windows:

- daily: 1, 2, 3, 7, 14, 28, 56
- retail: 7, 14, 30, 60, 140, 365
- hourly energy: 24, 48, 72, 168, 336
- market: recent rows/days with purge where labels overlap

Add differences and ratios:

- short mean - long mean
- short mean / long mean
- decayed mean
- days since last positive sale/activity
- count of active/sale/promo days

### 4.7 Historical aggregate before cutoff

For labels or future horizons, build features from history before a cutoff.

`Implementation Pattern`:

```python
hist = data[data[date_col] < cutoff_date]

agg = (
    hist.groupby(group_cols)[target]
        .agg(["min", "mean", "median", "max", "count", "std"])
        .add_prefix(f"{target}_hist_")
        .reset_index()
)

features = labels.merge(agg, on=group_cols, how="left")
```

This pattern applies to restaurant visitors, store-item forecasting, energy, customer
revenue, and entity-level future targets.

### 4.8 Calendar, holidays, promotions, and known future

Useful features:

- day of week, week, month, quarter, year
- day of year and cyclical sin/cos variants
- holiday, before holiday, after holiday, bridge day, transferred holiday
- store open/closed, school holiday, public holiday
- promotion flags before and during forecast horizon
- price/discount known at prediction time
- weather or external calendar known before prediction

Use only features known at inference. If a future promotion schedule is known in
test, it can be used; if it is not known in production, do not synthesize it
from labels.

### 4.9 Text, sparse, and multimodal features

For listings/essays/logs:

- normalize text fields and concatenate selected combinations
- word and character TF-IDF/count n-grams
- SVD/TruncatedSVD of sparse matrices for GBDT
- Ridge/ElasticNet/SVR/FTRL/FM over sparse features
- sparse MLP/CNN with text and category embeddings
- pretrained text embeddings or transformer OOF predictions
- image embeddings or confidence/quality features
- metadata GBDT over dense features plus OOF modality predictions

`Implementation Pattern`:

```python
price_log = np.log1p(train["price"])

text_matrix = FeatureUnion([
    ("title_tfidf", title_vectorizer),
    ("description_tfidf", description_vectorizer),
    ("category_count", category_vectorizer),
]).fit_transform(all_rows)

ridge_oof, ridge_test = cross_fit_ridge(
    text_matrix[:len(train)],
    price_log,
    text_matrix[len(train):],
)

dense_train["ridge_text_pred"] = ridge_oof
dense_test["ridge_text_pred"] = ridge_test
X_train = scipy.sparse.hstack([dense_sparse_train, text_matrix[:len(train)]])
X_test = scipy.sparse.hstack([dense_sparse_test, text_matrix[len(train):]])
```

Adaptation notes:

- the snippet is a pattern, not exact runnable code
- use OOF text predictions before feeding them into a dense model
- preserve memory by using sparse matrices and limited SVD dimensions

### 4.10 High-dimensional and multioutput features

For sparse matrices, fMRI, climate columns, or gene outputs:

- standardize or normalize per feature family
- PCA/SVD/ICA/IncrementalPCA for large feature blocks
- per-block statistics: mean, std, quantiles, norms
- batch/site/donor/day indicators and grouped validation
- target compression for thousands of target columns
- per-target weights and masks

`Implementation Pattern`:

```python
input_svd = TruncatedSVD(n_components=128, random_state=42)
target_svd = TruncatedSVD(n_components=128, random_state=42)

X_low = input_svd.fit_transform(sparse_inputs)
Y_low = target_svd.fit_transform(sparse_targets)

model.fit(X_low[tr_idx], Y_low[tr_idx])
Y_pred_low = model.predict(X_low[va_idx])
Y_pred = Y_pred_low @ target_svd.components_
```

Use this only when reconstructing dense target vectors is the scored output.

## 5. Model Families

### 5.1 LightGBM

Use LightGBM when:

- rows are large and features are mixed numeric/categorical encodings
- fast iteration and feature importance matter
- missing values and nonlinear interactions are common
- panel forecasting uses many lag features

Useful objectives:

- `regression`, `rmse`, `l2`
- `mae`, `regression_l1`
- `huber`, `fair`
- `tweedie`, `poisson`, `gamma`
- quantile for interval routes

Tune:

- `num_leaves`, `max_depth`
- `learning_rate`, `n_estimators`
- `min_child_samples`
- `feature_fraction`, `bagging_fraction`
- `lambda_l1`, `lambda_l2`
- categorical handling or encoded categories

### 5.2 XGBoost

Use XGBoost when:

- strong regularized trees are needed
- GPU hist is available
- categorical support can be used safely
- target aggregates and feature interactions are many
- robust early stopping is important

Common parameters:

- `max_depth`
- `learning_rate`
- `n_estimators`
- `subsample`
- `colsample_bytree` or `colsample_bynode`
- `min_child_weight`
- `reg_alpha`, `reg_lambda`
- `tree_method="hist"` or GPU equivalent

XGBoost was repeatedly strong in insurance-loss regression, synthetic-tabular, pricing, and finance
tasks. It is especially useful when feature engineering creates many
interaction-like statistics.

### 5.3 CatBoost

Use CatBoost when:

- categorical features are important
- target leakage from encodings is risky
- quantile objectives are needed
- data is medium-sized and CPU/GPU runtime is acceptable

Useful settings:

- `loss_function="RMSE"` or `"MAE"`
- `loss_function="Quantile:alpha=0.05"` for intervals
- `cat_features=[...]`
- `depth` or `grow_policy`
- `l2_leaf_reg`
- `random_strength`
- `bootstrap_type`

CatBoost is often a reliable first model for categorical-heavy regression and a
strong diversity member in GBDT blends.

### 5.4 Regularized linear and sparse models

Use Ridge/ElasticNet/Lasso/Huber/KRR/SVR when:

- features are sparse text/OHE
- the dataset is small
- the target is log-transformed and mostly additive
- base predictions need a stable positive linear blend
- high-dimensional PCA/SVD features are used

For text-heavy product listings, sparse linear models are often not just
baselines; they become strong first-layer models.

### 5.5 Neural tabular models

Use NNs when:

- categorical embeddings are important
- text/image/scientific arrays need representation learning
- DAE/SAE features add diversity
- finance or synthetic data benefits from smooth nonlinear representations
- multioutput target vectors need shared heads

Model families:

- MLP with embeddings for categorical features
- sparse MLP/FM/FFM for product text/category matrices
- DAE/SAE/Transformer autoencoder features followed by MLP/Ridge
- TabNet and TabM/RealMLP-style tabular nets
- 1D CNN/U-Net/ConvNeXt/BiLSTM/Transformer for vertical scientific profiles

Delay NNs until GBDT validation and preprocessing are correct, unless text,
arrays, or multioutput structure make NNs the natural first model.

### 5.6 Forecasting-specific models

Tabular forecasting usually starts with GBDT over engineered lags. Add sequence
models when:

- series are long
- horizons are multiple and correlated
- many entities share patterns
- recursive inference is manageable
- validation supports the added complexity

Model options:

- direct LightGBM/XGBoost per horizon
- one GBDT with horizon/day features
- recursive GBDT/NN that updates lag state
- MLP/LSTM/GRU/Temporal CNN/WaveNet-like models
- per-store/item/item-class models when behavior differs strongly

### 5.7 Finance models

Robust finance stacks usually include:

- LightGBM/XGBoost/CatBoost
- Ridge/ElasticNet on standardized features
- MLP/GRU/TabNet for diversity
- fold/seed/time-window ensembles
- prediction clipping
- neutralization or standardization only if OOF improves and it is legal

Optimize the group-level metric and information flow, not raw model loss.

## 6. Forecasting And Online Inference

### 6.1 Direct vs recursive horizons

Direct:

- train separate model per horizon/day
- less error accumulation
- more models and memory
- good when horizon-specific patterns differ

Recursive:

- train one-step or short-step model and feed predictions forward
- can use many past lag values
- risk of drift and error accumulation
- requires local replay that updates lag features exactly

One model with horizon feature:

- simpler than many direct models
- can share data across horizons
- may underfit day-specific behavior

### 6.2 store-item retail feature windows

High-value feature families:

- nearest-day windows: 1, 3, 7, 14, 30, 60, 140
- same-day-of-week windows: last 4/20 occurrences
- short minus long means
- decayed means
- promotion-specific means
- no-promotion means
- zero-sale proportion
- days since first/last positive sale
- item/store/class aggregate windows
- holiday/promo/open calendar features

Validation should use a historical cutoff and official forecast horizon.

### 6.3 Known-future covariates

Known future features can be decisive:

- promotion schedule
- calendar and holidays
- open/closed schedule
- weather forecast if available at prediction time
- price if test includes it
- reservations already made before prediction

If a covariate would not be known at prediction time, simulate the missingness
in validation before using it.

For reservation, booking, order, quote, or forecast tables, filter by both the
future target date and the information timestamp. A row is usable only if it
would already be known at the forecast cutoff.

`Implementation Pattern`:

```python
reserve_window = reserve[
    (reserve[visit_date_col] >= cutoff_date)
    & (reserve[visit_date_col] < label_end_date)
    & (reserve[reserve_date_col] < cutoff_date)
]

reserve_features = (
    reserve_window.groupby(group_cols)
        .agg({"reserve_visitors": ["sum", "mean", "count"]})
        .reset_index()
)
```

### 6.4 Online lag state

For API tasks, maintain state exactly as the evaluator gives it.

`Implementation Pattern`:

```python
lag_state = None

def predict(test, lags):
    global lag_state

    if lags is not None:
        lag_state = (
            lags.group_by(["date_id", "symbol_id"], maintain_order=True)
                .last()
        )

    if lag_state is not None:
        test = test.join(lag_state, on=["date_id", "symbol_id"], how="left")

    preds = model.predict(test[features])
    return preds
```

Adaptation notes:

- update state only when new lag labels are provided
- never peek into future API batches
- replay the local validation loop in the same order
- do not fill missing lags with future-derived values

For online forecasting tasks where each batch reveals new target, weather, or
forecast records, append the newly revealed tables, rebuild features, predict,
clip if required, and submit. Do not recompute history from future batches.

`Implementation Pattern`:

```python
df_forecast = pl.concat([df_forecast, df_new_forecast]).unique()
df_historical = pl.concat([df_historical, df_new_historical]).unique()
df_target = pl.concat([df_target, df_new_target]).unique()

X_test = make_features(
    test_batch,
    df_forecast=df_forecast,
    df_historical=df_historical,
    df_target=df_target,
)

sample_prediction["target"] = model.predict(X_test[features]).clip(0)
env.predict(sample_prediction)
```

### 6.5 Panel submission reconstruction

Forecasting submissions often require exact row order and horizon columns.

Checks:

- every sample id appears once
- horizon/day columns are in the right order
- negative sales/count predictions are clipped if metric/submission forbids
- missing store-item combinations are filled with a validated fallback
- recursive predictions update only internal lag state, not validation labels
- known future promotions/calendar align to the correct dates

## 7. Ensembling, Stacking, And Blending

### 7.1 OOF discipline

For every serious model, save:

- fold assignment
- OOF prediction at the official scoring unit
- test prediction
- target transform used
- feature list
- validation score
- random seed and model config

Do not stack without OOF predictions.

### 7.2 OOF stacking

`Implementation Pattern`:

```python
train_stack = np.column_stack([oof_preds[name] for name in model_names])
test_stack = np.column_stack([test_preds[name] for name in model_names])

meta = RidgeCV(alphas=[0.1, 1.0, 10.0])
meta.fit(train_stack, y)
pred = meta.predict(test_stack)
```

Use positive Ridge/ElasticNet when weights should be nonnegative. For RMSLE,
consider stacking in log space and inverse-transforming after the blend.

### 7.3 Constrained blend search

`Implementation Pattern`:

```python
def blend_loss(w):
    return metric(y_valid, oof_matrix @ w)

res = minimize(
    blend_loss,
    x0=np.ones(oof_matrix.shape[1]) / oof_matrix.shape[1],
    bounds=[(0, 1)] * oof_matrix.shape[1],
    constraints={"type": "eq", "fun": lambda w: w.sum() - 1},
)

blended_test = test_matrix @ res.x
```

Use this late, after the model pool is strong. If public/private drift is high,
prefer simpler blends with stable OOF evidence.

### 7.4 Hill climbing

Hill climbing can work when many OOF/test prediction files exist:

1. start from the strongest OOF model
2. try adding one model at a time
3. keep a model only if OOF metric improves
4. optimize weight on the new member
5. stop when no remaining model helps

Risks:

- overfits one validation split
- rewards correlated noise
- can overweight public predictions
- needs stable OOF and a plausible split

### 7.5 Metric-specific blends

Blend geometry:

- RMSE: average or Ridge on original/target-transformed predictions
- RMSLE: blend in log space or original space, whichever OOF validates
- Pearson: standardize/rank predictions per group if scale is unstable
- C-index: rank-average risk scores
- prediction intervals: blend lower and upper quantile predictions separately,
  then conformalize
- multioutput: blend per target or target family

## 8. Nonstandard Route Libraries

### 8.1 Text/multimodal price and demand

Strong first implementation:

- parse text/category/image/metadata fields
- log-transform price/revenue if metric supports it
- TF-IDF word and char n-grams for title/description/name
- Ridge/ElasticNet/SVR OOF text model
- LightGBM/CatBoost over metadata plus text OOF predictions
- sparse MLP/CNN/FM if text matrix is large and runtime allows
- simple OOF blend

Try next:

- different tokenization/stemming/no stemming datasets
- numeric vectorization from item descriptions
- image embeddings from pretrained CNNs
- FastText/SVD embeddings
- category-specific models only if a global model cannot learn interactions

Avoid:

- hand-engineered text features that do not improve OOF
- huge transformer finetuning as first round when sparse text is strong
- treating rows as independent if user/item repeats create leakage

### 8.2 Scientific multioutput

Strong first implementation:

- group/batch/site-aware split
- normalize inputs and targets
- low-rank feature or target compression if matrices are huge
- MLP/1D CNN/TabNet/GBDT baseline depending on input geometry
- per-target score report
- OOF and test predictions in exact column order

Try next:

- per-target-family models
- target-specific clipping or masks
- autoencoder or denoising pretraining
- batch-specific finetuning only if validation supports it
- ensemble architectures that specialize by target family

Avoid:

- random row split across donor/day/site when the test is grouped
- one global metric hiding bad target columns
- unscored targets dominating loss

### 8.3 Survival and C-index

Strong first implementation:

- event/censoring audit
- race/fairness-group fold diagnostics when metric is stratified
- risk labels from Kaplan-Meier/Nelson-Aalen/Cox or rank transforms
- GBDT risk-score model and C-index scorer
- rank-average OOF/test predictions

Try next:

- pairwise ranking NN with censored-pair mask
- event classifier mask plus risk model
- separate models for event probability and time/risk
- fairness-group calibration only if it improves stratified metric without
  leakage

Avoid:

- direct RMSE on censored times
- interpreting censored rows as event-free forever
- improving mean C-index while worsening group dispersion

### 8.4 Prediction intervals

Strong first implementation:

- train 5th and 95th quantile models
- hold out calibration set
- conformalize lower/upper residuals
- validate coverage and official Winkler/pinball score
- enforce lower <= upper

Try next:

- mean plus residual variance model
- cross-conformal calibration
- multiquantile NN/TabM
- detrended target then add trend back
- separate interval widths by segment if calibration supports it

Avoid:

- submitting point predictions as intervals
- tuning only interval width without coverage
- using calibration data for model training and calibration simultaneously

### 8.5 Geolocation and trajectory correction

Strong first implementation:

- start from provided baseline coordinates if available
- group validation by phone/route/site
- train correction models with sensor and context features
- remove outliers and interpolate only in validation-safe ways
- smooth trajectories and score distance percentiles

Try next:

- Kalman filtering or weighted least squares
- median/variance fusion across devices
- stop-state averaging
- snap-to-grid/shape only when legal and validated

Avoid:

- row RMSE validation when metric is percentile distance per route/phone
- using ground-truth snapping from test data

### 8.6 Runtime/config ranking

Strong first implementation:

- group by graph/model
- train runtime scorer
- validate ranking metric per graph
- output sorted configuration ids
- compare point loss and rank loss

Try next:

- graph neural networks or graph pooling
- configuration embeddings
- pairwise/listwise ranking losses
- ensemble GBDT and neural scorers

Avoid:

- treating each config row as independent when output is a per-graph
  permutation
- optimizing RMSE without checking top-k slowdown or Kendall tau

## 9. Route-Specific Playbooks

### 9.1 Flat regression

Strong first implementation:

- KFold or target-binned KFold
- exact metric and submission validator
- LGBM, XGB, CatBoost
- missing flags, row stats, categorical frequencies
- OOF/test prediction files
- simple average or Ridge blend

Try next:

- target aggregates with nested folds
- feature crosses and ratios
- TabM/MLP/DAE diversity
- hill-climbing blend

Avoid:

### 9.2 Skewed positive regression

Strong first implementation:

- log1p target
- GBDT on transformed target
- inverse-transform OOF and score official metric
- nonnegative clipping
- robust outlier audit

Try next:

- Tweedie/Gamma/Poisson objectives
- shifted log or power transforms
- quantile/Huber/Fair objectives
- segment-specific residual models

Avoid:

- negative submissions under RMSLE
- deleting many high-target rows without OOF proof

### 9.3 Categorical-heavy regression

Strong first implementation:

- CatBoost native categorical
- XGBoost categorical or LGBM with encoded categories
- count/frequency encodings
- fold-safe target statistics
- category interactions with smoothing

Try next:

- entity embeddings
- high-order interactions selected by OOF
- sparse OHE + Ridge/Vowpal/FTRL/FM
- target histograms/quantiles for key categories

Avoid:

- target encoding fitted on validation target
- uncontrolled category combinations that memorize rows

### 9.4 Entity/relational regression

Strong first implementation:

- define entity id and prediction window
- aggregate child tables to entity rows
- group/time validation
- recency/frequency/monetary/status aggregates
- GBDT ensemble

Try next:

- two-stage positive-probability times amount
- sequence features from recent events
- entity embeddings or user2vec/item2vec
- adversarial validation for train/test entity shift

Avoid:

- random session rows when submission is per user
- using target-period transactions in features

### 9.5 Panel forecasting

Strong first implementation:

- rolling-origin validation
- direct LGBM/XGB per horizon or one model with horizon feature
- lag/rolling/decay/calendar/known-future features
- nonnegative clipping for sales/counts
- official horizon submission reconstruction

Try next:

- recursive model with local replay
- NN/WaveNet/LSTM diversity
- per-store/item/class models
- weighted metric sample weights
- promotion/open/holiday special handling

Avoid:

- random row folds
- future leakage in rolling features
- last-day public-LB tricks that validation cannot support

### 9.6 Finance and online regression

Strong first implementation:

- forward or purged group-time validation
- exact metric: Pearson/R2/RMSPE/Sharpe-like/rank
- LGBM/XGB/CatBoost plus Ridge baseline
- lag-safe cross-sectional and rolling features
- API replay with state
- prediction clipping where appropriate

Try next:

- TabNet/MLP/GRU diversity
- time-window ensembles
- per-time standardization/ranking if metric supports it
- neutralization or rank postprocess only with OOF proof

Avoid:

- random folds as primary evidence
- hidden test order reconstruction
- public-test-in-train leakage

### 9.7 Text/multimodal regression

Strong first implementation:

- sparse text Ridge OOF
- metadata GBDT
- image/text embeddings if available
- log target for prices/revenue
- OOF blend

Try next:

- sparse MLP/CNN/FM
- multiple tokenization datasets
- pretrained transformer OOF predictions
- image quality/object confidence features

Avoid:

- ignoring text when it is the primary signal
- directly stacking in-sample text predictions

### 9.8 Multioutput scientific regression

Strong first implementation:

- group/batch/site split
- input/target normalization
- MLP/1D CNN/GBDT baseline
- per-target OOF metrics
- target masks/weights

Try next:

- SVD/PCA target compression
- per-target-family ensembles
- domain-specific physics features
- autoencoder pretraining

Avoid:

- one-size-fits-all target loss
- random split across known domain-shift groups

### 9.9 Survival/risk

Strong first implementation:

- route as survival/ranking
- risk-target transform
- C-index scorer
- event/fairness group diagnostics
- GBDT risk model and rank ensemble

Try next:

- pairwise ranking NN
- Cox/KM/NA label variants
- event-probability mask

Avoid:

- direct time RMSE
- ignoring censoring and stratification

### 9.10 Prediction intervals

Strong first implementation:

- quantile lower and upper models
- calibration split
- conformal correction
- interval submission validator

Try next:

- cross-conformal
- residual variance model
- multiquantile NN/TabM

Avoid:

- point-estimate-only route
- unconstrained lower/upper crossing

### 9.11 Geolocation/trajectory

Strong first implementation:

- route/phone/site groups
- baseline correction model
- outlier removal and smoothing
- percentile-distance metric

Try next:

- Kalman/WLS
- device fusion
- stop averaging

Avoid:

- ground-truth/test snapping unless explicitly provided and legal

### 9.12 Config ranking/runtime

Strong first implementation:

- group by graph/config set
- runtime scorer
- ranking output builder
- top-k/Kendall metric

Try next:

- GNN or pairwise ranker
- graph feature pooling
- ensemble scorer

Avoid:

- row-wise RMSE as the only validation
