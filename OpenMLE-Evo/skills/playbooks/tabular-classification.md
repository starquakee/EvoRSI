# Tabular Classification Category Playbook

This playbook is the full method library for Kaggle-style `tabular_classification`
tasks. It is intended as direct reading material for a skill-generating agent:
after reading a task description, the agent should route the task, select the
right validation strategy, then extract only the applicable recipes.

Do not use this playbook to justify leaderboard probing, public-test label
inference, leakage-based recipes, public/private split reconstruction,
sample-submission label updates, manually inferred hidden labels, or disallowed
external data.

## 1. Task Routing

### 1.1 Flat row-wise classification

Use for ordinary CSV/parquet rows with independent-looking samples and binary or
multiclass target.

Default plan:

- stratified 5-fold CV
- LightGBM, CatBoost, XGBoost
- missingness and row-stat features
- OOF predictions for all models
- simple average/rank average before stackers

### 1.2 Categorical-heavy classification

Use when most signal is in categorical/object/id columns.

Default plan:

- CatBoost native categorical baseline
- sparse one-hot or label encoding for linear/GBDT models
- count/frequency encodings
- fold-safe target encodings only when CV design is stable
- feature crosses for meaningful category pairs

### 1.3 Relational entity classification

Use when the task has multiple tables per entity: credit applications, insurance
claims, customer risk, medical history, previous transactions.

Default plan:

- define the entity key before feature engineering
- build one row per prediction entity
- aggregate child tables by entity
- create status-specific aggregates such as active/closed/refused/approved
- validate by entity/time if repeated entities or temporal drift exist

### 1.4 Temporal, fraud, click, session, and online tasks

Use when rows have event time, user/card/device/session IDs, market dates, or a
hidden online test API.

Default plan:

- sort by time and create forward validation
- use group-time or purged validation if entities repeat
- compute lag/rolling/count features without future targets
- maintain online state exactly as the evaluator reveals information
- avoid random KFold as primary evidence

### 1.5 Multi-label and high-dimensional scientific tabular

Use for many simultaneous binary targets, assays, gene/cell features, fMRI, or
molecular target panels.

Default plan:

- multilabel or group-aware multilabel folds
- normalized continuous features, PCA or low-rank summaries
- NN + GBDT/linear diversity
- per-target log loss/AUC audit
- group-aware validation by drug/compound/batch when available

### 1.6 Ranking or recommender-like tabular

Use only for the candidate scoring layer of recommender/matching tasks. If the
core problem is candidate generation or a full ranking system, route out to a
recsys/matching playbook.

Default plan:

- build candidate pairs/groups first
- validate candidate recall separately from scorer quality
- train a classifier/ranker on candidate rows
- enforce group-wise top-k/rank output format

### 1.7 Sports and tournament forecasting

Use for game/team matchup probability forecasting.

Default plan:

- season holdout
- one row per matchup
- seed/rating/Elo/recent-form differences
- logistic regression or GBDT
- calibration for log loss/Brier

### 1.8 Route out

Route out:

- interactive game-policy tasks, constrained-permutation optimization
- survey-only or visualization-only competitions
- CTF/exploit tasks
- pure clustering/matching/recommender tasks when no row-wise classifier is the
  main solution
- image/video/NLP tasks that are only accidentally under this category

## 2. First-Pass Checklist

Before modeling, extract:

- prediction unit: row, entity, event, pair, session, game, user-item candidate,
  compound-target, or online batch
- metric: AUC/Gini, log loss, balanced log loss, accuracy, F1/MCC, MAP/NDCG,
  QWK, custom utility
- target shape: binary, multiclass, ordinal, multi-label, ranking
- validation axis: time, entity, user/card/customer, session, season, compound,
  batch, generated source
- feature sources: single table, multiple tables, event logs, time series, text
  or SMILES fields, external allowed tables
- inference mode: static CSV, hidden online API, ranking groups, top-k rows
- leakage risk: duplicates, future data, target-like fields, public test
  structure, sample submission labels, repeated entities

Then create:

- `folds.csv`
- exact metric implementation
- submission validator
- one fast baseline with OOF predictions
- feature importance and train/test shift audit

## 3. Validation And Leakage Control

### 3.1 Stratified CV

Use only when rows are plausibly independent. Reuse the same fold indices for all
base models if stacking.

- Otto used fixed folds for L1 models.
- synthetic tabular tasks used repeated seed/fold checks.
- Santander transaction reused folds to avoid ensemble leakage.

### 3.2 Group CV

Use when groups repeat across rows: customer, card, user, session, patient,
compound, drug, source, season.

For grouped targets, check fold positive counts and target distribution. If
group-stratified folds are unavailable or unstable, manually repair folds.

```python
# stratified folds grouped by week for stability-style credit risk.
X = df_train.drop(columns=["target", "case_id", "WEEK_NUM"])
y = df_train["target"]
weeks = df_train["WEEK_NUM"]

cv = StratifiedGroupKFold(n_splits=5, shuffle=False)
for tr_idx, va_idx in cv.split(X, y, groups=weeks):
    X_train, y_train = X.iloc[tr_idx], y.iloc[tr_idx]
    X_valid, y_valid = X.iloc[va_idx], y.iloc[va_idx]
    model.fit(X_train, y_train)
    oof[va_idx] = model.predict_proba(X_valid)[:, 1]
```

### 3.3 Time and group-time CV

Use when test is later than train, hidden API is chronological, or features are
rolling/lagged.

```python
# chronological session validation.
train_df = train_df.sort_values(by="time1")
time_split = TimeSeriesSplit(n_splits=10)
cv_scores = cross_val_score(model, X_train, y_train, cv=time_split,
                            scoring="roc_auc")
```

Use forward validation for model selection. If using whole-period GroupKFold,
state why it approximates the hidden test.

```python
# month-grouped validation for fraud.
oof = np.zeros(len(X_train))
preds = np.zeros(len(X_test))
cv = GroupKFold(n_splits=6)

for fold, (tr_idx, va_idx) in enumerate(cv.split(X_train, y_train, groups=X_train["DT_M"])):
    clf = xgb.XGBClassifier(
        n_estimators=5000,
        max_depth=12,
        learning_rate=0.02,
        subsample=0.8,
        colsample_bytree=0.4,
        eval_metric="auc",
        tree_method="gpu_hist",
    )
    clf.fit(
        X_train[cols].iloc[tr_idx], y_train.iloc[tr_idx],
        eval_set=[(X_train[cols].iloc[va_idx], y_train.iloc[va_idx])],
        verbose=100,
        early_stopping_rounds=200,
    )
    oof[va_idx] = clf.predict_proba(X_train[cols].iloc[va_idx])[:, 1]
    preds += clf.predict_proba(X_test[cols])[:, 1] / cv.n_splits
```

### 3.4 Online state validation

For online streaming API tasks, validation must replay the information
release schedule.

```python
# update state only after prior answers are revealed.
previous_test_df[TARGET] = eval(test_df["prior_group_answers_correct"].iloc[0])
update_features(
    previous_test_df,
    answered_correctly_u_sum,
    answered_correctly_q_sum,
    timestamp_u_incorrect,
)
```

Do not use future labels. If the competition explicitly reveals previous batch
labels at inference, simulate exactly that behavior locally.

### 3.5 Adversarial validation

Use adversarial validation to detect train/test shift, not to infer test labels.
High adversarial AUC means random CV is weak evidence.

```python
# cross-validated train-vs-test classifier.
X_adv = pd.concat([X_train_processed, X_test_processed], axis=0)
y_adv = np.r_[np.zeros(len(X_train_processed)), np.ones(len(X_test_processed))]

adversarial_model = RandomForestClassifier(random_state=seed)
adv_pred = cross_val_predict(
    adversarial_model, X_adv, y_adv, cv=5, n_jobs=-1, method="predict_proba"
)
adv_auc = roc_auc_score(y_adv, adv_pred[:, 1])
```

### 3.6 CV/LB mismatch

Common causes:

- test is later in time
- generated/synthetic train differs from original test
- hidden public/private split is not random
- group leakage inflates CV
- target encodings leak
- pseudo-labels reinforce public-only artifacts
- stacker trained on in-fold predictions

Reduce reliance on public LB. Use robust folds, adversarial validation, and OOF
diagnostics.

## 4. Data Audit

### 4.1 Basic audit

Check:

- row and ID uniqueness
- train/test column parity
- target distribution and metric direction
- missingness per feature and per row
- sentinel values: `-1`, `999999`, `365243`, `-999`
- duplicate rows or duplicate entities
- time ranges by split
- categorical cardinality and unseen test categories
- numerical ranges and train/test drift

### 4.2 Leakage audit

Search for:

- target-like future columns
- row order leakage
- public/private split reconstruction
- identifiers that encode labels
- child-table rows after prediction time
- test rows duplicated from train
- sample submission with known labels

Exclude task-specific identity and join leaks.

### 4.3 Memory and dtype audit

Large click/fraud tasks require dtype control.

Use smaller integer dtypes for IDs, categorical codes, and binary targets.
Persist intermediate features with parquet/feather when repeated feature
engineering is expensive.

## 5. Feature Engineering

### 5.1 Categorical encoding

Use method by model and cardinality:

- one-hot for sparse linear models and low-cardinality categoricals
- CatBoost native categoricals for raw categorical-heavy tables
- label encoding for GBDT only when order is not interpreted by the model in a
  harmful way
- frequency/count encodings for high-cardinality IDs
- fold-safe target encoding when validation is stable
- entity embeddings for neural nets

```python
# fold-safe target encoding with prior fallback.
def oof_target_encode(train, test, col, target_col, fold_col, smoothing=20):
    prior = train[target_col].mean()
    train_encoded = pd.Series(prior, index=train.index, dtype=float)

    for fold in sorted(train[fold_col].dropna().unique()):
        tr_mask = train[fold_col] != fold
        va_mask = train[fold_col] == fold
        stats = train.loc[tr_mask].groupby(col)[target_col].agg(["mean", "count"])
        enc = (stats["mean"] * stats["count"] + prior * smoothing) / (
            stats["count"] + smoothing
        )
        train_encoded.loc[va_mask] = train.loc[va_mask, col].map(enc).fillna(prior)

    stats = train.groupby(col)[target_col].agg(["mean", "count"])
    enc = (stats["mean"] * stats["count"] + prior * smoothing) / (
        stats["count"] + smoothing
    )
    test_encoded = test[col].map(enc).fillna(prior)
    return train_encoded, test_encoded
```

Call this separately for each encoded column. Never compute target means from
validation rows or hidden test labels.

```python
# smoothed target encoding.
averages = temp.groupby(trn_series.name)[target.name].agg(["mean", "count"])
smoothing_value = 1 / (1 + np.exp(-(averages["count"] - min_samples_leaf) / smoothing))
prior = target.mean()
averages[target.name] = prior * (1 - smoothing_value) + averages["mean"] * smoothing_value
```

```python
# embedding dimension cap for high-cardinality categoricals.
embed_dim = int(min(np.ceil(num_unique_values / 2), 50))
inp = layers.Input(shape=(1,))
out = layers.Embedding(num_unique_values + 1, embed_dim, name=col)(inp)
out = layers.SpatialDropout1D(0.3)(out)
```

### 5.2 Frequency and count encoding

Frequency encodings are target-free and often strong for fraud/click/category
tasks.

```python
# frequency encoding using train+test feature values.
def encode_fe(train_df, test_df, cols):
    for col in cols:
        values = pd.concat([train_df[col], test_df[col]], axis=0)
        vc = values.value_counts(dropna=True, normalize=True).to_dict()
        train_df[f"{col}_FE"] = train_df[col].map(vc).fillna(-1).astype("float32")
        test_df[f"{col}_FE"] = test_df[col].map(vc).fillna(-1).astype("float32")
```

Using test feature values is transductive. Use only when rules allow feature-only
test distribution use.

### 5.3 Relational group aggregates

Group child tables to the prediction entity.

```python
# many-to-one customer history aggregation.
nb_bureau = bureau_full[["SK_ID_CURR", "SK_ID_BUREAU"]].groupby("SK_ID_CURR").count()
bureau_full["SK_ID_BUREAU"] = bureau_full["SK_ID_CURR"].map(nb_bureau["SK_ID_BUREAU"])
avg_bureau = bureau_full.groupby("SK_ID_CURR").mean()
data = data.merge(avg_bureau.reset_index(), how="left", on="SK_ID_CURR")
```

For Home-Credit-like data, also aggregate status-specific subsets:

- active vs closed bureau credits
- approved vs refused previous applications
- late vs on-time installments
- last record, trend, and ratio features

### 5.4 UID and fraud aggregates

Fraud solutions often create domain-specific entity keys and aggregate by them.

```python
# UID construction plus aggregate features.
X_train["day"] = X_train.TransactionDT / (24 * 60 * 60)
X_train["uid"] = (
    X_train.card1_addr1.astype(str) + "_" +
    np.floor(X_train.day - X_train.D1).astype(str)
)
encode_fe(X_train, X_test, ["uid"])
encode_ag(["TransactionAmt", "D4", "D9", "D10", "D15"],
          ["uid"], ["mean", "std"], fillna=True, usena=True)
```

Do not use hidden test labels or future outcomes in UID postprocessing. UID
features must be validated under the same information availability as inference.

### 5.5 Click/session time features

Use count, nunique, cumcount, next/previous event deltas, and rolling windows.

```python
# declarative click-fraud aggregate feature specs.
GROUPBY_AGGREGATIONS = [
    {"groupby": ["ip", "day", "hour"], "select": "channel", "agg": "count"},
    {"groupby": ["ip"], "select": "channel", "agg": "nunique"},
    {"groupby": ["ip", "device", "os"], "select": "app", "agg": "cumcount"},
]
```

```python
# next-click delta feature.
click_buffer = np.full(D, 3000000000, dtype=np.uint32)
next_clicks.append(click_buffer[category] - time)
click_buffer[category] = time
```

Future-looking deltas are only safe if they are target-free and available at
inference for the full batch. Prefer past-only lags for online tasks.

### 5.6 Session text/ngram features

For ordered categorical sessions, sparse text-style models can be strong.

```python
# web session sites as ngram text.
sites = [f"site{i}" for i in range(1, 11)]
train_df[sites].fillna(0).astype("int").to_csv("train_sessions_text.txt",
                                                sep=" ", index=None, header=None)
cv = CountVectorizer(ngram_range=(1, 3), max_features=50000)
X_train = cv.fit_transform(open("train_sessions_text.txt"))
```

### 5.7 Domain features

Small domain features can be worth more than model complexity.

```python
# compact family features.
dataset["FamilySize"] = dataset["SibSp"] + dataset["Parch"] + 1
dataset["IsAlone"] = 0
dataset.loc[dataset["FamilySize"] == 1, "IsAlone"] = 1
```

```r
# date decomposition.
train$month <- as.integer(format(train$Original_Quote_Date, "%m"))
train$year <- as.integer(format(train$Original_Quote_Date, "%y"))
train$day <- weekdays(as.Date(train$Original_Quote_Date))
```

### 5.8 Scientific and molecular features

```python
# rank-gauss normalization for high-dimensional assay columns.
transformer = QuantileTransformer(
    n_quantiles=100,
    random_state=0,
    output_distribution="normal",
)
train_features[col] = transformer.transform(raw_vec).reshape(1, vec_len)[0]
```

```python
# ECFP fingerprints from SMILES.
df["molecule"] = df["molecule_smiles"].apply(Chem.MolFromSmiles)

def generate_ecfp(molecule, radius=2, bits=1024):
    return list(AllChem.GetMorganFingerprintAsBitVect(molecule, radius, nBits=bits))
```

```python
# batch SMILES token IDs.
encoded_batch = tokenizer_fast.batch_encode_plus(
    smiles_chunk,
    max_length=max_length,
    padding="max_length",
    return_tensors="np",
    truncation=True,
)["input_ids"]
```

### 5.9 Sports matchup features

```python
# Elo update in season/day order using only known games.
historical_tournament = tournament[
    (tournament["Season"] < prediction_season)
    | ((tournament["Season"] == prediction_season) & (tournament["DayNum"] < prediction_daynum))
]
regular_before_prediction = regular_season[
    (regular_season["Season"] < prediction_season)
    | ((regular_season["Season"] == prediction_season) & (regular_season["DayNum"] < prediction_daynum))
]
games = pd.concat([regular_before_prediction, historical_tournament, seed_games])
games = games.sort_values(["Season", "DayNum"])
win_expect = 1.0 / (1.0 + 10.0 ** ((elo[loser] - elo[winner] - home_adv) / 400.0))
```

Features should be computed only from games before the matchup date or from
pre-tournament information.

## 6. Model Families

### 6.1 LightGBM

Default workhorse for most tabular tasks. Strong for mixed numeric/categorical
features after encoding and for large feature sets.

Use:

- early stopping
- OOF predictions
- feature importance and drift audit
- high `min_child_samples` for noisy large data
- class weights or downsampling for extreme imbalance

### 6.2 CatBoost

Use when:

- raw categorical columns dominate
- target encoding leakage risk is high
- categories have missing/unseen values
- a diverse GBDT ensemble member is needed

CatBoost can be weaker in some high-cardinality categorical settings, so keep OOF
evidence.

### 6.3 XGBoost

Use when:

- sparse one-hot matrices are large
- missing values should be handled natively
- depth and interaction control are critical
- GPU hist is available

many large categorical tasks used XGBoost effectively.

### 6.4 Linear and sparse models

Use when:

- features are high-cardinality sparse one-hot/ngrams
- signal is additive
- data is small relative to categories
- fast reliable baseline is needed

high-cardinality categorical tasks and session ngram tasks show linear models can beat heavier
models when representation is right.

### 6.5 Neural nets and embeddings

Use when:

- high-dimensional continuous assay data
- multi-label targets
- categorical embeddings add diversity
- entity embeddings are meaningful
- GBDTs plateau and OOF diversity is needed

Do not start with TabNet/FT-Transformer/MLP for ordinary tabular unless the
task shape suggests it.

### 6.6 Special models

Use selectively:

- QDA/GMM for Gaussian-style synthetic clusters
- TabPFN for tiny tabular tasks if installed and within size limits
- CatBoostRanker/LGBMRanker/XGBRanker for grouped ranking
- RDKit fingerprints plus RF/GBDT/NN for molecule tasks

## 7. Training, Metrics, And Postprocessing

### 7.1 AUC and Gini

AUC/Gini reward ranking. Rank averaging often helps. Probability calibration is
secondary unless the competition also uses thresholds or log loss.

### 7.2 Log loss

Use calibrated probabilities, clip only at the metric/submission boundary, and
keep probability sums valid for multiclass.

### 7.3 Balanced log loss

Small medical tasks such as ICR use class-balanced log loss. Class prior
reweighting can improve the metric but must be tuned on OOF.

### 7.4 rank-weighted top-fraction metric

```python
# rank-weighted top-fraction metric shape.
g = normalized_weighted_gini(y_true, y_pred)
d = top_four_percent_captured(y_true, y_pred)
amex_score = 0.5 * (g + d)
```

Implement the full official helper locally when solving rank-weighted credit-risk tasks. Make
score direction explicit.

### 7.5 QWK and ordinal classes

Use regression/ordinal models plus threshold optimization.

```python
# optimized thresholds for QWK.
X_p = pd.cut(X, [-np.inf] + list(np.sort(coef)) + [np.inf], labels=[0, 1, 2, 3])
self.coef_ = sp.optimize.minimize(loss_partial, [0.5, 1.5, 2.5], method="nelder-mead")
```

Thresholds must be fitted on OOF or a calibration split.

### 7.6 Multi-label log loss

For multi-target molecular-assay tasks, average per-target log loss and inspect target-level losses.

Control rows with known zero targets can be zeroed only when the task definition
explicitly supports that rule.

### 7.7 Ranking metrics

For MAP@K, HitRate@K, NDCG, or ranking submissions:

- validate candidate recall
- train scorer/ranker inside groups
- output valid per-group rank permutations
- evaluate the exact top-k metric

```python
# grouped rank output.
submission["selected"] = submission.groupby("ranker_id")["pred_score"].rank(
    ascending=False,
    method="first",
).astype(int)
```

### 7.8 Threshold tuning

Tune thresholds only on OOF/holdout.

```python
# threshold choice from validation ROC curve.
fpr, tpr, thresholds = roc_curve(y_valid, valid_pred_prob)
youden_index = tpr - fpr
optimal_threshold = thresholds[np.argmax(youden_index)]
```

## 8. Ensembling And Stacking

### 8.1 OOF discipline

OOF predictions are mandatory for stacking, blend-weight optimization,
calibration, threshold search, pseudo-label selection, and label cleaning.

### 8.2 OOF stacking

```python
# OOF stacking matrix.
S_train = np.zeros((X.shape[0], len(base_models)))
S_test = np.zeros((T.shape[0], len(base_models)))

for i, clf in enumerate(base_models):
    S_test_i = np.zeros((T.shape[0], n_splits))
    for j, (tr_idx, va_idx) in enumerate(folds):
        clf.fit(X[tr_idx], y[tr_idx])
        S_train[va_idx, i] = clf.predict_proba(X[va_idx])[:, 1]
        S_test_i[:, j] = clf.predict_proba(T)[:, 1]
    S_test[:, i] = S_test_i.mean(axis=1)
```

The meta-model must train on `S_train`, not in-fold predictions.

### 8.3 Metric-optimized blend weights

```python
# log-loss blend objective.
def log_loss_func(weights):
    final_prediction = 0
    for weight, prediction in zip(weights, predictions):
        final_prediction += weight * prediction
    return log_loss(valid_y, final_prediction)
```

Use constraints such as sum(weights)=1 and bounds [0,1]. Optimize on OOF/holdout
only.

### 8.4 Hill climbing

Useful when many OOF prediction files exist. Optimize on OOF, apply the same
weights to test, and freeze before final submission. Do not hill-climb public LB.

### 8.5 Rank averaging

For AUC/Gini, rank average can beat probability average when models have
different calibration.

```python
# rank-average prediction files.
from scipy.stats import rankdata

predictions = np.zeros_like(predict_list[0])
for predict in predict_list:
    for i in range(predictions.shape[1]):
        predictions[:, i] += rankdata(predict[:, i]) / predictions.shape[0]
predictions /= len(predict_list)
```

### 8.6 Diversity

High-value diversity:

- LGBM + CatBoost + XGBoost
- different feature sets
- sparse linear + GBDT
- NN embeddings + GBDT
- different folds/seeds when variance is high
- ranking/regression formulation plus classifier for sports and ordinal tasks

Low-value diversity:

- dozens of near-identical seeds
- public-LB-weighted blends
- stackers with leaked in-fold predictions

## 9. Pseudo-Labels And External Data

Pseudo-labeling can help when:

- validation is trustworthy
- unlabeled/test-like data is allowed
- labels are high confidence
- class balance is controlled

```python
# high-confidence pseudo-labeling.
test_conf = test_pred[(test_pred["target"] <= 0.01) | (test_pred["target"] >= 0.99)].copy()
test_conf.loc[test_conf["target"] >= 0.5, "target"] = 1
test_conf.loc[test_conf["target"] < 0.5, "target"] = 0
train_aug = pd.concat([train_fold, test_conf], axis=0)
```

Risks:

- confident majority classes dominate
- pseudo-labels capture public-test artifacts
- CV improves while private score drops
- rule violations if test labels are inferred or probed

External data:

- use only if allowed
- validate domain shift
- keep internal-only baseline
- never use hidden labels, public-test rows, or external leaks

## 10. Subtype Cookbooks

### 10.1 synthetic-tabular / synthetic tabular

Methods:

- simple strong GBDT baseline
- row missing count
- original-data matching only if rules permit and not hidden-test leakage
- repeated folds/seeds
- OOF stacking or hill climbing
- adversarial validation for generated train/test mismatch

Cautions:

### 10.2 Categorical-heavy

Methods:

- CatBoost native categoricals
- sparse one-hot + logistic regression
- count/frequency encodings
- fold-safe target encoding
- category embeddings as NN diversity
- feature crosses for meaningful pairs

Cautions:

- target encoding leakage is easy
- ordinal encoding can invent false order
- train/test category drift needs unknown handling

### 10.3 Relational credit/insurance

Methods:

- entity-level table assembly
- child-table aggregates
- status-specific aggregates
- ratios such as payment rate, income/credit, overdue rate
- time/entity validation
- LGBM/CatBoost/XGB stack

Cautions:

- joining child rows directly duplicates targets
- future records can leak
- instability metrics may penalize week-to-week variance

### 10.4 Fraud/click/session

Methods:

- time/group-time validation
- UID/entity construction
- count/frequency/nunique/cumcount
- next/previous click deltas
- target-free historical rates
- class weights/downsampling
- AUC/rank blends

Cautions:

- random CV is often invalid
- postprocessing with hidden labels is excluded
- future-looking features require proof of inference availability

### 10.5 Market/online

Methods:

- local replay of hidden API
- purged/group time split
- rolling and lag features
- state dictionaries updated only when labels become available
- simple robust threshold policy
- ensemble only if latency allows

Cautions:

- shuffled CV is misleading
- online state bugs silently ruin submissions
- utility metrics may not track AUC/log loss

### 10.6 Multi-label multi-target molecular-assay

Methods:

- MultilabelStratifiedKFold
- drug/compound-aware folds when repeated compounds matter
- QuantileTransformer/rank-gauss
- PCA/variance threshold on gene/cell blocks
- BCEWithLogits NN
- non-scored target pretraining
- diverse NN/TabNet/GBDT averaging

```python
# keep repeated drug_ids in the same fold when possible.
vc = train["drug_id"].value_counts()
repeated_drugs = vc[vc > 1].index
single_drugs = vc[vc == 1].index

drug_targets = train.groupby("drug_id")[target_cols].mean().loc[repeated_drugs]
drug_folds = np.zeros(len(drug_targets), dtype=int)

mskf = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
for fold, (_, va_idx) in enumerate(mskf.split(drug_targets, drug_targets[target_cols])):
    drug_folds[va_idx] = fold

fold_map = dict(zip(drug_targets.index, drug_folds))
train["kfold"] = train["drug_id"].map(fold_map)

single_rows = train["drug_id"].isin(single_drugs)
for fold, (_, va_idx) in enumerate(mskf.split(train.loc[single_rows, feature_cols],
                                             train.loc[single_rows, target_cols])):
    train.loc[train.loc[single_rows].index[va_idx], "kfold"] = fold
```

```python
# auxiliary-label transfer.
train_model(pretrained_model, "ALL_TARGETS", all_target_cols)
final_model = fine_tune_scheduler.copy_without_top(
    pretrained_model,
    num_features,
    num_all_targets,
    num_targets,
)
oof = train_model(final_model, "SCORED_ONLY", target_cols, fine_tune_scheduler)
```

Cautions:

- blend weights across incompatible CV schemes can overfit
- rare targets need per-target diagnostics
- known-control zeroing must be task-defined

### 10.7 Molecular Binding

Methods:

- split by building block/compound if hidden test has unseen chemistry
- ECFP fingerprints
- SMILES tokenization
- 1D CNN/Transformer/GNN as specialized models
- protein-target multi-output head
- average precision by target/split group

```python
# 1D CNN predicts three protein targets.
x = tf.keras.layers.Embedding(input_dim=36, output_dim=hidden_dim, mask_zero=True)(inputs)
x = tf.keras.layers.Conv1D(filters=NUM_FILTERS, kernel_size=3, activation="relu")(x)
outputs = tf.keras.layers.Dense(3, activation="sigmoid")(x)
```

Cautions:

- ordinary tabular GBDT may miss chemistry structure
- random split can overestimate generalization to unseen building blocks

### 10.8 Small biomedical tabular

Methods:

- repeated stratified/group CV
- robust imputation
- XGB/LGBM/CatBoost
- TabPFN if available and size-appropriate
- balanced log loss or AUC metric implementation
- probability reweighting calibrated on OOF

```python
# small biomedical ensemble ingredients.
self.imputer = SimpleImputer(missing_values=np.nan, strategy="median")
self.classifiers = [
    xgboost.XGBClassifier(...),
    xgboost.XGBClassifier(),
    TabPFNClassifier(N_ensemble_configurations=24),
]
```

Cautions:

- tiny public LB is unstable
- later-time hidden tests require time/epsilon awareness
- aggressive feature selection can be high variance

### 10.9 Sports tournament

Methods:

- season holdout
- seed differential
- Elo/team-quality features
- point differential and recent form
- GBDT/regression-to-margin then calibration
- log loss/Brier calibration

```python
# matchup features and calibrated probabilities.
features = ["men_women", "T1_seed", "T2_seed", "Seed_diff",
            "T1_elo", "T2_elo", "elo_diff", "T1_quality", "T2_quality"]
params["objective"] = "reg:squarederror"
probs = np.clip(spline_model(np.clip(margin_preds, -t, t)), 0.01, 0.99)
```

Cautions:

- future tournament results must not enter regular-season features
- random game split across seasons is weak
- bracket portfolio tasks are not ordinary row-wise classification

### 10.10 Ranking/candidate scoring

Methods:

- candidate generation first
- one row per candidate
- group-wise validation
- classifier or ranker scorer
- top-k/rank submission validator

Cautions:

- This playbook is not a full recommender/matching playbook
- candidate recall can dominate final score
- valid rank permutations matter

## 11. Excluded And Route-Out Materials

Exclude as recipes:

Route out:

- interactive game-policy tasks
- constrained optimization tasks
- pure survey storytelling
- CTF/exploit challenges
- image/video/text tasks in neighboring folders
- pure EDA judged-methodology tasks without a prediction target
