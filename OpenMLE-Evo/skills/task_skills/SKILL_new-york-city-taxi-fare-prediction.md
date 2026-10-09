# Skill: new-york-city-taxi-fare-prediction

## 1. Task-Specific Reading
- Predict `fare_amount` for each taxi ride from pickup/dropoff coordinates, pickup timestamp, and passenger count.
- This is large-scale geospatial tabular regression, not generic anonymous tabular regression. Raw coordinates alone are weak; the model needs route geometry, temporal fare-regime signals, and aggressive but validated cleanup of impossible trips.
- The metric is RMSE on fare dollars. Large errors are punished quadratically, so high-fare trips, invalid rows, coordinate mistakes, and extreme predicted values matter much more than median absolute error.
- The target includes tolls. The model should expect discontinuities for bridge/tunnel/airport-like trips and cannot be only a smooth distance-to-fare function.
- Training data is extremely large and test is small. Use enough rows to cover rare locations, long trips, years, hours, and passenger counts. A carefully cleaned multi-million-row sample with strong features can beat a naively trained full-data model with noisy rows.
- Dominant score levers:
  - Domain-valid filtering of fares, passenger counts, coordinates, zero/near-zero trips, and implausible distance/fare combinations.
  - Geospatial features: haversine distance, Manhattan-style distance, coordinate deltas, bearing, endpoint coordinates, route orientation, and dense-location/cluster proximity features.
  - Temporal features that capture fare schedule changes and demand/traffic patterns: year, month, hour, day-of-week, weekend, cyclical encodings, and interactions with distance/location.
  - Strong GBDT regression, primarily LightGBM on CPU, plus XGBoost/CatBoost diversity if time allows.
  - OOF RMSE-driven blending and postprocessing: nonnegative clipping, conservative upper caps, and optional route/segment-specific caps validated on OOF.

## 2. Highest-Expected-Score Strategy
- Converge toward a large-sample, feature-rich GBDT ensemble trained on cleaned raw fares with RMSE-aligned validation.
- Treat the problem as noisy fare-table-plus-geography regression:
  - Linear distance explains much of fare but leaves systematic residuals from traffic, tolls, pickup year, time of day, direction, and airport/outer-borough trips.
  - A tree ensemble should receive explicit geometry and temporal primitives rather than being forced to discover spherical distance from lat/lon splits.
- Primary model family:
  - LightGBM regressor is the workhorse because it handles many rows and engineered numeric features efficiently under CPU constraints.
  - XGBoost hist/gpu-hist is a valuable second model if runtime permits; use it for diversity with shallower/deeper regimes and different subsampling.
  - CatBoost can add diversity even with mostly numeric features, especially when binned/cluster/time categories are added, but it is not mandatory before a strong LightGBM baseline.
  - A neural network is low priority. With these low-dimensional, engineered, tabular/geospatial features, GBDTs should dominate for return per hour.
- Target handling:
  - Optimize RMSE on original fare as the default. RMSE is on dollars, not log dollars; log targets can underfit expensive/toll trips and reduce public RMSE only if OOF confirms.
  - Use raw-target models as the main blend. Add a log1p or sqrt-target model only as an ensemble diversity member after validating inverse-transformed OOF RMSE.
- Preprocessing and cleanup:
  - Remove impossible rows before training: nonpositive fares, invalid passenger counts, coordinates outside the plausible NYC-area bounding region implied by train/test, missing coordinates, zero pickup/dropoff coordinate artifacts, and trips with nonsensical distance/fare ratios.
  - Do not over-clean long legitimate trips. RMSE makes rare expensive trips important, and toll-inclusive fares can look high relative to simple distance.
  - Filter using broad domain constraints and OOF feedback, not a narrow Manhattan-only box.
- Feature strategy:
  - Keep raw pickup/dropoff lat/lon because trees can learn local fare geography from them once geometry features are present.
  - Add `abs_lon_delta`, `abs_lat_delta`, signed deltas, haversine distance, equirectangular/Manhattan proxy distance, bearing, centerpoint lat/lon, and endpoint distance from the coordinate center of the test/train region.
  - Add speed/traffic proxy interactions even without trip duration: distance by hour/day/year, binned distance by year/hour, route direction by hour, and endpoint-cluster by hour.
  - Learn dense pickup/dropoff hubs from the available coordinates rather than relying on external maps. KMeans or quantile/grid bins over endpoints can produce distances to high-density centroids and pickup/dropoff cluster IDs.
  - Add count/frequency encodings for spatial bins, endpoint clusters, and time-spatial combinations. These are unlabeled distribution features and are safe when computed from train+test coordinates without using target.
  - Add OOF-safe target encodings only for coarse spatial/time bins if the implementation is already stable; they can be strong but are leakage-prone if done casually.
- Inference:
  - Average fold predictions for each model.
  - Blend OOF predictions by RMSE with simple nonnegative weights or Ridge on OOF predictions.
  - Clip predictions to positive fare values and validate any upper cap on OOF. Do not round; fare is continuous.

## 3. Strong First Implementation Plan
- Build one complete first-round script around a cleaned, engineered LightGBM pipeline plus optional second LightGBM seed/model variant.
- Training sample:
  - Use a large cleaned subset if full training is too slow. Prefer several million to tens of millions of rows over small toy samples.
  - Preserve distribution coverage by sampling across years/months, distance bins, fare bins, and long-trip rows. Do not accidentally sample away high fares or rare geographies.
  - Keep all valid rows in the validation fold from the chosen training subset; do not validate on rows filtered differently from training.
- Cleaning:
  - Drop rows with missing target/features, `fare_amount <= 0`, passenger counts outside plausible taxi values, invalid lat/lon, and coordinates far outside the train/test support.
  - Drop exact or near-zero trips with high fare unless OOF diagnostics suggest they are legitimate waiting/toll cases; keep cheap short trips.
  - Remove extreme mismatches such as very long coordinates with tiny fare or very short distance with enormous fare using broad distance/fare ratio rules. Tune conservatively; false removal of valid toll trips can hurt.
- First feature set:
  - Parse pickup datetime into year, month, day, day-of-week, hour, weekend, and cyclic hour/day/month features.
  - Distance/geometry: haversine, squared/log distance variants, Manhattan proxy from separate lat/lon components, Euclidean degree distance, signed and absolute deltas, bearing, pickup/dropoff raw coordinates, midpoint coordinates.
  - Location bins: rounded lat/lon at multiple precisions or grid indices for pickup, dropoff, and midpoint. Use them as numeric/category-like features and for count encodings.
  - Distribution counts: pickup-bin count, dropoff-bin count, pickup-dropoff-bin pair count, hour-bin count, year-bin count, and cluster counts from combined train/test coordinates.
  - Cluster features: fit small/medium KMeans on pickup and dropoff coordinates from train+test or a large sample; add pickup cluster, dropoff cluster, cluster pair, distances to nearest centroids, and cluster-to-cluster distance.
  - Interaction features: distance × year, distance × hour, distance × weekend, long-trip flag, short-trip flag, same-area flag, and hub-like flags derived from high-density coordinate clusters. If high-fare cluster labels are used later, build them only with OOF-safe target encoding.
- Model:
  - Train LightGBM `objective=regression`, `metric=rmse`, CPU device, many estimators with early stopping, moderate leaves, row/column subsampling, and enough `min_child_samples` to smooth noisy fare artifacts.
  - Use 3 to 5 folds for first round. If using a very large sample, 3 folds may be enough for model selection; use 5 folds for final OOF/blend if time permits.
  - Use raw target first. Train one diversity model on `log1p(fare)` only if there is time, inverse-transform predictions, and keep it only if OOF RMSE improves the blend.
- Validation:
  - Use deterministic shuffled KFold over the cleaned sample for the first implementation, while monitoring year/time/location distribution per fold.
  - Also compute diagnostic RMSE by distance bin, year, hour, fare quantile, and spatial cluster. The biggest gains usually come from fixing high-error slices, not global hyperparameter tuning.
- Postprocessing:
  - Clip predictions to a small positive minimum.
  - Cap extreme predictions only if OOF shows the model overpredicts. Consider caps conditioned on distance bins rather than a single hard maximum.

## 4. High-ROI Upgrades Across Rounds
- Round 2:
  - Strengthen geospatial representation before tuning dozens of model parameters. Add multi-resolution grid bins, pickup/dropoff/midpoint clusters, centroid-distance features, cluster-pair counts, and route-bearing interactions.
  - Add OOF-safe target encodings for coarse pickup bin, dropoff bin, pickup-dropoff cluster pair, year-hour, distance bin, and cluster pair × year. Smooth heavily and validate on identical folds.
  - Train multiple LightGBM variants: raw target, log1p target, higher `num_leaves`/lower learning rate, shallower/high-min-child model, and feature-subset variants. Blend only variants with useful OOF diversity.
  - Try XGBoost hist/gpu-hist with the same engineered matrix. It often captures different split patterns from LightGBM and can improve a blend even if standalone RMSE is slightly worse.
  - Tune cleaning thresholds using OOF slices: coordinate bounds, fare bounds, passenger bounds, distance/fare mismatch thresholds, and treatment of zero-distance trips.
- Round 3:
  - Add CatBoost on a compact feature set with categorical versions of year/month/hour, grid bins, clusters, and distance bins. Use it as a blend member, not a replacement for LightGBM.
  - Build a Ridge or constrained linear stack over OOF predictions from LightGBM/XGBoost/CatBoost/raw/log variants. Keep meta-features limited to base predictions and possibly distance/year bins if using a simple segment-aware correction.
  - Add residual correction models: train a second LightGBM on OOF residual patterns by distance bin, year, hour, bearing, and clusters, then validate the combined prediction strictly through OOF.
  - Try sample weighting to emphasize medium/high fares or long trips if their RMSE contribution dominates. Avoid overweighting obvious label noise.
  - Increase training data volume after feature/model design stabilizes. Fuller data can help more than small hyperparameter changes, especially for rare geographies.
- Late round:
  - Multi-seed bag the best LightGBM and XGBoost configurations and average predictions.
  - Perform segment-specific blend weights by distance/fare-proxy bins, year, or cluster group only if OOF sample sizes are large and segment RMSE improves reliably.
  - Add conservative pseudo-labeling only if model disagreement on test is low and train/test adversarial shift suggests benefit; keep pseudo-labeled rows out of validation and give them low weight.
  - Use leaderboard feedback only to arbitrate close OOF ties or suspected distribution mismatch. Do not chase one-off public changes with aggressive clipping or cleaning.

## 5. Validation and Metric Optimization
- Use RMSE on original dollar fares for every model-selection decision. If a model trains in transformed space, inverse-transform before scoring OOF.
- Shuffled KFold is a reasonable default because the test set is a small sample from the same competition distribution, but it can hide temporal/geographic mismatch. Always inspect fold balance across year, month, distance, fare, and spatial clusters.
- Add a secondary validation view:
  - Time-based holdout by later pickup dates to test fare-regime robustness.
  - Spatial/cluster diagnostics to detect overfitting target encodings or hub-specific rules.
  - Distance/fare quantile RMSE to verify that global gains are not caused by sacrificing expensive trips.
- Trust improvements that hold on the same folds and across key slices. A tiny overall OOF gain that worsens long trips, high fares, or recent years is risky for RMSE.
- RMSE implications:
  - Outliers in target and invalid coordinates dominate the loss; cleaning is metric optimization, not cosmetic preprocessing.
  - Predictions must stay positive and not explode on abnormal coordinate combinations.
  - Log target can improve common short-trip behavior but can underpredict high fares. Keep it as a blend candidate, not the default final truth.
- OOF usage:
  - Use OOF predictions for blend weights, postprocessing caps, target-encoding decisions, and residual correction.
  - For target encodings, generate train encodings fold-by-fold and test encodings by averaging fold encoders or using full-train smoothed statistics only for test. Never compute target means on validation rows.
  - Optimize any clipping/capping thresholds on OOF RMSE, ideally by distance or prediction quantile.

## 6. Model, Feature, and Preprocessing Priorities
- Highest-priority features:
  - Haversine distance; Manhattan/equirectangular proxy; signed and absolute coordinate deltas; bearing; midpoint coordinates.
  - Raw pickup/dropoff lat/lon retained alongside engineered geometry.
  - Pickup year, month, day-of-week, hour, weekend, and cyclical encodings.
  - Distance interactions with year, hour, weekend, and route direction.
  - Multi-resolution spatial grids for pickup, dropoff, and midpoint.
  - Coordinate-cluster IDs, cluster-pair IDs, nearest-centroid distances, cluster-pair counts, and location frequency encodings.
  - Distance bins, short/medium/long trip flags, and same-bin/same-cluster flags.
- Highest-priority preprocessing:
  - Broad coordinate bounds aligned to train/test support.
  - Fare and passenger cleanup.
  - Conservative distance/fare consistency filtering.
  - Memory-conscious numeric dtypes are useful because the dataset is huge, but do not let implementation convenience force a tiny sample.
- Highest-priority models:
  - LightGBM raw-target RMSE model with strong geospatial features.
  - Additional LightGBM variants for target transform, feature subsets, and regularization regimes.
  - XGBoost hist/gpu-hist for blend diversity.
  - CatBoost with cluster/grid/time categorical features after the main pipeline is stable.
  - Ridge/constrained OOF stack or optimized weighted average for final predictions.
- Highest-priority diagnostics:
  - RMSE by distance bin, year, hour, passenger count, pickup/dropoff cluster, and prediction quantile.
  - Feature importance sanity: distance and geometry should dominate, with time/location features providing residual improvements.
  - Model disagreement on test, especially for long trips and rare clusters.

## 7. Avoid or Delay
- Avoid a generic anonymous-tabular pipeline that only feeds raw columns to GBDT. It wastes the strongest domain signal.
- Avoid relying on external maps, external fare tables, private labels, or manually looked-up landmark coordinates as the default plan. Derive hubs and spatial structure from available train/test coordinates.
- Avoid narrow coordinate filtering that removes legitimate outer-area or airport/toll trips. Broad, validated filters are safer than Manhattan-only assumptions.
- Avoid optimizing MAE, RMSLE, or median error as the main objective. They can make common trips look better while worsening RMSE.
- Avoid hard rounding. Fare is continuous and RMSE does not reward integer snapping.
- Avoid aggressive log-target-only training. It can underpredict expensive trips that matter heavily under RMSE.
- Avoid leaky target encodings on spatial bins or cluster pairs. If target encoding is used, it must be OOF-safe and smoothed.
- Avoid overcomplicated neural networks, sequence models, image/text ideas, or deep route reconstruction as a first route. The data is low-dimensional structured geospatial regression.
- Delay stacking until base OOF predictions are clean and fold-consistent. A simple strong LightGBM blend with excellent features is a better first competition route than a large fragile stack.
- Delay pseudo-labeling and segment-specific correction until after stable OOF diagnostics show where they help. Test has few rows; overfitting its quirks through pseudo-labeling or manual caps is easy.
