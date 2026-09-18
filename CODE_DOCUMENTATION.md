# Code Documentation — Richter's Predictor: Earthquake Damage Modeling

This document explains the full modeling notebook (`assets/earthquake_damage.ipynb`): what each
stage does, why each technique/parameter/model was chosen, what the alternatives were, and why
those alternatives were rejected. It picks up where [`EDA_Summary.md`](EDA_Summary.md) leaves
off — that document covers the exploratory analysis (Sections 1–9) in depth; this one covers
**cleaning, feature engineering, preprocessing, feature selection, modeling, tuning, evaluation,
and explainability** (everything from "Data Retention Decision" onward in the notebook), and
also explains *why* the statistical/ML tooling was chosen over the obvious alternatives.

---

## 1. Problem framing

**Task:** predict `damage_grade` (1 = low, 2 = medium, 3 = near-total destruction) for 86,868
buildings, trained on 260,601 labeled buildings, from 38 structural/geographic/ownership
features. This is Kathmandu Living Labs' *Richter's Predictor* competition, built from the 2015
Nepal earthquake survey.

**Why this is treated as classification, not regression, despite the target being ordinal:**
`damage_grade` (1/2/3) has a natural order, so it could be framed as ordinal regression. The
notebook frames it as **multi-class classification** because:
- The competition's own scoring metric (micro-F1) is a classification metric, not an ordinal
  loss (e.g., MAE on grade), so optimizing directly for the classification metric is more
  aligned with the actual objective than optimizing an ordinal proxy and hoping it correlates.
- Tree ensembles (the model family that ultimately wins the candidate comparison, see #7) handle
  multi-class classification natively and capture non-linear/interaction effects; a strict
  ordinal-regression formulation (e.g., proportional-odds logistic regression) would only pay
  off if the *relationship* between features and grade were closer to monotonic/linear, which
  the EDA's weak Pearson correlations (#9 of the EDA) suggest it is not.

**Why micro-F1 (not accuracy or macro-F1):** The target is imbalanced (57% grade 2, 10% grade 1
— EDA #4). Accuracy would reward a model that just predicts the majority class. Macro-F1 weighs
every class equally regardless of size, which over-corrects for imbalance in the *opposite*
direction. Micro-F1 pools TP/FP/FN across all classes before computing precision/recall — this
is literally the competition's own metric (`performance_metric.md`), so every comparison,
tuning objective, and reported score in the notebook uses `scoring="f1_micro"` for consistency
between validation and the real leaderboard metric.

---

## 2. Data retention: why nothing is dropped

The EDA flagged three candidate cleaning actions (drop `age == 995` sentinel rows, drop ~28.5K
feature-level duplicates, IQR-cap `area_percentage`/`height_percentage`). **None are applied.**
Reasoning:

| Candidate action | Why it was rejected |
|---|---|
| Drop `age == 995` rows | The code appears in *both* train and test; dropping it from train only would bias the training distribution away from what the model sees at inference. Instead it's turned into a feature (`age_is_placeholder`, `age_bucket="unknown"`) — see #3. |
| Drop feature-level duplicate rows | These are different buildings that happen to share a feature profile (confirmed in EDA #3) — they are real, independent observations, not data-entry errors, so removing them would just be throwing away data. |
| IQR-cap `area_percentage`/`height_percentage` | Large-but-plausible buildings, not entry errors. Tree ensembles split on thresholds rather than computing distances/gradients over raw magnitudes, so they are far less sensitive to long tails than a linear or distance-based model would be — capping only pays off for scale-sensitive model families the notebook wasn't planning to rely on. |

This is a **model-family-driven decision**: because the intended models are tree ensembles, several
"textbook" cleaning steps that matter for linear/distance-based models are unnecessary here and
would only remove real information.

---

## 3. Feature engineering — 8 new columns, all leakage-free

All engineered features are **deterministic functions of raw columns only** (no target
information), so they can be computed identically and independently on train and test with zero
leakage risk. This is an explicit design constraint, separate from the encoders in #5 which *do*
touch the target and therefore need out-of-fold handling.

| Feature | What it does | Why |
|---|---|---|
| `age_is_placeholder` | Binary flag for `age == 995` | Gives the model an explicit signal to separate the sentinel from genuinely old buildings, instead of leaving it to distort a continuous `age` term. |
| `age_bucket` | `age` binned into new/moderate/old/very_old/unknown (bins: -1,5,15,30,994,10000) | Captures non-linear age thresholds a single continuous term can't; the `(994, 10000]` bin only ever catches the 995 sentinel (confirmed max age is exactly 995), giving it its own category label rather than lumping it with genuinely old buildings. |
| `age_x_floors` | `age × count_floors_pre_eq` | Both are individually weak predictors (EDA #9); an older *and* taller building plausibly compounds risk in a way neither term alone captures — trees can approximate interactions like this from the raw columns too, but giving it explicitly saves the model from needing many splits to rediscover it. |
| `height_area_ratio` | `height_percentage / max(area_percentage, 1)` | A slenderness proxy. `height_percentage` correlates +0.05 with damage and `area_percentage` correlates −0.13 (EDA #9); a tall/narrow footprint should combine both signals directionally. |
| `family_density` | `count_families / max(count_floors_pre_eq, 1)` | Occupants per floor — a proxy for building use-intensity. |
| `superstructure_material_count` | Row-sum of the 11 binary superstructure flags | EDA #8 found 68% of buildings report exactly one material — mixed-material construction may behave structurally differently. |
| `secondary_use_count` | Row-sum of the 10 secondary-use *sub*-flags (excludes the `has_secondary_use` umbrella, which is a pure logical OR of these and would just duplicate the same information) | Avoids double-encoding a value that's already 100% derivable from other columns in the same table. |
| `foundation_position` | String concatenation `foundation_type + "_" + position` | `foundation_type` was the strongest single categorical predictor in the EDA (#9), `position` a modest one; their interaction may capture a combined structural-vulnerability pattern neither shows alone. This is a case where a **tree could learn the interaction on its own** given both raw columns, but explicitly materializing it as a single categorical (then one-hot encoded) removes the burden of the model needing to find the right compound split, at the cost of a larger one-hot block. |

A ninth engineered column, **`geo2_freq`** (row-count frequency of `geo_level_2_id`, fit on
train only), is computed separately because it's part of the geo-encoding comparison (#4), not
the deterministic `engineer_features()` function.

---

## 4. Geo-encoding: an empirical comparison, not an assumption

The EDA argued (on cardinality/unseen-category grounds) that the 3 geo columns should be target-
or frequency-encoded rather than one-hot encoded. Rather than taking that on faith, the notebook
**tests it empirically**: three full pipeline variants, differing only in how the geo columns are
encoded, are each 5-fold cross-validated with the same fast proxy classifier, and the winner
becomes `GEO_ENCODING_STRATEGY` for the real pipeline.

| Strategy | `geo_level_1_id` (31 levels) | `geo_level_2_id` (1,418) | `geo_level_3_id` (11,861) |
|---|---|---|---|
| `target` | out-of-fold smoothed target mean | out-of-fold smoothed target mean | out-of-fold smoothed target mean |
| `frequency` | row-count frequency | row-count frequency | row-count frequency |
| `onehot_geo1_freq_rest` | one-hot (31 dummies) | row-count frequency | row-count frequency |

**Why `LogisticRegression` as the proxy classifier for this comparison, not the eventual tree
models:** it's fast (essential when comparing 3 variants × 5 folds × 260K rows), and — being
linear — it's *more* sensitive to encoding quality than a tree ensemble would be, since it can't
compensate for a poorly-encoded column via arbitrary splits. If a linear model shows a clear
winner among encodings, that's a conservative, trustworthy signal to carry into the tree models.
`class_weight="balanced"` is set because the same class imbalance (EDA #4) applies here too.

**Why full one-hot for `geo_level_2_id`/`geo_level_3_id` is excluded entirely:** at 1,418 and
11,861 levels respectively, one-hot would blow up the feature space by four orders of magnitude
and — critically — cannot represent the 4 / 266 categories in test that were never seen in
train (EDA #5). Target and frequency encoding both degrade gracefully for unseen categories
(global mean / zero count); one-hot has no such fallback beyond dropping the column entirely.

**Why `geo_level_1_id` gets a dedicated one-hot variant to test against:** at only 31 levels it's
cheap enough that one-hot is *feasible*, unlike the finer levels — so it's worth checking
empirically whether preserving each region as an independent dummy beats encoding it into a
single ordered/frequency value, rather than assuming the finer-level argument automatically
applies to the coarsest level too.

### `KFoldTargetEncoder` — the custom transformer

A `BaseEstimator`/`TransformerMixin` implementing **out-of-fold smoothed target-mean encoding**:

- `fit_transform(X, y)` (used on train, inside `Pipeline.fit_transform`) computes encodings via
  `StratifiedKFold` — each fold's rows are encoded using only the *other* folds' target means, so
  no row ever sees a mapping derived even partly from its own label. This is what makes
  target encoding safe against leakage: a naive `groupby(col)[target].mean()` fit and applied to
  the same rows would leak the label directly through the encoded value.
- `transform(X)` (used on test/validation) applies a mapping fit on the *entire* training set,
  falling back to the global train mean for any category unseen during fit.
- **Smoothing**: `(count × mean + smoothing × global_mean) / (count + smoothing)`, with
  `TARGET_ENCODING_SMOOTHING = 10`. This shrinks small-sample category means toward the global
  mean — without it, a `geo_level_3_id` value that appears only once or twice in a fold would get
  an encoding of exactly 0, 1, or the observed class, wildly overfitting a near-noise estimate.
  `smoothing=10` means a category needs roughly 10+ observations before its own mean starts to
  dominate the global prior — a reasonable default given `geo_level_3_id` has a median of far
  fewer than that per training fold.
- `StratifiedKFold` (not plain `KFold`) is used for the internal fold split so each fold
  preserves the same class-imbalance ratio as the whole dataset — otherwise a fold could, by
  chance, under-represent grade 1, and its out-of-fold target means for that fold would be
  systematically biased.

`GeoFrequencyEncoder` (row-count frequency) needs no such out-of-fold machinery, because it never
touches the target column at all — a `value_counts()` fit on train and applied to
train/val/test is leakage-safe by construction, which is exactly the point made explicit in the
notebook's own comment.

---

## 5. Preprocessing pipeline

Implemented as a two-stage `sklearn.pipeline.Pipeline` of `ColumnTransformer`s so the exact same
fitted encoders/scaler apply to train and test, and the whole thing can be dropped into
cross-validation as a single object (avoiding the classic mistake of fitting a scaler/encoder on
the full dataset before splitting).

**Stage 1 — Encode** (`ColumnTransformer`, `verbose_feature_names_out=False` for readable output
column names):
- geo columns → whichever strategy won #4 (`get_geo_transformers`)
- `CONTINUOUS_NUMERIC_COLS` → passthrough (scaled in Stage 2)
- `ONE_HOT_COLS` (8 raw categoricals + `foundation_position` + `age_bucket`) →
  `OneHotEncoder(handle_unknown="ignore")`
- `BINARY_COLS` (22 raw flags + `age_is_placeholder`) → passthrough (already 0/1)

**Stage 2 — Scale**: `MinMaxScaler` on the continuous numeric + geo-encoded columns; everything
else passes through unscaled.

**Why `MinMaxScaler` over `StandardScaler`:** Tree ensembles are scale-invariant (splits only
depend on relative order, not magnitude), so *in principle* neither the model would care.
Scaling here exists mainly for the `LogisticRegression` proxy/baseline comparisons in #4 and #7,
where scale matters. `MinMaxScaler` was chosen over `StandardScaler` because several scaled
columns are counts/ratios bounded at zero with long right tails (`age`, `height_area_ratio`) —
compressing them into a fixed `[0, 1]` range keeps outliers from dominating the scale the way
they can under `StandardScaler`'s mean/variance normalization, without needing the outlier
removal step that was deliberately skipped in #2.

**Why `OneHotEncoder(handle_unknown="ignore")` specifically:** the EDA (#7) confirmed all 8 raw
categorical columns have zero unseen categories in test, so `handle_unknown="ignore"` is a safety
net rather than a load-bearing requirement for those columns — but it *is* load-bearing for the
engineered `foundation_position` column, since a `foundation_type × position` combination absent
from train could plausibly appear in test even though each individual column's values are fully
covered.

---

## 6. Feature selection

After encoding, the feature table has more columns than the model needs, including at least one
pair that's redundant *by construction* (`age_bucket_unknown` and `age_is_placeholder` both
encode the 995 sentinel). Two complementary, independently-computed checks are used rather than
one, so the final `SELECTED_FEATURES` set isn't an artifact of a single method's biases:

1. **Redundancy check — pairwise Pearson correlation, threshold 0.85.** Stricter than the 0.77
   max correlation already logged in the EDA appendix (between `count_floors_pre_eq` and
   `height_percentage`), so it's deliberately conservative — it flags pairs for *awareness*
   without automatically dropping either column, since tree ensembles tolerate correlated
   features reasonably well (unlike linear/distance-based models) and Pearson correlation only
   captures linear relationships, not general redundancy.

2. **Embedded importance — `RandomForestClassifier` + `SelectFromModel(threshold="median")`.**
   A 300-tree forest (`class_weight="balanced_subsample"` for the imbalance, same as elsewhere)
   is fit on the full engineered feature set, and any column at or above the *median* importance
   is kept. `threshold="median"` (rather than a fixed cutoff or top-K) scales automatically with
   however many features exist — it always keeps roughly half the feature space regardless of
   how many engineered columns are added later, and is a common, defensible default for this
   estimator-based selection method.

3. **Mutual-information cross-check** (`mutual_info_classif`) — computed independently and
   compared against the RF importance ranking's top 15 via set overlap. This exists because
   `SelectFromModel` inherits whatever biases a single `RandomForestClassifier` fit has (e.g.
   a mild bias toward high-cardinality features); mutual information makes no assumption about
   *how* a feature relates to the target (linear, tree-splittable, or otherwise) and doesn't
   depend on a single model fit, so strong agreement between the two rankings is reassuring that
   the selection isn't an artifact of the RF's particular splits. `discrete_features` is set
   per-column based on whether it has ≤20 unique values, since `mutual_info_classif` estimates
   discrete and continuous features with different underlying algorithms (contingency-table-based
   vs. k-NN-based estimation) and mixing them up would bias the scores.

`ALL_FEATURES` is retained alongside `SELECTED_FEATURES` specifically so the modeling section can
compare the reduced set against the full set (`FEATURE_SET = SELECTED_FEATURES` is the default,
with a one-line swap documented in the code to try the alternative) — feature selection isn't
treated as an irreversible, unverified decision.

---

## 7. Modeling

### 7.1 Naive baseline

`DummyClassifier(strategy="most_frequent")` — always predicts grade 2. This exists purely to
contextualize every later score: any candidate model that doesn't clear this bar is worse than
doing nothing, which is a real risk for naïve accuracy-style thinking on an imbalanced target
(the majority-class baseline already scores ~57% "accuracy" despite being useless — restating
this in `f1_micro` terms keeps the baseline honest).

### 7.2 Candidate comparison — why these four models and not others

| Model | Role | Why included |
|---|---|---|
| `LogisticRegression` (`class_weight="balanced"`, `max_iter=2000`) | Linear baseline | Establishes whether the non-linear/interaction structure the EDA implied (weak Pearson correlations, #9) is actually necessary — if a linear model matched the tree ensembles, that would be a strong reason to prefer it for simplicity/interpretability. |
| `RandomForestClassifier` (`class_weight="balanced_subsample"`) | Bagged tree ensemble | Handles non-linearities/interactions and mixed categorical/numeric features natively; `balanced_subsample` reweights classes *within each bootstrap sample*, which is a better fit for bagging than a single global `class_weight` since each tree sees its own resample. |
| `XGBClassifier` (`objective="multi:softprob"`) | Gradient-boosted trees | Typically the strongest tabular-data performer in practice for structured, mixed-type data at this scale; boosting (sequential error-correction) often outperforms bagging (RF) when there's a meaningful amount of learnable non-linear signal, as the EDA's importance/interaction findings suggest here. |
| `LGBMClassifier` (`class_weight="balanced"`) | Gradient-boosted trees (histogram-based) | A second boosting implementation, included alongside XGBoost rather than instead of it because the two use different tree-growth strategies (LightGBM grows leaf-wise, XGBoost historically level-wise) and different regularization defaults — at 260K rows with high-cardinality-derived features, it's cheap to check both rather than assume one dominates. |

**Models deliberately not tried, and why:**
- **SVM / KNN** — both scale poorly to 260K rows × dozens of features (SVM training is
  super-linear in row count; KNN prediction cost scales with dataset size at inference), and
  neither offers a clear advantage over tree ensembles on this kind of mixed
  categorical/numeric/hierarchical feature set.
- **Naive Bayes** — assumes feature independence, which is directly violated here (e.g.
  `count_floors_pre_eq` and `height_percentage` correlate at 0.77 per the EDA appendix; the geo
  hierarchy is nested by construction).
- **Neural networks** — plausible in principle, but tabular data of this size and mostly
  low-cardinality categorical structure is exactly the regime where gradient-boosted trees
  usually match or beat neural nets, without the extra tuning/regularization burden.
- **Ordinal/proportional-odds regression** — see #1: the ordinal structure exists, but the
  scoring metric doesn't reward it directly, and tree ensembles can still exploit ordinal
  information implicitly through split thresholds if it's useful.

**Why `StratifiedKFold` (5-fold) for cross-validation, not plain `KFold`:** identical reasoning
to #4 — with a 57/10/33 class split, an unstratified fold could easily under- or over-represent
grade 1, making the reported CV variance partly an artifact of fold composition rather than model
instability.

**XGBoost's label shift:** `XGBClassifier`'s `multi:softprob` objective requires 0-indexed class
labels internally, so `y_tr - 1` is passed at fit time and predictions are shifted back with
`+ label_offset` afterward. `LogisticRegression`, `RandomForestClassifier`, and `LGBMClassifier`
all handle raw 1/2/3 labels via their own internal label encoding, so this shift is XGBoost-specific
plumbing, not a general requirement — the notebook tracks it explicitly per-model
(`tuned_searches` stores a `label_offset` alongside each search) rather than assuming a shared
convention.

### 7.3 Hyperparameter tuning — why `RandomizedSearchCV` over `GridSearchCV`

Each of the three tree ensembles is tuned with `RandomizedSearchCV(n_iter=25, cv=3-fold)`.
`RandomizedSearchCV` was chosen over exhaustive `GridSearchCV` because:
- The parameter spaces below mix continuous distributions (e.g. `learning_rate` ~
  `uniform(0.01, 0.29)`) with integer ranges — a grid would require manually discretizing every
  continuous parameter, losing resolution or exploding the grid size.
- With ~220K training rows, a 3-fold CV pass over even a modest grid (e.g. 4×4×4×4 = 256
  combinations) would be far more expensive than 25 randomly sampled draws, and random search is
  known (Bergstra & Bengio, 2012) to find comparably good regions of a high-dimensional
  hyperparameter space with far fewer evaluations than grid search, because most hyperparameters
  in these grids contribute unevenly to performance — a few dominate, and random sampling
  explores the full range of the *important* ones more efficiently than a grid that spends most
  of its combinations varying unimportant ones.

`cv=3` (rather than the 5-fold used in the candidate comparison) is a deliberate cost/precision
tradeoff — tuning multiplies cost by `n_iter`, so a cheaper inner CV is used once the candidate
shortlist is already fixed, and 3-fold CV means score estimates are averaged over fewer, larger
folds, trading a little variance for a large speedup, given 25 draws × 3 models already comprises
a substantial amount of compute.

| Model | Parameters tuned | Rationale |
|---|---|---|
| Random Forest | `n_estimators` (200–600), `max_depth` (None or 8–40), `min_samples_split` (2–20), `min_samples_leaf` (1–10), `max_features` (sqrt/log2/0.3/0.5) | Ensemble size, tree complexity/overfitting controls, and per-split feature subsampling — the parameters that matter most for a bagged forest's bias/variance tradeoff. |
| XGBoost | `n_estimators` (200–600), `max_depth` (3–12), `learning_rate` (0.01–0.3), `subsample`/`colsample_bytree` (0.6–1.0), `min_child_weight` (1–10), `gamma` (0–0.5) | Boosting's core knobs: tree count/depth, learning rate, row/column subsampling (reduces overfitting via stochasticity), and the two regularization terms that most directly control leaf-split aggressiveness. |
| LightGBM | `n_estimators` (200–600), `num_leaves` (15–127), `max_depth` (-1 or 3–12), `learning_rate` (0.01–0.3), `subsample`/`colsample_bytree` (0.6–1.0), `min_child_samples` (5–50), `reg_alpha`/`reg_lambda` (0–1) | Same boosting fundamentals as XGBoost, plus `num_leaves` (LightGBM's primary complexity control under leaf-wise growth) and explicit L1/L2 regularization terms, since leaf-wise growth without a leaf cap can overfit faster than XGBoost's more constrained level-wise default. |

All three use `scoring="f1_micro"` — the same metric as everywhere else in the notebook, so the
tuning objective, CV comparison, and final leaderboard metric are never mismatched.

### 7.4 Model selection, held-out evaluation, and final refit

- The three tuned searches' `best_score_` (CV `f1_micro`) are compared, and the winner becomes
  `best_model`.
- That model is checked **once** against `X_val`/`y_val` — a stratified 15% split carved out
  *before* any CV or tuning ever touched it (#7.2's train/val split). This is a deliberate
  separation of concerns: CV during tuning estimates generalization for *model selection*, while
  the held-out split is a single, untouched check that the selected model+hyperparameters
  generalize beyond the folds they were tuned against, catching any subtle tuning-induced
  overfitting that repeated CV scoring can accumulate.
- A confusion matrix (`ConfusionMatrixDisplay.from_predictions`) is plotted on this held-out
  check to look for *systematic* error patterns (e.g. grade 2↔3 confusion) rather than just a
  single aggregate score.
- For the actual submission, the same tuned hyperparameters are `clone()`d (not refit via
  `.fit()` on the already-fit estimator, which would just retrain on the same data — `clone()`
  produces a fresh, unfitted estimator with identical hyperparameters) and refit on **all** of
  `train_final` (`X_tr` + `X_val` combined), so the final model sees every labeled row before
  predicting on the real test set. This is standard practice: the held-out split is only useful
  *before* the model is finalized — once a model is selected, throwing away 15% of labeled data
  for the real submission would be pure waste.

### 7.5 Submission

Predictions are written as `building_id, damage_grade` per `submission_format.md`. The predicted
class distribution is compared against the training distribution (~9.6% / 56.9% / 33.5% for
grades 1/2/3) as a sanity check — a wildly different predicted split would be a red flag for a
label-shift bug (e.g. forgetting `label_offset` for XGBoost) rather than a legitimate model
finding.

---

## 8. Explainability — SHAP

`shap.TreeExplainer` is applied to whichever `final_model` won #7 (it works natively on all three
tree ensembles — no special-casing needed per model type), computed on a 2,000-row random sample
of the held-out validation set.

**Why SHAP over permutation importance or LIME:**
- **Permutation importance** answers "how much does shuffling this column hurt the *aggregate*
  score" — a single global number per feature. SHAP additionally explains *individual*
  predictions and *direction* (does a high value of this feature push toward or away from severe
  damage?), which the beeswarm plot in the notebook relies on directly.
- **`TreeExplainer` specifically (vs. `KernelExplainer`/LIME)** exploits the internal tree
  structure to compute *exact* Shapley values in polynomial time, rather than approximating them
  via local perturbation sampling (what `KernelExplainer`/LIME do, which is both slower and only
  approximate). Since all three candidate models are tree ensembles, `TreeExplainer` is strictly
  the better tool here — there's no accuracy/speed tradeoff to make.
- **Why validation-set sample, not the full training set:** SHAP's per-row computation cost adds
  up at 260K rows; 2,000 rows is large enough to produce a stable beeswarm/importance ranking
  while computing quickly, and using the *validation* (not training) sample avoids explaining
  predictions the model was fit on, which is more representative of how the model behaves on
  data it hasn't memorized.

The per-class SHAP values are re-indexed from each model's *internal* class labels (0/1/2 for
XGBoost after its label shift, 1/2/3 for RandomForest/LightGBM) back to the real `damage_grade`
values using the same `label_offset` tracked since #7.2, so the explainability code works
unchanged regardless of which of the three models won.

The **grade-3 beeswarm** and the **mean-|SHAP| ranking across all classes** close the loop back
to the EDA's own bivariate findings (#9: `foundation_type` and geography as the strongest single
predictors) — but computed on the actual tuned model driving the submission, not just raw
association strength computed before any modeling happened. Agreement between the two increases
confidence that the model is learning genuine structural signal rather than an artifact of
encoding or tuning.

---

## 9. Summary: statistical/ML tool choices at a glance

| Decision | Chosen | Rejected alternative(s) | Core reason |
|---|---|---|---|
| Scoring metric | micro-F1 | Accuracy, macro-F1 | Matches the competition's actual metric; robust to class imbalance without over-correcting like macro-F1 |
| Geo encoding | Target encoding (empirically confirmed) | One-hot, ordinal/label encoding | High cardinality + unseen test categories rule out one-hot; empirically tested against frequency encoding and geo1-only one-hot rather than assumed |
| Target-encoding leakage control | Out-of-fold via `StratifiedKFold` | Plain in-sample `groupby` mean | In-sample mean leaks the label directly into its own encoded value |
| Categorical (low-cardinality) encoding | One-hot (`handle_unknown="ignore"`) | Target/frequency encoding | Fully covered in test (EDA #7), low cardinality — one-hot is safe and interpretable |
| Numeric scaling | `MinMaxScaler` | `StandardScaler`, no scaling | Bounded range tolerates skewed/long-tailed columns without needing outlier removal; matters for the LR proxy/baseline, not the tree models |
| CV splitting | `StratifiedKFold` | Plain `KFold` | Preserves class ratio per fold under imbalance |
| Redundancy detection | Pearson correlation, 0.85 threshold | VIF, hierarchical clustering | Simple, consistent with the correlation metric already used in EDA; deliberately conservative (flag, don't auto-drop) |
| Embedded feature importance | RF + `SelectFromModel(threshold="median")` | Fixed top-K, L1-regularized linear model | Scales automatically with feature-set size; cross-checked against MI so it isn't a single-model artifact |
| Importance cross-check | Mutual information | — | Model-agnostic, captures non-linear dependence, doesn't share RF's biases |
| Candidate models | LogisticRegression, RandomForest, XGBoost, LightGBM | SVM, KNN, Naive Bayes, neural nets, ordinal regression | Spans linear baseline → two boosting variants; matches the mixed-type, large-row-count, weak-linear-signal profile from EDA |
| Hyperparameter search | `RandomizedSearchCV` (25 draws, 3-fold) | `GridSearchCV`, Bayesian optimization | Handles continuous distributions natively; more efficient than grid search at this dimensionality/budget; simpler than Bayesian methods for a 25-draw budget |
| Explainability | SHAP `TreeExplainer` | Permutation importance, LIME/`KernelExplainer` | Exact (not approximate) for tree models; explains direction and individual predictions, not just aggregate importance |

---

## 10. Reproducibility

A single `RANDOM_STATE = 42` seeds every stochastic step in the notebook (data splits, K-fold
generation, model initialization, `RandomizedSearchCV` sampling, SHAP subsampling) so the entire
pipeline — from geo-encoding comparison through final submission — is deterministic and
re-runnable end to end.

## 11. File/output map

| Output | Produced by | Purpose |
|---|---|---|
| `outputs/train_processed.csv`, `outputs/test_processed.csv` | Preprocessing section | Fully encoded/scaled feature tables (pre-feature-selection) |
| `outputs/submission.csv` | Step 25 | Final `building_id, damage_grade` predictions in competition format |
| `assets/figures/*.png` | Throughout | EDA and modeling plots (target distribution, geo cardinality, feature importance, confusion matrix, SHAP summaries, etc.) |
