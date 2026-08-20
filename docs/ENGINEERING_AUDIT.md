# PricePredict AI — Engineering Audit

Audit date: 2026-08-19

## Executive finding

The repository already had a sound leakage-safe preprocessing foundation, schema-driven prediction, domain/granularity detection, an AutoML catalog, explainability, local artifact persistence, and meaningful tests. It was a strong capstone but not yet an evidence-backed startup prototype because benchmark, fault-injection, model-selection-regret, controlled shift, and default-user simplicity were missing.

## Working correctly

- Streamlit prediction inputs are generated from persisted schema contracts; the application does not import or depend on the notebook's California-specific helper.
- Imputation, encoding, scaling, and optional IQR clipping are fitted inside model pipelines and validation folds.
- California Housing is detected as `Census Block / Geographic Area`, with area-level rather than individual-house language.
- Leakage checks identify price-derived fields and near-perfect numeric proxies.
- Single and batch predictions validate names, order, numeric conversion, missing fields, extra fields, and unseen categories.
- Optional model packages are represented explicitly and do not terminate an AutoML run when unavailable.
- Model artifacts are created locally with schema, metrics, target, features, model ID, limitations, and training metadata.
- The Docker image starts Streamlit as a non-interactive service and includes a health check.

## Partially implemented before this upgrade

- Model confidence existed but did not explicitly include validation stability and train/test generalization.
- Grouped and chronological validation existed, but geographic validation was not user-selectable and temporal fields outside model features were ignored.
- Ensemble candidates existed, but they were retained without a minimum evidence threshold.
- Data issues distinguished extremes from invalid values, but there was no reusable fault-scenario matrix or BLOCKER/HIGH/MEDIUM/LOW recovery contract.
- The model catalog recorded exclusion reasons, but no persisted cross-dataset benchmark database exercised those rules.

## Incorrect or fragile findings

- Commercial data containing road width could be mislabeled as land because one land-like field was sufficient.
- Commercial lease data could be mislabeled as rental data.
- The Model page exposed detailed model, validation, transformation, optimization, and ensemble controls to every user.
- Training and prediction pages could display raw exception text.
- Composite ranking weights were fixed despite being presented as a general policy.
- Ensembling added complexity even when validation improvement was negligible.

## Missing methodology and tests before this upgrade

- Causally structured apartment, villa, mixed residential, land, commercial, and rental generators.
- Multiple deterministic seeds and configurable market assumptions.
- MCAR, MAR, and feature-dependent missingness; impossible values; valid extremes; duplicates; leakage; and market-shift scenarios.
- AutoML selection regret against the holdout oracle.
- Explainability recovery against known synthetic drivers.
- Ranking sensitivity analysis and evidence-based ensemble acceptance.
- Persisted benchmark results and manifests.
- Automated tests covering all synthetic asset classifications and a broad fault matrix.

## Notebook and California dataset verification

- `houseprediction.csv` and the bundled demo both contain 20,640 rows and 10 columns.
- `total_bedrooms` contains 207 missing values.
- `ocean_proximity` contains five categories: `<1H OCEAN`, `INLAND`, `ISLAND`, `NEAR BAY`, and `NEAR OCEAN`.
- Notebook cells 71–72 contain a manually defined `predict_house_price(...)` helper with nine California-specific inputs. It is retained only as educational demonstration code.
- Production inference uses `ModelSchemaContract.feature_order` and dynamic forms; no production module imports the notebook.

## Security and operational limitations

- Joblib artifacts use pickle-family serialization and must never be loaded from untrusted sources; the UI communicates this.
- The application is session-local: it has no authentication, tenant isolation, database, remote artifact store, audit service, or encrypted secrets workflow.
- Training runs in the Streamlit process. Large production workloads still require a job queue, resource limits, cancellation, and external observability.
- Residual ranges are heuristic model-based ranges, not calibrated legal appraisal intervals.
