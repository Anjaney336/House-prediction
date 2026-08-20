# PricePredict AI — Universal Platform Final Engineering Report

## Executive Summary

PricePredict AI is now a universal, contract-driven real-estate ML platform prototype rather than a California-shaped demo. Dataset activation resets stale state, every dataframe projection is validated, market and currency are evidence-based hypotheses requiring confirmation, and model delivery is governed by explicit lifecycle, provenance, routing, OOD, and publication rules. The supplied Noida dataset and California sample both complete the same workflow without market-specific branches in production code.

The final code passes 72 automated tests and all 11 Streamlit routes. The API and Streamlit application are running locally. No model is currently exposed to customers because the only operator models in the persistent database were legacy synthetic/demo artifacts; the old active record was deprecated. This is the correct safe state until a licensed real-market model is validated, approved, and published.

## Bugs Found

- Streamlit retained California target/features after a different dataset was uploaded, then indexed the new dataframe with stale columns and raised `KeyError`.
- A stale caller passed `metadata` to an incompatible loaded `train_regressors` function.
- Unique continuous fields such as area and coordinates were classified as identifiers and silently excluded.
- CSV date fields were treated as high-cardinality categories rather than temporal values.
- Active legacy models could be called without a strict `PUBLISHED` status check.
- A trivial LightGBM smoke test passed even though realistic mixed-feature training raised a native Windows access violation.
- Customer delivery still depended on a previously activated synthetic demo model.

## Architectural Problems

- Dataset/session state did not have an atomic activation boundary.
- Schema, market, currency, asset type, property type, and transaction scope were not captured in a versioned model contract.
- Model activation did not represent validation, approval, publication, shadow comparison, or deprecation.
- Customer delivery did not route through an exact market/property model segment.
- Optional native estimators could threaten the long-running application process.

## ML Problems

- Random validation could be used where temporal or geographic structure existed.
- Prediction compatibility did not block missing critical fields, unseen property types, or severe range violations.
- Explanation recovery was invalid because the strongest continuous driver was omitted before training.
- Market-shift degradation was not separated from in-distribution accuracy.
- Model-selection regret and ensemble value were not independently audited.

## Startup/Product Problems

- The customer experience could imply model availability where no approved real-market model existed.
- Model provenance and publication eligibility were not visible to operators.
- No supported-market catalog or exact router contract existed for an embeddable widget.
- The product lacked a maintained production risk register and transparent unsupported-market response.

## Security Problems

- Raw Streamlit exception details could expose stack traces and filesystem paths.
- Production publication did not require real/authorized data provenance.
- Customer and operator model scopes were insufficiently separated.
- Legacy activation state could expose a synthetic artifact as if it were production-ready.

## Performance Problems

- Unrestricted LightGBM OpenMP execution was unstable on the installed Windows binary.
- Some comprehensive linear candidates emit convergence warnings on a small, high-dimensional benchmark.
- Full model-zoo and multi-seed audits are intentionally expensive and should be background jobs in production.

## Fixes Implemented

- Added atomic dataset activation and downstream session reset.
- Added safe schema projection and explicit unavailable-column messages across cleaning, quality, training, and prediction.
- Added a universal real-estate ontology, domain analysis, property-type discovery, target/currency hypotheses, and market evidence.
- Isolated California compatibility literals under `src/demo`; production-code audit finds no California-only field names elsewhere.
- Added semantic date promotion, datetime preprocessing, and datetime prediction validation.
- Corrected identifier detection so valid continuous and coordinate fields remain eligible.
- Added a version 2 model schema contract with feature roles, units, ranges, vocabularies, required fields, market, region, property types, transaction type, currency, target unit, and dataset fingerprint.
- Added dataset provenance, permission, retention, tenant, model scope, and publication eligibility.
- Added lifecycle transitions, exact router, optional explicit regional fallback, shadow gate, deprecation, drift report, OOD rejection, and comparable-coverage confidence.
- Added registry-driven customer widget and supported-market API; removed hardcoded synthetic customer routing.
- Added isolated realistic native-dependency preflight. The unsafe LightGBM binary is `UNAVAILABLE`; XGBoost and CatBoost remain usable.
- Hid detailed Streamlit errors in production UI and retained structured server-side diagnostics.

## Model Registry

Lifecycle: `DRAFT → TRAINING → VALIDATED → APPROVED → PUBLISHED → DEPRECATED`. Platform publication requires real data, owned/licensed/authorized permission, confirmed market, confirmed currency, validation evidence, and a passing shadow decision when replacing production. Private customer models use an isolated scope and never enter the platform router. Public prediction requires both `is_active = 1` and `status = PUBLISHED`.

Registry identity includes market, region, asset type, property type, transaction type, version timestamp, dataset fingerprint, training coverage, metrics, and provenance. Exact market/property routing is preferred; regional fallback is disabled unless both request and model explicitly permit it.

## Dataset Compatibility Matrix

| Dataset | Rows used | Workflow | Result |
|---|---:|---|---|
| Supplied Noida listings | 3,015 | Detect → contract → train → single/batch predict → OOD | Pass |
| California housing sample/full benchmark | 600 / full sample | Aggregate-domain detection → train → predict | Pass |
| Apartments | 500 benchmark | Temporal validation | Pass |
| Villas | 500 benchmark | Temporal validation | Pass |
| Mixed residential | 500 benchmark | Temporal validation | Pass |
| Land/plots | 500 benchmark | Temporal validation | Pass |
| Commercial | 500 benchmark | Temporal validation | Pass |
| Rentals | 500 benchmark | Temporal validation | Pass |
| Unrelated/missing/malformed schemas | Fault matrix | Safe reject or actionable warning | Pass |

## Noida Test Results

The exact supplied file `noida_real_estate_synthetic.csv` was loaded: 3,015 rows, 20 columns, five observed property types. It was recognized as property-listing real estate; `price_inr` was selected as an INR target; `listing_id` was excluded; the Noida market hypothesis required confirmation; and no California fields entered the feature set. The same contract supported one-row and eight-row prediction. A `Warehouse` request was rejected because it was outside covered property types.

The final lightweight run selected Ridge with CV RMSE about INR 4.01M, test RMSE INR 11.29M, test MAE INR 2.80M, test R² 0.500, and 92.5% empirical 95% interval coverage. These numbers are QA evidence only: the filename declares the dataset synthetic, so publication as a platform production model is blocked.

## California Test Results

California remains a compatibility demo and is explicitly labeled `Census Block / Geographic Area`, never an individual-property valuation. The full benchmark selected Histogram Gradient Boosting with CV RMSE 48,749, holdout RMSE 48,726, and holdout R² 0.819. California-only aliases live under the demo adapter and do not define universal defaults.

## Customer Workflow Test

Admins and customers now use the same Streamlit dashboard. The separate customer-facing HTML page was removed; the legacy `/customer` URL redirects to `http://127.0.0.1:8501` so old bookmarks reach the shared interface. Tenant-isolated customer API workflows remain available for future integrations, but there is no second customer dashboard to maintain or confuse with the main product.

## Admin Workflow Test

Automated API tests cover real licensed ingestion, target/market/currency confirmation, training, validation, approval, publication, exact routing, prediction, OOD blocking, unsupported routing, synthetic publication rejection, private-customer activation, and router isolation. The Streamlit admin page exposes model cards, lifecycle controls, dataset/model inventory, routing embed guidance, and production risks.

## Fault Injection Results

Tests cover stale California fields on a generic dataframe, missing required columns, extra columns, malformed numeric/date values, unseen categories, out-of-range values, incompatible property type, missing/duplicate/constant targets, mixed currency, inconsistent units, tiny datasets, high cardinality, leakage proxies, native dependency access violation, drift, and unsupported model routes. Failures are contained and converted into actionable errors; no raw dataframe `KeyError` remains in the tested workflow.

## Model Selection Results

The final seven-dataset matrix completed successfully. Synthetic winners are data-dependent rather than hardcoded. Mixed-residential selection regret remains the largest observed case at 25.3%; the six-seed audit produced mean regret 10.24%, median 8.55%, and maximum 25.26%, with mean conformal coverage 92.8%. The full model zoo produced 23 successful models, two resource exclusions, one safely unavailable LightGBM binary, and three rejected ensembles. Selected Lasso regret was 3.13%; all ensembles were rejected because validation RMSE worsened by 1.13%–5.44%.

After correcting feature exclusion, explanation recovery achieved Spearman rank correlation 0.75 and recovered all three expected top drivers (`area_sqft`, city, locality). This is synthetic association evidence, not causal proof.

## Performance Results

The final matrix took roughly 83 seconds for California and 19–23 seconds per 500-row synthetic market on this machine. Prediction latency, training time, dataframe memory, fold variance, and interval evidence are persisted. The platform uses bounded candidate eligibility and rejects high-cost polynomial models when dimensionality exceeds policy.

## Remaining Risks

- No licensed real transaction dataset has been externally validated or published; the live customer catalog is intentionally empty.
- Market shift remains severe: final matched-row shift degradation ranged from 45% to 81% across synthetic markets. OOD/routing reduce misuse but do not create cross-market validity.
- Split-conformal intervals provide marginal, not subgroup-conditional, coverage and require ongoing recalibration.
- Model-selection regret remains seed-sensitive; leaderboard ranking is decision support, not an oracle.
- SQLite, local object storage, in-process rate limiting, and synchronous training are pilot-grade. Production requires managed PostgreSQL/object storage, shared rate limiting, identity/RBAC, secret management, worker queues, monitoring, backups, and disaster recovery.
- The installed LightGBM binary is unsafe and remains disabled until replaced and revalidated.
- In-app browser automation was blocked by the local browser plugin’s trusted-path configuration; UI validation was completed with Streamlit’s official AppTest harness (11 routes) plus live HTTP checks.

## Final Validation

- `python -m pytest -q --disable-warnings`: **72 passed, 0 failed**.
- `python -m compileall -q app backend src scripts tests`: passed.
- Streamlit AppTest: main plus 10 pages, zero uncaught exceptions.
- Live API `/health`: HTTP 200, `status=ok`.
- Live Streamlit root: HTTP 200.
- Legacy `/customer` URL: HTTP 307 redirect to the shared Streamlit dashboard.
- Live published market catalog for operator: empty by design.
- Validation matrix, shift benchmark, robustness validation, explainability validation, multi-seed selection audit, and full model-zoo audit: completed.

## Runbook

```powershell
python -m streamlit run app\main.py --server.address 127.0.0.1 --server.port 8501
python -m uvicorn backend.api:app --host 127.0.0.1 --port 8765
python -m pytest -q --disable-warnings
python -m scripts.run_validation_matrix
python -m scripts.run_shift_benchmark
python -m scripts.run_robustness_validation
python -m scripts.run_explainability_validation
python -m scripts.run_multiseed_selection_validation
python -m scripts.run_full_model_zoo_benchmark
```
