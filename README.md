# Dynamic Pricing Engine

A production-oriented machine learning system for **Airbnb demand forecasting and assumption-based pricing decision support**.

The project combines public marketplace data, weather and holiday context, leakage-aware temporal modeling, explainability, anomaly detection, scenario analysis, a FastAPI service, a Streamlit dashboard, MLflow experiment tracking, drift monitoring, Docker, and CI.

> **Important modeling boundary:** the system does **not** claim to estimate causal price elasticity from observational Airbnb data. The predictive model estimates a calendar-unavailability probability proxy, while pricing recommendations are conditional on explicit price-sensitivity assumptions.

---

## What the project does

The pipeline has two deliberately separate layers:

1. **Predictive layer**
   - Ingests Inside Airbnb listings/calendar data plus weather and holiday context.
   - Builds listing, temporal, location, market, weather, and holiday features.
   - Trains an XGBoost classifier to estimate the probability that a listing-date is unavailable.
   - Evaluates the model using chronological validation and a strictly later future holdout.
   - Uses TreeSHAP for prediction explanations and Isolation Forest for anomaly detection.

2. **Decision-support layer**
   - Starts from the model-derived demand/unavailability proxy and a reference market price.
   - Evaluates explicit **low**, **moderate**, and **high** price-sensitivity scenarios.
   - Searches within configured price bounds.
   - Returns a scenario-based price recommendation and revenue proxy.
   - Makes the assumptions visible instead of presenting observational correlation as causal elasticity.

The result is a system that is useful for pricing experimentation and ML engineering demonstrations without overstating what observational marketplace data can identify.

---

## Architecture

```text
Inside Airbnb ───────────────┐
Weather data ────────────────┼──> ingestion / validation
Public holidays ─────────────┘
                                      │
                                      ▼
                             feature engineering
                                      │
                                      ▼
                           leakage-safe XGBoost
                    calendar-unavailability probability
                                      │
                         chronological validation
                                      │
                         ┌────────────┴────────────┐
                         ▼                         ▼
                     TreeSHAP              Isolation Forest
                 prediction explanations       anomalies
                         │                         │
                         └────────────┬────────────┘
                                      ▼
                         pricing scenario engine
                       low / moderate / high
                         sensitivity assumptions
                                      │
                                      ▼
                         bounded price search
                                      │
                         ┌────────────┴────────────┐
                         ▼                         ▼
                      FastAPI                  Streamlit
                         │
                         ▼
                 MLflow + Evidently + CI
```

---

## Modeling methodology

### Target semantics

The source dataset contains a historical column named `was_booked` for backward compatibility with the original project.

In this implementation it is treated as a **calendar-unavailability proxy**, not proof that a confirmed booking occurred. Airbnb calendar availability can change for multiple reasons, so the project avoids claiming that every unavailable date represents realized demand.

### Leakage controls

The demand model excludes target-history features that would leak information unavailable at prediction time, including rolling occupancy-style variables derived from the target.

Validation is chronological rather than random:

- complete dates stay together;
- cross-validation advances through time;
- the final benchmark is measured on a strictly later holdout period that is not used for model fitting.

### Future-holdout benchmark

Current validated future-holdout performance:

| Metric | Score |
|---|---:|
| ROC-AUC | **0.6897** |
| PR-AUC | **0.7884** |
| Log loss | **0.6040** |
| Brier score | **0.2093** |

The probability metrics are reported together because ranking performance alone does not describe probability quality.

### Why the old elasticity model was removed

The original implementation estimated a single log-log price coefficient from observational marketplace data and used it as if it were causal elasticity.

That approach was retired because host prices are endogenous: prices change with expected demand, seasonality, events, location, listing quality, host strategy, and other factors. A predictive association between price and availability is therefore not automatically the causal effect of changing price.

The current project keeps prediction and decision support separate:

- **XGBoost:** predictive probability model.
- **Scenario engine:** assumption-based price response.
- **Causal elasticity:** not claimed.

---

## Pricing scenario engine

For a reference price \(P_0\), base demand proxy \(D_0\), and an assumed sensitivity \(s\), the scenario engine uses:

For a reference price \(P_0\), base demand proxy \(D_0\), and an assumed sensitivity \(s\), the scenario engine uses:

$$
D(P) = D_0 \exp\left[-s\left(\frac{P}{P_0}-1\right)\right]
$$

and scores candidate prices with:

$$
R_{\text{proxy}}(P) = P \times D(P)
$$

The engine searches within configured price bounds and compares low, moderate, and high sensitivity assumptions.

These are **scenario outputs**, not forecasts of realized revenue. Their purpose is to answer a conditional question:

> If demand responded to price according to this stated sensitivity assumption, what price would maximize the corresponding revenue proxy within the allowed bounds?

The scenario implementation is covered by synthetic tests, including recovery of a known analytical optimum.

---

## Tech stack

| Area | Tools |
|---|---|
| Data | pandas, NumPy, PyArrow, requests |
| ML | XGBoost, scikit-learn, LightGBM |
| Explainability | SHAP / TreeSHAP |
| Anomaly detection | Isolation Forest |
| Scenario analysis | NumPy, SciPy |
| Experiment tracking | MLflow with SQLite backend |
| Monitoring | Evidently |
| API | FastAPI, Pydantic, Uvicorn |
| Dashboard | Streamlit, Plotly |
| Testing | pytest, pytest-cov |
| Quality | Ruff |
| Containers | Docker, Docker Compose |
| CI | GitHub Actions |

---

## Repository structure

```text
.
├── config.yaml
├── dashboard/
│   └── app.py
├── docker/
│   ├── Dockerfile.api
│   └── Dockerfile.dashboard
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_experiments.ipynb
│   └── 04_optimization_logic.ipynb
├── src/
│   ├── api/
│   ├── data/
│   ├── models/
│   │   ├── anomaly_detector.py
│   │   ├── demand_forecaster.py
│   │   ├── scenario_engine.py
│   │   └── train_models.py
│   ├── monitoring/
│   └── utils/
├── tests/
│   ├── integration/
│   └── unit/
├── docker-compose.yml
├── Makefile
├── pyproject.toml
└── requirements.txt
```

Generated datasets, trained model artifacts, local MLflow state, caches, and other runtime outputs are not part of the core source tree.

---

## Data sources

The project uses free public sources and does not require paid APIs:

- **Inside Airbnb** — listing, calendar, and review data.
- **Open-Meteo** — historical weather context.
- **Nager.Date** — public-holiday dates.

The checked-in configuration targets a New York City Inside Airbnb snapshot and corresponding NYC weather coordinates.

---

## Installation

Python **3.11** is the primary local development version.

```bash
python -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If using Conda:

```bash
conda create -n dynamic-pricing python=3.11 -y
conda activate dynamic-pricing
python -m pip install -r requirements.txt
```

### macOS OpenMP note

XGBoost and LightGBM may require an OpenMP runtime on macOS. With Conda:

```bash
conda install -c conda-forge llvm-openmp -y
```

The Docker images install the Linux OpenMP runtime automatically.

---

## End-to-end workflow

### 1. Download data

```bash
make data
```

### 2. Train production models

```bash
make train
```

The training pipeline persists the production demand forecaster and anomaly detector and logs experiment metadata through MLflow.

### 3. Start the API

```bash
make run
```

API documentation is available through FastAPI's generated docs once the service is running.

### 4. Start the dashboard

In a second terminal:

```bash
make dashboard
```

### 5. View MLflow experiments

```bash
make mlflow
```

The project uses a local SQLite tracking backend (`sqlite:///mlflow.db`) rather than the legacy MLflow filesystem tracking backend.

---

## API

The FastAPI service exposes the main production workflow.

| Endpoint | Purpose |
|---|---|
| `POST /predict` | Demand prediction plus pricing-scenario recommendation |
| `POST /explain` | SHAP-based prediction explanation |
| `GET /health` | Service/model health |
| `GET /metrics` | Model performance metadata |
| `GET /drift` | Drift-monitoring report |

The pricing endpoint validates listing and date inputs with Pydantic and returns model output together with scenario-analysis information rather than a causal elasticity estimate.

---

## Docker

The API and dashboard are containerized separately and orchestrated with Docker Compose.

```bash
docker compose up --build -d
```

Services:

- API: port `8000`
- Dashboard: port `8501`

Check status:

```bash
docker compose ps
```

Stop the stack:

```bash
docker compose down
```

The Dockerfiles use:

- `python:3.11-slim`;
- `libgomp1` for the Linux OpenMP runtime;
- BuildKit pip cache mounts;
- extended pip retry/timeout settings for more resilient builds;
- container health checks for both services.

The API and dashboard images have been smoke-tested together with trained production model artifacts mounted through Docker Compose.

---

## Testing and code quality

Run the full suite:

```bash
python -m pytest tests/ -q
```

Run the same coverage gate enforced by CI:

```bash
python -m pytest \
  tests/ \
  --cov=src \
  --cov-report=term-missing \
  --cov-fail-under=75
```

Run lint checks:

```bash
python -m ruff check src tests dashboard
```

Current verified state:

- **144 tests passing**
- **82% total coverage**
- Ruff clean
- Docker API + dashboard smoke test passing

CI runs on Python 3.11 and 3.12 and treats lint, unit tests, integration tests, and the coverage threshold as blocking checks.

---

## Notebooks

| Notebook | Purpose |
|---|---|
| `01_eda.ipynb` | Exploratory analysis of listings, prices, geography, and temporal structure |
| `02_feature_engineering.ipynb` | Feature construction and inspection |
| `03_model_experiments.ipynb` | Leakage-safe XGBoost training, temporal evaluation, feature importance, SHAP, anomaly detection, and rationale for retiring observational elasticity |
| `04_optimization_logic.ipynb` | Pricing-scenario mathematics, sensitivity curves, bounded optimization, and synthetic known-optimum validation |

---

## Design decisions

### Whole-date temporal validation

Random train/test splits can leak future marketplace conditions into the past. The evaluation strategy therefore preserves chronology and holds out later dates.

### Probability-quality metrics

ROC-AUC and PR-AUC are accompanied by log loss and Brier score so the project evaluates probability quality rather than only ranking ability.

### Explicit assumptions instead of hidden causal claims

Pricing sensitivity is an input to scenario analysis rather than a coefficient presented as causal elasticity.

### Models loaded once at API startup

FastAPI lifespan handling loads production artifacts once rather than reloading them on every request.

### SQLite-backed MLflow tracking

Experiment metadata uses a database tracking backend suitable for local development and avoids dependence on MLflow's legacy filesystem backend.

### Reproducible engineering checks

The repository combines unit tests, integration tests, coverage enforcement, Ruff, GitHub Actions, container health checks, and a Docker Compose smoke-test path.

---

## Limitations

This repository is a decision-support and ML-engineering project, not a controlled pricing experiment.

Important limitations:

- calendar unavailability is only a proxy for booking/demand;
- observational price is affected by host behavior and expected demand;
- scenario sensitivities are assumptions, not estimated causal effects;
- revenue proxy is not realized revenue;
- the current benchmark is based on one market/data snapshot;
- model quality and scenario behavior should be revalidated before use on another city or time period.

A causal pricing system would require stronger identification, such as randomized price experiments or another defensible causal design.

---

## License

MIT
