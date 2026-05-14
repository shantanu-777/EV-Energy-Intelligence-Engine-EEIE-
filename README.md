EEIE - EV Energy intelligence Engine

A runnable end-to-end platform: simulator produces realistic synthetic data into TimescaleDB, all engines train on it, FastAPI serves predictions, Streamlit visualizes them. Depth comes later; breadth and structural integrity come now.

<img width="2158" height="1349" alt="Screenshot 2026-05-13 at 21 20 25" src="https://github.com/user-attachments/assets/3cb729be-bdd2-4d3f-a93e-0e483a7d0a10" />
<img width="1500" height="1271" alt="Screenshot 2026-05-13 at 21 22 09" src="https://github.com/user-attachments/assets/3742d063-da46-4d40-835d-6632fc80d54c" />
<img width="1485" height="1321" alt="Screenshot 2026-05-13 at 21 23 22" src="https://github.com/user-attachments/assets/5b57c404-c110-428d-bcdb-717cba107af9" />




layout

```
EV-Energy-Intelligence-Engine-EEIE/
├── pyproject.toml                  # uv-managed, Python 3.11
├── uv.lock
├── docker-compose.yml             # timescaledb, api, ui
├── .env.example
├── .pre-commit-config.yaml        # ruff, mypy, pytest-on-push
├── .github/
│   └── workflows/
│       └── ci.yml                 # lint + type + tests
│
├── docker/
│   ├── api.Dockerfile
│   └── ui.Dockerfile
│
├── eeie/                          # Main Python package
│   ├── config/                    # pydantic-settings, tariff/vehicle profiles
│   ├── db/                        # SQLAlchemy models, Alembic, hypertable bootstrap
│   ├── ingestion/                 # source → DB adapters + pydantic schemas
│   ├── features/                  # feature engineering (versioned)
│   ├── simulation/                # synthetic data generator (CLI)
│   │
│   ├── models/
│   │   ├── range/
│   │   │   ├── xgb.py
│   │   │   ├── lstm.py
│   │   │   └── predict.py
│   │   │
│   │   ├── demand/
│   │   │   ├── xgb.py
│   │   │   ├── tft.py
│   │   │   └── predict.py
│   │   │
│   │   ├── battery/
│   │   │   ├── empirical.py
│   │   │   ├── correction.py
│   │   │   └── predict.py
│   │   │
│   │   ├── optimization/
│   │   │   ├── milp.py
│   │   │   ├── env.py
│   │   │   ├── rl.py
│   │   │   └── optimize.py
│   │   │
│   │   └── behavior/
│   │       ├── cluster.py
│   │       ├── consumption.py
│   │       └── analyze.py
│   │
│   ├── explainability/
│   │   ├── shap_engine.py
│   │   ├── pdp.py
│   │   └── insight.py
│   │
│   ├── api/                       # FastAPI app, routers, schemas, deps
│   └── evaluation/                # metrics, ablation harness
│
├── ui_streamlit/                  # multi-page app, API-only
├── tests/                         # pytest smoke coverage per module
├── scripts/                       # CLIs: simulate, train_all, evaluate
└── data/                          # gitignored Parquet snapshots
```

Build order (module dependencies)

<img width="1577" height="1168" alt="Screenshot 2026-05-13 at 15 01 25" src="https://github.com/user-attachments/assets/d992ac8d-5198-4e58-bbce-9675420153fe" />


Key technical decisions

- Python 3.11, dependency manager uv (with pyproject.toml + lockfile); also generate a requirements.txt export for non-uv users.
- Postgres 15 + TimescaleDB in Docker, schema managed via SQLAlchemy 2.x + Alembic, hypertables for telemetry, weather, tariffs, charging_events.
- Simulator is physics-lite but realistic: per-vehicle battery capacity, kWh/km consumption modulated by temperature (cold-weather range loss), driver profile (efficient/moderate/aggressive), commuter/weekend driving patterns, ToU tariff schedule (EU-style peak/off-peak/shoulder), seasonal weather curves. Outputs hourly resolution.
- Range: XGBoost baseline + PyTorch LSTM sequence model. Both train on simulator output.
- Demand: XGBoost baseline + Temporal Fusion Transformer via pytorch-forecasting.
- Battery: empirical calendar+cycle aging model + XGBoost residual correction layer (hybrid as the doc specifies).
- Optimization: OR-Tools CP-SAT/MILP solver as primary; stable-baselines3 PPO agent on a custom gymnasium.Env charging environment as advanced track. Both selectable via config.
- Behavior: K-Means clustering + XGBoost energy-per-100km regression.
- Explainability: SHAP TreeExplainer for tree models, DeepExplainer/KernelExplainer for nets, partial dependence utility, and a unified Insight pydantic model returned alongside every prediction (top features + financial impact + battery impact + confidence).
- API: FastAPI with five endpoints from the architecture doc (/predict_range, /optimize_charging, /battery_health, /cost_analysis, /behavior_analysis), each returning prediction + insight + confidence; OpenAPI docs auto-generated.
- UI: Streamlit multi-page (one page per engine), strictly an HTTP client of the API.
- Evaluation: naive vs rule-based vs EEIE ablation harness over the synthetic dataset, computing cost reduction %, peak-hour avoidance %, degradation reduction %, RMSE/MAE/calibration error, inference latency.


The data layer is **PostgreSQL + TimescaleDB**, run in Docker. Hypertables
store telemetry, weather, tariffs, and charging events at hourly resolution.




## Quickstart

### 1. Bring up the stack

```bash
cp .env.example .env
docker compose up --build -d
```

This starts three services:

- `timescaledb` (Postgres 15 + TimescaleDB)
- `api` (FastAPI on `http://localhost:8000`, docs at `/docs`)
- `ui` (Streamlit on `http://localhost:8501`)

### 2. Bootstrap the database and populate it

```bash
docker compose exec api alembic upgrade head
docker compose exec api python -m eeie.simulation.run --vehicles 100 --months 12
```

The simulator generates 100 vehicles x 12 months of hourly telemetry,
tariff schedules, weather, and charging events, writing both to TimescaleDB
hypertables and to Parquet snapshots in `./data/`.

### 3. Train the models

```bash
docker compose exec api python -m scripts.train_all
```

Trains every engine (XGBoost + LSTM + TFT + battery hybrid + MILP/PPO
configs + K-Means/XGBoost). Checkpoints are persisted to `./checkpoints/`.

### 4. Use the platform

- API explorer: <http://localhost:8000/docs>
- Streamlit UI: <http://localhost:8501>
- Run the ablation: `docker compose exec api python -m scripts.evaluate`

## dataset ingestion

The ingestion CLI tracks several public datasets that ultimately feed the
same schema as the built-in simulator. Downloads are manual: unzip into
`data/raw/<slug>/`:

| Slug | Source | Primary file | Target canonical table(s) |
|---|---|---|---|
| `charging_patterns` | [Electric Vehicle Charging Patterns](https://www.kaggle.com/datasets/valakhorasani/electric-vehicle-charging-patterns) | `ev_charging_patterns.csv` | `vehicles`, `charging_events` |
| `battery_charging` | [EV Battery Charging Dataset](https://www.kaggle.com/datasets/programmer3/ev-battery-charging-dataset) | `nev_battery_charging.csv` | `telemetry` |
| `station_availability` | [EV Charging Station Availability Tracking](https://www.kaggle.com/datasets/likithagedipudi/ev-charging-station-availability-tracking) | `ev_charging_station_data.csv` | `station_state` |
| `germany_charging` | [Electric Vehicle Charging in Germany](https://www.kaggle.com/datasets/mexwell/electric-vehicle-charging-in-germany) | `charging_data.csv` | `station_state` (static catalog) |

Verify the on-disk layout before running adapters:

```bash
python -m eeie.ingestion.cli verify --all
# or per dataset:
python -m eeie.ingestion.cli verify --slug charging_patterns
```

The verifier checks each slug folder, the primary CSV filename, and a
minimal header signature. Anything under `data/` stays untracked in git.

Once verified, run the adapters to materialise curated parquet snapshots
under `data/curated/<slug>/`. Each adapter projects its source into the
canonical schema (`vehicles`, `telemetry`, `charging_events`, ...) so the
curated frames are drop-in compatible with the simulator's tables.

```bash
python -m eeie.ingestion.cli run --all
python -m eeie.ingestion.cli run --slug charging_patterns
```

`USD` cost columns are converted to `EUR` at the rate configured by
`EEIE_USD_TO_EUR` (default `0.92`).

## Local development (no Docker)

```bash
# Install uv (https://docs.astral.sh/uv/)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync --extra dev

# Run linters and tests
uv run ruff check .
uv run mypy eeie
uv run pytest

# Start API and UI locally (point them at a local Postgres)
uv run uvicorn eeie.api.main:app --reload
uv run streamlit run eeie/ui_streamlit/Home.py
```

You will still need a TimescaleDB instance reachable via `EEIE_DATABASE_URL`.
The Compose file's `timescaledb` service can be used standalone:

```bash
docker compose up timescaledb -d
EEIE_DATABASE_URL=postgresql+psycopg://eeie:eeie_dev_password@localhost:5432/eeie uv run ...
```

## Engineering standards

- Python 3.11, dependencies managed with [uv](https://docs.astral.sh/uv/).
- Linting: `ruff`. Types: `mypy`. Tests: `pytest`.
- Pre-commit hooks enforce formatting, linting, and type checks.
- CI (GitHub Actions) runs lint + type + tests on every push.
- All ML inference paths emit a structured `Insight` (top features +
  financial impact + battery impact + confidence) alongside the prediction.

## Roadmap beyond Phase 1

- RL convergence and reward shaping for `optimization.rl`
- TFT hyperparameter sweeps and quantile forecasting
- Real-vehicle telemetry adapters (replace simulator)
- Carbon-intensity-aware charging
- Vehicle-to-Grid simulation
- Fleet-level scheduling
- Solar self-consumption integration
