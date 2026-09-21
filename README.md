# Solar-PV-Forecasting-ML
Machine Learning-Based Solar PV Power Generation Forecasting Using Weather Data (Project 2)

---

# Machine Learning-Based Solar PV Power Generation Forecasting Using Weather Data

A machine learning project that forecasts the next 15-minute AC power output of a solar photovoltaic (PV) plant using co-located weather sensor data and recent power output history.

**Author:** Ishra Ismat Kamal  
**Department:** Electrical and Electronic Engineering (EEE)  
**Institution:** Independent University, Bangladesh (IUB)  
**Project Type:** Undergraduate ML Research Project (Project 2)

---

## 1. Project Overview

Solar PV power generation is inherently intermittent — it depends on solar irradiance, module temperature, and other environmental conditions. Accurate short-term forecasting of PV power output is important for grid stability, energy scheduling, and plant operation.

This project investigates whether **incorporating recent power output history improves short-term solar PV power forecasting compared with using weather-sensor data alone.**

The project is developed as an independent machine learning study, building on the author's EEE background with an emphasis on:
- Time-series aware data handling
- Chronological train/validation/test splitting
- Data leakage prevention
- Reproducible workflow

---

## 2. Research Question

> **Does incorporating recent AC power history improve short-term (next 15-minute) solar PV power forecasting compared with weather-sensor data alone?**

Two forecasting models will be developed and compared on the same chronological split:

| Model | Input Features | Target |
|---|---|---|
| **Model A (Weather-only)** | `IRRADIATION(t)`, `AMBIENT_TEMPERATURE(t)`, `MODULE_TEMPERATURE(t)` | `AC_POWER(t+1)` |
| **Model B (Weather + Historical Power)** | Model A features + `AC_POWER(t)`, `AC_POWER(t-1)`, `AC_POWER(t-2)`, `AC_POWER(t-3)` | `AC_POWER(t+1)` |

Models will be evaluated using **MAE**, **RMSE**, and **R²** on a held-out chronological test set. Comparison will use identical data splits and identical evaluation metrics.

---


## 3. Dataset

### Source
- **Name:** Solar Power Generation Data
- **Author:** Ani Kannal
- **Platform:** Kaggle
- **URL:** https://www.kaggle.com/anikannal/solar-power-generation-data
- **License:** Data files © Original Authors

### How the Data Was Accessed
The four CSV files were downloaded from the **Input** tab of the Kaggle notebook
["Forecasting Solar Power: XGBoost vs LSTM"](https://www.kaggle.com/code/joshmoore0/forecasting-solar-power-xgboost-vs-lstm)
by Josh Moore. That notebook links directly to the original Kaggle dataset by
Ani Kannal — no modification or subsetting of the original data was performed
during download. The column names, row counts, date ranges, and units in this
project match the original dataset as published by Ani Kannal.

### Dataset Description
The dataset contains **solar power generation and weather sensor data** collected 
from **two solar power plants in India** over a **34-day period** 
(2020-05-15 to 2020-06-17), sampled at **15-minute intervals**.

The dataset consists of four CSV files — one pair per plant:

| File | Content |
|---|---|
| `Plant_1_Generation_Data.csv` | Inverter-level DC/AC power for Plant 1 |
| `Plant_1_Weather_Sensor_Data.csv` | Plant-level weather sensor data for Plant 1 |
| `Plant_2_Generation_Data.csv` | Inverter-level DC/AC power for Plant 2 |
| `Plant_2_Weather_Sensor_Data.csv` | Plant-level weather sensor data for Plant 2 |

### Actual Columns (Verified from CSV)

**Generation files:**
`DATE_TIME`, `PLANT_ID`, `SOURCE_KEY`, `DC_POWER`, `AC_POWER`, `DAILY_YIELD`, `TOTAL_YIELD`

**Weather sensor files:**
`DATE_TIME`, `PLANT_ID`, `SOURCE_KEY`, `AMBIENT_TEMPERATURE`, `MODULE_TEMPERATURE`, `IRRADIATION`

### Units
| Variable | Unit |
|---|---|
| `DC_POWER`, `AC_POWER` | kW |
| `DAILY_YIELD`, `TOTAL_YIELD` | kWh (cumulative) |
| `AMBIENT_TEMPERATURE`, `MODULE_TEMPERATURE` | °C |
| `IRRADIATION` | kW/m² |

### Important Note on Date Format
The `DATE_TIME` column uses **different formats across files**, which was verified from raw CSV strings:

| File | Actual Format | String Length |
|---|---|---|
| `Plant_1_Generation_Data.csv` | `DD-MM-YYYY HH:MM` | 16 |
| `Plant_2_Generation_Data.csv` | `YYYY-MM-DD HH:MM:SS` | 19 |
| `Plant_1_Weather_Sensor_Data.csv` | `YYYY-MM-DD HH:MM:SS` | 19 |
| `Plant_2_Weather_Sensor_Data.csv` | `YYYY-MM-DD HH:MM:SS` | 19 |

**All timestamps are parsed explicitly using `pd.to_datetime(..., format=...)`** — automatic parsing is avoided because it can silently misinterpret dates.

---

## 4. Plant Selection (Data-Driven)

Both plants were compared on multiple dimensions before selecting one:

| Criterion | Plant 1 | Plant 2 |
|---|---|---|
| Number of inverters | 22 | 22 |
| Weather sensors | 1 | 1 |
| Total generation rows | 68,778 | 67,698 |
| Weather rows | 3,182 | 3,259 |
| Unique generation timestamps | 3,158 | 3,259 |
| Common timestamps (gen ∩ weather) | 3,157 | 3,259 |
| Missing values | 0 | 0 |
| Negative AC_POWER | 0 | 0 |
| Inverter timestamp consistency | High (1718–1734 rows each) | Low (four inverters have only 1275 rows) |
| Daytime mean AC_POWER (plant-level) | ~12,195 kW | ~9,260 kW |

### Decision: **Plant 1** is used.

**Reason:** Plant 1 provides:
1. More consistent inverter-level data (all inverters contribute comparable row counts).
2. Higher average power output (better signal-to-noise ratio).
3. No inverter with significant missing periods.

**Trade-off:** Plant 1 loses 1 timestamp (3158 → 3157) when merged with weather data — a 0.03% loss, which is negligible.

---

## 5. Methodology (Current Stage)

### Step 1 — Data Audit (Completed)
- Loaded all four CSV files.
- Inspected shape, columns, dtypes, and first rows.
- Verified date formats from raw strings (not through automatic parsing).
- Checked for missing values, duplicates, and negative power values.
- Compared Plant 1 and Plant 2 across multiple data-quality dimensions.

### Step 2 — Plant Selection (Completed)
- Plant 1 selected based on data-driven comparison (not on convention).

### Step 3 — Plant-Level Aggregation (Completed)
- Generation data (inverter-level, 22 rows per timestamp) aggregated to plant level using `groupby("DATE_TIME").sum()`.
- Result: 3,158 plant-level rows.

### Step 4 — Merge with Weather Data (Completed)
- Inner join on `DATE_TIME`.
- Result: **3,157 rows × 6 columns** — every row has generation and weather data.
- Verified zero missing values after merge.

### Step 5 — Exploratory Data Analysis 
- Descriptive statistics for all variables.
- Timestamp continuity check (gaps detected and noted).
- Hourly-average profile (clear daily solar cycle).
- Correlation analysis (IRRADIATION is strongly correlated with AC_POWER).
- Distribution analysis (bimodal, dominated by nighttime zeros).
- Visual inspection: time series, scatter plots, correlation heatmap, histogram.

**EDA findings will be available in `figures/eda_initial.png`.**

### Step 6 — Cleaning and Feature Engineering (In Progress)
Upcoming tasks:
- Handle the small number of timestamp gaps detected during EDA.
- Construct target: `AC_POWER(t+1)`.
- Construct lag features for Model B: `AC_POWER(t-1)`, `AC_POWER(t-2)`, `AC_POWER(t-3)`.
- Drop columns that leak future information (`DAILY_YIELD`, `TOTAL_YIELD`).
- Drop highly collinear column (`DC_POWER`) to avoid redundancy with `AC_POWER`.

### Step 7 — Chronological Split (Planned)
- Train: first ~70% of timestamps.
- Validation: next ~15%.
- Test: last ~15%.
- No shuffling. Ever.

### Step 8 — Modeling (Planned)
- Model A (Weather-only) and Model B (Weather + Historical) trained on the same split.
- Algorithms: Linear Regression (baseline), Random Forest, Gradient Boosting.
- Advanced models (e.g., LSTM) not planned — dataset size (~3,157 samples) is modest and would not justify them.

### Step 9 — Evaluation (Planned)
- Metrics: MAE, RMSE, R².
- MAPE will be avoided (or used cautiously) because `AC_POWER` has many near-zero values at night.
- Actual vs predicted plots.
- Feature importance analysis (tree-based models).

---

## 6. Data Leakage Prevention Strategy

Time-series forecasting requires strict care to avoid leakage. The following rules are applied throughout:

1. **Chronological splitting only** — no random shuffle at any stage.
2. **Target shifting is done explicitly** — `y(t) = AC_POWER(t+1)`.
3. **Lag features use only past values** — `AC_POWER(t-1)` uses data from time `t-1` and earlier.
4. **Cumulative columns are dropped** — `DAILY_YIELD` and `TOTAL_YIELD` encode future information relative to earlier timestamps.
5. **Scaling is fitted on training data only** — validation and test sets are transformed using training-set statistics.
6. **No cross-plant mixing** — only Plant 1 data is used.
7. **No future weather is used** — only weather measurements already available at time `t` are used to predict `AC_POWER(t+1)`.

---

## 7. Known Limitations

The project's claims are bounded by the following limitations:

1. **Short time coverage:** ~34 days. No seasonal or year-round claims can be made.
2. **Single location:** Both plants are in India; generalization to other climates is not claimed.
3. **Sensor-based forecasting only:** The model uses currently observed weather, not weather forecasts. This is a one-step-ahead sensor-based forecast, not a weather-forecast-driven system.
4. **Plant-level aggregation:** Inverter-level dynamics are averaged out.
5. **Potential sensor and inverter errors:** Data was checked for obvious issues, but real-world measurement noise is unavoidable.
6. **No production deployment:** This is a research/study project on historical data.
7. **Weather variables limited to three:** The dataset does not include humidity, wind speed, or cloud cover. The project does not fabricate these.

---

## 8. Repository Structure

solar-pv-forecasting-ml/
├── README.md
├── notebooks/
│ └── 01_data_audit_eda.ipynb Data loading, audit, merge, EDA
├── figures/
│ └── eda_initial.png EDA composite plot
├── data/
│ └── README.md Dataset source and licensing info
└── results/
└── (to be added) Model results, metrics, comparison


**Raw CSV files are NOT uploaded to this repository** — they belong to the original Kaggle dataset. Only the download link and license note are provided.

---

## 9. Tools and Libraries

- **Python 3** (Google Colab)
- **pandas** — data loading, manipulation, aggregation, merging
- **NumPy** — numerical operations
- **matplotlib** — visualization
- **scikit-learn** — model training and evaluation (upcoming)

---

## 10. Reproducibility

To reproduce this project:

1. Download the dataset from Kaggle:  
   https://www.kaggle.com/anikannal/solar-power-generation-data
2. Upload the four CSV files to a local folder (or Google Drive).
3. Open `notebooks/01_data_audit_eda.ipynb` in Google Colab or Jupyter.
4. Update the `folder` variable to point to the CSV location.
5. Run the notebook cells sequentially.

All random seeds are fixed (`random_state=42`) where used.

---

## 11. Progress Tracker

- [x] Dataset audit and verification
- [x] Plant selection (Plant 1, data-driven)
- [x] Plant-level aggregation
- [x] Merge with weather data
- [x] Exploratory Data Analysis
- [ ] Cleaning and feature engineering
- [ ] Chronological split
- [ ] Model training (Model A vs Model B)
- [ ] Evaluation and comparison
- [ ] Final report

---

## 12. Acknowledgments

- Dataset provided by **Ani Kannal** on Kaggle.
- Guidance on research methodology: project supervisor.
- Academic context: Department of Electrical and Electronic Engineering, Independent University, Bangladesh (IUB).

---

## 13. Contact

**Ishra Ismat Kamal**  
B.Sc. in Electrical and Electronic Engineering  
Independent University, Bangladesh (IUB)  
GitHub: [Ishra18-hub](https://github.com/Ishra18-hub)
