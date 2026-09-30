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

Solar PV power generation is inherently intermittent, it depends on solar irradiance, module temperature, and other environmental conditions. Accurate short-term forecasting of PV power output is important for grid stability, energy scheduling, and plant operation.

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

### Step 5 — Exploratory Data Analysis (Completed)
- Descriptive statistics for all variables.
- Timestamp continuity check — precise gap detection found 10 irregular timestamp transitions (largest gap: 9 hours).
- Hourly-average profile (clear daily solar cycle).
- Correlation analysis (IRRADIATION IRRADIATION strongly correlated with AC_POWER, correlation ≈ 1.00).
- Distribution analysis (bimodal, dominated by nighttime zeros).
- Visual inspection: time series, scatter plots, correlation heatmap, histogram.

**EDA findings available in `figures/eda_initial.png`.**

### Step 6 — Cleaning and Feature Engineering (Completed)
- Dropped leakage-prone and redundant columns: DAILY_YIELD, TOTAL_YIELD (cumulative, leak future info), DC_POWER (perfectly correlated with AC_POWER).
- Detected 10 irregular timestamp gaps using explicit .diff()-based checking.
- Constructed target: AC_POWER(t+1) using shift(-1), marked invalid where the (t → t+1) step crossed a gap.
- Constructed lag features for Model B: AC_POWER(t-1), AC_POWER(t-2), AC_POWER(t-3), marked invalid where any lag step crossed a gap.
- Dropped 54 rows whose target or lag features were affected by a timestamp gap.
- **Verified correctness** of target and lag construction via a timestamp-based lookup check (not positional) — confirmed **0 mismatches** for both target and lag1 alignment.
Final cleaned dataset: **3,103 rows × 9 columns**, saved as df_clean_step6.csv.

### Step 7 — Chronological Split (Planned)
- Train: first ~70% of timestamps.
- Validation: next ~15%.
- Test: last ~15%.
- No shuffling. Ever.

### Step 8 — Modeling (Planned)
- Model A (Weather-only) and Model B (Weather + Historical) trained on the same split.
- Algorithms: Linear Regression (baseline), Random Forest, Gradient Boosting.
- Advanced models (e.g., LSTM) not planned — dataset size (~3,103 samples after cleaning) is modest and would not justify them.

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
4. **Timestamp gaps are explicitly handled** — rows whose target or lag features would cross a non-15-minute gap are dropped rather than interpolated, to avoid fabricating data.
5. **Cumulative columns are dropped** — `DAILY_YIELD` and `TOTAL_YIELD` encode future information relative to earlier timestamps.
6. **Scaling is fitted on training data only** — validation and test sets are transformed using training-set statistics.
7. **No cross-plant mixing** — only Plant 1 data is used.
8. **No future weather is used** — only weather measurements already available at time `t` are used to predict `AC_POWER(t+1)`.

---

## 7. Related Work

This project's dataset (Ani Kannal's Solar Power Generation Data on Kaggle) is widely used, with 400+ public Kaggle notebooks and multiple academic papers using it as a benchmark, including deep-learning approaches such as explainable LSTM-based models. Most published work on this and similar datasets focuses on deep learning architectures (LSTM, CNN, Transformer) for multi-step or probabilistic forecasting.

This project differs by:

- Using a simpler, interpretable tree-based / linear modeling approach appropriate for a modest dataset size (~3,100 samples).
- Explicitly isolating the marginal contribution of historical power lags versus weather-only features through a controlled A/B model comparison on an identical chronological split.
- Applying explicit, verified data-leakage prevention (timestamp-based verification of target/lag construction) rather than relying on standard train-test splitting alone.

**(Full citation list to be added in the final report.)**

---

## 8. Known Limitations

The project's claims are bounded by the following limitations:

1. **Short time coverage:** ~34 days. No seasonal or year-round claims can be made.
2. **Single location:** Both plants are in India; generalization to other climates is not claimed.
3. **Sensor-based forecasting only:** The model uses currently observed weather, not weather forecasts. This is a one-step-ahead sensor-based forecast, not a weather-forecast-driven system.
4. **Plant-level aggregation:** Inverter-level dynamics are averaged out.
5. **Potential sensor and inverter errors:** Data was checked for obvious issues, but real-world measurement noise is unavoidable.
6. **No production deployment:** This is a research/study project on historical data.
7. **Weather variables limited to three:** The dataset does not include humidity, wind speed, or cloud cover. The project does not fabricate these.
8. **Row loss from gap handling:** 54 rows (~1.7%) were dropped during Step 6 due to timestamp gaps; this is a controlled, documented decision rather than an incidental loss.

---

## 9. Repository Structure

solar-pv-forecasting-ml/
├── README.md
├── notebooks/
│   ├── 00_data_exploration_plant_selection.ipynb   Initial audit, plant comparison, format verification
│   ├── 01_data_audit_eda.ipynb                     Aggregation, merge, EDA
│   └── 02_feature_engineering.ipynb                Cleaning, target/lag construction, verification
├── figures/
│   └── eda_initial.png                             EDA composite plot
├── data/
│   └── README.md                                   Dataset source and licensing info
└── results/
    └── (to be added)                               Model results, metrics, comparison


**Raw CSV files are NOT uploaded to this repository** — they belong to the original Kaggle dataset. Only the download link and license note are provided. they belong to the original Kaggle dataset. Only the download link and license note are provided. The cleaned dataset (df_clean_step6.csv) is also not committed to the repository, following the same principle — it is fully reproducible from 02_feature_engineering.ipynb.

---

## 10. Tools and Libraries

- **Python 3** (Google Colab)
- **pandas** — data loading, manipulation, aggregation, merging
- **NumPy** — numerical operations
- **matplotlib** — visualization
- **scikit-learn** — model training and evaluation (upcoming)

---

## 11. Reproducibility

To reproduce this project:

1. Download the dataset from Kaggle:  
   https://www.kaggle.com/anikannal/solar-power-generation-data
2. Upload the four CSV files to a local folder (or Google Drive).
3. Open notebooks/00_data_exploration_plant_selection.ipynb and notebooks/01_data_audit_eda.ipynb in Google Colab or Jupyter to reproduce the audit, plant selection, aggregation, merge, and EDA.
4. Open notebooks/02_feature_engineering.ipynb to reproduce the cleaning, target/lag construction, gap handling, and verification steps.
5. Update the `folder` variable to point to the CSV location.
6. Run the notebook cells sequentially.

All random seeds are fixed (`random_state=42`) where used.

---

## 11. Progress Tracker

- [x] Dataset audit and verification
- [x] Plant selection (Plant 1, data-driven)
- [x] Plant-level aggregation
- [x] Merge with weather data
- [x] Exploratory Data Analysis
- [x] Cleaning and feature engineering
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
