# Machine Learning-Based Solar PV Power Generation Forecasting Using Weather Data

A machine learning project that forecasts the AC power output of a solar photovoltaic (PV) plant **15, 30 and 60 minutes ahead** using co-located weather sensor data and recent power output history.

**Author:** Ishra Ismat Kamal  
**Department:** Electrical and Electronic Engineering (EEE)  
**Institution:** Independent University, Bangladesh (IUB)  
**Project Type:** Undergraduate ML Research Project (Project 2)

---

## 1. Project Overview

Solar PV power generation is intermittent: it depends on solar irradiance, module temperature, and other environmental conditions. Accurate short-term forecasting of PV power output is important for grid stability, energy scheduling, and plant operation.

This project investigates whether **incorporating recent power output history improves short-term solar PV power forecasting compared with using weather-sensor data alone**, and how that effect changes with the forecast horizon.

The project is developed as an independent machine learning study, building on the author's EEE background with an emphasis on:
- Time-series aware data handling
- Chronological train/validation/test splitting
- Data leakage prevention
- A controlled, reproducible model comparison

**Main finding (on this dataset):** adding recent AC power history did **not** produce a statistically detectable improvement at any of the three horizons. Knowing the time of day helped more than knowing past power (Section 6).

---

## 2. Research Question

> **Does incorporating recent AC power history improve short-term (15-, 30- and 60-minute-ahead) solar PV power forecasting compared with weather-sensor data alone?**

Four feature sets are compared on the same chronological split, together with a no-training persistence benchmark. For a horizon of `h` steps (1 step = 15 minutes) every model predicts `AC_POWER(t+h)`.

| Name | Input features (all measured at time `t`) | Purpose |
|---|---|---|
| **Model A** (weather-only) | `IRRADIATION`, `AMBIENT_TEMPERATURE`, `MODULE_TEMPERATURE` | Weather-only baseline |
| **Model A+time** | Model A + `hour_sin`, `hour_cos` | Control: does knowing the time of day alone help? |
| **Model B** (weather + power history) | Model A + `AC_POWER(t)`, `AC_POWER(t-1)`, `AC_POWER(t-2)`, `AC_POWER(t-3)` | Tests the research question |
| **Model B+time** | Model B + `hour_sin`, `hour_cos` | Fair partner of A+time |
| **Persistence** | none: `AC_POWER(t+h) = AC_POWER(t)` | Standard benchmark for short-term forecasts |

| Horizon | Steps `h` | Target |
|---|---|---|
| 15 min | 1 | `AC_POWER(t+1)` |
| 30 min | 2 | `AC_POWER(t+2)` |
| 60 min | 4 | `AC_POWER(t+4)` |

The most direct test of the research question is **A vs B**. The fairest test of whether history helps *beyond time of day* is **A+time vs B+time**.

Models are evaluated using **MAE**, **RMSE** and **R²** on a held-out chronological test set, on **all rows** and on **daytime rows only**.

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
by Josh Moore. That notebook links directly to the original Kaggle dataset by Ani Kannal. No modification or subsetting of the original data was performed during download. The column names, row counts, date ranges, and units in this project match the original dataset as published by Ani Kannal.

### Dataset Description
The dataset contains **solar power generation and weather sensor data** collected from **two solar power plants in India** over a **34-day period** (2020-05-15 to 2020-06-17), sampled at **15-minute intervals**.

| File | Content |
|---|---|
| `Plant_1_Generation_Data.csv` | Inverter-level DC/AC power for Plant 1 |
| `Plant_1_Weather_Sensor_Data.csv` | Plant-level weather sensor data for Plant 1 |
| `Plant_2_Generation_Data.csv` | Inverter-level DC/AC power for Plant 2 |
| `Plant_2_Weather_Sensor_Data.csv` | Plant-level weather sensor data for Plant 2 |

### Actual Columns (Verified from CSV)

**Generation files:** `DATE_TIME`, `PLANT_ID`, `SOURCE_KEY`, `DC_POWER`, `AC_POWER`, `DAILY_YIELD`, `TOTAL_YIELD`

**Weather sensor files:** `DATE_TIME`, `PLANT_ID`, `SOURCE_KEY`, `AMBIENT_TEMPERATURE`, `MODULE_TEMPERATURE`, `IRRADIATION`

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

**All timestamps are parsed explicitly using `pd.to_datetime(..., format=...)`**. Automatic parsing is avoided because it can silently misinterpret dates.

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
2. A larger signal amplitude (higher average power output).
3. No inverter with significant missing periods.

**Trade-off:** Plant 1 loses 1 timestamp (3,158 → 3,157) when merged with weather data, a 0.03% loss, which is negligible.

---

## 5. Methodology

### Step 1: Data Audit (Completed)
- Loaded all four CSV files; inspected shape, columns, dtypes, and first rows.
- Verified date formats from raw strings (not through automatic parsing).
- Checked for missing values, duplicates, and negative power values.

### Step 2: Plant Selection (Completed)
- Plant 1 selected based on the data-driven comparison in Section 4.

### Step 3: Plant-Level Aggregation (Completed)
- Inverter-level generation data (22 rows per timestamp) aggregated to plant level using `groupby("DATE_TIME")` with sum.
- Result: 3,158 plant-level rows.

### Step 4: Merge with Weather Data (Completed)
- Inner join on `DATE_TIME`.
- Result: **3,157 rows**, each with generation and weather data; zero missing values after the merge.

### Step 5: Exploratory Data Analysis (Completed)
- Descriptive statistics for all variables.
- Timestamp continuity check: **10 irregular timestamp transitions** (largest gap: 9 hours).
- Hourly profile shows a clear daily solar cycle.
- Correlation of `AC_POWER` with `IRRADIATION` = **0.996**, with `MODULE_TEMPERATURE` = 0.961, with `AMBIENT_TEMPERATURE` = 0.726.
- Distribution of `AC_POWER` is bimodal: **about 46.6% of rows are exactly zero** (night). This is why daytime-only metrics are reported alongside all-hours metrics.
- Visual inspection: time series, scatter plots, correlation heatmap, histogram (`figures/eda_initial.png`).

### Step 6: Cleaning and Feature Engineering, first version (Completed, notebook 02)
- Dropped leakage-prone and redundant columns: `DAILY_YIELD`, `TOTAL_YIELD` (cumulative), `DC_POWER` (almost perfectly correlated with `AC_POWER`).
- Detected 10 irregular timestamp gaps using explicit `.diff()` checking.
- Built the 15-minute target `AC_POWER(t+1)` and lags `AC_POWER(t-1..t-3)`, marking rows invalid when a step crossed a gap.
- Dropped 54 rows → 3,103 rows × 9 columns, verified with a timestamp-based lookup.

> **Correction found later (Step 6b):** the validity rule in notebook 02 contained one term too many (`gap_before.shift(3)`), which discarded one perfectly valid row after every gap. This is conservative: it causes no leakage and no wrong values, only the loss of 10 valid rows. The corrected count is **3,113 rows kept, 44 dropped** (10 gaps × 4 rows + 3 rows at the start + 1 row at the end). Notebook 02 is kept unchanged as a record; Step 6b supersedes it.

### Step 6b: Multi-Horizon Feature Engineering (Completed, notebook 02b)
- The target and lag construction is wrapped in one function, `build_dataset(base, h)`, producing the 15-, 30- and 60-minute datasets with identical logic.
- Target and lags are looked up **by timestamp** (`t + h·15 min`, `t − k·15 min`). If the exact timestamp does not exist the row is dropped (never interpolated).
- Added time-of-day features `hour_sin`, `hour_cos` and a `TARGET_TIME` column (used for daytime masks and leakage checks).
- Verified with an independent dictionary-lookup check: **0 mismatches** for the target and for lag1, lag2 and lag3 at every horizon.
- Verified that every row kept by notebook 02 is present with identical values.

| Horizon | Rows kept | Rows dropped | Columns |
|---|---|---|---|
| 15 min | 3,113 | 44 | 12 |
| 30 min | 3,105 | 52 | 12 |
| 60 min | 3,092 | 65 | 12 |

### Step 7: Chronological Split (Completed, notebook 03b)
- The split is made **on calendar-day boundaries** (34 days → 24 train / 5 validation / 5 test) and is identical for all horizons.
- No shuffling at any stage.
- **Boundary purge:** a row is kept only if its target time falls in the same split as its feature time, so no training target lies inside the validation period (and none in validation lies inside the test period).
- Verified programmatically with `assert` statements for every horizon.

| Split | Days | Period | Rows (15 / 30 / 60 min) |
|---|---|---|---|
| Train | 24 | 2020-05-15 to 2020-06-07 | 2,159 / 2,152 / 2,139 |
| Validation | 5 | 2020-06-08 to 2020-06-12 | 479 / 478 / 476 |
| Test | 5 | 2020-06-13 to 2020-06-17 | 473 / 471 / 469 |

Daily insolation (sum of `IRRADIATION` × 0.25 h) averages 5.37 kWh/m² in train, 5.06 in validation and 5.05 in test (test range 4.16–6.35), so the test window is neither unusually clear nor unusually cloudy.

> The original row-index split (70/15/15, notebook 03) cut the data at 05:00 and 01:15. It is kept as a record but is not used for the final results.

### Step 8: Modeling (Completed, notebook 04)
- **Algorithms:** Linear Regression (inside a `Pipeline` with `StandardScaler`, so scaling is fitted on training data only), Random Forest, Gradient Boosting.
- **Hyperparameter tuning on the validation days only** (small grids: Random Forest `max_depth` and `min_samples_leaf`; Gradient Boosting `n_estimators`, `learning_rate`, `max_depth`). Selection criterion: daytime RMSE on validation.
- The best settings are **refit on train + validation** (both strictly earlier than test) and the **test set is used once**.
- For each horizon and feature set, the reported algorithm is the one with the best **validation** score (never chosen on test).
- Predictions are clipped at 0 kW (power cannot be negative).
- Advanced models (e.g., LSTM) are not used: ~3,100 samples would not justify them.

### Step 9: Evaluation (Completed, notebook 04)
- Metrics: MAE, RMSE, R² on all rows and on **daytime rows** (06:00 ≤ target time < 18:00).
- MAPE is not used because `AC_POWER` is zero at night.
- Skill score versus persistence: `1 − RMSE_model / RMSE_persistence` (daytime).
- **Paired moving-block bootstrap** (2,000 resamples, blocks of 16 consecutive daytime rows = 4 hours) for the error difference between models with and without power history. Block resampling is used because errors in neighbouring 15-minute rows are strongly correlated.
- Error by test day, actual-vs-predicted plots and tree-based feature importance.
- A diagnostic ("oracle") analysis explains why history adds so little (Section 6).

---

## 6. Results

All numbers are on the **held-out test set (5 days)**. Daytime = 06:00–18:00 at target time. Full tables are in `results/`.

### 6.1 Daytime test performance

| Horizon | Model | Algorithm (chosen on validation) | MAE (kW) | RMSE (kW) | R² | Skill vs persistence |
|---|---|---|---|---|---|---|
| 15 min | Persistence | none | 2,395 | 3,296 | 0.802 | n/a |
| 15 min | A | Random Forest | 2,209 | 3,232 | 0.810 | 1.9% |
| 15 min | A+time | Gradient Boosting | 2,117 | 3,027 | 0.833 | 8.1% |
| 15 min | B | Random Forest | 2,201 | 3,154 | 0.819 | 4.3% |
| 15 min | B+time | Random Forest | 2,114 | 3,000 | 0.836 | 9.0% |
| 30 min | Persistence | none | 3,487 | 4,472 | 0.637 | n/a |
| 30 min | A | Gradient Boosting | 3,027 | 4,166 | 0.685 | 6.9% |
| 30 min | A+time | Gradient Boosting | 2,668 | 3,692 | 0.752 | 17.4% |
| 30 min | B | Random Forest | 2,957 | 4,181 | 0.683 | 6.5% |
| 30 min | B+time | Gradient Boosting | 2,634 | 3,660 | 0.757 | 18.2% |
| 60 min | Persistence | none | 4,676 | 5,733 | 0.412 | n/a |
| 60 min | A | Random Forest | 3,536 | 4,945 | 0.562 | 13.7% |
| 60 min | A+time | Gradient Boosting | 2,908 | 3,932 | 0.723 | 31.4% |
| 60 min | B | Random Forest | 3,452 | 4,740 | 0.598 | 17.3% |
| 60 min | B+time | Gradient Boosting | 3,004 | 4,002 | 0.713 | 30.2% |

All-hours metrics are in `results/results_selected.csv`. They are better for every model (night rows are trivially zero) and are therefore less informative about the research question.

![Daytime error vs horizon](results/horizon_comparison.png)

### 6.2 Does power history help? (paired block bootstrap, daytime)

Gain = error without history − error with history, in kW. **Positive means power history helps.** 95% confidence intervals in brackets.

| Horizon | Comparison (without → with history) | MAE gain | RMSE gain |
|---|---|---|---|
| 15 min | A → B | +8 [−73, +105] | +78 [−55, +229] |
| 15 min | A+time → B+time | +4 [−81, +82] | +28 [−93, +154] |
| 30 min | A → B | +70 [−53, +233] | −15 [−167, +184] |
| 30 min | A+time → B+time | +35 [−40, +127] | +32 [−68, +139] |
| 60 min | A → B | +84 [−94, +282] | +205 [0, +405] |
| 60 min | A+time → B+time | −96 [−179, −4] | −70 [−180, +24] |

(The extra comparison A+time → B is in `results/bootstrap_comparison.csv`.)

### 6.3 Interpretation

1. **Power history gave no reliably detectable benefit.** For MAE, every A → B interval includes zero. At 15 minutes the interval for A → B is about −73 to +105 kW on an error of roughly 2,200 kW, so any real benefit is probably below about 5%. At 60 minutes the A → B RMSE interval only just reaches zero (0 to +405 kW) while the MAE interval includes zero; this is treated as inconclusive. "No significant difference" is not proof of "no effect", especially with a 5-day test window.
2. **Time of day mattered more than power history**, especially at longer horizons: at 60 minutes, A+time reduces daytime MAE by about 18% relative to A. When time of day is available, adding history does not help (and at 60 minutes B+time is slightly worse than A+time).
3. **All models beat persistence, but only modestly at 15 minutes** (skill 2–9%). The advantage grows with the horizon because persistence degrades quickly.
4. **Why history adds so little:** `AC_POWER(t)` is almost a function of `IRRADIATION(t)` (correlation 0.996), so power lags carry nearly the same information as the weather features already in Model A. A diagnostic where a simple model is given the irradiation at the *target* time (not a valid forecast) reaches daytime MAE ≈ 751–755 kW and R² ≈ 0.98 at every horizon, versus ≈ 2,100–3,500 kW for real models. The limit is therefore the unpredictability of future irradiation (cloud-driven fluctuations), not the choice of model:

| Horizon | Mean abs. irradiation change over the horizon (kW/m²) | Share of daytime rows with change > 0.1 kW/m² | Oracle daytime MAE (kW) | Oracle daytime R² |
|---|---|---|---|---|
| 15 min | 0.091 | 31% | 751 | 0.981 |
| 30 min | 0.123 | 47% | 754 | 0.980 |
| 60 min | 0.170 | 65% | 755 | 0.980 |

(Mean daytime irradiation is about 0.45 kW/m².) See `results/diagnostic_oracle_irradiation.csv`.

---

## 7. Data Leakage Prevention Strategy

Time-series forecasting requires strict care to avoid leakage. The following rules are applied throughout:

1. **Chronological splitting only**, on calendar-day boundaries; no random shuffle at any stage. Verified with assertion checks for every horizon.
2. **Boundary purge:** rows whose target time falls in a different split than their feature time are removed.
3. **Target shifting is done explicitly** with exact timestamps: `y = AC_POWER(t+h)`.
4. **Lag features use only past values:** `AC_POWER(t-k)` uses data from time `t-k` and earlier.
5. **Timestamp gaps are handled explicitly:** rows whose target or lag timestamp does not exist are dropped rather than interpolated.
6. **Cumulative columns are dropped:** `DAILY_YIELD` and `TOTAL_YIELD` encode future information relative to earlier timestamps.
7. **Scaling is fitted on training data only** (inside a scikit-learn `Pipeline`); trees do not need scaling.
8. **Hyperparameters are tuned on validation days only;** the test set is used once.
9. **The reported algorithm for each model is selected on validation, not test.**
10. **No cross-plant mixing:** only Plant 1 data is used.
11. **No future weather is used:** only measurements available at time `t` predict `AC_POWER(t+h)`. Time-of-day features are derived from the timestamp at `t`.

---

## 8. Related Work

This project's dataset (Ani Kannal's Solar Power Generation Data on Kaggle) is widely used in public Kaggle notebooks and in published studies, many of which apply deep-learning architectures (LSTM, CNN, Transformer) to multi-step or probabilistic forecasting.

This project differs by:

- Using a simple, interpretable linear / tree-based approach suited to a modest dataset (~3,100 samples).
- Isolating the marginal contribution of power-history features through a controlled comparison (A vs B, with and without time-of-day features) on an identical chronological split, at three horizons.
- Benchmarking against persistence and quantifying uncertainty with a block bootstrap.
- Applying explicit, verified data-leakage prevention (timestamp-based verification of target and lag construction).

**(Full citation list to be added in the final report.)**

---

## 9. Known Limitations

1. **Short time coverage:** ~34 days, test window only 5 days. No seasonal or year-round claims can be made, and bootstrap intervals describe this test window only.
2. **Single location:** the plants are in India; generalization to other climates is not claimed.
3. **Sensor-based forecasting only:** the models use currently observed weather, not weather forecasts. This is not a weather-forecast-driven system.
4. **Plant-level aggregation:** inverter-level dynamics are averaged out.
5. **Potential sensor and inverter errors:** the data was checked for obvious issues, but real-world measurement noise is unavoidable.
6. **No production deployment:** this is a study on historical data.
7. **Weather variables limited to three:** the dataset does not include humidity, wind speed, or cloud cover, and none are fabricated.
8. **Row loss from gap handling:** 44 rows (~1.4%) are dropped at 15 minutes (up to 65 rows, ~2.1%, at 60 minutes) because of timestamp gaps. This is a controlled, documented decision.
9. **Small hyperparameter grids and one split:** with ~2,100 training rows, differences between algorithms are often within noise.
10. **Daytime definition:** daytime is fixed at 06:00–18:00 at target time, following the window used in the plant-selection analysis.
11. **Negative result is conditional:** the absence of a detectable benefit from power history applies to this dataset, these features (three lags) and these horizons.

---

## 10. Repository Structure

```
solar-pv-forecasting-ml/
├── README.md
├── notebooks/
│   ├── 00_data_exploration_plant_selection.ipynb   Initial audit, plant comparison, format verification
│   ├── 01_data_audit_eda.ipynb                     Aggregation, merge, EDA
│   ├── 02_feature_engineering.ipynb                First version (15-min only), kept as a record
│   ├── 02b_multihorizon_features.ipynb             Multi-horizon targets/lags, time features, verification
│   ├── 03_Chronological_split.ipynb                First split (row index), kept as a record
│   ├── 03b_chronological_split_multihorizon.ipynb  Day-boundary split with boundary purge
│   └── 04_modeling_evaluation.ipynb                Models, evaluation, bootstrap, diagnostic
├── figures/
│   └── eda_initial.png                             EDA composite plot
├── data/
│   └── README.md                                   Dataset source and licensing info
└── results/
    ├── results_selected.csv                        Selected model per horizon and feature set
    ├── results_all_models.csv                      All algorithms
    ├── results_persistence.csv                     Persistence benchmark
    ├── bootstrap_comparison.csv                    Paired block-bootstrap comparisons
    ├── error_by_test_day.csv                       Daytime MAE per test day
    ├── diagnostic_oracle_irradiation.csv           Oracle diagnostic
    ├── horizon_comparison.png
    ├── actual_vs_predicted_h1.png
    └── feature_importance_B_h1.png
```

**Raw CSV files are NOT uploaded to this repository.** They belong to the original Kaggle dataset; only the download link and license note are provided. The intermediate CSV files created by the notebooks (`df_clean_h*.csv`, `ds_h*_split.csv`) are also not committed, because they are fully reproducible from notebooks `02b` and `03b`.

---

## 11. Tools and Libraries

- **Python 3** (Google Colab)
- **pandas**: data loading, manipulation, aggregation, merging
- **NumPy**: numerical operations
- **matplotlib**: visualization
- **scikit-learn**: `LinearRegression`, `RandomForestRegressor`, `GradientBoostingRegressor`, `StandardScaler`, `Pipeline`, evaluation metrics

---

## 12. Reproducibility

1. Download the dataset from Kaggle: https://www.kaggle.com/anikannal/solar-power-generation-data
2. Place the CSV files in one folder (e.g., Google Drive `Project2_SolarPV/`).
3. In each notebook, set the `folder` variable to that location.
4. Run the notebooks in this order:
   - `00` and `01`: audit, plant selection, aggregation, merge, EDA
   - `02b`: multi-horizon features and verification (writes `df_clean_h1/h2/h4.csv`)
   - `03b`: day-boundary split (writes `ds_h1/h2/h4_split.csv`)
   - `04`: modeling, evaluation, bootstrap, diagnostic (writes the `results/` folder, about 2 minutes)
5. Notebooks `02` and `03` document the first 15-minute-only version and are not needed for the final results.

All random seeds are fixed (`random_state=42` for Random Forest and Gradient Boosting; the bootstrap uses seed 42).

---

## 13. Progress Tracker

- [x] Dataset audit and verification
- [x] Plant selection (Plant 1, data-driven)
- [x] Plant-level aggregation
- [x] Merge with weather data
- [x] Exploratory Data Analysis
- [ ] Cleaning and feature engineering (single and multi-horizon)
- [ ] Chronological split (day-boundary, with purge)
- [ ] Model training (A, A+time, B, B+time; three horizons)
- [ ] Evaluation and comparison (persistence, daytime metrics, block bootstrap)
- [ ] Diagnostic analysis of why history adds little
- [ ] Final report (with full citation list)

---

## 14. Acknowledgments

- Dataset provided by **Ani Kannal** on Kaggle.
- Guidance on research methodology: project supervisor.
- Academic context: Department of Electrical and Electronic Engineering, Independent University, Bangladesh (IUB).

---

## 15. Contact

**Ishra Ismat Kamal**  
B.Sc. in Electrical and Electronic Engineering  
Independent University, Bangladesh (IUB)  
GitHub: [Ishra18-hub](https://github.com/Ishra18-hub)
