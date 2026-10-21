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

**Main finding (on this dataset):** adding recent AC power history did **not** produce a statistically detectable improvement at any of the three horizons. Knowing the time of day helped more than knowing past power, and a simple smart-persistence benchmark is hard to beat at 15 minutes (Section 6).

---

## 2. Research Question

> **Does incorporating recent AC power history improve short-term (15-, 30- and 60-minute-ahead) solar PV power forecasting compared with weather-sensor data alone?**

Four feature sets are compared on the same chronological split, together with three no-training benchmarks. For a horizon of `h` steps (1 step = 15 minutes) every model predicts `AC_POWER(t+h)`.

| Name | Input features (all measured at time `t`) | Purpose |
|---|---|---|
| **Model A** (weather-only) | `IRRADIATION`, `AMBIENT_TEMPERATURE`, `MODULE_TEMPERATURE` | Weather-only baseline |
| **Model A+time** | Model A + `hour_sin`, `hour_cos` | Control: does knowing the time of day alone help? |
| **Model B** (weather + power history) | Model A + `AC_POWER(t)`, `AC_POWER(t-1)`, `AC_POWER(t-2)`, `AC_POWER(t-3)` | Tests the research question |
| **Model B+time** | Model B + `hour_sin`, `hour_cos` | Fair partner of A+time |
| **Persistence** | none: `AC_POWER(t+h) = AC_POWER(t)` | Naive benchmark |
| **Same time yesterday** | none: `AC_POWER(t+h) = AC_POWER(t+h−24h)` | Diurnal-cycle benchmark |
| **Smart persistence** | none: persistence of the clear-sky index, with an empirical clear-sky envelope per time of day (95th percentile of AC power over the training/validation days) | Stronger benchmark that removes the daily cycle (the dataset has no coordinates, so a physical clear-sky model is not available) |

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
- Inverter-level generation data (22 rows per timestamp) aggregated to plant level.
- In notebook 01 this was a plain sum (`groupby("DATE_TIME")` with sum). **Correction (notebook 02b):** 101 of 3,158 timestamps (3.2%) have fewer than 22 inverters reporting (10 of them in daytime, where the plant total was understated by 22% on average and by up to 82%). Plant-level AC power is therefore computed as **mean inverter AC power × 22**, which stays on the same scale when an inverter is missing.
- Sensitivity check: re-running all models with the original sum aggregation changed daytime MAE by at most 0.4% (RMSE by at most 0.6%), selected the same algorithms, and did not change any conclusion.
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
- Three benchmarks (persistence, same time yesterday, smart persistence) and skill scores against persistence and against smart persistence. The smart-persistence envelope uses training and validation days only.
- **Paired moving-block bootstrap** (2,000 resamples, blocks of 16 consecutive daytime rows = 4 hours) for error differences (model without vs with power history; smart persistence vs ML models), reported in kW and as a percentage of the reference error. Block resampling is used because errors in neighbouring 15-minute rows are strongly correlated.
- Error by test day, actual-vs-predicted plots and tree-based feature importance.
- A diagnostic ("oracle") analysis explains why history adds so little (Section 6).

---

## 6. Results

All numbers are on the **held-out test set (5 days)**. Daytime = 06:00–18:00 at target time. Full tables are in `results/`.
Skill = 1 − RMSE_model / RMSE_benchmark (daytime); positive means better than the benchmark.

### 6.1 Daytime test performance (benchmarks and models)

| Horizon | Model | Algorithm (chosen on validation) | MAE (kW) | RMSE (kW) | R² | Skill vs persistence | Skill vs smart persistence |
|---|---|---|---|---|---|---|---|
| 15min | Persistence | none | 2,395 | 3,296 | 0.802 | n/a | -5.3% |
| 15min | Same time yesterday | none | 3,983 | 5,450 | 0.459 | -65.4% | -74.2% |
| 15min | Smart persistence | none | 2,182 | 3,129 | 0.822 | +5.0% | n/a |
| 15min | A | Random Forest | 2,210 | 3,233 | 0.810 | +1.9% | -3.3% |
| 15min | A+time | Gradient Boosting | 2,111 | 3,046 | 0.831 | +7.6% | +2.7% |
| 15min | B | Random Forest | 2,200 | 3,156 | 0.819 | +4.2% | -0.9% |
| 15min | B+time | Random Forest | 2,111 | 2,984 | 0.838 | +9.4% | +4.6% |
| 30min | Persistence | none | 3,487 | 4,472 | 0.637 | n/a | -11.1% |
| 30min | Same time yesterday | none | 3,998 | 5,462 | 0.458 | -22.1% | -35.7% |
| 30min | Smart persistence | none | 2,932 | 4,025 | 0.706 | +10.0% | n/a |
| 30min | A | Gradient Boosting | 3,036 | 4,184 | 0.682 | +6.4% | -4.0% |
| 30min | A+time | Gradient Boosting | 2,675 | 3,700 | 0.751 | +17.3% | +8.1% |
| 30min | B | Random Forest | 2,955 | 4,181 | 0.682 | +6.5% | -3.9% |
| 30min | B+time | Gradient Boosting | 2,644 | 3,672 | 0.755 | +17.9% | +8.8% |
| 60min | Persistence | none | 4,676 | 5,733 | 0.412 | n/a | -28.4% |
| 60min | Same time yesterday | none | 3,983 | 5,458 | 0.467 | +4.8% | -22.2% |
| 60min | Smart persistence | none | 3,301 | 4,465 | 0.643 | +22.1% | n/a |
| 60min | A | Random Forest | 3,534 | 4,944 | 0.563 | +13.8% | -10.7% |
| 60min | A+time | Gradient Boosting | 2,908 | 3,918 | 0.725 | +31.7% | +12.2% |
| 60min | B | Random Forest | 3,451 | 4,738 | 0.598 | +17.3% | -6.1% |
| 60min | B+time | Gradient Boosting | 3,005 | 4,003 | 0.713 | +30.2% | +10.3% |

All-hours metrics are in `results/results_selected.csv` and `results/results_baselines.csv`. They look better for every model (night rows are trivially zero) and are therefore less informative about the research question.

![Daytime error vs horizon](results/horizon_comparison.png)

### 6.2 Does power history help? (paired block bootstrap, daytime)

Gain = error of the reference − error of the compared model, in kW. **Positive means the compared model (with power history) is better.** The % column divides by the reference model's error, so it states how large an improvement could plausibly be.

| Horizon | Comparison (reference → compared) | MAE gain, kW [95% CI] | MAE gain as % of reference (95% CI) | RMSE gain, kW [95% CI] |
|---|---|---|---|---|
| 15min | A → B | +9 [-74, +107] | -3.3% to +4.8% | +76 [-58, +229] |
| 15min | A+time → B+time | 0 [-69, +71] | -3.2% to +3.4% | +62 [-72, +197] |
| 30min | A → B | +81 [-40, +256] | -1.3% to +8.4% | +3 [-150, +228] |
| 30min | A+time → B+time | +32 [-51, +123] | -1.9% to +4.6% | +28 [-74, +133] |
| 60min | A → B | +84 [-103, +291] | -2.9% to +8.2% | +205 [-1, +408] |
| 60min | A+time → B+time | -97 [-191, +20] | -6.6% to +0.7% | -85 [-190, +30] |

### 6.3 Does machine learning beat smart persistence? (paired block bootstrap, daytime)

| Horizon | Comparison (reference → compared) | MAE gain, kW [95% CI] | MAE gain as % of reference (95% CI) | RMSE gain, kW [95% CI] |
|---|---|---|---|---|
| 15min | SmartPersistence → A+time | +71 [-89, +236] | -4.1% to +10.8% | +83 [-76, +274] |
| 15min | SmartPersistence → B+time | +71 [-62, +222] | -2.8% to +10.2% | +145 [0, +322] |
| 30min | SmartPersistence → A+time | +257 [+33, +483] | +1.1% to +16.5% | +325 [+97, +575] |
| 30min | SmartPersistence → B+time | +289 [+99, +505] | +3.4% to +17.2% | +353 [+169, +577] |
| 60min | SmartPersistence → A+time | +393 [+67, +751] | +2.0% to +22.7% | +547 [+219, +924] |
| 60min | SmartPersistence → B+time | +296 [+16, +608] | +0.5% to +18.4% | +462 [+173, +804] |

(All paired comparisons, including A+time → B, are in `results/bootstrap_comparison.csv`.)

### 6.4 Interpretation

1. **Power history gave no reliably detectable benefit.** Every A → B and A+time → B+time MAE interval includes zero. The 95% intervals for the A → B MAE gain correspond to relative changes of -3.3% to +4.8% at 15 minutes, -1.3% to +8.4% at 30 minutes and -2.9% to +8.2% at 60 minutes. In words: **in this test window, an MAE improvement from power history larger than about 4.8% at 15 minutes is excluded, but at 30 and 60 minutes gains of up to about 8% cannot be ruled out.** The RMSE intervals are wider (upper bound at 15 minutes: +7.1%), and at 60 minutes the A → B RMSE interval only just reaches zero (about −1 to +408 kW), which is treated as inconclusive. "No significant difference" is not proof of "no effect", especially with a 5-day test window.
2. **Time of day mattered more than power history.** Adding `hour_sin`/`hour_cos` to Model A reduced daytime MAE by 4.5% (15 min), 11.9% (30 min) and 17.7% (60 min). Once time of day is included, adding history does not help: A+time → B+time has MAE intervals that include zero at every horizon, with a negative point estimate at 60 minutes (-97 kW).
3. **Plain persistence is a weak benchmark; smart persistence is strong.** Same time yesterday is poor at 15 and 30 minutes (daytime MAE 3,983 and 3,998 kW) and only beats plain persistence at 60 minutes. Smart persistence (clear-sky-index persistence with an empirical envelope) reaches a daytime MAE of 2,182 kW at 15 minutes, **better than weather-only Model A (2,210 kW) and Model B (2,200 kW)**. Against smart persistence, Models A and B have negative skill at every horizon (-3.3% / -0.9% at 15 min; -10.7% / -6.1% at 60 min).
4. **Machine learning reliably beats smart persistence only at 30–60 minutes, and only with time of day.** At 15 minutes the A+time and B+time gains over smart persistence (+71 kW for A+time, interval -89 to +236) are not distinguishable from zero. At 30 minutes (+257 kW for A+time, +289 kW for B+time) and 60 minutes (+393 and +296 kW) the MAE intervals exclude zero.
5. **Why history adds so little:** `AC_POWER(t)` is almost a function of `IRRADIATION(t)` (correlation 0.996), so power lags carry nearly the same information as the weather features already in Model A. A diagnostic where a simple model is given the irradiation at the *target* time (not a valid forecast) reaches daytime MAE ≈ 750–754 kW and R² ≈ 0.98 at every horizon, versus ≈ 2,100–3,500 kW for real models. The limit is the unpredictability of future irradiation (cloud-driven fluctuations), not the choice of model:

| Horizon | Mean abs. irradiation change over the horizon (kW/m²) | Share of daytime rows with change > 0.1 kW/m² | Oracle daytime MAE (kW) | Oracle daytime R² |
|---|---|---|---|---|
| 15min | 0.091 | 31% | 750 | 0.981 |
| 30min | 0.123 | 47% | 752 | 0.981 |
| 60min | 0.170 | 65% | 754 | 0.980 |

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
12. **Benchmark envelope uses training/validation days only:** the smart-persistence clear-sky envelope is never estimated from test days.
13. **Same-time-yesterday reads only past values** (`t+h−24h` is earlier than `t` for all horizons used).

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
12. **Smart persistence is empirical:** its clear-sky envelope is estimated from ~29 days of plant data, not from a physical clear-sky model, and may be noisy.
13. **Single plant so far:** Plant 2 has not been evaluated; it has 28% of timestamps with fewer than 22 inverters and needs the same corrected aggregation.

---

## 10. Repository Structure

```
solar-pv-forecasting-ml/
├── README.md
├── notebooks/
│   ├── 00_data_exploration_plant_selection.ipynb   Initial audit, plant comparison, format verification
│   ├── 01_data_audit_eda.ipynb                     Aggregation, merge, EDA
│   ├── 02_feature_engineering.ipynb                First version (15-min only), kept as a record
│   ├── 02b_multihorizon_features.ipynb             Plant aggregation fix, multi-horizon targets/lags, time features, verification
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
    ├── results_baselines.csv                       Persistence, same-time-yesterday, smart persistence
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
- [ ] Additional benchmarks (same time yesterday, smart persistence) and aggregation fix with sensitivity check
- [ ] Plant 2 robustness check
- [ ] Literature review for Related Work (with verified citations)
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
