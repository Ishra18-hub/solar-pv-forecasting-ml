# Machine Learning-Based Solar PV Power Generation Forecasting Using Weather Data

A machine learning study forecasting solar photovoltaic (PV) plant AC power 15, 30, 45, and 60 minutes ahead using weather sensor data and recent power output history.

**Author:** Ishra Ismat Kamal  
**Department:** Electrical and Electronic Engineering (EEE)  
**Institution:** Independent University, Bangladesh (IUB)  
**Project type:** Undergraduate ML research project

## 1. Project overview

Solar PV generation varies with irradiance, module temperature, and other environmental conditions. This project examines whether recent AC power history improves short-term forecasts beyond weather measurements, and whether that effect changes with forecast horizon.

The workflow emphasizes chronological splitting, timestamp-based feature construction, leakage prevention, and controlled comparisons between feature sets.

**Current Plant 1 result:** Weather and time-of-day features give the strongest daytime forecasts in most comparisons. The added value of recent power history is mixed: some nominal bootstrap intervals favor history, others include zero, and the 60-minute all-row comparison with time-of-day features favors the model without history. These results come from a five-day test window and multiple unadjusted comparisons, so they are exploratory rather than general evidence of a consistent history benefit.

Plant 2 is a separate robustness analysis. Its earlier results show positive but imprecise point estimates for power history; that analysis has not yet been rerun with the updated Plant 1 evaluation code described below.

## 2. Research question and feature sets

> Does recent AC power history improve short-term PV power forecasts compared with weather-sensor data alone?

Each observation uses features available at time `t` to predict `AC_POWER(t+h)`. One step is 15 minutes.

| Name | Input features measured at time `t` | Purpose |
|---|---|---|
| **A_weather** | `IRRADIATION`, `AMBIENT_TEMPERATURE`, `MODULE_TEMPERATURE` | Weather-only baseline |
| **A_time** | A_weather + `hour_sin`, `hour_cos` | Measures the value of time of day |
| **B_history** | A_weather + `AC_POWER_lag1`, `AC_POWER_lag2`, `AC_POWER_lag3` | Tests the contribution of recent power history |
| **B_time** | B_history + `hour_sin`, `hour_cos` | Tests history beyond weather and time of day |

| Forecast horizon | Steps `h` | Target |
|---|---:|---|
| 15 minutes | 1 | `AC_POWER(t+1)` |
| 30 minutes | 2 | `AC_POWER(t+2)` |
| 45 minutes | 3 | `AC_POWER(t+3)` |
| 60 minutes | 4 | `AC_POWER(t+4)` |

The direct comparisons are **A_weather vs B_history** and **A_time vs B_time**. Three no-training baselines are also evaluated: persistence, same time yesterday, and smart persistence. Metrics are MAE, RMSE, and R², reported for all test rows and for daytime targets (06:00 inclusive to 18:00 exclusive).

## 3. Dataset

The dataset is **Solar Power Generation Data** by Ani Kannal, hosted on Kaggle: [dataset page](https://www.kaggle.com/anikannal/solar-power-generation-data). The four CSVs were obtained from the Input tab of Josh Moore's Kaggle notebook, [Forecasting Solar Power: XGBoost vs LSTM](https://www.kaggle.com/code/joshmoore0/forecasting-solar-power-xgboost-vs-lstm), which links to the original dataset. Raw data files are not included in this repository; consult the Kaggle page for usage terms.

The data contains generation and weather measurements from two solar plants in India, sampled at 15-minute intervals from 2020-05-15 through 2020-06-17.

| File | Contents |
|---|---|
| `Plant_1_Generation_Data.csv` | Inverter-level DC/AC generation for Plant 1 |
| `Plant_1_Weather_Sensor_Data.csv` | Plant 1 weather sensor measurements |
| `Plant_2_Generation_Data.csv` | Inverter-level DC/AC generation for Plant 2 |
| `Plant_2_Weather_Sensor_Data.csv` | Plant 2 weather sensor measurements |

The generation files contain `DATE_TIME`, `PLANT_ID`, `SOURCE_KEY`, `DC_POWER`, `AC_POWER`, `DAILY_YIELD`, and `TOTAL_YIELD`. Weather files contain `DATE_TIME`, `PLANT_ID`, `SOURCE_KEY`, `AMBIENT_TEMPERATURE`, `MODULE_TEMPERATURE`, and `IRRADIATION`.

The project treats `AC_POWER` as kW, temperatures as °C, irradiation as kW/m², and cumulative yields as kWh. Plant 1 `DC_POWER` is on a scale inconsistent with its AC values, so it is excluded from modeling. Date strings are parsed with explicit formats: Plant 1 generation uses `%d-%m-%Y %H:%M`; the other three files use `%Y-%m-%d %H:%M:%S`.

See [`data/README.md`](data/README.md) for file placement and regeneration notes.

## 4. Plant selection

Both plants were audited before Plant 1 was selected for the main analysis.

| Criterion | Plant 1 | Plant 2 |
|---|---:|---:|
| Inverters | 22 | 22 |
| Generation rows | 68,778 | 67,698 |
| Weather rows | 3,182 | 3,259 |
| Unique generation timestamps | 3,158 | 3,259 |
| Shared generation/weather timestamps | 3,157 | 3,259 |
| Missing values / negative AC power | 0 / 0 | 0 / 0 |
| Daytime inverter record completeness (06:00–18:00) | 1,593–1,599 for all 22 | 1,171–1,628; 10 of 22 incomplete |
| Mean daytime AC power per inverter | ~601 kW | ~481 kW |

Plant 1 has more consistent inverter records and a larger daytime signal. Plant 2 is retained as a robustness check, with caveats about its missing inverter records and lower weather-power relationship.

## 5. Methodology

### Data preparation and feature engineering

1. Plant 1 inverter readings are aggregated by timestamp. The corrected AC total is the **mean AC power of reporting inverters × 22**, which avoids mechanically lowering the total when an inverter is absent. A sensitivity analysis compares this with the original sum aggregation.
2. The generation and weather tables are joined on `DATE_TIME`, producing 3,157 shared timestamps.
3. `DAILY_YIELD`, `TOTAL_YIELD`, and `DC_POWER` are excluded from model features.
4. Targets and power lags are looked up by exact timestamp. Missing target or lag timestamps are dropped, not interpolated. The feature-generation notebook verifies targets and all three lags at each horizon with an independent timestamp lookup.
5. Cyclical `hour_sin` and `hour_cos` variables encode time of day at feature time `t`.

| Horizon | Rows before split | Dropped during feature creation |
|---|---:|---:|
| 15 min | 3,113 | 44 |
| 30 min | 3,105 | 52 |
| 45 min | 3,098 | 59 |
| 60 min | 3,092 | 65 |

### Chronological split
