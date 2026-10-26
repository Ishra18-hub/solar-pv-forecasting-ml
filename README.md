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

The 34 calendar days are divided into 24 training days (2020-05-15–2020-06-07), 5 validation days (2020-06-08–2020-06-12), and 5 test days (2020-06-13–2020-06-17). The same day boundaries are used for all horizons. Rows are purged when the target time falls outside the split containing the feature time.

| Split | 15 min | 30 min | 45 min | 60 min |
|---|---:|---:|---:|---:|
| Train | 2,159 | 2,152 | 2,145 | 2,139 |
| Validation | 479 | 478 | 477 | 476 |
| Test | 473 | 471 | 470 | 469 |
| Boundary rows purged | 2 | 4 | 6 | 8 |

Assertions confirm chronological ordering and that retained targets do not cross split boundaries. Lag features in validation or test can use earlier observations from a prior split because those observations would be available at prediction time.

Daily irradiation (sum of irradiation × 0.25 h) averages 5.37 kWh/m² in train, 5.06 in validation, and 5.05 in test; the test range is 4.16–6.35 kWh/m².

### Modeling and evaluation

Three model families are compared: Linear Regression (with scaling in a pipeline), Random Forest, and Gradient Boosting. Hyperparameters are searched on the training data and selected by **daytime validation RMSE**. The chosen model for each horizon and feature set is refit on train plus validation, then evaluated on the held-out test set. Predictions are clipped at zero.

Baselines are persistence (`AC_POWER(t)`), same time yesterday (`AC_POWER(t+h−24h)`), and smart persistence. The smart-persistence envelope is the 95th percentile of AC power by time of day, estimated from training and validation rows only. Its score may have fewer valid rows than the learned models when its envelope is unavailable.

Uncertainty for paired MAE differences is estimated with 2,000 paired moving-block resamples using a 16-row block size and seed 42. Blocks are resampled within each test day. Negative `MAE_difference` means the model with power history has lower MAE; positive means it has higher MAE. Intervals are nominal 95% intervals and are not adjusted for multiple comparisons.

## 6. Results

The results below are from the updated Plant 1 modeling/evaluation run provided for this README. The test period contains five days. MAE and RMSE are in kW. Model abbreviations: A = A_weather, A+time = A_time, B = B_history, B+time = B_time.

### Plant 1: daytime test performance

| Horizon | Model | Selected algorithm | MAE | RMSE | R² |
|---|---|---|---:|---:|---:|
| 15 min | A | Gradient Boosting | 2,241.410 | 3,205.526 | 0.813 |
| 15 min | A+time | Gradient Boosting | 2,128.267 | 3,036.006 | 0.832 |
| 15 min | B | Random Forest | 2,252.711 | 3,270.058 | 0.805 |
| 15 min | B+time | Gradient Boosting | 2,126.907 | 3,023.286 | 0.834 |
| 30 min | A | Gradient Boosting | 3,035.845 | 4,183.955 | 0.682 |
| 30 min | A+time | Gradient Boosting | 2,703.016 | 3,708.492 | 0.750 |
| 30 min | B | Random Forest | 2,965.797 | 4,207.205 | 0.678 |
| 30 min | B+time | Gradient Boosting | 2,639.568 | 3,651.401 | 0.758 |
| 45 min | A | Random Forest | 3,394.374 | 4,762.022 | 0.592 |
| 45 min | A+time | Gradient Boosting | 2,806.198 | 3,842.788 | 0.734 |
| 45 min | B | Gradient Boosting | 3,254.171 | 4,474.916 | 0.639 |
| 45 min | B+time | Gradient Boosting | 2,815.622 | 3,861.820 | 0.731 |
| 60 min | A | Random Forest | 3,514.880 | 4,958.719 | 0.560 |
| 60 min | A+time | Gradient Boosting | 2,907.837 | 3,917.961 | 0.725 |
| 60 min | B | Random Forest | 3,419.036 | 4,718.675 | 0.601 |
| 60 min | B+time | Gradient Boosting | 3,024.305 | 3,981.092 | 0.716 |

### Plant 1: daytime baselines

| Horizon | Baseline | MAE | RMSE | R² | Valid rows |
|---|---|---:|---:|---:|---:|
| 15 min | Persistence | 2,395.227 | 3,295.760 | 0.802 | 234 |
| 15 min | Same time yesterday | 3,983.165 | 5,450.428 | 0.459 | 234 |
| 15 min | Smart persistence | 2,330.876 | 3,329.368 | 0.789 | 229 |
| 30 min | Persistence | 3,486.617 | 4,472.088 | 0.637 | 233 |
| 30 min | Same time yesterday | 3,997.940 | 5,461.997 | 0.458 | 233 |
| 30 min | Smart persistence | 3,047.255 | 4,186.869 | 0.655 | 224 |
| 45 min | Persistence | 4,112.134 | 5,182.708 | 0.516 | 233 |
| 45 min | Same time yesterday | 3,989.883 | 5,458.734 | 0.463 | 233 |
| 45 min | Smart persistence | 3,328.631 | 4,500.427 | 0.592 | 220 |
| 60 min | Persistence | 4,676.150 | 5,732.855 | 0.412 | 233 |
| 60 min | Same time yesterday | 3,982.910 | 5,457.510 | 0.467 | 233 |
| 60 min | Smart persistence | 3,484.376 | 4,670.488 | 0.547 | 215 |

Baseline valid-row counts differ when a historical value or smart-persistence envelope is unavailable; compare methods with this in mind.

### Does power history help on Plant 1?

`MAE_difference = MAE(with history) − MAE(reference)`. Negative values favor the model with history. The table reports daytime comparisons.

| Horizon | Comparison | MAE difference (kW) | Nominal 95% interval |
|---|---|---:|---:|
| 15 min | B − A | +11.301 | [−71.864, +101.261] |
| 15 min | B+time − A+time | −1.360 | [−50.687, +48.973] |
| 30 min | B − A | −70.048 | [−231.442, +39.366] |
| 30 min | B+time − A+time | −63.448 | [−155.382, −27.949] |
| 45 min | B − A | −140.203 | [−352.780, −66.047] |
| 45 min | B+time − A+time | +9.424 | [−35.038, +54.502] |
| 60 min | B − A | −95.844 | [−238.015, −7.867] |
| 60 min | B+time − A+time | +116.468 | [−33.443, +189.513] |

The point estimates and intervals are mixed. The intervals for 30-minute B+time vs A+time and for 45- and 60-minute B vs A exclude zero in favor of history; the other daytime intervals include zero. These are multiple, unadjusted comparisons on the same five test days, so the intervals should be treated as exploratory and not as independent confirmation. The 60-minute B+time point estimate is worse than A+time, though its interval includes zero.

Time-of-day features improve the selected daytime test MAE relative to weather-only features at each horizon. Smart persistence is a useful benchmark, but it does not outperform A_weather at 15, 30, or 60 minutes in this run; it has lower MAE than A_weather at 45 minutes. A_time and B_time have lower MAE than smart persistence at every horizon. Differences should be interpreted with the differing valid-row counts in mind.

### Plant 2 robustness analysis

The existing Plant 2 analysis uses a fixed fleet of 12 inverters as its main run and mean inverter power × 22 as a sensitivity run. It was not rerun as part of the updated Plant 1 evaluation above; the values below are retained from the prior README and should not be treated as directly method-matched until notebook 05 is rerun with the same updated selection and bootstrap procedure.

| | Plant 1 | Plant 2 |
|---|---:|---:|
| AC power / irradiation correlation | 0.996 | 0.89 (fixed fleet), 0.91 (all-inverter sum) |
| 15-minute daytime persistence R² | 0.802 | 0.588 (fixed fleet) |
| 15-minute oracle daytime R² | 0.98 | 0.56 |
| Timestamps with fewer than 22 inverters | 3.2% | 27.7% |

Prior Plant 2 bootstrap results had positive MAE-gain point estimates for history across the tested comparisons, but intervals were mostly wide. Four of 16 intervals excluded zero: one in the fixed-fleet run and three in the mean × 22 sensitivity run. Plant 2 remains a weaker robustness check because its weather-power relationship is noisier and inverter completeness is lower. Rerun notebook 05 before using these values as final results.

## 7. Leakage prevention

- Splits are chronological and based on complete calendar days; no random shuffling is used.
- Rows whose target time crosses a split boundary are purged.
- Targets and lags use exact timestamps; missing timestamps are never interpolated.
- Lag features use only power observed at or before feature time `t`.
- Cumulative yield columns are excluded from modeling.
- Scaling is fitted inside a pipeline on training data; model selection uses validation data only.
- Selected models are refit on train plus validation, and the test set is used for final evaluation.
- No future weather is used. Time-of-day features use the timestamp at `t`.
- Smart-persistence envelopes use training and validation data only.
- Same-time-yesterday values are from a timestamp earlier than `t` for all horizons in this project.

## 8. Related work

Public work using this Kaggle dataset includes deep-learning approaches such as LSTM, CNN, and Transformer models. This project instead focuses on interpretable linear and tree-based models, controlled feature-set comparisons, chronological evaluation, persistence benchmarks, and timestamp-verified leakage prevention. A full verified citation list will be added to the final report.

## 9. Limitations

1. The dataset covers about 34 days; the held-out test period is only five days. No seasonal or year-round generalization is claimed.
2. Both plants are from one location in India; generalization to other locations is untested.
3. The models use current sensor readings, not weather forecasts.
4. Plant-level aggregation averages over inverter-level behavior.
5. The dataset has only three weather variables and may contain sensor or inverter errors.
6. This is a historical-data study, not a deployed forecasting system.
7. Rows are dropped when exact target or lag timestamps are unavailable: 44 at 15 minutes, 52 at 30 minutes, 59 at 45 minutes, and 65 at 60 minutes before the split purge.
8. The grids are small and there is one chronological split; conclusions are conditional on this dataset, these features, horizons, and test window.
9. The empirical smart-persistence envelope is estimated from a limited number of days and may be noisy.
10. The Plant 2 robustness analysis has not yet been rerun with the updated Plant 1 evaluation procedure.

## 10. Repository structure

```text
solar-pv-forecasting-ml/
├── README.md
├── notebooks/
│   ├── 00_data_exploration_plant_selection.ipynb
│   ├── 01_data_audit_eda.ipynb
│   ├── 02_feature_engineering.ipynb                # first 15-minute version, kept as a record
│   ├── 02b_multihorizon_features.ipynb             # final feature construction
│   ├── 03_Chronological_split.ipynb                # original row-index split, kept as a record
│   ├── 03b_chronological_split_multihorizon.ipynb  # final day-boundary split
│   ├── 04_modeling_evaluation.ipynb
│   └── 05_plant2_robustness.ipynb
├── data/
│   └── README.md
├── figures/
│   └── eda_initial.png
├── results/
│   ├── results_selected.csv
│   ├── results_baselines.csv
│   ├── research_question_comparison.csv
│   ├── model_and_baseline_comparison.csv
│   └── bootstrap_comparison.csv
├── results_plant2/
└── results_plant2_mean22/
```

Raw CSV files are not committed; they belong to the original Kaggle dataset. Intermediate CSVs (`df_clean_h*.csv`, `ds_h*_split.csv`) are reproducible from notebooks 02b and 03b and are not committed.

## 11. Tools and libraries

- Python 3 in Google Colab
- pandas and NumPy for data preparation
- matplotlib for visualizations
- scikit-learn for pipelines, models, and metrics

## 12. Reproducibility

1. Download the four CSV files from the Kaggle dataset page.
2. Put them in one folder, such as Google Drive `Project2_SolarPV/`.
3. Set `folder` in each notebook to that location.
4. Run notebooks in order:
   - `00` and `01`: audit, plant selection, merge, and EDA.
   - `02b`: generate and verify 15-, 30-, 45-, and 60-minute feature data.
   - `03b`: create day-based train/validation/test splits and apply the boundary purge.
   - `04`: fit and evaluate Plant 1 models and baselines.
   - `05`: rerun the Plant 2 robustness analysis; its outputs should be refreshed to match the updated Plant 1 protocol.
5. Do not commit raw or intermediate generated CSV files.

Random seeds are fixed at 42 in the model and bootstrap code.

## 13. Progress

- [x] Dataset audit and plant selection
- [x] Plant-level aggregation, merge, and EDA
- [x] Four-horizon feature engineering and timestamp verification
- [x] Day-boundary chronological splits with target-boundary purge
- [x] Plant 1 model and baseline evaluation (updated protocol)
- [ ] Rerun Plant 2 with the updated evaluation and bootstrap procedure
- [ ] Complete related-work citations
- [ ] Complete final report

## 14. Acknowledgments

- Dataset provided by Ani Kannal on Kaggle.
- Research methodology guidance from the project supervisor.
- Department of Electrical and Electronic Engineering, Independent University, Bangladesh.

## 15. Contact

**Ishra Ismat Kamal**  
B.Sc. in Electrical and Electronic Engineering  
Independent University, Bangladesh (IUB)  
GitHub: [Ishra18-hub](https://github.com/Ishra18-hub)
