# Data

The raw data is **not included** in this repository. It belongs to the original Kaggle dataset.

- **Dataset:** Solar Power Generation Data
- **Author:** Ani Kannal
- **Platform:** Kaggle
- **Link:** https://www.kaggle.com/anikannal/solar-power-generation-data
- **License:** data files © original authors; see the Kaggle page for the terms of use

## Files expected by the notebooks

| File | Used by |
|---|---|
| `Plant_1_Generation_Data.csv` | notebooks 00, 01, 02b |
| `Plant_1_Weather_Sensor_Data.csv` | notebooks 00, 01, 02b |
| `Plant_2_Generation_Data.csv` | notebooks 00, 05 |
| `Plant_2_Weather_Sensor_Data.csv` | notebooks 00, 05 |

Put all four files in one folder (for example Google Drive `Project2_SolarPV/`) and set the `folder` variable at the top of each notebook.

## Notes

- The date format differs between files (`DD-MM-YYYY HH:MM` in `Plant_1_Generation_Data.csv`, `YYYY-MM-DD HH:MM:SS` in the other three), so the notebooks parse dates with explicit formats.
- Files created by the notebooks (`df_clean_h*.csv`, `ds_h*_split.csv`) are not committed; run notebooks `02b` and `03b` to regenerate them.
