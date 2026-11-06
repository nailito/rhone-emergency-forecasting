# rhone-emergency-forecasting

Is it possible to forecast daily emergency department visits in the Rhône region early enough and with a known degree of accuracy, based on data available before the day in question?

## Result

A one-day-ahead linear model using the two previous visit counts and the calendar reaches an **MAE of 60.7 visits per day** (3.9% of the daily mean) on the untouched 2023 test year, against 86.8 for weekly persistence. The 80% and 90% conformal intervals cover 74.5% and 86.6% of days respectively.

See [analysis/05-final-evaluation.ipynb](analysis/05-final-evaluation.ipynb) for the final test, and [analysis/features.py](analysis/features.py) for the shared feature engineering.
