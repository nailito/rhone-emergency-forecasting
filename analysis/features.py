"""Feature engineering shared by the Rhône emergency-department notebooks.

The 13 predictors used by every model are defined once, here, so the notebooks
cannot drift apart. A new forecast is made at J-1 for the target day J: the
recent-visit lags and the calendar of day J are both known at that point.
"""
from __future__ import annotations

import holidays
import pandas as pd

VALIDATION_START = pd.Timestamp("2022-01-01")
FINAL_TEST_START = pd.Timestamp("2023-01-01")

WEEKDAY_COLUMNS = [f"dow_{day}" for day in range(1, 7)]  # Monday (0) is the reference
CALENDAR_COLUMNS = ["is_holiday", "day_before_holiday", "day_after_holiday", "bridge_day"]
FEATURE_COLUMNS = [
    "visits_lag1",
    "visits_lag7",
    *WEEKDAY_COLUMNS,
    *CALENDAR_COLUMNS,
    "christmas_eve",
]


def lag_features(visits: pd.Series) -> pd.DataFrame:
    """Visits observed one day and seven days before each date."""
    return pd.DataFrame(
        {"visits_lag1": visits.shift(1), "visits_lag7": visits.shift(7)},
        index=visits.index,
    )


def weekday_indicators(dates: pd.DatetimeIndex) -> pd.DataFrame:
    """One-hot weekday indicators, with Monday as the reference level."""
    indicators = pd.get_dummies(dates.day_of_week, prefix="dow", drop_first=True, dtype=int)
    indicators.index = dates
    return indicators.reindex(columns=WEEKDAY_COLUMNS, fill_value=0)


def holiday_calendar(dates: pd.DatetimeIndex) -> pd.DataFrame:
    """French public holidays and the days immediately around them.

    A bridge day is a Friday after a Thursday holiday, or a Monday before a
    Tuesday holiday. Both are known in advance, so they are usable at J-1.
    """
    years = range(dates.min().year - 1, dates.max().year + 2)
    holiday_dates = pd.DatetimeIndex(sorted(holidays.France(years=years).keys()))
    one_day = pd.Timedelta(days=1)
    tomorrow_is_holiday = (dates + one_day).isin(holiday_dates)
    yesterday_was_holiday = (dates - one_day).isin(holiday_dates)
    weekday = dates.day_of_week
    bridge_day = ((weekday == 4) & yesterday_was_holiday) | (
        (weekday == 0) & tomorrow_is_holiday
    )
    return pd.DataFrame(
        {
            "is_holiday": dates.isin(holiday_dates),
            "day_before_holiday": tomorrow_is_holiday,
            "day_after_holiday": yesterday_was_holiday,
            "bridge_day": bridge_day,
        },
        index=dates,
    ).astype(int)


def eve_indicator(dates: pd.DatetimeIndex) -> pd.Series:
    """Christmas Eve and New Year's Eve share one indicator."""
    values = dates.strftime("%m-%d").isin(["12-24", "12-31"]).astype(int)
    return pd.Series(values, index=dates, name="christmas_eve")


def build_features(visits: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    """Return the 13 predictors and the matching target, one row per date.

    The first week of the series is dropped because it has no lag-7 value.
    """
    visits = visits.sort_index()
    dates = visits.index
    X = pd.concat(
        [
            lag_features(visits),
            weekday_indicators(dates),
            holiday_calendar(dates),
            eve_indicator(dates),
        ],
        axis=1,
    ).dropna()
    assert list(X.columns) == FEATURE_COLUMNS
    return X, visits.loc[X.index]
