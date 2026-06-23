import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class BikeFeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, comfort_temp=24, extreme_humidity=70):
        self.comfort_temp = comfort_temp
        self.extreme_humidity = extreme_humidity

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        X["holiday"] = X["holiday"].replace(
            {
                "Holiday": 1,
                "No Holiday": 0,
            }
        ).astype(float)

        X["functioning_day"] = X["functioning_day"].replace(
            {
                "Yes": 1,
                "No": 0,
            }
        ).astype(float)

        time_columns = [
            "time_period_evening",
            "time_period_late_evening",
            "time_period_morning",
            "time_period_night",
        ]
        condlist = [
            X["time_period_evening"] == True,
            X["time_period_late_evening"] == True,
            X["time_period_morning"] == True,
            X["time_period_night"] == True,
            X[time_columns].sum(axis=1) == 0,
        ]
        choicelist = [
            "evening",
            "late_evening",
            "morning",
            "night",
            "day",
        ]
        X["time_period"] = np.select(condlist, choicelist, "unknown")

        X["temp_zone"] = pd.cut(
            x=X["temperature"],
            bins=[-np.inf, 5, 15, 27, +np.inf],
            labels=["cold", "cool", "comfortable", "hot"],
        ).astype("object").fillna("unknown")

        X["comfort_distance"] = (X["temperature"] - self.comfort_temp).abs()

        X["dew_point_distance"] = X["temperature"] - X["dew_point_temperature"]

        X["extreme_humidity_flag"] = (
            X["humidity"] > self.extreme_humidity
        ).astype(float)

        return X.drop(columns=time_columns)
