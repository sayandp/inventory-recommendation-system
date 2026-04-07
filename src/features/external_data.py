# src/features/external_data.py

import pandas as pd
import requests
import meteostat
from meteostat import Point


# =========================
# 1. UK HOLIDAYS
# =========================
def add_holidays(df):
    print("Adding UK holiday data...")

    # UK = GB
    url = "https://date.nager.at/api/v3/PublicHolidays/2011/GB"
    holidays = requests.get(url).json()

    holiday_df = pd.DataFrame(holidays)[["date"]]
    holiday_df["date"] = pd.to_datetime(holiday_df["date"])
    holiday_df["is_holiday"] = 1

    df = df.merge(holiday_df, on="date", how="left")
    df["is_holiday"] = df["is_holiday"].fillna(0)

    return df


# =========================
# 2. UK WEATHER 
# =========================

from meteostat import daily
import pandas as pd


def add_weather(df):
    print("Adding REAL UK weather data...")

    df["date"] = pd.to_datetime(df["date"])

    start = df["date"].min()
    end = df["date"].max()

    station = "03772"  # Heathrow

    weather = daily(station, start, end)
    weather = weather.fetch().reset_index()

    if weather.empty:
        print("⚠️ Weather data empty → using fallback")
        df["temperature"] = 10
        df["rain"] = 0
        return df

    # HANDLE MISSING COLUMNS
    if "tavg" not in weather.columns:
        if "tmin" in weather.columns and "tmax" in weather.columns:
            weather["tavg"] = (weather["tmin"] + weather["tmax"]) / 2
        else:
            print("⚠️ No temperature data → fallback")
            weather["tavg"] = 10

    if "prcp" not in weather.columns:
        weather["prcp"] = 0

    weather = weather[["time", "tavg", "prcp"]]
    weather.columns = ["date", "temperature", "rain"]

    weather["date"] = pd.to_datetime(weather["date"])

    df = df.merge(weather, on="date", how="left")

    df["temperature"] = df["temperature"].fillna(df["temperature"].mean())
    df["rain"] = df["rain"].fillna(0)

    return df