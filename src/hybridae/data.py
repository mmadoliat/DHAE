"""Data loaders: U.S. age-specific fertility (HFD-style wide CSV) and the UCI
Metro Interstate Traffic Volume data (daily curve + 49 nonfunctional variables)."""
from __future__ import annotations
import gzip, zipfile
from pathlib import Path
from typing import List, Tuple
import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline

# ------------------------------------------------------------------ fertility
def load_usa_fertility(csv_path, smooth_s: float | None = 50.0):
    """Returns ages [P], years [N], raw rates [N,P], smoothed rates [N,P] (rows = years).

    smooth_s : smoothing factor of scipy's UnivariateSpline applied to each year (the
    notebooks used 50, which on these per-woman rates returns the cubic least-squares
    polynomial; see REPRODUCIBILITY_AUDIT.md). None = no pre-smoothing (smoothed = raw)."""
    df = pd.read_csv(csv_path)
    ages = df["Age"].to_numpy()
    year_cols = [c for c in df.columns if c != "Age"]
    years = np.array([int(c) for c in year_cols])
    rates = df[year_cols].to_numpy()                       # [P, N]
    smoothed = np.zeros_like(rates, dtype=float)
    for i in range(rates.shape[1]):
        smoothed[:, i] = rates[:, i] if smooth_s is None else UnivariateSpline(ages, rates[:, i], s=smooth_s)(ages)
    order = np.argsort(years)
    return ages, years[order], rates.T[order].astype(float), smoothed.T[order].astype(float)


# ------------------------------------------------------------------ Metro
REQUIRED = {"holiday", "temp", "rain_1h", "snow_1h", "clouds_all", "weather_main",
            "weather_description", "date_time", "traffic_volume"}
WEATHER_CATS = ["Clear", "Clouds", "Rain", "Mist", "Snow", "Drizzle", "Haze", "Fog", "Thunderstorm"]


def load_hourly_traffic(path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as zf:
            member = [n for n in zf.namelist() if n.lower().endswith((".csv", ".csv.gz"))][0]
            with zf.open(member) as f:
                if member.lower().endswith(".gz"):
                    with gzip.open(f, mode="rt") as g:
                        df = pd.read_csv(g)
                else:
                    df = pd.read_csv(f)
    else:
        df = pd.read_csv(path, compression="infer")
    missing = sorted(REQUIRED.difference(df.columns))
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    df = df.copy()
    df["date_time"] = pd.to_datetime(df["date_time"])
    df = df.sort_values("date_time").reset_index(drop=True)
    df["date"] = df["date_time"].dt.floor("D")
    df["hr"] = df["date_time"].dt.hour.astype(int)
    return df


def _mode(x: pd.Series):
    m = x.dropna().mode()
    return np.nan if len(m) == 0 else m.iloc[0]


def _one_hot(values: pd.Series, prefix: str, cats: List[str]) -> pd.DataFrame:
    d = pd.get_dummies(values.fillna("Missing").astype(str), prefix=prefix)
    return d.reindex(columns=[f"{prefix}_{c}" for c in cats], fill_value=0.0).astype(np.float32)


def build_metro_daily(df_raw: pd.DataFrame):
    """Port of build_daily_functional_and_nonfunctional_data. Returns
    hour_grid [24], x_counts [n,24], y_raw [n,p], feature names, dates."""
    agg = {"traffic_volume": "mean", "temp": "mean", "rain_1h": "mean", "snow_1h": "mean",
           "clouds_all": "mean", "holiday": _mode, "weather_main": _mode, "weather_description": _mode}
    df_hour = (df_raw.groupby(["date", "hr"], as_index=False).agg(agg)
               .sort_values(["date", "hr"]).reset_index(drop=True))
    curve = df_hour.pivot_table(index="date", columns="hr", values="traffic_volume", aggfunc="mean")
    curve = curve.reindex(columns=range(24)).dropna(axis=0)
    dates = pd.DatetimeIndex(curve.index)
    g = df_hour[df_hour["date"].isin(dates)].copy().groupby("date")
    cal = pd.DataFrame(index=dates)
    cal["is_holiday"] = (g["holiday"].agg(_mode).reindex(dates).fillna("None").astype(str) != "None").astype(float)
    cal["is_weekend"] = (dates.day_of_week >= 5).astype(float)
    cal["year_index"] = (dates.year - dates.year.min()).astype(float)
    cal["month_sin"] = np.sin(2 * np.pi * dates.month / 12.0); cal["month_cos"] = np.cos(2 * np.pi * dates.month / 12.0)
    cal["weekday_sin"] = np.sin(2 * np.pi * dates.day_of_week / 7.0); cal["weekday_cos"] = np.cos(2 * np.pi * dates.day_of_week / 7.0)
    cal["doy_sin"] = np.sin(2 * np.pi * dates.day_of_year / 365.25); cal["doy_cos"] = np.cos(2 * np.pi * dates.day_of_year / 365.25)
    cont = pd.DataFrame(index=dates)
    for col in ["temp", "rain_1h", "snow_1h", "clouds_all"]:
        cont[f"{col}_mean"] = g[col].mean().reindex(dates); cont[f"{col}_min"] = g[col].min().reindex(dates)
        cont[f"{col}_max"] = g[col].max().reindex(dates); cont[f"{col}_std"] = g[col].std().reindex(dates).fillna(0.0)
    cont["rain_any"] = (g["rain_1h"].max().reindex(dates).fillna(0.0) > 0).astype(float)
    cont["snow_any"] = (g["snow_1h"].max().reindex(dates).fillna(0.0) > 0).astype(float)
    cont["temp_range"] = cont["temp_max"] - cont["temp_min"]
    cont["clouds_range"] = cont["clouds_all_max"] - cont["clouds_all_min"]
    parts = [cal.astype(np.float32), cont.astype(np.float32),
             _one_hot(g["weather_main"].agg(_mode).reindex(dates).fillna("Missing").astype(str), "weather", WEATHER_CATS)]
    hol = g["holiday"].agg(_mode).reindex(dates).fillna("None").astype(str)
    hcats = sorted([x for x in hol.unique().tolist() if x != "None"])[:12]
    if hcats:
        parts.append(_one_hot(hol, "holiday", hcats))
    y_df = pd.concat(parts, axis=1)
    valid = ~y_df.isna().any(axis=1)
    if not valid.all():
        y_df = y_df.loc[valid]; curve = curve.loc[y_df.index]; dates = pd.DatetimeIndex(y_df.index)
    return (np.arange(24, dtype=np.float32), curve.to_numpy(dtype=np.float32),
            y_df.to_numpy(dtype=np.float32), list(y_df.columns), dates)


def weather_target_columns(names: List[str]) -> List[int]:
    """Indices of the 29 weather variables (targets); the rest are calendar/holiday metadata."""
    cols = []
    for v in ["temp", "rain_1h", "snow_1h", "clouds_all"]:
        cols += [f"{v}_{s}" for s in ["mean", "min", "max", "std"]]
    cols += ["temp_range", "clouds_range", "rain_any", "snow_any"] + [f"weather_{w}" for w in WEATHER_CATS]
    return [names.index(c) for c in cols]
