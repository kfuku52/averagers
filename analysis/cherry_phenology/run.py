#!/usr/bin/env python3
"""Reproduce the averagers x JMA cherry-flowering pilot analysis.

This is a feasibility study rather than a biological flowering model.  It asks
whether reconstructing daily mean temperature from daily extrema changes
temperature error, flowering-date predictions, or inferred thermal
requirements.  NASA POWER T2M is the daily-mean reference; JMA observations
provide flowering dates and station coordinates.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

import averagers  # noqa: E402


DEFAULT_CACHE_DIR = REPOSITORY_ROOT / ".cache" / "cherry_phenology"
DEFAULT_OUTPUT_DIR = REPOSITORY_ROOT / "report" / "cherry_phenology"

JMA_PHENOLOGY_URL = "https://www.data.jma.go.jp/sakura/data/ruinenchi/004.csv"
JMA_STATIONS_URL = "https://www.jma.go.jp/bosai/amedas/const/amedastable.json"

START_YEAR = 1981
END_YEAR = 2025
SELECTED_STATIONS = [
    "札幌",
    "仙台",
    "新潟",
    "東京",
    "名古屋",
    "京都",
    "大阪",
    "広島",
    "福岡",
    "鹿児島",
]
ENGLISH_NAMES = {
    "札幌": "Sapporo",
    "仙台": "Sendai",
    "新潟": "Niigata",
    "東京": "Tokyo",
    "名古屋": "Nagoya",
    "京都": "Kyoto",
    "大阪": "Osaka",
    "広島": "Hiroshima",
    "福岡": "Fukuoka",
    "鹿児島": "Kagoshima",
}
METHOD_COLUMNS = {
    "POWER daily mean": "Ave",
    "Simple min-max mean": "Ave_simple",
    "DH2006": "Ave_dh2006",
    "Diurnal3": "Ave_diurnal3",
}
BASE_TEMPERATURES = (0.0, 5.0, 10.0)
RANDOM_SEED = 20260722


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help="Download and reconstructed-temperature cache directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory in which summary.json is written",
    )
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=5000,
        help="Number of paired station/year bootstrap replicates",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Ignore the cached reconstructed-temperature table",
    )
    return parser.parse_args()


def download_if_missing(url: str, path: Path) -> None:
    if path.exists() and path.stat().st_size > 0:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "averagers-pilot"})
    with urllib.request.urlopen(request, timeout=60) as response:
        path.write_bytes(response.read())


def load_phenology(raw_dir: Path) -> pd.DataFrame:
    path = raw_dir / "jma_sakura_flowering_004.csv"
    download_if_missing(JMA_PHENOLOGY_URL, path)
    with path.open(encoding="cp932", newline="") as handle:
        rows = list(csv.reader(handle))

    header = rows[1]
    year_columns = {
        year: header.index(str(year)) for year in range(START_YEAR, END_YEAR + 1)
    }
    records = []
    for row in rows[2:]:
        station = row[1].strip()
        if station not in SELECTED_STATIONS:
            continue
        for year, column in year_columns.items():
            raw_value = int(row[column] or 0)
            remark = int(row[column + 1] or 0)
            if raw_value == 0 or remark not in {8, 9}:
                continue
            month, day = divmod(raw_value, 100)
            flowering_date = pd.Timestamp(year=year, month=month, day=day)
            records.append(
                {
                    "Station": station,
                    "Station_en": ENGLISH_NAMES[station],
                    "Year": year,
                    "Flowering_date": flowering_date,
                    "Flowering_DOY": flowering_date.dayofyear,
                }
            )

    phenology = pd.DataFrame.from_records(records)
    expected_years = END_YEAR - START_YEAR + 1
    counts = phenology.groupby("Station")["Year"].nunique()
    incomplete = [
        station
        for station in SELECTED_STATIONS
        if counts.get(station, 0) != expected_years
    ]
    if incomplete:
        raise ValueError(f"Incomplete standard-species records: {incomplete}")
    return phenology.sort_values(["Station", "Year"]).reset_index(drop=True)


def load_coordinates(raw_dir: Path) -> pd.DataFrame:
    path = raw_dir / "jma_amedas_station_table.json"
    download_if_missing(JMA_STATIONS_URL, path)
    stations = json.loads(path.read_text(encoding="utf-8"))
    records = []
    for station_id, metadata in stations.items():
        station = metadata.get("kjName")
        if station not in SELECTED_STATIONS:
            continue
        records.append(
            {
                "Station": station,
                "Station_en": ENGLISH_NAMES[station],
                "Amedas_station_id": station_id,
                "Latitude": metadata["lat"][0] + metadata["lat"][1] / 60,
                "Longitude": metadata["lon"][0] + metadata["lon"][1] / 60,
                "Elevation_m": metadata["alt"],
            }
        )
    coordinates = pd.DataFrame.from_records(records)
    if set(coordinates["Station"]) != set(SELECTED_STATIONS):
        raise ValueError("Not all selected stations were found in the AMeDAS table")
    return coordinates.sort_values("Latitude", ascending=False).reset_index(drop=True)


def reconstruct_station(row: pd.Series, power_cache_dir: Path) -> pd.DataFrame:
    print(f"Fetching and fitting {row['Station_en']}", flush=True)
    weather = averagers.fetch_power_daily_temperature(
        start_date=f"{START_YEAR}-01-02",
        end_date=f"{END_YEAR}-12-31",
        lat=float(row["Latitude"]),
        lon=float(row["Longitude"]),
        add_max_prev=True,
        timeout=90,
        retries=3,
        retry_delay=2,
        cache_dir=power_cache_dir,
    )
    weather["Date"] = pd.to_datetime(weather["Date"])
    weather["Year"] = weather["Date"].dt.year
    weather["Month"] = weather["Date"].dt.month
    photoperiod = averagers.get_photoperiod(
        start_date=weather["Date"].min(),
        end_date=weather["Date"].max(),
        lat=float(row["Latitude"]),
        lon=float(row["Longitude"]),
        timezone=9,
        elevation=float(row["Elevation_m"]),
    )
    weather = weather.join(
        photoperiod[
            ["Sunrise_nondimensional", "Sunset_nondimensional", "Daytime"]
        ]
    )
    reconstructed, _ = averagers.cross_validate_estimates(
        weather,
        specs=[
            {"name": "Simple min-max mean", "column": "Ave_simple", "kind": "simple"},
            {
                "name": "DH2006",
                "column": "Ave_dh2006",
                "kind": "yearly",
                "method": "DH2006",
                "optimizer": "least_squares",
            },
            {
                "name": "Diurnal3",
                "column": "Ave_diurnal3",
                "kind": "yearly",
                "method": "Diurnal3",
                "optimizer": "least_squares",
            },
        ],
        observed_column="Ave",
        fold_column="Year",
    )
    reconstructed["Station"] = row["Station"]
    reconstructed["Station_en"] = row["Station_en"]
    reconstructed["Latitude"] = float(row["Latitude"])
    return reconstructed[
        [
            "Station",
            "Station_en",
            "Latitude",
            "Date",
            "Year",
            "Min",
            "Max",
            "Ave",
            "Ave_simple",
            "Ave_dh2006",
            "Ave_diurnal3",
            "Daytime",
        ]
    ]


def load_or_reconstruct_weather(
    coordinates: pd.DataFrame,
    cache_dir: Path,
    refresh: bool,
) -> pd.DataFrame:
    path = cache_dir / "processed" / "temperature_reconstructions.csv.gz"
    if path.exists() and not refresh:
        print(f"Reading cached temperatures: {path}", flush=True)
        return pd.read_csv(path, parse_dates=["Date"])

    power_cache_dir = cache_dir / "cache"
    frames = [
        reconstruct_station(row, power_cache_dir)
        for _, row in coordinates.iterrows()
    ]
    weather = pd.concat(frames, ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    weather.to_csv(path, index=False, compression="gzip", date_format="%Y-%m-%d")
    return weather


def temperature_metrics(weather: pd.DataFrame) -> pd.DataFrame:
    records = []
    for method, column in METHOD_COLUMNS.items():
        if column == "Ave":
            continue
        error = (weather[column] - weather["Ave"]).dropna()
        records.append(
            {
                "Method": method,
                "N_days": len(error),
                "RMSE_C": math.sqrt(float(np.mean(error**2))),
                "MAE_C": float(np.mean(np.abs(error))),
                "Mean_error_C": float(np.mean(error)),
            }
        )
    return pd.DataFrame.from_records(records)


def thermal_time(weather: pd.DataFrame) -> pd.DataFrame:
    warm_season = weather.loc[weather["Date"].dt.month <= 6].copy()
    frames = []
    for method, column in METHOD_COLUMNS.items():
        for base_temperature in BASE_TEMPERATURES:
            current = warm_season[
                ["Station", "Station_en", "Latitude", "Date", "Year", column]
            ].copy()
            current["Method"] = method
            current["Base_temperature_C"] = base_temperature
            current["Daily_GDD"] = (current[column] - base_temperature).clip(lower=0)
            current["Cumulative_GDD"] = current.groupby(
                ["Station", "Year"]
            )["Daily_GDD"].cumsum()
            frames.append(current.drop(columns=[column]))
    return pd.concat(frames, ignore_index=True)


def predict_flowering(
    thermal: pd.DataFrame,
    phenology: pd.DataFrame,
) -> pd.DataFrame:
    events = phenology.merge(
        thermal,
        left_on=["Station", "Year", "Flowering_date"],
        right_on=["Station", "Year", "Date"],
        how="left",
        validate="one_to_many",
    )
    if events["Cumulative_GDD"].isna().any():
        raise ValueError("Missing thermal time on one or more flowering dates")

    daily_groups = {
        key: group[["Date", "Cumulative_GDD"]].sort_values("Date")
        for key, group in thermal.groupby(
            ["Station", "Year", "Method", "Base_temperature_C"],
            sort=False,
        )
    }
    predictions = []
    for keys, group in events.groupby(
        ["Station", "Station_en_x", "Method", "Base_temperature_C"],
        sort=False,
    ):
        station, station_en, method, base_temperature = keys
        for event in group.itertuples(index=False):
            threshold = float(
                group.loc[group["Year"] != event.Year, "Cumulative_GDD"].median()
            )
            daily = daily_groups[(station, event.Year, method, base_temperature)]
            reached = daily.loc[daily["Cumulative_GDD"] >= threshold, "Date"]
            predicted_date = reached.iloc[0] if not reached.empty else pd.NaT
            predicted_doy = (
                predicted_date.dayofyear if pd.notna(predicted_date) else np.nan
            )
            error = predicted_doy - event.Flowering_DOY
            predictions.append(
                {
                    "Station": station,
                    "Station_en": station_en,
                    "Latitude": event.Latitude,
                    "Year": event.Year,
                    "Method": method,
                    "Base_temperature_C": base_temperature,
                    "Observed_DOY": event.Flowering_DOY,
                    "Event_GDD": event.Cumulative_GDD,
                    "Predicted_DOY": predicted_doy,
                    "Date_error_days": error,
                    "Absolute_error_days": abs(error),
                }
            )
    return pd.DataFrame.from_records(predictions)


def flowering_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    records = []
    for (method, base_temperature), group in predictions.groupby(
        ["Method", "Base_temperature_C"]
    ):
        error = group["Date_error_days"].dropna()
        records.append(
            {
                "Method": method,
                "Base_temperature_C": base_temperature,
                "N_station_years": len(error),
                "RMSE_days": math.sqrt(float(np.mean(error**2))),
                "MAE_days": float(np.mean(np.abs(error))),
                "Mean_error_days": float(np.mean(error)),
                "Pearson_r": float(
                    np.corrcoef(
                        group.loc[error.index, "Observed_DOY"],
                        group.loc[error.index, "Predicted_DOY"],
                    )[0, 1]
                ),
            }
        )
    return pd.DataFrame.from_records(records)


def summarize_thresholds(predictions: pd.DataFrame) -> pd.DataFrame:
    thresholds = (
        predictions.groupby(
            ["Station", "Station_en", "Latitude", "Method", "Base_temperature_C"]
        )["Event_GDD"]
        .median()
        .rename("Median_event_GDD")
        .reset_index()
    )
    reference = thresholds.loc[
        thresholds["Method"] == "POWER daily mean",
        ["Station", "Base_temperature_C", "Median_event_GDD"],
    ].rename(columns={"Median_event_GDD": "Reference_event_GDD"})
    thresholds = thresholds.merge(
        reference,
        on=["Station", "Base_temperature_C"],
        validate="many_to_one",
    )
    thresholds["GDD_shift_percent"] = 100 * (
        thresholds["Median_event_GDD"] - thresholds["Reference_event_GDD"]
    ) / thresholds["Reference_event_GDD"]
    return thresholds


def bootstrap_penalties(
    predictions: pd.DataFrame,
    n_bootstrap: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED)
    reference = predictions.loc[
        predictions["Method"] == "POWER daily mean",
        ["Station", "Year", "Base_temperature_C", "Absolute_error_days"],
    ].rename(columns={"Absolute_error_days": "Reference_absolute_error_days"})
    comparison = predictions.merge(
        reference,
        on=["Station", "Year", "Base_temperature_C"],
        validate="many_to_one",
    )
    comparison["Absolute_error_penalty_days"] = (
        comparison["Absolute_error_days"]
        - comparison["Reference_absolute_error_days"]
    )

    records = []
    for (method, base_temperature), group in comparison.groupby(
        ["Method", "Base_temperature_C"]
    ):
        matrix = group.pivot(
            index="Year",
            columns="Station",
            values="Absolute_error_penalty_days",
        )
        if matrix.isna().any().any():
            raise ValueError("Bootstrap matrix must contain every station-year")
        values = matrix.to_numpy()
        estimates = np.empty(n_bootstrap)
        for i in range(n_bootstrap):
            sampled_years = rng.integers(0, values.shape[0], size=values.shape[0])
            sampled_stations = rng.integers(0, values.shape[1], size=values.shape[1])
            estimates[i] = values[np.ix_(sampled_years, sampled_stations)].mean()
        records.append(
            {
                "Method": method,
                "Base_temperature_C": base_temperature,
                "Mean_absolute_error_penalty_days": float(values.mean()),
                "CI95_low": float(np.quantile(estimates, 0.025)),
                "CI95_high": float(np.quantile(estimates, 0.975)),
            }
        )
    return pd.DataFrame.from_records(records)


def threshold_shift_summary(thresholds: pd.DataFrame) -> list[dict[str, object]]:
    comparison = thresholds.loc[thresholds["Method"] != "POWER daily mean"]
    comparison = comparison.replace([np.inf, -np.inf], np.nan)
    result = (
        comparison.groupby(["Method", "Base_temperature_C"])["GDD_shift_percent"]
        .agg(["count", "median", "min", "max"])
        .reset_index()
        .rename(
            columns={
                "count": "N_stations",
                "median": "Median_shift_percent",
                "min": "Minimum_shift_percent",
                "max": "Maximum_shift_percent",
            }
        )
    )
    return result.to_dict(orient="records")


def main() -> None:
    args = parse_args()
    if args.bootstrap < 1:
        raise ValueError("--bootstrap must be at least 1")

    raw_dir = args.cache_dir / "raw"
    phenology = load_phenology(raw_dir)
    coordinates = load_coordinates(raw_dir)
    weather = load_or_reconstruct_weather(coordinates, args.cache_dir, args.refresh)

    complete_columns = list(METHOD_COLUMNS.values())
    weather = weather.loc[weather[complete_columns].notna().all(axis=1)].copy()
    daily_metrics = temperature_metrics(weather)
    predictions = predict_flowering(thermal_time(weather), phenology)
    prediction_metrics = flowering_metrics(predictions)
    thresholds = summarize_thresholds(predictions)
    bootstrap = bootstrap_penalties(predictions, args.bootstrap)

    summary = {
        "design": {
            "stations": int(phenology["Station"].nunique()),
            "year_start": START_YEAR,
            "year_end": END_YEAR,
            "phenology_records": int(len(phenology)),
            "temperature_days_complete": int(len(weather)),
            "temperature_reference": "NASA POWER T2M",
            "validation": "leave-one-year-out",
            "bootstrap_replicates": args.bootstrap,
        },
        "daily_temperature_metrics": daily_metrics.to_dict(orient="records"),
        "flowering_prediction_metrics": prediction_metrics.to_dict(orient="records"),
        "bootstrap_error_penalty": bootstrap.to_dict(orient="records"),
        "thermal_requirement_shift": threshold_shift_summary(thresholds),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "summary.json"
    output_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
