import hashlib
import json
from pathlib import Path
import time
import urllib.error
import urllib.parse
import urllib.request

import pandas


POWER_DAILY_POINT_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
POWER_TEMPERATURE_PARAMETERS = ("T2M_MIN", "T2M_MAX", "T2M")
POWER_MISSING_VALUE = -999


def _format_power_date(value):
    date = pandas.Timestamp(value)
    if pandas.isna(date):
        raise ValueError("Date values must not be missing")
    return date.strftime("%Y%m%d")


def _validate_coordinates(lat, lon):
    if not -90 <= lat <= 90:
        raise ValueError("lat must be between -90 and 90")
    if not -180 <= lon <= 180:
        raise ValueError("lon must be between -180 and 180")


def _build_power_daily_url(start_date, end_date, lat, lon, community, time_standard):
    query = {
        "parameters": ",".join(POWER_TEMPERATURE_PARAMETERS),
        "community": community,
        "longitude": lon,
        "latitude": lat,
        "start": _format_power_date(start_date),
        "end": _format_power_date(end_date),
        "format": "JSON",
        "time-standard": time_standard,
    }
    return f"{POWER_DAILY_POINT_URL}?{urllib.parse.urlencode(query)}"


def _open_json_url(url, timeout, urlopen, retries=2, retry_delay=1.0, sleep=None):
    opener = urlopen or urllib.request.urlopen
    request = urllib.request.Request(url, headers={"User-Agent": "averagers"})
    if retries < 0:
        raise ValueError("retries must be >= 0")
    sleep = time.sleep if sleep is None else sleep
    last_error = None
    for attempt in range(retries + 1):
        try:
            with opener(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except (TimeoutError, urllib.error.URLError) as exc:
            last_error = exc
            if attempt < retries and retry_delay > 0:
                sleep(retry_delay)
    raise RuntimeError(f"NASA POWER request failed: {last_error}") from last_error


def _cache_path_for_url(cache_dir, url):
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()
    return Path(cache_dir).expanduser() / f"{digest}.json"


def _read_cached_json(cache_path):
    try:
        return json.loads(cache_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except json.JSONDecodeError:
        return None


def _write_cached_json(cache_path, payload):
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _temperature_frame_from_power_response(payload):
    try:
        parameters = payload["properties"]["parameter"]
    except KeyError as exc:
        raise ValueError("Unexpected NASA POWER response: missing parameter data") from exc

    missing_parameters = [name for name in POWER_TEMPERATURE_PARAMETERS if name not in parameters]
    if missing_parameters:
        raise ValueError(f"NASA POWER response is missing: {', '.join(missing_parameters)}")

    records = []
    for date_key in sorted(parameters["T2M_MIN"]):
        date = pandas.to_datetime(date_key, format="%Y%m%d")
        records.append(
            {
                "Date": date,
                "Year": date.year,
                "Month": date.month,
                "Day": date.day,
                "Min": parameters["T2M_MIN"].get(date_key),
                "Max": parameters["T2M_MAX"].get(date_key),
                "Ave": parameters["T2M"].get(date_key),
            }
        )

    df = pandas.DataFrame.from_records(records)
    if df.empty:
        raise ValueError("NASA POWER response did not contain any daily records")

    for column in ["Min", "Max", "Ave"]:
        df[column] = pandas.to_numeric(df[column], errors="coerce")
        df.loc[df[column] == POWER_MISSING_VALUE, column] = pandas.NA

    return df


def fetch_power_daily_temperature(
    start_date,
    end_date,
    lat,
    lon,
    community="AG",
    time_standard="LST",
    add_min_next=True,
    add_max_prev=False,
    timeout=30,
    retries=2,
    retry_delay=1.0,
    cache_dir=None,
    force_refresh=False,
    urlopen=None,
    sleep=None,
):
    start = pandas.Timestamp(start_date)
    end = pandas.Timestamp(end_date)
    if pandas.isna(start) or pandas.isna(end):
        raise ValueError("start_date and end_date must be valid dates")
    if start > end:
        raise ValueError("start_date must be earlier than or equal to end_date")

    _validate_coordinates(lat, lon)

    query_start = start - pandas.Timedelta(days=1) if add_max_prev else start
    query_end = end + pandas.Timedelta(days=1) if add_min_next else end
    url = _build_power_daily_url(
        start_date=query_start,
        end_date=query_end,
        lat=lat,
        lon=lon,
        community=community,
        time_standard=time_standard,
    )
    cache_path = None
    cache_status = None
    if cache_dir is not None:
        cache_path = _cache_path_for_url(cache_dir, url)
        if not force_refresh:
            payload = _read_cached_json(cache_path)
            if payload is not None:
                cache_status = "hit"
            else:
                cache_status = "miss"
        else:
            payload = None
            cache_status = "refresh"
    else:
        payload = None

    if payload is None:
        payload = _open_json_url(
            url,
            timeout=timeout,
            urlopen=urlopen,
            retries=retries,
            retry_delay=retry_delay,
            sleep=sleep,
        )
        if cache_path is not None:
            _write_cached_json(cache_path, payload)

    df = _temperature_frame_from_power_response(payload)

    if add_min_next:
        df["Min_next"] = df["Min"].shift(-1)
    if add_max_prev:
        df["Max_prev"] = df["Max"].shift(1)

    requested = df.loc[df["Date"].between(start, end)].copy()
    requested.attrs["source"] = "NASA POWER Daily API"
    requested.attrs["url"] = url
    if cache_path is not None:
        requested.attrs["cache_path"] = str(cache_path)
        requested.attrs["cache_status"] = cache_status
    return requested.reset_index(drop=True)
