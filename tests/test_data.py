import json
import urllib.error
import urllib.parse

import pandas
import pytest

import averagers


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


def test_fetch_power_daily_temperature_parses_power_response():
    payload = {
        "properties": {
            "parameter": {
                "T2M_MIN": {
                    "20200601": 15.5,
                    "20200602": 16.2,
                    "20200603": 17.1,
                },
                "T2M_MAX": {
                    "20200601": 25.4,
                    "20200602": 26.1,
                    "20200603": 26.8,
                },
                "T2M": {
                    "20200601": 20.0,
                    "20200602": 21.1,
                    "20200603": 22.0,
                },
            }
        }
    }
    captured = {}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        return FakeResponse(payload)

    df = averagers.fetch_power_daily_temperature(
        start_date="2020-06-01",
        end_date="2020-06-02",
        lat=35.681,
        lon=139.767,
        timeout=7,
        urlopen=fake_urlopen,
    )

    query = urllib.parse.parse_qs(urllib.parse.urlparse(captured["url"]).query)
    assert captured["timeout"] == 7
    assert query["parameters"] == ["T2M_MIN,T2M_MAX,T2M"]
    assert query["start"] == ["20200601"]
    assert query["end"] == ["20200603"]
    assert query["format"] == ["JSON"]

    assert list(df.columns) == ["Date", "Year", "Month", "Day", "Min", "Max", "Ave", "Min_next"]
    assert list(df["Date"]) == [pandas.Timestamp("2020-06-01"), pandas.Timestamp("2020-06-02")]
    assert list(df["Min_next"]) == [16.2, 17.1]
    assert df.attrs["source"] == "NASA POWER Daily API"


def test_fetch_power_daily_temperature_can_add_previous_max():
    payload = {
        "properties": {
            "parameter": {
                "T2M_MIN": {
                    "20200531": 14.0,
                    "20200601": 15.5,
                    "20200602": 16.2,
                    "20200603": 17.1,
                },
                "T2M_MAX": {
                    "20200531": 24.0,
                    "20200601": 25.4,
                    "20200602": 26.1,
                    "20200603": 26.8,
                },
                "T2M": {
                    "20200531": 19.0,
                    "20200601": 20.0,
                    "20200602": 21.1,
                    "20200603": 22.0,
                },
            }
        }
    }
    captured = {}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        return FakeResponse(payload)

    df = averagers.fetch_power_daily_temperature(
        start_date="2020-06-01",
        end_date="2020-06-02",
        lat=35.681,
        lon=139.767,
        add_max_prev=True,
        urlopen=fake_urlopen,
    )

    query = urllib.parse.parse_qs(urllib.parse.urlparse(captured["url"]).query)
    assert query["start"] == ["20200531"]
    assert query["end"] == ["20200603"]
    assert list(df["Max_prev"]) == [24.0, 25.4]
    assert list(df["Min_next"]) == [16.2, 17.1]


def test_fetch_power_daily_temperature_validates_inputs():
    with pytest.raises(ValueError, match="start_date"):
        averagers.fetch_power_daily_temperature("2020-06-02", "2020-06-01", 35, 139)

    with pytest.raises(ValueError, match="lat"):
        averagers.fetch_power_daily_temperature("2020-06-01", "2020-06-02", 91, 139)


def test_fetch_power_daily_temperature_retries_failed_request():
    payload = {
        "properties": {
            "parameter": {
                "T2M_MIN": {"20200601": 15.5, "20200602": 16.2},
                "T2M_MAX": {"20200601": 25.4, "20200602": 26.1},
                "T2M": {"20200601": 20.0, "20200602": 21.1},
            }
        }
    }
    calls = []

    def flaky_urlopen(request, timeout):
        calls.append(request.full_url)
        if len(calls) == 1:
            raise urllib.error.URLError("temporary failure")
        return FakeResponse(payload)

    df = averagers.fetch_power_daily_temperature(
        start_date="2020-06-01",
        end_date="2020-06-01",
        lat=35.681,
        lon=139.767,
        retries=1,
        retry_delay=0,
        urlopen=flaky_urlopen,
    )

    assert len(calls) == 2
    assert list(df["Ave"]) == [20.0]


def test_fetch_power_daily_temperature_uses_cache(tmp_path):
    payload = {
        "properties": {
            "parameter": {
                "T2M_MIN": {"20200601": 15.5, "20200602": 16.2},
                "T2M_MAX": {"20200601": 25.4, "20200602": 26.1},
                "T2M": {"20200601": 20.0, "20200602": 21.1},
            }
        }
    }
    calls = []

    def fake_urlopen(request, timeout):
        calls.append(request.full_url)
        return FakeResponse(payload)

    first = averagers.fetch_power_daily_temperature(
        start_date="2020-06-01",
        end_date="2020-06-01",
        lat=35.681,
        lon=139.767,
        cache_dir=tmp_path,
        urlopen=fake_urlopen,
    )
    second = averagers.fetch_power_daily_temperature(
        start_date="2020-06-01",
        end_date="2020-06-01",
        lat=35.681,
        lon=139.767,
        cache_dir=tmp_path,
        urlopen=lambda request, timeout: pytest.fail("cache should avoid network"),
    )

    assert len(calls) == 1
    assert first.attrs["cache_status"] == "miss"
    assert second.attrs["cache_status"] == "hit"
    assert first.attrs["cache_path"] == second.attrs["cache_path"]
    assert list(second["Ave"]) == [20.0]
