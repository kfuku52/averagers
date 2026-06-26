import json
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


def test_fetch_power_daily_temperature_validates_inputs():
    with pytest.raises(ValueError, match="start_date"):
        averagers.fetch_power_daily_temperature("2020-06-02", "2020-06-01", 35, 139)

    with pytest.raises(ValueError, match="lat"):
        averagers.fetch_power_daily_temperature("2020-06-01", "2020-06-02", 91, 139)
