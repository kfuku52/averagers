import averagers


def main():
    start_date = "2020-06-01"
    end_date = "2020-06-03"
    lat = 35.681
    lon = 139.767
    timezone = 9

    weather = averagers.fetch_power_daily_temperature(
        start_date=start_date,
        end_date=end_date,
        lat=lat,
        lon=lon,
    )

    photoperiod = averagers.get_photoperiod(
        start_date=start_date,
        end_date=end_date,
        lat=lat,
        lon=lon,
        timezone=timezone,
    )
    weather = weather.join(photoperiod[["Sunset_nondimensional"]])

    weather["Ave_sim"] = averagers.get_average_temperature(
        weather,
        params={"CD": 0.5, "CN": 0.5},
        method="DH2006",
    )

    print(weather[["Min", "Max", "Min_next", "Ave_sim"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
