import pandas as pd

from experiments.rolling_origin import generate_rolling_origins


def frame():
    dates = pd.date_range("1990-01-01", "2016-12-01", freq="MS")
    return pd.DataFrame({
        "date": dates, "site": "a", "feature": 1.0,
        "spei_3": 0.0, "spei_6": 0.0, "spei_12": 0.0,
    })


def test_origin_generation_is_deterministic_and_aligned():
    first, first_excluded = generate_rolling_origins(frame(), origins=["2012-10-01"], features=["feature"])
    second, second_excluded = generate_rolling_origins(frame(), origins=["2012-10-01"], features=["feature"])
    assert first.equals(second)
    assert first_excluded.equals(second_excluded)
    assert len(first) == 12
    assert first["input_end"].eq("2012-10").all()
    assert first["target_date"].min() == "2012-11"


def test_future_final_evaluation_and_missing_targets_are_excluded():
    data = frame()
    data.loc[data.date.eq(pd.Timestamp("2016-08-01")), "spei_12"] = None
    rows, exclusions = generate_rolling_origins(data, origins=["2015-12-01", "2016-06-01"])
    assert not rows[(rows.origin == "2016-06") & (rows.scale == "spei_12")].any().any()
    assert (exclusions.reason == "missing SPEI target").any()
