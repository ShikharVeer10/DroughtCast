import argparse

import pandas as pd

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", nargs="?", default="data/monthly_sites.csv")
    args = parser.parse_args()
    frame = pd.read_csv(args.path, parse_dates=["date"])
    print(
        frame.groupby("site").agg(
            first_date=("date", "min"),
            last_date=("date", "max"),
            months=("date", "size"),
        )
    )
    print("month-13 rows:", int((frame.date.dt.month == 13).sum()))
    print(
        "target missingness:",
        frame[["spei_3", "spei_6", "spei_12"]].isna().mean().to_dict(),
    )
