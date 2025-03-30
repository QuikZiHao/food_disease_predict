import pandas as pd
from typing import Dict


def get_province(df:pd.DataFrame, province:str)-> pd.DataFrame:
    return df[df["省"] == province]

def get_hash_by_date(df:pd.DataFrame) -> Dict:
    hash_map = {}
    df["日期"] = pd.to_datetime(df["日期"])
    hash_map = df.groupby(["日期", "市"])["降水量"].unstack().T.to_dict()
    print(f"Total unique dates stored in hash_map: {len(hash_map)}")  # Debugging step
    return hash_map

def get_waterdrop_history(row, weather_dict, days=7):
    date = row["发病日期"]
    city = row["现在住址地市"]

    # Create a list to store past precipitation values
    past_precipitations = []

    for i in range(1, days + 1):
        past_date = date - pd.Timedelta(days=i)
        precipitation = weather_dict.get(past_date, {}).get(city, 0)  # Default to 0 if not found
        past_precipitations.append(precipitation)

    return past_precipitations