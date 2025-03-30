import pandas as pd
from typing import Dict


def get_province(df:pd.DataFrame, province:str)-> pd.DataFrame:
    return df[df["省"] == province]

def get_hash_by_date(df:pd.DataFrame) -> Dict:
    hash_map = df.groupby(["日期", "市"])["降水量"].sum().unstack().T.to_dict()
    return hash_map

def get_waterdrop(row, weather_dict)->float:
    date = row["发病日期"]
    city = row["现在住址地市"]
    return weather_dict.get(date, {}).get(city, 0)  # Default to 0 if not found