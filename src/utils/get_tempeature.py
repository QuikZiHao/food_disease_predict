import pandas as pd
from typing import Dict


def get_province(df:pd.DataFrame, province:str)-> pd.DataFrame:
    return df[df["省"] == province]

def convert_column_name(col):
    try:
        return pd.to_datetime(col, format="%Y%m%d")  # Convert if possible
    except ValueError:
        return col

def get_temperature_history(df:pd.DataFrame, df_temperature:pd.DataFrame, day:int=7)->pd.DataFrame:
    df["发病日期"] = pd.to_datetime(df["发病日期"])

    for i in range(1, day + 1):
        df[f"Temperature-before{i}day"] = pd.NA 

    for index, row in df.iterrows():
        city = row["现在住址地市"]
        date = row["发病日期"]
        
        # Find city's temperature row (avoid memory-heavy merge)
        temp_row = df_temperature[df_temperature["市"] == city]

        if not temp_row.empty:  # Only proceed if city exists
            temp_row = temp_row.iloc[0]  # Get first match
            
            for i in range(1, day + 1):
                past_date = date - pd.Timedelta(days=i)
                if past_date in temp_row.index:
                    df.at[index, f"Temperature-before{i}day"] = temp_row[past_date]
                else:
                    df.at[index, f"Temperature-before{i}day"] = None  # Ensure None for missing dates

    return df