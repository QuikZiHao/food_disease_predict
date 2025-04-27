import pandas as pd


def clear_data(df:pd.DataFrame, key:str, year:int)-> pd.DataFrame:
    df[key] = pd.to_datetime(df[key])
    df_year = df[df[key].dt.year == year]
    df = df.drop(df_year.index)
    return df

