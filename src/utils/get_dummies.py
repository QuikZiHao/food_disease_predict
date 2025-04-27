import pandas as pd
from typing import List, Dict


def get_dummies(columns_to_encode:List, df:pd.DataFrame)->pd.DataFrame:
    df_encoded = pd.get_dummies(df, columns=columns_to_encode,drop_first=True)
    return df_encoded


def get_bool(df:pd.DataFrame, key:str, mapping_address:Dict)->pd.DataFrame:
    df[key] = df[key].map(mapping_address)
    return df

