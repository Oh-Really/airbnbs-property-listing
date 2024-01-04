
# %%
import os
import sys
from pathlib import Path
import pandas as pd
import ast
import numpy as np

print(sys.path)

def load_data(filename : str, index_col=False) -> pd.DataFrame:
    root_dir = os.getcwd()
    csv_path = Path(root_dir, 'airbnb-property-listings', 'tabular_data', filename)
    if csv_path.is_file():
        df = pd.read_csv(csv_path, index_col=index_col)
    else:
        print(f'File {csv_path} does not exist')
    return df

def remove_rows_with_missing_ratings(df: pd.DataFrame) -> pd.DataFrame:
    return df.dropna(subset=['Cleanliness_rating', 'Accuracy_rating', 'Communication_rating', 'Location_rating', 'Check-in_rating', 'Value_rating'])

def remove_whitespace(list: list) -> list:
    for string in list:
        #Checks to see if string is empty
        if not string.strip():
            try:
                list.remove(string)
            except:
                pass
    return list

def combine_description_strings(df: pd.DataFrame) -> pd.DataFrame:
    '''Cleans up Description column of the dataframe'''
    null_desc = df['Description'].isna()
    df = df[~null_desc]
    df.loc[:, 'Description'] = df['Description'].apply(convert_description_strings)
    return df

def convert_description_strings(desc):
    '''Converts description values to list, removes first item and any whitespace, and concatenates the string'''
    try:
        desc = ast.literal_eval(desc) # convert to list
    except SyntaxError:
        print('Skipping conversion to string; already a string')
    try:
        desc.remove(desc[0])
    except AttributeError:
        pass

    desc = remove_whitespace(desc)
    desc = ' '.join(desc)
    return desc

def set_default_feature_values(df: pd.DataFrame) -> pd.DataFrame:
    return df.fillna(value={'guests': 1, 'beds': 1, 'bathrooms': 1, 'bedrooms': 1})


def clean_tabular_data(df: pd.DataFrame) -> pd.DataFrame:
    df = remove_rows_with_missing_ratings(df)
    df = combine_description_strings(df)
    df = set_default_feature_values(df)
    return df


def load_airbnb(df, label : str) -> tuple[pd.DataFrame, pd.Series]:
    if label == 'Price_Night':
        features = df.select_dtypes(include=[np.number])
        labels = df.pop(label)
        return features, labels
    else:
        features = df.select_dtypes(include=[np.number])
        labels = df[label]
        return features, labels



if __name__ == "__main__":
    df = load_data('listing.csv')
    df = clean_tabular_data(df)
    output_filepath = Path(os.getcwd(), 'airbnb-property-listings', 'tabular_data', 'clean_tabular_data.csv')
    df.to_csv(output_filepath)
# %%
