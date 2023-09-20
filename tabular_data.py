
# %%
import os
from pathlib import Path
import pandas as pd

root_dir = os.getcwd()
csv_path = Path(root_dir, 'airbnb-property-listings', 'tabular_data', 'listing.csv')
if csv_path.is_file():
    listings_df = pd.read_csv(csv_path)
else:
    print(f'File {csv_path} does not exist')


def remove_rows_with_missing_ratings(df: pd.DataFrame) -> pd.DataFrame:
    return df.dropna(subset=['Cleanliness_rating', 'Accuracy_rating', 'Communication_rating', 'Location_rating', 'Check-in_rating', 'Value_rating'], inplace=True)