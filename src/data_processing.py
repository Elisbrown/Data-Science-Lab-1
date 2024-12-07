import pandas as pd
import datetime

def load_and_explore_data(filepath):
    """
    Load housing data and perform initial exploratory data analysis.
    """
    home_data = pd.read_csv(filepath)
    print("Dataset Dimensions:", home_data.shape)
    print("\nDataset Summary Statistics:\n", home_data.describe())

    avg_lot_size = round(home_data['Lot Area'].mean())
    current_year = datetime.datetime.now().year
    newest_home_age = current_year - home_data['Year Built'].max()

    print(f"Average Lot Size: {avg_lot_size} sq ft")
    print(f"Age of Newest Home: {newest_home_age} years")

    return home_data
