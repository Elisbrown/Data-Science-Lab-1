def prepare_features(home_data):
    """
    Prepare features and target variable for modeling.
    """
    y = home_data['SalePrice']
    feature_columns = ['Lot Area', 'Year Built', '1st Flr SF', '2nd Flr SF', 'Full Bath', 'Bedroom AbvGr', 'TotRms AbvGrd']
    X = home_data[feature_columns]
    return X, y
