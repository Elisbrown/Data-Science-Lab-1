
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

def train_and_evaluate_models(X, y):
    """
    Train models and evaluate their performance.
    Also return the Random Forest model and validation data for further use.
    """
    # Split the data
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, random_state=1)

    # Track model results
    model_results = {}

    # Decision Tree Model
    dt_model = DecisionTreeRegressor(max_leaf_nodes=100, random_state=1)
    dt_model.fit(X_train, y_train)
    dt_mae = mean_absolute_error(y_valid, dt_model.predict(X_valid))
    model_results['DecisionTree'] = dt_mae

    # Random Forest Model
    rf_model = RandomForestRegressor(random_state=1)
    rf_model.fit(X_train, y_train)
    rf_mae = mean_absolute_error(y_valid, rf_model.predict(X_valid))
    model_results['RandomForest'] = rf_mae

    # Return results, Random Forest model, and validation datasets
    return model_results, rf_model, X_valid, y_valid


def display_validation_predictions(model, X_valid, y_valid):
    """
    Display predicted and actual prices for validation data.
    """
    # Make predictions for the validation set
    predictions = model.predict(X_valid)[:5]  # Get the first 5 predictions
    actual_values = y_valid[:5].to_numpy()   # Get the first 5 actual values

    # Print results
    print("\nValidation Output:")
    print(f"Predicted Prices: {list(map(int, predictions))}")
    print(f"Actual Prices:    {list(map(int, actual_values))}")
