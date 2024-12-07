from src.data_processing import load_and_explore_data
from src.feature_engineering import prepare_features
from src.model_training import train_and_evaluate_models, display_validation_predictions
from src.visualization import visualize_model_performance

def main():
    # File path for dataset
    filepath = "data/iowaHousing.csv"

    # Step 1: Load and explore the data
    home_data = load_and_explore_data(filepath)

    # Step 2: Prepare features and target variable
    X, y = prepare_features(home_data)

    # Step 3: Train and evaluate models
    model_results, rf_model, X_valid, y_valid = train_and_evaluate_models(X, y)

    # Step 4: Display validation predictions
    display_validation_predictions(rf_model, X_valid, y_valid)

    # Step 5: Visualize model performance
    visualize_model_performance(model_results)

if __name__ == "__main__":
    main()
