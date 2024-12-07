import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
import matplotlib.pyplot as plt

def visualize_model_performance(model_results):
    """
    Visualize model performance using a bar chart and save the output as an image.
    """
    plt.figure(figsize=(10, 6))
    plt.bar(model_results.keys(), model_results.values())
    plt.title('Model Performance Comparison')
    plt.xlabel('Models')
    plt.ylabel('Mean Absolute Error')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('model_performance.png')  # Save the plot as a PNG file
    print("Model performance plot saved as 'model_performance.png'")
