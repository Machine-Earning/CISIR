import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report

from modules.training.ts_modeling import plot_confusion_matrix


def main():
    """
    Test function to demonstrate confusion matrix plotting with dummy data
    """
    # Generate dummy data
    np.random.seed(42)
    n_samples = 1000
    
    # Generate random true and predicted classes (0, 1, 2)
    y_true = np.random.randint(0, 3, n_samples)
    y_pred = np.random.randint(0, 3, n_samples)
    
    # Class names for the plot
    class_names = ['minus', 'zero', 'plus']

    # Create and save confusion matrix plot
    cm_fig = plot_confusion_matrix(
        y_pred,  # Predicted values on y-axis
        y_true,  # Actual values on x-axis
        class_names=class_names,
        title="Test Confusion Matrix",
        xlabel="Actual",
        ylabel="Predicted"
    )

    # Calculate accuracy
    accuracy = accuracy_score(y_true, y_pred)

    # Get detailed classification report
    report = classification_report(y_true, y_pred)

    print(f"\nAccuracy: {accuracy}")
    print("\nClassification Report:")
    print(report)

    # Save the plot
    plt.savefig('confusion_matrix.png')

    # Close figure to free memory
    plt.close(cm_fig)


if __name__ == '__main__':
    main()
