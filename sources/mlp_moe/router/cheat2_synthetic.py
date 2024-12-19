from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import wandb
from sklearn.metrics import classification_report, accuracy_score
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow_addons.optimizers import AdamW
from wandb.integration.keras import WandbCallback

from modules.training.phase_manager import TrainingPhaseManager, IsTraining
from modules.training.smooth_early_stopping import SmoothEarlyStopping, find_optimal_epoch_by_smoothing
from modules.training.ts_modeling import (
    create_mlp,
    plot_confusion_matrix,
    create_metrics_table,
)

def generate_quadrant_data(n_samples=1000, noise=0.1):
    """Generate synthetic data with 4 classes based on quadrants"""
    X = np.random.uniform(-1, 1, (n_samples, 2))
    
    # Add some noise
    X += np.random.normal(0, noise, X.shape)
    
    # Assign classes based on quadrants (0: top-right, 1: top-left, 2: bottom-left, 3: bottom-right)
    y = np.zeros(n_samples)
    y[(X[:, 0] < 0) & (X[:, 1] > 0)] = 1  # top-left
    y[(X[:, 0] < 0) & (X[:, 1] < 0)] = 2  # bottom-left
    y[(X[:, 0] > 0) & (X[:, 1] < 0)] = 3  # bottom-right
    
    # Convert to one-hot encoding
    y_onehot = np.zeros((n_samples, 4))
    y_onehot[np.arange(n_samples), y.astype(int)] = 1
    
    return X, y_onehot

def main():
    """
    Main function to run the Router model with synthetic data
    """
    # Generate synthetic data
    X_train, y_train = generate_quadrant_data(n_samples=1000)
    X_test, y_test = generate_quadrant_data(n_samples=200)

    # Model parameters
    n_features = 2  # x and y coordinates
    output_dim = 4  # 4 quadrants
    hiddens = [64, 32]  # Simplified architecture
    learning_rate = 0.001
    batch_size = 32
    epochs = 50
    
    # Initialize wandb
    experiment_name = f'synthetic_quadrant_router_{datetime.now().strftime("%Y%m%d-%H%M%S")}'
    wandb.init(project="synthetic-quadrant-router", name=experiment_name)

    pm = TrainingPhaseManager()

    # Create model
    model = create_mlp(
        input_dim=n_features,
        hiddens=hiddens,
        output_dim=output_dim,
        dropout=0.1,
        activation='relu',
        output_activation='softmax'
    )

    model.compile(
        optimizer=AdamW(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Callbacks
    early_stopping = SmoothEarlyStopping(
        monitor='val_loss',
        patience=10,
        mode='min'
    )

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6
    )

    # Train model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[
            early_stopping,
            reduce_lr,
            WandbCallback(save_model=False),
            IsTraining(pm)
        ],
        verbose=1
    )

    # Get predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Convert predictions to class labels
    y_train_pred_classes = np.argmax(y_train_pred, axis=1)
    y_test_pred_classes = np.argmax(y_test_pred, axis=1)
    y_train_true_classes = np.argmax(y_train, axis=1)
    y_test_true_classes = np.argmax(y_test, axis=1)

    # Calculate accuracies
    train_accuracy = accuracy_score(y_train_true_classes, y_train_pred_classes)
    test_accuracy = accuracy_score(y_test_true_classes, y_test_pred_classes)

    print(f"\nFinal Training Accuracy: {train_accuracy:.4f}")
    print(f"Final Test Accuracy: {test_accuracy:.4f}")

    # Plot confusion matrices
    class_names = ['Q1', 'Q2', 'Q3', 'Q4']
    
    train_cm_fig = plot_confusion_matrix(
        y_train_pred_classes,
        y_train_true_classes,
        class_names=class_names,
        title="Training Confusion Matrix"
    )

    test_cm_fig = plot_confusion_matrix(
        y_test_pred_classes,
        y_test_true_classes,
        class_names=class_names,
        title="Test Confusion Matrix"
    )

    # Plot data distribution
    plt.figure(figsize=(10, 10))
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test_pred_classes, cmap='viridis')
    plt.title('Predicted Classes Distribution')
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    plt.colorbar(label='Quadrant')
    plt.grid(True)
    plt.savefig('predicted_distribution.png')
    
    # Log results to wandb
    wandb.log({
        "train_accuracy": train_accuracy,
        "test_accuracy": test_accuracy,
        "train_confusion_matrix": wandb.Image(train_cm_fig),
        "test_confusion_matrix": wandb.Image(test_cm_fig),
        "predicted_distribution": wandb.Image('predicted_distribution.png')
    })

    wandb.finish()

if __name__ == '__main__':
    main()
