import tensorflow as tf
from tensorflow.keras.utils import plot_model
from graphviz import Digraph
import os


def visualize_model(model, save_dir="model_visualizations"):
    """
    Visualize a Keras model using reliable visualization methods.

    Args:
        model: Keras model instance
        save_dir: Directory to save visualization files
    """
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # 1. Basic Keras plot_model visualization
    plot_model(
        model,
        to_file=f"{save_dir}/model_architecture.png",
        show_shapes=True,
        show_layer_names=True,
        show_layer_activations=True,
        expand_nested=True,
        dpi=300,  # Higher DPI for better quality
        rankdir='TB'  # Top to bottom layout
    )

    # 2. Print model summary to text file
    with open(f"{save_dir}/model_summary.txt", 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))

    # 3. Create detailed Graphviz visualization
    dot = Digraph(comment='Model Architecture')
    dot.attr(rankdir='TB')

    # Define color scheme
    colors = {
        'input': '#FFB6C1',  # Light pink
        'dense': '#98FB98',  # Pale green
        'dropout': '#87CEFA',  # Light sky blue
        'norm': '#DDA0DD',  # Plum
        'activation': '#FFA07A',  # Light salmon
        'add': '#B0C4DE',  # Light steel blue
        'output': '#FFD700'  # Gold
    }

    # Add nodes for each layer
    layers = model.layers
    for i, layer in enumerate(layers):
        # Create detailed label
        label = f"{layer.__class__.__name__}\n"

        # Add shape information
        if hasattr(layer, 'input_shape'):
            label += f"In: {layer.input_shape}\n"
        if hasattr(layer, 'output_shape'):
            label += f"Out: {layer.output_shape}\n"

        # Add layer-specific details
        if hasattr(layer, 'units'):
            label += f"Units: {layer.units}\n"
        if hasattr(layer, 'rate'):
            label += f"Rate: {layer.rate}\n"
        if isinstance(layer, (tf.keras.layers.BatchNormalization, tf.keras.layers.LayerNormalization)):
            label += f"Norm\n"

        # Style node based on layer type
        if isinstance(layer, tf.keras.layers.InputLayer):
            style = {'fillcolor': colors['input'], 'style': 'filled', 'shape': 'ellipse'}
        elif isinstance(layer, tf.keras.layers.Dense):
            if i == len(layers) - 1:  # Last dense layer
                style = {'fillcolor': colors['output'], 'style': 'filled', 'shape': 'box'}
            else:
                style = {'fillcolor': colors['dense'], 'style': 'filled', 'shape': 'box'}
        elif isinstance(layer, tf.keras.layers.Dropout):
            style = {'fillcolor': colors['dropout'], 'style': 'filled', 'shape': 'box'}
        elif isinstance(layer, (tf.keras.layers.BatchNormalization, tf.keras.layers.LayerNormalization)):
            style = {'fillcolor': colors['norm'], 'style': 'filled', 'shape': 'box'}
        elif isinstance(layer, tf.keras.layers.Add):
            style = {'fillcolor': colors['add'], 'style': 'filled', 'shape': 'diamond'}
        else:
            style = {'fillcolor': colors['activation'], 'style': 'filled', 'shape': 'box'}

        # Add node
        dot.node(f"layer_{i}", label, **style)

        # Add edges
        if i > 0:
            if isinstance(layer, tf.keras.layers.Add):
                # Regular connection from previous layer
                dot.edge(f"layer_{i - 1}", f"layer_{i}")
                # Find and add residual connection
                for j in range(i - 2, -1, -1):
                    if isinstance(layers[j], tf.keras.layers.Dense):
                        dot.edge(f"layer_{j}", f"layer_{i}", style='dashed', color='blue')
                        break
            else:
                dot.edge(f"layer_{i - 1}", f"layer_{i}")

    # Save the Graphviz visualization
    dot.render(f"{save_dir}/model_detailed", format='png', cleanup=True)

    print(f"\nVisualizations saved in {save_dir}:")
    print("1. model_architecture.png - Basic Keras visualization")
    print("2. model_summary.txt - Detailed model summary")
    print("3. model_detailed.png - Detailed Graphviz visualization")
    print("\nVisualization features:")
    print("- Node colors indicate layer types")
    print("- Dashed blue lines show residual connections")
    print("- Node shapes vary by layer type")
    print("- Labels include shape and parameter information")
