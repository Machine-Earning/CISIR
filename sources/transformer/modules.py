from typing import List, Optional, Tuple

import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, Multiply


class AttentionBlock(Layer):
    """
    A custom attention layer that applies a series of dense layers followed by an output layer.

    This layer can be used to compute attention scores or feature transformations
    in attention-based neural network architectures.

    Attributes:
        input_dim (int): The dimensionality of the input.
        hidden_units (List[int]): A list of integers, where each integer is the number of units
                                  in the corresponding dense layer.
        output_dim (int): The dimensionality of the output.
        activation (str): The activation function to use in the hidden layers.
        dense_layers (List[Dense]): A list of Dense layers used in the attention mechanism.
        output_layer (Dense): The final Dense layer that produces the output.
    """

    def __init__(self, input_dim: int, hidden_units: List[int], output_dim: int, activation: str = 'tanh'):
        """
        Initialize the AttentionBlock.

        Args:
            input_dim (int): The dimensionality of the input.
            hidden_units (List[int]): A list of integers, where each integer is the number of units
                                      in the corresponding dense layer.
            output_dim (int): The dimensionality of the output.
            activation (str, optional): The activation function to use in the hidden layers.
                                        Defaults to 'tanh'.
        """
        super(AttentionBlock, self).__init__()
        self.input_dim = input_dim
        self.hidden_units = hidden_units
        self.output_dim = output_dim
        self.activation = activation
        self.dense_layers: List[Dense] = []

        # Create dense layers based on hidden_units list
        for units in hidden_units:
            self.dense_layers.append(Dense(units, activation=self.activation))
        self.output_layer = Dense(output_dim)

    def build(self, input_shape: tf.TensorShape) -> None:
        """
        Build the layer. This method is called automatically by Keras when the layer is first used.

        In this implementation, the layers are created in __init__, so this method is empty.

        Args:
            input_shape (tf.TensorShape): The shape of the input tensor.
        """
        # The layers are created in __init__, so no need to create them here.
        pass

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        Perform the forward pass of the attention layer.

        This method applies a series of dense layers to the input, followed by a final output layer.

        Args:
            inputs (tf.Tensor): The input tensor.

        Returns:
            tf.Tensor: The output tensor after applying the attention mechanism.
        """
        x = inputs
        # Apply each dense layer in sequence
        for dense in self.dense_layers:
            x = dense(x)
        # Apply the final output layer
        return self.output_layer(x)


class BlockBase(Layer):
    """
    Base class for attention blocks with common methods.
    """

    def __init__(self,
                 attn_hidden_units: Optional[List[int]] = None,
                 activation: str = 'tanh',
                 output_activation: Optional[str] = None):
        super(BlockBase, self).__init__()
        self.attn_hidden_units = attn_hidden_units or [3]
        self.activation = activation
        self.output_activation = output_activation
        self.attention_scores = None
        self.attention_block = None

    def get_attention_scores(self) -> tf.Tensor:
        """
        Retrieve the most recently computed attention scores.

        Returns:
            tf.Tensor: The attention scores from the last forward pass.
                       Returns None if the layer hasn't been called yet.
        """
        return self.attention_scores

    def get_config(self) -> dict:
        """
        Get the config of the layer.

        Returns:
            dict: Configuration of the layer.
        """
        config = super().get_config()
        config.update({
            'attn_hidden_units': self.attn_hidden_units,
            'activation': self.activation,
            'output_activation': self.output_activation
        })
        return config


class BlockT1(BlockBase):
    """
    A custom layer that applies attention mechanism followed by a dense layer.

    This block consists of an attention layer that computes attention scores,
    applies these scores to the input via element-wise multiplication,
    and then passes the result through a dense layer.

    y = w0 + w1 * a1 * x1 + w2 * a2 * x2 + ... + wn * an * xn

    Attributes:
        attn_hidden_units (List[int]): A list of integers representing the number of units
                                       in each hidden layer of the attention mechanism.
        activation (str): The activation function to use in the attention layer.
        output_activation (Optional[str]): The activation function to use in the final dense layer.
        attention_block (AttentionBlock): The layer used to compute attention scores.
        dense_layer (Dense): The final dense layer that produces the output.
        attention_scores (tf.Tensor): The most recently computed attention scores.
    """

    def __init__(self,
                 attn_hidden_units: Optional[List[int]] = None,
                 activation: str = 'tanh',
                 output_activation: Optional[str] = None):
        super().__init__(attn_hidden_units, activation, output_activation)
        self.dense_layer = None

    def build(self, input_shape: tf.TensorShape) -> None:
        """
        Build the layer. This method is called automatically by Keras when the layer is first used.

        Args:
            input_shape (tf.TensorShape): The shape of the input tensor.
        """
        # Create the attention layer
        self.attention_block = AttentionBlock(
            input_dim=input_shape[-1],
            hidden_units=self.attn_hidden_units,
            output_dim=input_shape[-1],
            activation=self.activation
        )
        # Create the final dense layer
        self.dense_layer = Dense(1, activation=self.output_activation)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        Perform the forward pass of the BlockT1 layer.

        This method applies the attention mechanism to the inputs,
        then passes the weighted inputs through a dense layer.

        Args:
            inputs (tf.Tensor): The input tensor.

        Returns:
            tf.Tensor: The output tensor after applying attention and the dense layer.
        """
        # Compute attention scores
        self.attention_scores = self.attention_block(inputs)

        # Apply attention scores to inputs via element-wise multiplication
        weighted_inputs = Multiply()([inputs, self.attention_scores])

        # Pass the weighted inputs through the final dense layer
        output = self.dense_layer(weighted_inputs)

        return output

    def get_dense_weights(self) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Retrieve the weights and bias of the output dense layer.

        Returns:
            Tuple[tf.Tensor, tf.Tensor]: A tuple containing the weights and bias of the dense layer.
                                         The first element is the weight tensor, and the second
                                         is the bias tensor.
        """
        return self.dense_layer.get_weights()


class BlockT2(BlockBase):
    """
    A custom layer that applies attention mechanism and combines the attention weights with input features.

    This block consists of an attention layer that computes attention scores,
    applies these scores to the input via element-wise multiplication,
    and then directly combines the weighted inputs to produce the output.

    y = a1 * x1 + a2 * x2 + ... + an * xn

    Attributes:
        attn_hidden_units (List[int]): A list of integers representing the number of units
                                       in each hidden layer of the attention mechanism.
        activation (str): The activation function to use in the attention layer.
        output_activation (Optional[str]): The activation function to use in the final output.
        attention_block (AttentionBlock): The layer used to compute attention scores.
        attention_scores (tf.Tensor): The most recently computed attention scores.
    """

    def build(self, input_shape: tf.TensorShape) -> None:
        """
        Build the layer. This method is called automatically by Keras when the layer is first used.

        Args:
            input_shape (tf.TensorShape): The shape of the input tensor.
        """
        # Create the attention layer
        self.attention_block = AttentionBlock(
            input_dim=input_shape[-1],
            hidden_units=self.attn_hidden_units,
            output_dim=input_shape[-1],
            activation=self.activation
        )

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        Perform the forward pass of the BlockT2 layer.

        This method applies the attention mechanism to the inputs,
        then combines the weighted inputs to produce the output.

        Args:
            inputs (tf.Tensor): The input tensor.

        Returns:
            tf.Tensor: The output tensor after applying attention and combining the inputs.
        """
        # Compute attention scores
        self.attention_scores = self.attention_block(inputs)

        # Apply attention scores to inputs via element-wise multiplication
        weighted_inputs = Multiply()([inputs, self.attention_scores])

        # Sum the weighted inputs to get the final output
        output = tf.reduce_sum(weighted_inputs, axis=-1, keepdims=True)

        if self.output_activation:
            output = tf.keras.activations.get(self.output_activation)(output)

        return output


class BlockT3(BlockBase):
    """
    A custom layer that applies an attention mechanism and combines attention scores with input features.

    This block computes y = w0 + a(x) + (w1 + a1(x)) * x1 + (w2 + a2(x)) * x2 + ...

    Attributes:
        attn_hidden_units (List[int]): A list of integers representing the number of units
                                       in each hidden layer of the attention mechanism.
        activation (str): The activation function to use in the attention block.
        output_activation (Optional[str]): The activation function to use in the final output.
        attention_block (AttentionBlock): The block used to compute attention scores.
        dense_layers (List[Dense]): The dense layers used to compute weights.
    """

    def __init__(self,
                 attn_hidden_units: Optional[List[int]] = None,
                 activation: str = 'tanh',
                 output_activation: Optional[str] = None):
        super().__init__(attn_hidden_units, activation, output_activation)
        self.dense_layers = None

    def build(self, input_shape: tf.TensorShape) -> None:
        """
        Build the layer. This method is called automatically by Keras when the layer is first used.

        Args:
            input_shape (tf.TensorShape): The shape of the input tensor.
        """
        # Create the attention block
        self.attention_block = AttentionBlock(
            input_dim=input_shape[-1],
            hidden_units=self.attn_hidden_units,
            output_dim=input_shape[-1],
            activation=self.activation
        )
        # Create the dense layers for weights
        self.dense_layers = [Dense(1) for _ in range(input_shape[-1] + 1)]

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        Perform the forward pass of the BlockT3 layer.

        This method applies the attention mechanism to the inputs,
        then combines the weighted inputs to produce the output.

        Args:
            inputs (tf.Tensor): The input tensor.

        Returns:
            tf.Tensor: The output tensor after applying attention and combining the inputs.
        """
        # Compute attention scores
        attention_scores = self.attention_block(inputs)

        # Initialize the output with w0
        output = self.dense_layers[0](inputs)

        # Iterate over each input feature and apply the attention scores
        for i in range(inputs.shape[-1]):
            attn_score = attention_scores[:, i:i + 1]
            x_i = inputs[:, i:i + 1]
            weight = self.dense_layers[i + 1](x_i)
            output += (weight + attn_score) * x_i

        if self.output_activation:
            output = tf.keras.activations.get(self.output_activation)(output)

        return output

    def get_config(self) -> dict:
        """
        Get the config of the layer.

        Returns:
            dict: Configuration of the layer.
        """
        config = super().get_config()
        config.update({
            'attn_hidden_units': self.attn_hidden_units,
            'activation': self.activation,
            'output_activation': self.output_activation
        })
        return config
