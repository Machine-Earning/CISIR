from typing import List, Optional, Tuple, Dict, Union, Any

import tensorflow as tf
from tensorflow import Tensor
from tensorflow.keras import backend as K
from tensorflow.keras import regularizers
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    BatchNormalization,
    LayerNormalization,
    Add,
    Softmax,
    Multiply,
    Layer,
    Activation
)


class AwayFrom1Regularizer(regularizers.Regularizer):
    """
    Custom regularizer that encourages weights to be away from the value 1.

    Attributes:
        strength (float): The regularization strength.
    """

    def __init__(self, strength: float = 0.01):
        self.strength = strength

    def __call__(self, x):
        return self.strength * K.sum(K.square(x - 1.0))

    def get_config(self):
        return {'strength': self.strength}


# class AttentionBlock(Layer):
#     """
#     A custom attention layer that applies a series of dense layers followed by an output layer.
#
#     This layer can be used to compute attention scores or feature transformations
#     in attention-based neural network architectures.
#
#     Attributes:
#         input_dim (int): The dimensionality of the input.
#         hidden_units (List[int]): A list of integers, where each integer is the number of units
#                                   in the corresponding dense layer.
#         output_dim (int): The dimensionality of the output.
#         activation (str): The activation function to use in the hidden layers.
#         dense_layers (List[Dense]): A list of Dense layers used in the attention mechanism.
#         output_layer (Dense): The final Dense layer that produces the output.
#     """
#
#     def __init__(self, input_dim: int, hidden_units: List[int], output_dim: int, activation: str = 'tanh'):
#         """
#         Initialize the AttentionBlock.
#
#         Args:
#             input_dim (int): The dimensionality of the input.
#             hidden_units (List[int]): A list of integers, where each integer is the number of units
#                                       in the corresponding dense layer.
#             output_dim (int): The dimensionality of the output.
#             activation (str, optional): The activation function to use in the hidden layers.
#                                         Defaults to 'tanh'.
#         """
#         super(AttentionBlock, self).__init__()
#         self.input_dim = input_dim
#         self.hidden_units = hidden_units
#         self.output_dim = output_dim
#         self.activation = activation
#         self.dense_layers: List[Dense] = []
#
#         # Create dense layers based on hidden_units list
#         for units in hidden_units:
#             self.dense_layers.append(Dense(units, activation=self.activation))
#         self.output_layer = Dense(output_dim)
#
#     def build(self, input_shape: tf.TensorShape) -> None:
#         """
#         Build the layer. This method is called automatically by Keras when the layer is first used.
#
#         In this implementation, the layers are created in __init__, so this method is empty.
#
#         Args:
#             input_shape (tf.TensorShape): The shape of the input tensor.
#         """
#         # The layers are created in __init__, so no need to create them here.
#         pass
#
#     def call(self, inputs: tf.Tensor) -> tf.Tensor:
#         """
#         Perform the forward pass of the attention layer.
#
#         This method applies a series of dense layers to the input, followed by a final output layer.
#
#         Args:
#             inputs (tf.Tensor): The input tensor.
#
#         Returns:
#             tf.Tensor: The output tensor after applying the attention mechanism.
#         """
#         x = inputs
#         # Apply each dense layer in sequence
#         for dense in self.dense_layers:
#             x = dense(x)
#         # Apply the final output layer
#         return self.output_layer(x)

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
        dropout_rate (float): The dropout rate to use in the hidden layers.
        norm (str): The type of normalization to use ('batch_norm' or 'layer_norm').
        residual (bool): Whether to use residual connections.
        skipped_layers (int): The number of layers between residual connections.
        dense_layers (List[Dense]): A list of Dense layers used in the attention mechanism.
        dropout_layers (List[Dropout]): A list of Dropout layers used in the attention mechanism.
        norm_layers (List[Layer]): A list of normalization layers used in the attention mechanism.
        output_layer (Dense): The final Dense layer that produces the output.
    """

    def __init__(
            self,
            input_dim: int,
            hidden_units: List[int],
            output_dim: int,
            activation: str = 'tanh',
            dropout_rate: float = 0.0,
            norm: str = None,
            residual: bool = False,
            skipped_layers: int = 2
    ):
        """
        Initialize the AttentionBlock.

        Args:
            input_dim (int): The dimensionality of the input.
            hidden_units (List[int]): A list of integers, where each integer is the number of units
                                      in the corresponding dense layer.
            output_dim (int): The dimensionality of the output.
            activation (str, optional): The activation function to use in the hidden layers.
                                        Defaults to 'tanh'.
            dropout_rate (float, optional): The dropout rate to use in the hidden layers. Defaults to 0.0.
            norm (str, optional): The type of normalization to use ('batch_norm' or 'layer_norm'). Defaults to None.
            residual (bool, optional): Whether to use residual connections. Defaults to False.
            skipped_layers (int, optional): The number of layers between residual connections. Defaults to 2.
        """
        super(AttentionBlock, self).__init__()
        self.input_dim = input_dim
        self.hidden_units = hidden_units
        self.output_dim = output_dim
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.norm = norm
        self.residual = residual
        self.skipped_layers = skipped_layers
        self.dense_layers: List[Dense] = []
        self.dropout_layers: List[Dropout] = []
        self.norm_layers: List[Layer] = []

        # Create dense, dropout, and normalization layers based on hidden_units list
        for i, units in enumerate(hidden_units):
            self.dense_layers.append(Dense(units, activation=None))
            if self.dropout_rate > 0.0:
                self.dropout_layers.append(Dropout(self.dropout_rate))
            if self.norm == 'batch_norm':
                self.norm_layers.append(BatchNormalization())
            elif self.norm == 'layer_norm':
                self.norm_layers.append(LayerNormalization())
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
        residual_layer = None

        for i, dense in enumerate(self.dense_layers):
            if i % self.skipped_layers == 0 and i > 0 and self.residual:
                if residual_layer is not None:
                    # Check if projection is needed
                    if x.shape[-1] != residual_layer.shape[-1]:
                        residual_layer = Dense(x.shape[-1], use_bias=False)(residual_layer)
                    x = Add()([x, residual_layer])
                residual_layer = x  # Update the starting point for the next residual connection
            else:
                if i % self.skipped_layers == 0 or residual_layer is None:
                    residual_layer = x

            x = dense(x)

            if self.norm is not None:
                x = self.norm_layers[i](x)

            x = Activation(self.activation)(x)

            if self.dropout_rate > 0.0:
                x = self.dropout_layers[i](x)

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
        # Create the final dense layer with custom regularization
        self.dense_layer = Dense(
            1,
            activation=self.output_activation,
            kernel_regularizer=AwayFrom1Regularizer(strength=0.001)  # Custom regularization
        )

    def call(self, inputs: tf.Tensor) -> Dict[str, Any]:
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

        return {'output': output, 'attention_scores': self.attention_scores}

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

    def call(self, inputs: tf.Tensor) -> Dict[str, Union[Tensor, Any]]:
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

        return {'output': output, 'attention_scores': self.attention_scores}


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
        w (tf.Variable): Fixed weights to be added to attention scores.
    """

    def __init__(self,
                 attn_hidden_units: Optional[List[int]] = None,
                 activation: str = 'tanh',
                 output_activation: Optional[str] = None):
        super().__init__(attn_hidden_units, activation, output_activation)
        self.w = None

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

        # Create fixed weights w (including w0)
        self.w = self.add_weight(
            name='fixed_weights',
            shape=(input_shape[-1] + 1,),
            initializer='random_normal',
            trainable=True
        )

    def call(self, inputs: tf.Tensor) -> Dict[str, Union[tf.Tensor, Any]]:
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

        self.attention_scores = attention_scores

        # Separate w0 from the rest of the weights
        w0 = self.w[0]
        w_rest = self.w[1:]

        # Add attention scores to the rest of the weights
        combined_weights = w_rest + attention_scores

        # Element-wise multiplication of combined weights with input features
        weighted_inputs = combined_weights * inputs

        # Sum up all components
        intermediate_output = w0 + tf.reduce_sum(weighted_inputs, axis=-1, keepdims=True)

        output = intermediate_output

        if self.output_activation:
            output = tf.keras.activations.get(self.output_activation)(output)

        return {'output': output, 'attention_scores': self.attention_scores}

    def get_dense_weights(self) -> tf.Tensor:
        """
        Retrieve the fixed weights (w0, w1, w2, ...).

        Returns:
            tf.Tensor: The fixed weights tensor.
        """
        return self.w


class BlockT4(BlockBase):
    """
        A custom layer that applies attention mechanism followed by a dense layer.

        This block consists of an attention layer that computes attention scores,
        applies these scores to the input via element-wise multiplication,
        and then passes the result through a dense layer.

        y = w0 + w1 * a1 * x1 + w2 * a2 * x2 + ... + wn * an * xn where softmax is applied to attention scores.

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
        """
        Initialize the BlockT4.

        Args:
            attn_hidden_units (Optional[List[int]]): List of integers for hidden layer units in attention mechanism.
            activation (str): Activation function to use in attention layers.
            output_activation (Optional[str]): Activation function to use in the final dense layer.
        """
        super().__init__(attn_hidden_units, activation, output_activation)
        self.dense_layer = None
        self.softmax = Softmax(axis=-1)

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
        # Create the final dense layer
        self.dense_layer = Dense(1, activation=self.output_activation)

    def call(self, inputs: tf.Tensor) -> Dict[str, Any]:
        """
        Perform the forward pass of the BlockT4 layer.

        This method applies the attention mechanism to the inputs,
        applies softmax to the attention scores, then passes the weighted inputs through a dense layer.

        Args:
            inputs (tf.Tensor): The input tensor.

        Returns:
            tf.Tensor: The output tensor after applying attention, softmax, and the dense layer.
        """
        # Compute attention scores
        self.attention_scores = self.attention_block(inputs)

        # Apply softmax to obtain attention probabilities
        attention_probs = self.softmax(self.attention_scores)

        self.attention_scores = attention_probs

        # Apply attention probabilities to inputs via element-wise multiplication
        weighted_inputs = Multiply()([inputs, attention_probs])

        # Pass the weighted inputs through the final dense layer
        output = self.dense_layer(weighted_inputs)

        return {'output': output, 'attention_scores': self.attention_scores}

    def get_dense_weights(self) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Retrieve the weights and bias of the output dense layer.

        Returns:
            Tuple[tf.Tensor, tf.Tensor]: A tuple containing the weights and bias of the dense layer.
                                         The first element is the weight tensor, and the second
                                         is the bias tensor.
        """
        return self.dense_layer.get_weights()


class BlockT5(BlockBase):
    """
    A custom layer that applies attention mechanism followed by a dense layer.

    This block consists of an attention layer that computes attention scores,
    applies these scores to the input via element-wise multiplication,
    and then passes the result through a dense layer.

    y = w0 + w1 * a1 * x1 + w2 * a2 * x2 + ... + wn * an * xn where sigmoid is applied to attention scores.

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
        """
        Initialize the BlockT5.

        Args:
            attn_hidden_units (Optional[List[int]]): List of integers for hidden layer units in attention mechanism.
            activation (str): Activation function to use in attention layers.
            output_activation (Optional[str]): Activation function to use in the final dense layer.
        """
        super().__init__(attn_hidden_units, activation, output_activation)
        self.dense_layer = None
        self.sigmoid = Activation('sigmoid')

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
        # Create the final dense layer
        self.dense_layer = Dense(1, activation=self.output_activation)

    def call(self, inputs: tf.Tensor) -> Dict[str, Any]:
        """
        Perform the forward pass of the BlockT5 layer.

        This method applies the attention mechanism to the inputs,
        applies sigmoid to the attention scores, then passes the weighted inputs through a dense layer.

        Args:
            inputs (tf.Tensor): The input tensor.

        Returns:
            tf.Tensor: The output tensor after applying attention, sigmoid, and the dense layer.
        """
        # Compute attention scores
        self.attention_scores = self.attention_block(inputs)

        # Apply sigmoid to obtain attention weights between 0 and 1
        attention_weights = self.sigmoid(self.attention_scores)

        self.attention_scores = attention_weights

        # Apply attention weights to inputs via element-wise multiplication
        weighted_inputs = Multiply()([inputs, attention_weights])

        # Pass the weighted inputs through the final dense layer
        output = self.dense_layer(weighted_inputs)

        return {'output': output, 'attention_scores': self.attention_scores}

    def get_dense_weights(self) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Retrieve the weights and bias of the output dense layer.

        Returns:
            Tuple[tf.Tensor, tf.Tensor]: A tuple containing the weights and bias of the dense layer.
                                         The first element is the weight tensor, and the second
                                         is the bias tensor.
        """
        return self.dense_layer.get_weights()


class BlockT7(BlockBase):
    """
    A custom layer that applies attention mechanism followed by a dense layer.

    This block consists of an attention layer that computes attention scores,
    applies these scores to the input via element-wise multiplication,
    and then passes the result through a dense layer.

    y = w0 + w1 * a1 * x1 + w2 * a2 * x2 + ... + wn * an * xn where tanh is applied to attention scores.

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
        """
        Initialize the BlockT5.

        Args:
            attn_hidden_units (Optional[List[int]]): List of integers for hidden layer units in attention mechanism.
            activation (str): Activation function to use in attention layers.
            output_activation (Optional[str]): Activation function to use in the final dense layer.
        """
        super().__init__(attn_hidden_units, activation, output_activation)
        self.dense_layer = None
        self.tanh = Activation('tanh')

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
        # Create the final dense layer
        self.dense_layer = Dense(1, activation=self.output_activation)

    def call(self, inputs: tf.Tensor) -> Dict[str, Any]:
        """
        Perform the forward pass of the BlockT5 layer.

        This method applies the attention mechanism to the inputs,
        applies sigmoid to the attention scores, then passes the weighted inputs through a dense layer.

        Args:
            inputs (tf.Tensor): The input tensor.

        Returns:
            tf.Tensor: The output tensor after applying attention, sigmoid, and the dense layer.
        """
        # Compute attention scores
        self.attention_scores = self.attention_block(inputs)

        # Apply tanh to obtain attention weights between -1 and 1
        attention_weights = self.tanh(self.attention_scores)

        self.attention_scores = attention_weights

        # Apply attention weights to inputs via element-wise multiplication
        weighted_inputs = Multiply()([inputs, attention_weights])

        # Pass the weighted inputs through the final dense layer
        output = self.dense_layer(weighted_inputs)

        return {'output': output, 'attention_scores': self.attention_scores}

    def get_dense_weights(self) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Retrieve the weights and bias of the output dense layer.

        Returns:
            Tuple[tf.Tensor, tf.Tensor]: A tuple containing the weights and bias of the dense layer.
                                         The first element is the weight tensor, and the second
                                         is the bias tensor.
        """
        return self.dense_layer.get_weights()


class BlockT6(BlockBase):
    """
    A custom layer that applies attention mechanism followed by a dense layer.

    This block consists of an attention layer that computes attention scores,
    applies these scores to the input via element-wise multiplication,
    normalizes the attention scores, and then passes the result through a dense layer.

    y = w0 + w1 * a1 * x1 + w2 * a2 * x2 + ... + wn * an * xn where sigmoid is applied to attention scores.
    Then attention scores are normalized to sum to 1.

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
        """
        Initialize the BlockT6.

        Args:
            attn_hidden_units (Optional[List[int]]): List of integers for hidden layer units in attention mechanism.
            activation (str): Activation function to use in attention layers.
            output_activation (Optional[str]): Activation function to use in the final dense layer.
        """
        super().__init__(attn_hidden_units, activation, output_activation)
        self.dense_layer = None
        self.sigmoid = Activation('sigmoid')

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
        # Create the final dense layer
        self.dense_layer = Dense(1, activation=self.output_activation)

    def call(self, inputs: tf.Tensor, training=None) -> Dict[str, Any]:
        """
        Perform the forward pass of the BlockT6 layer.

        This method applies the attention mechanism to the inputs,
        applies sigmoid to the attention scores, normalizes them to sum to 1,
        then passes the weighted inputs through a dense layer.

        Args:
            inputs (tf.Tensor): The input tensor.

        Returns:
            tf.Tensor: The output tensor after applying attention, sigmoid, normalization, and the dense layer.
        """
        # Compute attention scores
        self.attention_scores = self.attention_block(inputs)

        # Apply sigmoid to obtain attention weights between 0 and 1
        attention_weights = self.sigmoid(self.attention_scores)

        # Normalize attention weights to sum to 1
        attention_weights_sum = K.sum(attention_weights, axis=-1, keepdims=True)
        normalized_attention_weights = attention_weights / attention_weights_sum

        self.attention_scores = normalized_attention_weights

        # Apply attention weights to inputs via element-wise multiplication
        weighted_inputs = Multiply()([inputs, normalized_attention_weights])

        # Pass the weighted inputs through the final dense layer
        output = self.dense_layer(weighted_inputs)

        return {'output': output, 'attention_scores': self.attention_scores}

    def get_dense_weights(self) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Retrieve the weights and bias of the output dense layer.

        Returns:
            Tuple[tf.Tensor, tf.Tensor]: A tuple containing the weights and bias of the dense layer.
                                         The first element is the weight tensor, and the second
                                         is the bias tensor.
        """
        return self.dense_layer.get_weights()


class BlockT0(BlockBase):
    """
    A simple dense layer block with no attention mechanism.

    This block consists of a dense layer that directly learns weights for the inputs.

    y = w0 + w1 * x1 + w2 * x2 + ... + wn * xn

    Attributes:
        dense_layer (Dense): The dense layer that produces the output.
    """

    def __init__(self,
                 attn_hidden_units: Optional[List[int]] = None,
                 activation: str = 'tanh',
                 output_activation: Optional[str] = None):
        super().__init__(attn_hidden_units=None, activation=None, output_activation=output_activation)
        self.dense_layer = None

    def build(self, input_shape: tf.TensorShape) -> None:
        """
        Build the layer. This method is called automatically by Keras when the layer is first used.

        Args:
            input_shape (tf.TensorShape): The shape of the input tensor.
        """
        # Create a single dense layer for weights (w0, w1, w2, ...)
        self.dense_layer = Dense(1, activation=self.output_activation)

    def call(self, inputs: tf.Tensor) -> Dict[str, Any]:
        """
        Perform the forward pass of the BlockT0 layer.

        This method directly applies the dense layer to the inputs.

        Args:
            inputs (tf.Tensor): The input tensor.

        Returns:
            tf.Tensor: The output tensor after applying the dense layer.
        """
        output = self.dense_layer(inputs)
        return {'output': output, 'attention_scores': tf.zeros_like(inputs)}

    def get_dense_weights(self) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Retrieve the weights and bias of the output dense layer.

        Returns:
            Tuple[tf.Tensor, tf.Tensor]: A tuple containing the weights and bias of the dense layer.
                                         The first element is the weight tensor, and the second
                                         is the bias tensor.
        """
        return self.dense_layer.get_weights()
