from typing import List, Optional, Tuple, Dict, Any

import tensorflow as tf
from tensorflow.keras.layers import (
    Input,
    Dense,
    Dropout,
    BatchNormalization,
    LayerNormalization,
    Add,
    Multiply,
    Layer,
    Activation,
    LeakyReLU
)
from tensorflow.keras.models import Model

from modules.training.cme_modeling import NormalizeLayer
from modules.training.sam_keras import SAMModel


class AttentionBlock(Layer):
    def __init__(
            self,
            input_dim: int,
            hidden_units: List[int],
            output_dim: int,
            hidden_activation: str = 'tanh',
            dropout_rate: float = 0.0,
            norm: str = None,
            residual: bool = False,
            skipped_layers: int = 2
    ):
        super(AttentionBlock, self).__init__()
        self.input_dim = input_dim
        self.hidden_units = hidden_units
        self.output_dim = output_dim
        self.hidden_activation = hidden_activation
        self.dropout_rate = dropout_rate
        self.norm = norm
        self.residual = residual
        self.skipped_layers = skipped_layers
        self.dense_layers: List[Dense] = []
        self.dropout_layers: List[Dropout] = []
        self.norm_layers: List[Layer] = []
        self.projection_layers: Dict[int, Dense] = {}

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
        for i, units in enumerate(self.hidden_units):
            if self.residual and i % self.skipped_layers == 0 and i > 0:
                self.projection_layers[i] = Dense(units, use_bias=False)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        x = inputs
        residual_layer = None

        for i, dense in enumerate(self.dense_layers):
            if i % self.skipped_layers == 0 and i > 0 and self.residual:
                if residual_layer is not None:
                    if x.shape[-1] != residual_layer.shape[-1]:
                        residual_layer = self.projection_layers[i](residual_layer)
                    x = Add()([x, residual_layer])
                residual_layer = x
            else:
                if i % self.skipped_layers == 0 or residual_layer is None:
                    residual_layer = x

            x = dense(x)

            if self.norm is not None:
                x = self.norm_layers[i](x)

            if self.hidden_activation == 'leaky_relu':
                x = LeakyReLU()(x)
            else:
                x = Activation(self.hidden_activation)(x)

            if self.dropout_rate > 0.0:
                x = self.dropout_layers[i](x)

        return self.output_layer(x)


class TanhAttentiveBlock(Layer):
    """
    A custom layer that applies an attention mechanism followed by a dense layer.

    This block consists of an attention layer that computes attention scores,
    applies these scores to the input via element-wise multiplication,
    and then passes the result through a dense layer.

    y = w0 + w1 * a1 * x1 + w2 * a2 * x2 + ... + wn * an * xn where tanh is applied to attention scores.

    Attributes:
        attn_hidden_units (List[int]): A list of integers representing the number of units
                                       in each hidden layer of the attention mechanism.
        attn_hidden_activation (str): The activation function to use in the hidden layers of the attention block.
        attn_dropout_rate (float): The dropout rate to use in the attention layer.
        attn_norm (str): The type of normalization to use in the attention layer ('batch_norm' or 'layer_norm').
        attn_residual (bool): Whether to use residual connections in the attention layer.
        attn_skipped_layers (int): The number of layers between residual connections in the attention layer.
        output_dim (int): The dimensionality of the output.
        output_activation (str): The activation function to use in the final dense layer.
        dropout_rate (float): The dropout rate to use in the TanhAttentiveBlock.
        norm (str): The type of normalization to use in the TanhAttentiveBlock ('batch_norm' or 'layer_norm').
        attention_block (AttentionBlock): The layer used to compute attention scores.
        dense_layer (Dense): The final dense layer that produces the output.
        attention_scores (tf.Tensor): The most recently computed attention scores.
    """

    def __init__(self,
                 attn_hidden_units: Optional[List[int]] = None,
                 attn_hidden_activation: str = 'leaky_relu',
                 attn_dropout_rate: float = 0,
                 attn_norm: Optional[str] = None,
                 attn_residual: bool = True,
                 attn_skipped_layers: int = 2,
                 output_dim: int = 1,
                 output_activation: Optional[str] = 'leaky_relu',
                 dropout_rate: float = 0.0,
                 norm: Optional[str] = None,
                 a: float = 1.5):
        """
        Initialize the TanhAttentiveBlock.

        Args:
            attn_hidden_units (Optional[List[int]]): List of integers for hidden layer units in attention mechanism.
            attn_hidden_activation (str): Activation function to use in hidden layers of the attention block.
            attn_dropout_rate (float, optional): The dropout rate to use in the attention layers. Defaults to 0.0.
            attn_norm (Optional[str], optional): The type of normalization to use ('batch_norm' or 'layer_norm'). Defaults to None.
            attn_residual (bool, optional): Whether to use residual connections in the attention layer. Defaults to False.
            attn_skipped_layers (int, optional): The number of layers between residual connections. Defaults to 2.
            output_dim (int, optional): The dimensionality of the output. Defaults to 1.
            output_activation (Optional[str], optional): Activation function to use in the final dense layer. Defaults to 'leaky_relu'.
            dropout_rate (float, optional): The dropout rate to use in the TanhAttentiveBlock. Defaults to 0.0.
            norm (Optional[str], optional): The type of normalization to use ('batch_norm' or 'layer_norm'). Defaults to None.
            a (float, optional): The parameter to scale the attention scores before applying tanh. Defaults to 0.5.
        """
        super(TanhAttentiveBlock, self).__init__()
        self.attn_hidden_units = attn_hidden_units or [3]
        self.attn_hidden_activation = attn_hidden_activation
        self.attn_dropout_rate = attn_dropout_rate
        self.attn_norm = attn_norm
        self.attn_residual = attn_residual
        self.attn_skipped_layers = attn_skipped_layers
        self.output_dim = output_dim
        self.output_activation = output_activation
        self.dropout_rate = dropout_rate
        self.norm = norm
        self.a = tf.Variable(a, trainable=True, dtype=tf.float32, name='attention_scale')
        self.attention_scores = None
        self.attention_block = None
        self.dense_layer = None
        self.tanh = Activation('tanh')
        self.dropout_layer = Dropout(self.dropout_rate) if self.dropout_rate > 0 else None
        self.norm_layer = None

    def build(self, input_shape: tf.TensorShape) -> None:
        """
        Build the layer. This method is called automatically by Keras when the layer is first used.

        Args:
            input_shape (tf.TensorShape): The shape of the input tensor.
        """
        # Create the attention block
        self.attention_block = AttentionBlock(
            input_dim=input_shape[-1],
            output_dim=input_shape[-1],
            hidden_units=self.attn_hidden_units,
            hidden_activation=self.attn_hidden_activation,
            dropout_rate=self.attn_dropout_rate,
            norm=self.attn_norm,
            residual=self.attn_residual,
            skipped_layers=self.attn_skipped_layers
        )
        # Create the final dense layer
        self.dense_layer = Dense(self.output_dim)

        # Set the normalization layer if specified
        if self.norm == 'batch_norm':
            self.norm_layer = BatchNormalization()
        elif self.norm == 'layer_norm':
            self.norm_layer = LayerNormalization()

    def call(self, inputs: tf.Tensor) -> Dict[str, Any]:
        """
        Perform the forward pass of the TanhAttentiveBlock layer.

        This method applies the attention mechanism to the inputs,
        applies tanh to the attention scores, then passes the weighted inputs through a dense layer.

        Args:
            inputs (tf.Tensor): The input tensor.

        Returns:
            tf.Tensor: The output tensor after applying attention, tanh, and the dense layer.
        """
        # Compute attention scores
        self.attention_scores = self.attention_block(inputs)

        # Apply scaling factor a and tanh to obtain attention weights between -1 and 1
        attention_weights = self.tanh(self.a * self.attention_scores)

        self.attention_scores = attention_weights

        # Apply attention weights to inputs via element-wise multiplication
        weighted_inputs = Multiply()([inputs, attention_weights])

        # Optionally apply dropout
        if self.dropout_layer is not None:
            weighted_inputs = self.dropout_layer(weighted_inputs)

        # Optionally apply normalization
        if self.norm_layer is not None:
            weighted_inputs = self.norm_layer(weighted_inputs)

        # Pass the weighted inputs through the final dense layer
        output = self.dense_layer(weighted_inputs)

        # Apply the output activation function
        if self.output_activation == 'leaky_relu':
            output = LeakyReLU()(output)
        elif self.output_activation:
            output = Activation(self.output_activation)(output)

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


class TanhAttentiveBlockV2(Layer):
    """
    A custom layer that applies an attention mechanism to modulate the weights of a dense layer.

    This block consists of an attention layer that computes attention scores,
    which are then applied to the weights of the final dense layer before producing the output.

    y = w0 + w1 * a1 + w2 * a2 + ... + wn * an where tanh is applied to attention scores.

    Attributes:
        attn_hidden_units (List[int]): A list of integers representing the number of units
                                       in each hidden layer of the attention mechanism.
        attn_hidden_activation (str): The activation function to use in the hidden layers of the attention block.
        attn_dropout_rate (float): The dropout rate to use in the attention layer.
        attn_norm (str): The type of normalization to use in the attention layer ('batch_norm' or 'layer_norm').
        attn_residual (bool): Whether to use residual connections in the attention layer.
        attn_skipped_layers (int): The number of layers between residual connections in the attention layer.
        output_dim (int): The dimensionality of the output.
        output_activation (str): The activation function to use in the final dense layer.
        dropout_rate (float): The dropout rate to use in the TanhAttentiveBlock.
        norm (str): The type of normalization to use in the TanhAttentiveBlock ('batch_norm' or 'layer_norm').
        attention_block (AttentionBlock): The layer used to compute attention scores.
        dense_layer (Dense): The final dense layer that produces the output.
        attention_scores (tf.Tensor): The most recently computed attention scores.
    """

    def __init__(self,
                 attn_hidden_units: Optional[List[int]] = None,
                 attn_hidden_activation: str = 'leaky_relu',
                 attn_dropout_rate: float = 0,
                 attn_norm: Optional[str] = None,
                 attn_residual: bool = True,
                 attn_skipped_layers: int = 2,
                 output_dim: int = 1,
                 output_activation: Optional[str] = 'leaky_relu',
                 dropout_rate: float = 0.0,
                 norm: Optional[str] = None,
                 a: float = 1.5):
        """
        Initialize the TanhAttentiveBlockV2.

        Args:
            attn_hidden_units (Optional[List[int]]): List of integers for hidden layer units in attention mechanism.
            attn_hidden_activation (str): Activation function to use in hidden layers of the attention block.
            attn_dropout_rate (float, optional): The dropout rate to use in the attention layers. Defaults to 0.0.
            attn_norm (Optional[str], optional): The type of normalization to use ('batch_norm' or 'layer_norm'). Defaults to None.
            attn_residual (bool, optional): Whether to use residual connections in the attention layer. Defaults to False.
            attn_skipped_layers (int, optional): The number of layers between residual connections. Defaults to 2.
            output_dim (int, optional): The dimensionality of the output. Defaults to 1.
            output_activation (Optional[str], optional): Activation function to use in the final dense layer. Defaults to 'leaky_relu'.
            dropout_rate (float, optional): The dropout rate to use in the TanhAttentiveBlockV2. Defaults to 0.0.
            norm (Optional[str], optional): The type of normalization to use ('batch_norm' or 'layer_norm'). Defaults to None.
            a (float, optional): The parameter to scale the attention scores before applying tanh. Defaults to 0.5.
        """
        super(TanhAttentiveBlockV2, self).__init__()
        self.attn_hidden_units = attn_hidden_units or [3]
        self.attn_hidden_activation = attn_hidden_activation
        self.attn_dropout_rate = attn_dropout_rate
        self.attn_norm = attn_norm
        self.attn_residual = attn_residual
        self.attn_skipped_layers = attn_skipped_layers
        self.output_dim = output_dim
        self.output_activation = output_activation
        self.dropout_rate = dropout_rate
        self.norm = norm
        self.a = tf.Variable(a, trainable=True, dtype=tf.float32, name='attention_scale')
        self.attention_scores = None
        self.attention_block = None
        self.dense_layer = None
        self.tanh = Activation('tanh')
        self.dropout_layer = Dropout(self.dropout_rate) if self.dropout_rate > 0 else None
        self.norm_layer = None

    def build(self, input_shape: tf.TensorShape) -> None:
        """
        Build the layer. This method is called automatically by Keras when the layer is first used.

        Args:
            input_shape (tf.TensorShape): The shape of the input tensor.
        """
        # Create the attention block with output dimension based on input_dim * output_dim
        self.attention_block = AttentionBlock(
            input_dim=input_shape[-1],
            output_dim=input_shape[-1] * self.output_dim,
            hidden_units=self.attn_hidden_units,
            hidden_activation=self.attn_hidden_activation,
            dropout_rate=self.attn_dropout_rate,
            norm=self.attn_norm,
            residual=self.attn_residual,
            skipped_layers=self.attn_skipped_layers
        )
        # Create the final dense layer
        self.dense_layer = Dense(self.output_dim)

        # Set the normalization layer if specified
        if self.norm == 'batch_norm':
            self.norm_layer = BatchNormalization()
        elif self.norm == 'layer_norm':
            self.norm_layer = LayerNormalization()

    def call(self, inputs: tf.Tensor) -> Dict[str, Any]:
        """
        Perform the forward pass of the TanhAttentiveBlockV2 layer.

        This method applies the attention mechanism to modulate the weights
        of the dense layer before producing the output.

        Args:
            inputs (tf.Tensor): The input tensor.

        Returns:
            tf.Tensor: The output tensor after applying attention, tanh, and the dense layer.
        """
        # Compute attention scores
        self.attention_scores = self.attention_block(inputs)

        # Apply scaling factor a and tanh to obtain attention weights between -1 and 1
        attention_weights = self.tanh(self.a * self.attention_scores)

        # Reshape attention weights to match the weights of the dense layer
        attention_weights = tf.reshape(attention_weights, (-1, self.output_dim, inputs.shape[-1]))

        # Get the original weights of the dense layer
        dense_weights, dense_bias = self.dense_layer.get_weights()

        # Reshape the dense weights to match the attention weights
        dense_weights = tf.reshape(dense_weights, (self.output_dim, -1))

        # Apply the attention weights to the dense weights
        modulated_weights = dense_weights * attention_weights

        # Calculate the final output by using the modulated weights
        modulated_output = tf.matmul(inputs, modulated_weights, transpose_b=True) + dense_bias

        # Optionally apply dropout
        if self.dropout_layer is not None:
            modulated_output = self.dropout_layer(modulated_output)

        # Optionally apply normalization
        if self.norm_layer is not None:
            modulated_output = self.norm_layer(modulated_output)

        # Apply the output activation function
        if self.output_activation == 'leaky_relu':
            modulated_output = LeakyReLU()(modulated_output)
        elif self.output_activation:
            modulated_output = Activation(self.output_activation)(modulated_output)

        return {'output': modulated_output, 'attention_scores': self.attention_scores}

    def get_dense_weights(self) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Retrieve the weights and bias of the output dense layer.

        Returns:
            Tuple[tf.Tensor, tf.Tensor]: A tuple containing the weights and bias of the dense layer.
                                         The first element is the weight tensor, and the second
                                         is the bias tensor.
        """
        return self.dense_layer.get_weights()


def create_attentive_model(
        input_dim: int = 25,
        output_dim: int = 1,
        hidden_blocks=None,
        attn_hidden_units: Optional[List[int]] = None,
        attn_hidden_activation: str = 'tanh',
        attn_dropout_rate: float = -1,
        attn_norm: Optional[str] = None,
        attn_residual: bool = False,
        attn_skipped_layers: int = 2,
        skipped_blocks: int = 1,
        repr_dim: int = 128,
        dropout_rate: float = 0.0,
        activation='leaky_relu',
        pds: bool = False,
        norm: str = None,
        residual: bool = False,
        sam_rho: float = 0.05,
        name: str = 'attentive_mlp'
) -> Model:
    """
    Create a model with stacked TanhAttentiveBlock layers, optional dropout, normalization, and residual connections.

    Parameters:
    - input_dim (int): The number of features in the input data.
    - output_dim (int): The dimension of the output layer. Default is 1 for regression tasks.
    - hidden_blocks (List[int]): List where each entry is the output_dim of the block at that position.
    - attn_hidden_units (List[int]): A list of integers where each integer is the number of units in a hidden layer of the attention block.
    - attn_hidden_activation (str): Activation function to use in hidden layers of the attention block.
    - attn_dropout_rate (float): The dropout rate to use in the attention layers. If 0.0, dropout is not used. If -1, use the same dropout rate as the model dropout rate.
    - attn_norm (str): The type of normalization to use in the attention layers ('batch_norm' or 'layer_norm'). Default is None.
    - attn_residual (bool): Whether to use residual connections in the attention layer. Default is False.
    - attn_skipped_layers (int): Number of layers between residual connections in the attention layer. Default is 2.
    - skipped_blocks (int): Number of blocks between residual connections. Default is 1.
    - repr_dim (int): The number of features in the final representation vector.
    - dropout_rate (float): The dropout rate to use in the model. Default is 0.0.
    - activation: Optional activation function to use in the blocks. Default is LeakyReLU.
    - pds (bool): If True, use a NormalizeLayer after the representation layer.
    - norm (str): The type of normalization to use ('batch_norm' or 'layer_norm'). Default is None.
    - residual (bool): If True, add residual connections for every 'skipped_blocks' hidden blocks. Default is False.
    - sam_rho (float): Size of the neighborhood for perturbation in SAM. Default is 0.05. If 0.0, SAM is not used.
    - name (str): Name of the model. Default is 'attentive_mlp'.

    Returns:
    - Model: A Keras model instance.
    """

    if hidden_blocks is None:
        hidden_blocks = [50, 50]

    input_layer = Input(shape=(input_dim,))
    x = input_layer
    skip_connection = None

    # Process all hidden blocks
    for i, block_output_dim in enumerate(hidden_blocks):
        # Set attn_hidden_units to three layers, each of the size of the input to the block if not provided
        if attn_hidden_units is None:
            attn_hiddens = [x.shape[-1]] * 3
            print('attn')
            print(attn_hidden_units)
        else:
            attn_hiddens = attn_hidden_units

        # Create a TanhAttentiveBlock for each specified hidden block
        block = TanhAttentiveBlock(
            attn_hidden_units=attn_hiddens,
            attn_hidden_activation=attn_hidden_activation,
            attn_dropout_rate=attn_dropout_rate,
            attn_norm=attn_norm,
            attn_residual=attn_residual,
            attn_skipped_layers=attn_skipped_layers,
            output_dim=block_output_dim,
            norm=norm,
            dropout_rate=dropout_rate,
            output_activation=activation
        )

        # Implement residual connections between blocks
        if i % skipped_blocks == 0 and i > 0 and residual:
            if skip_connection is not None:
                if x.shape[-1] != skip_connection.shape[-1]:
                    skip_connection = Dense(x.shape[-1], use_bias=False)(skip_connection)
                x = Add()([x, skip_connection])
            skip_connection = x
        else:
            if i % skipped_blocks == 0 or skip_connection is None:
                skip_connection = x

        x = block(x)['output']

    if attn_hidden_units is None:
        attn_hiddens = [x.shape[-1]] * 3
        print('attn')
        print(attn_hidden_units)
    else:
        attn_hiddens = attn_hidden_units

    # Add the final block with repr_dim output
    final_block = TanhAttentiveBlock(
        attn_hidden_units=attn_hiddens,
        attn_hidden_activation=attn_hidden_activation,
        attn_dropout_rate=attn_dropout_rate,
        attn_norm=attn_norm,
        attn_residual=attn_residual,
        attn_skipped_layers=attn_skipped_layers,
        output_dim=repr_dim,
        norm=norm,
        dropout_rate=dropout_rate,
        output_activation=activation
    )

    final_repr = final_block(x)['output']

    if pds:
        final_repr_output = NormalizeLayer(name='normalize_layer')(final_repr)
    else:
        final_repr_output = final_repr

    if output_dim > 0:
        output_layer = Dense(output_dim, name='forecast_head')(final_repr_output)
        model_output = [final_repr_output, output_layer]
    else:
        model_output = final_repr_output

    if sam_rho > 0.0:
        model = SAMModel(inputs=input_layer, outputs=model_output, rho=sam_rho, name=name)
    else:
        model = Model(inputs=input_layer, outputs=model_output, name=name)

    return model
