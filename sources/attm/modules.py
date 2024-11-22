from typing import List, Optional, Tuple, Dict, Any, Callable

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

from modules.shared.globals import LEAKY_RELU_ALPHA
from modules.training.cme_modeling import NormalizeLayer
from modules.training.sam_keras import SAMModel


def create_attentive_model(
        input_dim: int = 25,
        output_dim: int = 1,
        hidden_blocks=None,
        attn_hidden_units: Optional[List[int]] = None,
        attn_hidden_activation: str = 'tanh',
        attn_dropout: float = -1,
        attn_norm: Optional[str] = None,
        attn_skipped_layers: int = 2,
        skipped_blocks: int = 1,
        repr_dim: int = 128,
        dropout: float = 0.0,
        activation='leaky_relu',
        pds: bool = False,
        norm: str = None,
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
    - attn_dropout (float): The dropout rate to use in the attention layers. If 0.0, dropout is not used. If -1, use the same dropout rate as the model dropout rate.
    - attn_norm (str): The type of normalization to use in the attention layers ('batch_norm' or 'layer_norm'). Default is None.
    - attn_skipped_layers (int): Number of layers between residual connections in the attention layer. Default is 2.
    - skipped_blocks (int): Number of blocks between residual connections. Default is 1.
    - repr_dim (int): The number of features in the final representation vector.
    - dropout (float): The dropout rate to use in the model. Default is 0.0.
    - activation: Optional activation function to use in the blocks. Default is LeakyReLU.
    - pds (bool): If True, use a NormalizeLayer after the representation layer.
    - norm (str): The type of normalization to use ('batch_norm' or 'layer_norm'). Default is None.
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
        # Create a TanhAttentiveBlock for each specified hidden block
        block = TanhAttentiveBlock(
            attn_hidden=attn_hidden_units,
            attn_hidden_activation=attn_hidden_activation,
            attn_dropout=attn_dropout,
            attn_norm=attn_norm,
            attn_skipped_layers=attn_skipped_layers,
            output_dim=block_output_dim,
            norm=norm,
            dropout=dropout,
            output_activation=activation
        )

        # Implement residual connections between blocks
        if i % skipped_blocks == 0 and i > 0 and skipped_blocks > 0:
            if skip_connection is not None:
                if x.shape[-1] != skip_connection.shape[-1]:
                    skip_connection = Dense(x.shape[-1], use_bias=False)(skip_connection)
                x = Add()([x, skip_connection])
            skip_connection = x
        else:
            if i % skipped_blocks == 0 or skip_connection is None:
                skip_connection = x

        x = block(x)['output']

    # Add the final block with repr_dim output
    final_block = TanhAttentiveBlock(
        attn_hidden=attn_hidden_units,
        attn_hidden_activation=attn_hidden_activation,
        attn_dropout=attn_dropout,
        attn_norm=attn_norm,
        attn_skipped_layers=attn_skipped_layers,
        output_dim=repr_dim,
        norm=norm,
        dropout=dropout,
        output_activation=activation
    )

    final = final_block(x)
    final_repr = final['output']
    last_attention_scores = final['attention_scores']

    if pds:
        final_repr_output = NormalizeLayer(name='normalize_layer')(final_repr)
    else:
        final_repr_output = final_repr

    if output_dim > 0:
        output_block = TanhAttentiveBlock(
            attn_hidden=attn_hidden_units,
            attn_hidden_activation=attn_hidden_activation,
            attn_dropout=attn_dropout,
            attn_norm=attn_norm,
            attn_skipped_layers=attn_skipped_layers,
            output_dim=output_dim,
            norm=norm,
            dropout=dropout,
            output_activation=activation
        )

        output = output_block(final_repr_output)
        output_layer = output['output']
        last_attention_scores = output['attention_scores']

        model_output = [final_repr_output, output_layer]
    else:
        model_output = final_repr_output

    if sam_rho > 0.0:
        model = SAMModel(inputs=input_layer, outputs=model_output, rho=sam_rho, name=name)
    else:
        model = Model(inputs=input_layer, outputs=model_output, name=name)

    return model


def add_proj_head(
        model: Model,
        output_dim: int = 1,
        freeze_features: bool = True,
        pds: bool = False,
        hidden_blocks=None,
        attn_hidden_units: Optional[List[int]] = None,
        attn_hidden_activation: str = 'tanh',
        attn_dropout: float = -1,
        attn_norm: Optional[str] = None,
        attn_skipped_layers: int = 2,
        skipped_blocks: int = 1,
        dropout: float = 0.0,
        activation='leaky_relu',
        norm: str = None,
        sam_rho: float = 0.05,
        name: str = 'mlp',
) -> Model:
    """
    Add a regression head with one output unit and a projection layer to an existing model,
    replacing the existing prediction layer and optionally the decoder layer.

    :param model: The existing model
    :param output_dim: The dimensionality of the output of the regression head.
    :param freeze_features: Whether to freeze the layers of the base model or not.
    :param pds: Whether to use a NormalizeLayer after the representation layer.
    :param hidden_blocks: List of hidden layer sizes for the attention blocks.
    :param attn_hidden_units: List of hidden layer sizes for the attention mechanism.
    :param attn_hidden_activation: Activation function for the attention mechanism.
    :param attn_dropout: Dropout rate for the attention mechanism.
    :param attn_norm: Type of normalization for the attention mechanism.
    :param attn_skipped_layers: Number of layers between residual connections in the attention mechanism.
    :param skipped_blocks: Number of blocks between residual connections.
    :param dropout: Dropout rate for the model.
    :param activation: Activation function for the model.
    :param norm: Type of normalization for the model.
    :param sam_rho: Rho value for the SAM model.
    :param name: Name of the model.

    :return: The modified model with a projection layer and a regression head.
    """

    if hidden_blocks is None:
        hiddens = [6]

    print(f'Features are frozen: {freeze_features}')

    # Determine the layer to be kept based on whether PDS representations are used
    layer_to_keep = 'normalize_layer' if pds else 'repr_layer'

    # Remove the last layer(s) to keep only the representation layer
    new_base_model = Model(inputs=model.input, outputs=model.get_layer(layer_to_keep).output)

    # If freeze_features is True, freeze the layers of the new base model
    if freeze_features:
        for layer in new_base_model.layers:
            layer.trainable = False

    # Count existing dropout layers to avoid naming conflicts
    dropout_count = sum(1 for layer in model.layers if isinstance(layer, Dropout))

    # Extract the output of the last layer of the new base model (representation layer)
    repr_output = new_base_model.output

    # Projection Layer(s)
    x_proj = repr_output
    skip_connection = None

    # Process all hidden blocks
    for i, block_output_dim in enumerate(hidden_blocks):
        # Set attn_hidden_units to three layers, each of the size of the input to the block if not provided
        if attn_hidden_units is None:
            attn_hiddens = [x_proj.shape[-1]] * 3
            print('attn')
            print(attn_hidden_units)
        else:
            attn_hiddens = attn_hidden_units

        # Create a TanhAttentiveBlock for each specified hidden block
        block = TanhAttentiveBlock(
            attn_hidden=attn_hiddens,
            attn_hidden_activation=attn_hidden_activation,
            attn_dropout=attn_dropout,
            attn_norm=attn_norm,
            attn_skipped_layers=attn_skipped_layers,
            output_dim=block_output_dim,
            norm=norm,
            dropout=dropout,
            output_activation=activation
        )

        # Implement residual connections between blocks
        if i % skipped_blocks == 0 and i > 0 and skipped_blocks > 0:
            if skip_connection is not None:
                if x_proj.shape[-1] != skip_connection.shape[-1]:
                    skip_connection = Dense(x_proj.shape[-1], use_bias=False)(skip_connection)
                x_proj = Add()([x_proj, skip_connection])
            skip_connection = x_proj
        else:
            if i % skipped_blocks == 0 or skip_connection is None:
                skip_connection = x_proj

        x_proj = block(x_proj)['output']

    if attn_hidden_units is None:
        attn_hiddens = [x_proj.shape[-1]] * 3
        print('attn')
        print(attn_hidden_units)
    else:
        attn_hiddens = attn_hidden_units

    output_block = TanhAttentiveBlock(
        attn_hidden=attn_hiddens,
        attn_hidden_activation=attn_hidden_activation,
        attn_dropout=attn_dropout,
        attn_norm=attn_norm,
        attn_skipped_layers=attn_skipped_layers,
        output_dim=output_dim,
        norm=norm,
        dropout=dropout,
        output_activation=activation
    )

    output = output_block(x_proj)
    output_layer = output['output']

    last_attention_scores = output['attention_scores']

    model_output = {
        'repr': repr_output,
        'output': output_layer,
        'attention_scores': last_attention_scores
    }

    if sam_rho > 0.0:
        # create the new extended SAM model
        extended_model = SAMModel(inputs=new_base_model.input, outputs=model_output, rho=sam_rho,
                                  name=name)
    else:
        # Create the new extended model
        extended_model = Model(inputs=new_base_model.input, outputs=model_output, name=name)

    # If freeze_features is False, make all layers trainable
    if not freeze_features:
        for layer in extended_model.layers:
            layer.trainable = True

    return extended_model


class AttentionBlock(Layer):
    def __init__(
            self,
            input_dim: int,
            hidden_units: List[int],
            repr_dim: int,
            output_dim: int,
            hidden_activation=None,
            dropout: float = 0.0,
            norm: str = None,
            skipped_layers: int = 0,  # 0 means no residual, >0 means residual is active
            skip_repr: bool = True,
    ):
        """
        Initializes a customizable attention-based neural network block with skip connections,
        normalization layers, and dropout, following the architectural flow of create_mlp2.

        Parameters:
        - input_dim (int): Dimensionality of input data.
        - hidden_units (List[int]): List of hidden layer units.
        - repr_dim (int): Dimensionality of the representation layer.
        - output_dim (int): Output dimensionality.
        - hidden_activation: Activation function for hidden layers. Default is LeakyReLU.
        - dropout (float): Dropout rate. Default is 0.0 (no dropout).
        - norm (str): Type of normalization ('batch_norm', 'layer_norm', or None). Default is None.
        - skipped_layers (int): Number of layers between residual connections. Default is 0 (no residual).
        - skip_repr (bool): If True, adds a residual connection to the representation layer.
        """
        super(AttentionBlock, self).__init__()
        self.repr_projection = None
        self.input_projection = None
        self.input_dim = input_dim
        self.hidden_units = hidden_units
        self.repr_dim = repr_dim
        self.output_dim = output_dim
        self.leaky_relu_alpha = LEAKY_RELU_ALPHA
        self.dropout = dropout
        self.norm = norm
        self.skipped_layers = skipped_layers
        self.skip_repr = skip_repr
        self.has_residuals = self.skipped_layers > 0

        # Activation function
        if hidden_activation is None:
            self.hidden_activation = LeakyReLU(alpha=self.leaky_relu_alpha)
        elif isinstance(hidden_activation, str):
            if hidden_activation.lower() == 'leaky_relu':
                self.hidden_activation = LeakyReLU(alpha=self.leaky_relu_alpha)
            else:
                self.hidden_activation = Activation(hidden_activation)
        elif callable(hidden_activation):
            self.hidden_activation = hidden_activation
        else:
            raise ValueError("Invalid hidden_activation parameter")

        # Layers
        self.dense_layers: List[Dense] = []
        self.norm_layers: List[Layer] = []
        self.dropout_layers: List[Dropout] = []
        self.projection_layers: Dict[int, Dense] = {}

        # Build hidden layers
        for units in self.hidden_units:
            self.dense_layers.append(Dense(units, activation=None))
            if self.norm == 'batch_norm':
                self.norm_layers.append(BatchNormalization())
            else:
                self.norm_layers.append(None)
            if self.dropout > 0.0:
                self.dropout_layers.append(Dropout(self.dropout))
            else:
                self.dropout_layers.append(None)

        # Representation layer
        self.repr_layer = Dense(self.repr_dim, activation=None, name='repr_layer')
        if self.norm == 'batch_norm':
            self.repr_norm_layer = BatchNormalization()
        else:
            self.repr_norm_layer = None

        # Output layer
        if self.output_dim > 0:
            self.output_layer = Dense(self.output_dim, name='output_layer')
        else:
            self.output_layer = None

    def build(self, input_shape: tf.TensorShape) -> None:
        # Projection layers for residual connections (if dimensions mismatch)
        if self.has_residuals:
            # For the first skip connection from input to first hidden layer
            if self.dense_layers[0].units != self.input_dim:
                self.input_projection = Dense(
                    self.dense_layers[0].units, use_bias=False, activation=None
                )
            else:
                self.input_projection = None

            # For subsequent skip connections
            for i in range(1, len(self.dense_layers)):
                if (i % self.skipped_layers == 0) and (self.skipped_layers > 0):
                    prev_units = self.dense_layers[i - self.skipped_layers].units
                    current_units = self.dense_layers[i].units
                    if prev_units != current_units:
                        self.projection_layers[i] = Dense(
                            current_units, use_bias=False, activation=None
                        )
                    else:
                        self.projection_layers[i] = None

            # Projection for residual to representation layer
            if self.skip_repr:
                last_residual_units = self.dense_layers[-1].units
                repr_units = self.repr_dim
                if last_residual_units != repr_units:
                    self.repr_projection = Dense(
                        repr_units, use_bias=False, activation=None
                    )
                else:
                    self.repr_projection = None
        else:
            self.input_projection = None
            self.repr_projection = None

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        x = inputs

        # First hidden layer
        x = self.dense_layers[0](x)
        if self.norm_layers[0]:
            x = self.norm_layers[0](x)
        x = self.hidden_activation(x)

        # First skip connection (from input)
        if self.has_residuals:
            if self.input_projection:
                residual = self.input_projection(inputs)
            else:
                residual = inputs
            x = Add()([x, residual])
            if self.norm == 'layer_norm':
                x = LayerNormalization()(x)
            if self.dropout_layers[0]:
                x = self.dropout_layers[0](x)
        elif self.dropout_layers[0]:
            x = self.dropout_layers[0](x)

        residual_layer = x

        # Remaining hidden layers
        for i in range(1, len(self.dense_layers)):
            x = self.dense_layers[i](x)
            if self.norm_layers[i]:
                x = self.norm_layers[i](x)
            x = self.hidden_activation(x)

            # Skip connection
            if self.has_residuals and (i % self.skipped_layers == 0):
                if residual_layer is not None:
                    if self.projection_layers.get(i):
                        residual = self.projection_layers[i](residual_layer)
                    else:
                        residual = residual_layer
                    x = Add()([x, residual])
                    if self.norm == 'layer_norm':
                        x = LayerNormalization()(x)
                    if self.dropout_layers[i]:
                        x = self.dropout_layers[i](x)
                residual_layer = x
            elif self.dropout_layers[i]:
                x = self.dropout_layers[i](x)

        # Representation layer
        x = self.repr_layer(x)
        if self.repr_norm_layer:
            x = self.repr_norm_layer(x)

        # Activation
        x = self.hidden_activation(x)

        # Residual connection to representation layer
        if self.skip_repr and self.has_residuals:
            if self.repr_projection:
                residual = self.repr_projection(residual_layer)
            else:
                residual = residual_layer
            x = Add(name='repr_layer')([x, residual])
            if self.norm == 'layer_norm':
                x = LayerNormalization()(x)
            if self.dropout > 0.0:
                x = Dropout(self.dropout)(x)
        elif self.dropout > 0.0:
            x = Dropout(self.dropout)(x)

        # Output layer
        if self.output_layer is not None:
            output = self.output_layer(x)
            return output
        else:
            return x


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
        attn_dropout (float): The dropout rate to use in the attention layer.
        attn_norm (str): The type of normalization to use in the attention layer ('batch_norm' or 'layer_norm').
        attn_skipped_layers (int): The number of layers between residual connections in the attention layer.
        output_dim (int): The dimensionality of the output.
        output_activation (str): The activation function to use in the final dense layer.
        dropout (float): The dropout rate to use in the TanhAttentiveBlock.
        norm (str): The type of normalization to use in the TanhAttentiveBlock ('batch_norm' or 'layer_norm').
        attention_block (AttentionBlock): The layer used to compute attention scores.
        dense_layer (Dense): The final dense layer that produces the output.
        attention_scores (tf.Tensor): The most recently computed attention scores.
    """

    def __init__(self,
                 attn_hidden: Optional[List[int]] = None,
                 attn_hidden_activation: str = 'leaky_relu',
                 attn_dropout: float = 0,
                 attn_norm: Optional[str] = None,
                 attn_skipped_layers: int = 2,
                 attn_skip_repr: bool = True,
                 output_dim: int = 1,
                 output_activation: Optional[str] = 'leaky_relu',
                 dropout: float = 0.0,
                 norm: Optional[str] = None,
                 a: float = 1):
        """
        Initialize the TanhAttentiveBlock.

        Args:
            attn_hidden (Optional[List[int]]): List of integers for hidden layer units in attention mechanism.
            attn_hidden_activation (str): Activation function to use in hidden layers of the attention block.
            attn_dropout (float, optional): The dropout rate to use in the attention layers. Defaults to 0.0.
            attn_norm (Optional[str], optional): The type of normalization to use ('batch_norm' or 'layer_norm'). Defaults to None.
            attn_skipped_layers (int, optional): The number of layers between residual connections. Defaults to 2.
            output_dim (int, optional): The dimensionality of the output. Defaults to 1.
            output_activation (Optional[str], optional): Activation function to use in the final dense layer. Defaults to 'leaky_relu'.
            dropout (float, optional): The dropout rate to use in the TanhAttentiveBlock. Defaults to 0.0.
            norm (Optional[str], optional): The type of normalization to use ('batch_norm' or 'layer_norm'). Defaults to None.
            a (float, optional): The parameter to scale the attention scores before applying tanh. Defaults to 0.5.
        """
        super(TanhAttentiveBlock, self).__init__()
        self.attn_hidden_units = attn_hidden or [3]
        self.attn_hidden_activation = attn_hidden_activation
        self.attn_dropout = attn_dropout
        self.attn_norm = attn_norm
        self.attn_skipped_layers = attn_skipped_layers
        self.output_dim = output_dim
        self.output_activation = output_activation
        self.dropout = dropout
        self.norm = norm
        self.a = tf.Variable(a, trainable=True, dtype=tf.float32, name='attention_scale')
        self.attention_scores = None
        self.attention_block = None
        self.dense_layer = None
        self.tanh = Activation('tanh')
        self.dropout_layer = Dropout(self.dropout) if self.dropout > 0 else None
        self.norm_layer = None
        self.skip_repr = attn_skip_repr

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
            repr_dim=input_shape[-1],
            output_dim=input_shape[-1],
            hidden_activation=self.attn_hidden_activation,
            dropout=self.attn_dropout,
            norm=self.attn_norm,
            skipped_layers=self.attn_skipped_layers,
            skip_repr=self.skip_repr
        )
        # Create the final dense layer
        self.dense_layer = Dense(self.output_dim)

        # Set the normalization layer if specified
        if self.norm == 'batch_norm':
            self.norm_layer = BatchNormalization()
        # elif self.norm == 'layer_norm':
        #     self.norm_layer = LayerNormalization()

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

        # Pass the weighted inputs through the final dense layer
        output = self.dense_layer(weighted_inputs)

        # Apply batch normalization if specified
        if self.norm == 'batch_norm':
            output = self.norm_layer(output)

        # Apply the output activation function
        if self.output_activation == 'leaky_relu':
            output = LeakyReLU(alpha=LEAKY_RELU_ALPHA)(output)
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


def create_attentive_model2_dict(
        input_dim: int = 25,
        output_dim: int = 1,
        hidden_blocks=None,
        attn_hidden_units: Optional[List[int]] = None,
        attn_hidden_activation: str = 'tanh',
        attn_dropout: float = -1,
        attn_norm: Optional[str] = None,
        attn_skipped_layers: int = 1,
        attn_skip_repr: bool = True,
        skipped_blocks: int = 1,
        repr_dim: int = 128,
        dropout: float = 0.0,
        activation='leaky_relu',
        skip_repr: bool = True,
        pds: bool = False,
        norm: str = None,
        sam_rho: float = 0.05,
        name: str = 'attentive_mlp'
) -> Model:
    """
    Create a model with stacked TanhAttentiveBlock layers, optional dropout, normalization, and residual connections.
    The final output layer is a Dense layer instead of a TanhAttentiveBlock.

    Parameters:
    - input_dim (int): The number of features in the input data. Default is 25.
    - output_dim (int): The dimension of the output layer. Default is 1 for regression tasks.
    - hidden_blocks (List[int]): List where each entry is the output_dim of the block at that position. Default is [50, 50].
    - attn_hidden_units (List[int]): A list of integers where each integer is the number of units in a hidden layer of the attention block.
    - attn_hidden_activation (str): Activation function to use in hidden layers of the attention block. Default is 'tanh'.
    - attn_dropout (float): The dropout rate to use in the attention layers. If 0.0, dropout is not used. If -1, use the same dropout rate as the model dropout rate.
    - attn_norm (str): The type of normalization to use in the attention layers ('batch_norm' or 'layer_norm'). Default is None.
    - attn_skipped_layers (int): Number of layers between residual connections in the attention layer. Default is 2.
    - attn_skip_repr (bool): Whether to add a residual connection to the representation layer in attention blocks. Default is True.
    - skipped_blocks (int): Number of blocks between residual connections. Default is 1.
    - repr_dim (int): The number of features in the final representation vector. Default is 128.
    - dropout (float): The dropout rate to use in the model. Default is 0.0.
    - activation (str): Activation function to use in the blocks. Default is 'leaky_relu'.
    - skip_repr (bool): Whether to add a residual connection to the representation layer. Default is True.
    - pds (bool): If True, use a NormalizeLayer after the representation layer. Default is False.
    - norm (str): The type of normalization to use ('batch_norm' or 'layer_norm'). Default is None.
    - sam_rho (float): Size of the neighborhood for perturbation in SAM. Default is 0.05. If 0.0, SAM is not used.
    - name (str): Name of the model. Default is 'attentive_mlp'.

    Returns:
    - Model: A Keras model instance with the following outputs:
        - 'repr': The representation layer output
        - 'output': The final model output
        - 'attention_scores': The attention scores from the last attention block
    """

    if hidden_blocks is None:
        hidden_blocks = [50, 50]

    input_layer = Input(shape=(input_dim,))
    x = input_layer

    # Process the first block
    block = TanhAttentiveBlock(
        attn_hidden=attn_hidden_units,
        attn_hidden_activation=attn_hidden_activation,
        attn_dropout=attn_dropout,
        attn_norm=attn_norm,
        attn_skipped_layers=attn_skipped_layers,
        attn_skip_repr=attn_skip_repr,
        output_dim=hidden_blocks[0],
        norm=norm,
        dropout=dropout,
        output_activation=activation
    )

    x = block(x)['output']

    # Add residual connection from input
    if x.shape[-1] != input_layer.shape[-1]:
        residual_proj = Dense(x.shape[-1], use_bias=False)(input_layer)
    else:
        residual_proj = input_layer

    x = Add()([x, residual_proj])

    if norm == 'layer_norm':
        x = LayerNormalization()(x)

    if dropout > 0:
        x = Dropout(dropout)(x)

    # Set skip_connection for future residuals
    skip_connection = x

    # Process the remaining hidden blocks
    for i, block_output_dim in enumerate(hidden_blocks[1:], start=1):
        # Create a TanhAttentiveBlock for each specified hidden block
        block = TanhAttentiveBlock(
            attn_hidden=attn_hidden_units,
            attn_hidden_activation=attn_hidden_activation,
            attn_dropout=attn_dropout,
            attn_norm=attn_norm,
            attn_skipped_layers=attn_skipped_layers,
            attn_skip_repr=attn_skip_repr,
            output_dim=block_output_dim,
            norm=norm,
            dropout=dropout,
            output_activation=activation
        )

        x = block(x)['output']

        # Implement residual connections between blocks
        if skipped_blocks > 0 and i % skipped_blocks == 0:
            if x.shape[-1] != skip_connection.shape[-1]:
                residual_proj = Dense(x.shape[-1], use_bias=False)(skip_connection)
            else:
                residual_proj = skip_connection

            x = Add()([x, residual_proj])

            if norm == 'layer_norm':
                x = LayerNormalization()(x)

            if dropout > 0:
                x = Dropout(dropout)(x)

            # Update skip_connection
            skip_connection = x

    # Add the final block with repr_dim output
    final_block = TanhAttentiveBlock(
        attn_hidden=attn_hidden_units,
        attn_hidden_activation=attn_hidden_activation,
        attn_dropout=attn_dropout,
        attn_norm=attn_norm,
        attn_skipped_layers=attn_skipped_layers,
        attn_skip_repr=attn_skip_repr,
        output_dim=repr_dim,
        norm=norm,
        dropout=dropout,
        output_activation=activation
    )

    final_output = final_block(x)
    final_repr = final_output['output']
    last_attention_scores = final_output['attention_scores']

    # Add residual connection to representation if skip_repr is True
    if skip_repr and skipped_blocks > 0:
        if final_repr.shape[-1] != skip_connection.shape[-1]:
            residual_proj = Dense(final_repr.shape[-1], use_bias=False)(skip_connection)
        else:
            residual_proj = skip_connection

        final_repr = Add(name='repr_layer')([final_repr, residual_proj])

    if norm == 'layer_norm':
        final_repr = LayerNormalization()(final_repr)

    if dropout > 0:
        final_repr = Dropout(dropout)(final_repr)

    # Handle PDS normalization if needed
    if pds:
        final_repr_output = NormalizeLayer(name='normalize_layer')(final_repr)
    else:
        final_repr_output = final_repr

    # Add output layer if output_dim > 0
    if output_dim > 0:
        output_layer = Dense(output_dim, activation=activation, name='forecast_head')(final_repr_output)
        model_output = {
            'repr': final_repr_output,
            'output': output_layer,
            'attention_scores': last_attention_scores
        }
    else:
        model_output = {
            'repr': final_repr_output,
            'attention_scores': last_attention_scores
        }

    # Create appropriate model type based on SAM parameter
    if sam_rho > 0.0:
        model = SAMModel(inputs=input_layer, outputs=model_output, rho=sam_rho, name=name)
    else:
        model = Model(inputs=input_layer, outputs=model_output, name=name)

    return model


class FeedForwardBlock(Layer):
    """
    A custom layer that applies a feed-forward network with optional residual connections,
    dropout, and normalization, similar in structure to the `create_mlp` function,
    including a skip connection from the input layer.
    """

    def __init__(
            self,
            input_dim: int,
            output_dim: int = 0,
            hiddens: Optional[List[int]] = None,
            skipped_layers: int = 1,
            repr_dim: int = 128,
            skip_repr: bool = True,
            pds: bool = False,
            activation: Optional[Callable] = None,
            norm: Optional[str] = 'batch_norm',
            dropout: float = 0.1,
            name: str = 'ff_block',
            **kwargs
    ):
        """
        Initialize the FeedForwardBlock.

        Args:
            input_dim (int): The number of features in the input data.
            output_dim (int): The dimension of the output layer. Default is 1.
            hiddens (Optional[List[int]]): List of integers for hidden layer units.
            skipped_layers (int): Number of layers between residual connections.
            repr_dim (int): The number of features in the final representation vector.
            skip_repr (bool): If True, adds a residual connection to the representation layer.
            pds (bool): If True, the model will use PDS and normalize its representations.
            activation (Optional[Callable]): Activation function to use. If None, defaults to LeakyReLU.
            norm (Optional[str]): Type of normalization to use ('batch_norm' or 'layer_norm').
            dropout (float): Dropout rate to apply after activations or residual connections.
            name (str): Name of the layer.
        """
        super(FeedForwardBlock, self).__init__(name=name, **kwargs)
        self.repr_batch_norm = None
        self.repr_residual_projection = None
        self.repr_layer_norm = None
        self.repr_dropout = None
        self.output_dense = None
        self.normalize_layer = None
        self.repr_dense = None
        self.repr_activation = None
        self.residual_projections = None
        self.dropout_layers = None
        self.activation_layers = None
        self.layer_norm_layers = None
        self.batch_norm_layers = None
        self.dense_layers = None
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hiddens = hiddens or [50, 50]
        self.skipped_layers = skipped_layers
        self.repr_dim = repr_dim
        self.skip_repr = skip_repr
        self.pds = pds
        self.activation = activation or LeakyReLU()
        self.norm = norm
        self.dropout = dropout
        self.residual = self.skipped_layers > 0

    def build(self, input_shape):
        """
        Build the feed-forward network layers.

        Args:
            input_shape (tf.TensorShape): The shape of the input tensor.
        """
        # Validate skipped_layers parameter
        if self.skipped_layers >= len(self.hiddens):
            raise ValueError(
                f"skipped_layers ({self.skipped_layers}) must be less than the number "
                f"of hidden layers ({len(self.hiddens)})"
            )

        # Initialize variables
        self.dense_layers = []
        self.batch_norm_layers = []
        self.layer_norm_layers = []
        self.activation_layers = []
        self.dropout_layers = []
        self.residual_projections = []

        previous_units = self.input_dim
        residual_layer_units = previous_units

        # Handle the first hidden layer separately to include skip from input
        # First Dense layer
        first_dense = Dense(self.hiddens[0])
        self.dense_layers.append(first_dense)

        # Batch normalization
        if self.norm == 'batch_norm':
            first_batch_norm = BatchNormalization()
            self.batch_norm_layers.append(first_batch_norm)
        else:
            self.batch_norm_layers.append(None)

        # Activation
        if callable(self.activation):
            first_activation = self.activation
        else:
            first_activation = LeakyReLU(alpha=LEAKY_RELU_ALPHA)
        self.activation_layers.append(first_activation)

        # Since we handle residuals after activation, we don't add dropout here yet

        # Residual projection from input if dimensions do not match
        if self.residual:
            if self.hiddens[0] != self.input_dim:
                first_residual_projection = Dense(self.hiddens[0], use_bias=False)
            else:
                first_residual_projection = None
            self.residual_projections.append(first_residual_projection)
        else:
            self.residual_projections.append(None)

        # Layer normalization (applied after residual connection)
        if self.norm == 'layer_norm':
            first_layer_norm = LayerNormalization()
            self.layer_norm_layers.append(first_layer_norm)
        else:
            self.layer_norm_layers.append(None)

        # Dropout after residual or activation
        if self.dropout > 0:
            first_dropout = Dropout(self.dropout)
        else:
            first_dropout = None
        self.dropout_layers.append(first_dropout)

        # Update residual_layer_units for next residual connections
        residual_layer_units = self.hiddens[0]

        # Build remaining hidden layers
        for i, units in enumerate(self.hiddens[1:], start=1):
            # Dense layer
            dense_layer = Dense(units)
            self.dense_layers.append(dense_layer)

            # Batch normalization
            if self.norm == 'batch_norm':
                batch_norm_layer = BatchNormalization()
                self.batch_norm_layers.append(batch_norm_layer)
            else:
                self.batch_norm_layers.append(None)

            # Activation
            if callable(self.activation):
                activation_layer = self.activation
            else:
                activation_layer = LeakyReLU()
            self.activation_layers.append(activation_layer)

            # Residual connection every 'skipped_layers' layers
            if self.residual and (i % self.skipped_layers == 0):
                # Projection layer if dimensions do not match
                if units != residual_layer_units:
                    projection = Dense(units, use_bias=False)
                else:
                    projection = None
                self.residual_projections.append(projection)
                residual_layer_units = units
            else:
                self.residual_projections.append(None)

            # Layer normalization
            if self.norm == 'layer_norm':
                layer_norm_layer = LayerNormalization()
                self.layer_norm_layers.append(layer_norm_layer)
            else:
                self.layer_norm_layers.append(None)

            # Dropout
            if self.dropout > 0:
                dropout_layer = Dropout(self.dropout)
            else:
                dropout_layer = None
            self.dropout_layers.append(dropout_layer)

        # Build representation layer
        self.repr_dense = Dense(self.repr_dim)
        if self.norm == 'batch_norm':
            self.repr_batch_norm = BatchNormalization()
        else:
            self.repr_batch_norm = None

        # Activation for representation layer
        if not self.skip_repr:
            if callable(self.activation):
                self.repr_activation = self.activation
            else:
                self.repr_activation = LeakyReLU(alpha=LEAKY_RELU_ALPHA)
        else:
            self.repr_activation = None

        # Residual projection for representation layer if skip_repr is True
        if self.skip_repr and self.residual:
            if self.repr_dim != residual_layer_units:
                self.repr_residual_projection = Dense(self.repr_dim, use_bias=False)
            else:
                self.repr_residual_projection = None
            if self.norm == 'layer_norm':
                self.repr_layer_norm = LayerNormalization()
            else:
                self.repr_layer_norm = None
            if self.dropout > 0:
                self.repr_dropout = Dropout(self.dropout)
            else:
                self.repr_dropout = None
        else:
            self.repr_residual_projection = None
            self.repr_layer_norm = None
            self.repr_dropout = None

        # PDS normalization
        if self.pds:
            self.normalize_layer = NormalizeLayer()
        else:
            self.normalize_layer = None

        # Output layer
        if self.output_dim > 0:
            self.output_dense = Dense(self.output_dim)
        else:
            self.output_dense = None

        super(FeedForwardBlock, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """
        Perform the forward pass of the FeedForwardBlock.

        Args:
            inputs (tf.Tensor): The input tensor.

        Returns:
            tf.Tensor: The output tensor after applying the feed-forward network.
        """
        x = inputs
        residual_layer = x if self.residual else None

        # Handle first hidden layer separately to include skip from input
        # Dense layer
        x = self.dense_layers[0](x)

        # Batch normalization
        if self.batch_norm_layers[0]:
            x = self.batch_norm_layers[0](x)

        # Activation
        x = self.activation_layers[0](x)

        # Residual connection from input
        if self.residual:
            # Project input if needed
            if self.residual_projections[0]:
                residual = self.residual_projections[0](inputs)
            else:
                residual = inputs
            x = Add()([x, residual])

            # Layer normalization
            if self.layer_norm_layers[0]:
                x = self.layer_norm_layers[0](x)

            # Dropout after residual
            if self.dropout_layers[0]:
                x = self.dropout_layers[0](x)

            # Update residual_layer
            residual_layer = x
        else:
            # Dropout after activation if no residual
            if self.dropout_layers[0]:
                x = self.dropout_layers[0](x)

        # Process remaining hidden layers
        for i in range(1, len(self.hiddens)):
            # Dense layer
            x = self.dense_layers[i](x)

            # Batch normalization
            if self.batch_norm_layers[i]:
                x = self.batch_norm_layers[i](x)

            # Activation
            x = self.activation_layers[i](x)

            # Residual connection every 'skipped_layers' layers
            if self.residual and (i % self.skipped_layers == 0):
                # Project residual_layer if needed
                if self.residual_projections[i]:
                    residual = self.residual_projections[i](residual_layer)
                else:
                    residual = residual_layer
                x = Add()([x, residual])

                # Layer normalization
                if self.layer_norm_layers[i]:
                    x = self.layer_norm_layers[i](x)

                # Dropout after residual
                if self.dropout_layers[i]:
                    x = self.dropout_layers[i](x)

                # Update residual_layer
                residual_layer = x
            else:
                # Dropout after activation if no residual
                if self.dropout_layers[i]:
                    x = self.dropout_layers[i](x)

        # Representation layer
        x = self.repr_dense(x)
        if self.repr_batch_norm:
            x = self.repr_batch_norm(x)

        # Activation or residual connection for representation layer
        if self.skip_repr:
            if callable(self.activation):
                x = self.activation(x)
            else:
                x = LeakyReLU()(x)
        else:
            if callable(self.activation):
                x = self.activation(x)
            else:
                x = LeakyReLU(name='repr_layer')(x)

        # Residual connection to representation layer
        if self.skip_repr and self.residual:
            # Project residual_layer if needed
            if self.repr_residual_projection:
                residual = self.repr_residual_projection(residual_layer)
            else:
                residual = residual_layer
            x = Add(name='repr_layer')([x, residual])

            # Layer normalization
            if self.repr_layer_norm:
                x = self.repr_layer_norm(x)

            # Dropout after residual
            if self.repr_dropout:
                x = self.repr_dropout(x)
        else:
            # Dropout after activation if no residual
            if self.dropout > 0:
                x = Dropout(self.dropout)(x)
            if self.norm == 'layer_norm':
                x = LayerNormalization()(x)

        # PDS normalization
        if self.pds and self.normalize_layer:
            final_repr_output = self.normalize_layer(x)
        else:
            final_repr_output = x

        # Output layer
        if self.output_dense:
            output = self.output_dense(final_repr_output)
            return [final_repr_output, output]
        else:
            return final_repr_output


def create_attentive_model3_dict(
        input_dim: int = 100,
        output_dim: int = 1,
        hidden_blocks=None,
        skipped_blocks: int = 1,
        dropout: float = 0.0,
        activation='leaky_relu',
        skip_repr: bool = True,
        pds: bool = False,
        norm: str = None,
        sam_rho: float = 0.05,
        name: str = 'attentive_mlp',
        # Parameters for TanhAttentiveBlock
        attn_hiddens: Optional[List[int]] = None,
        attn_skipped_layers: int = 1,
        attn_skip_repr: bool = True,
        attn_activation: str = 'leaky_relu',
        attn_dropout: float = -1,
        attn_norm: Optional[str] = 'batch_norm',
        # Parameters for FeedForwardBlock
        ff_hiddens: Optional[List[int]] = None,
        ff_skipped_layers: int = 1,
        ff_skip_repr: bool = True,
        ff_activation: Optional[Callable] = None,
        ff_dropout: float = 0.1,
        ff_norm: Optional[str] = 'batch_norm',
) -> Model:
    """
    Create a model with stacked TanhAttentiveBlock and FeedForwardBlock layers, optional dropout, normalization,
    and residual connections. The residual flow will be attentive block, then ff_block, then next layer attentive block,
    then ff_block, etc., with an initial residual connection from the input layer.

    Parameters:
    - input_dim (int): The number of features in the input data. Default is 25.
    - output_dim (int): The dimension of the output layer. Default is 1 for regression tasks.
    - hidden_blocks (List[int]): List where each entry is the output_dim of the block at that position. Default is [50, 50].
    - attn_*: Parameters for the TanhAttentiveBlock.
    - ff_*: Parameters for the FeedForwardBlock.
    - Other parameters as in create_attentive_model2_dict.

    Returns:
    - Model: A Keras model instance with the following outputs:
        - 'repr': The representation layer output.
        - 'output': The final model output.
        - 'attention_scores': The attention scores from the last attention block.
    """

    global last_attention_scores
    if hidden_blocks is None:
        hidden_blocks = [50, 50]

    input_layer = Input(shape=(input_dim,))
    x = input_layer
    skip_connection = x  # Initialize skip_connection with input_layer

    # Process the blocks
    for i, block_output_dim in enumerate(hidden_blocks):
        # --- TanhAttentiveBlock ---
        attn_block = TanhAttentiveBlock(
            attn_hidden=attn_hiddens,
            attn_hidden_activation=attn_activation,
            attn_dropout=attn_dropout if attn_dropout >= 0 else dropout,
            attn_norm=attn_norm,
            attn_skipped_layers=attn_skipped_layers,
            attn_skip_repr=attn_skip_repr,
            output_dim=block_output_dim,
            norm=norm,
            dropout=dropout,
            output_activation=activation
        )

        attn_output = attn_block(x)
        x = attn_output['output']
        last_attention_scores = attn_output['attention_scores']  # Update attention scores

        # --- FeedForwardBlock ---
        ff_block = FeedForwardBlock(
            input_dim=x.shape[-1],
            output_dim=0,  # Output_dim is 0 since it's for representation
            hiddens=ff_hiddens,
            skipped_layers=ff_skipped_layers,
            repr_dim=block_output_dim,  # repr_dim matches block_output_dim
            skip_repr=ff_skip_repr,
            activation=ff_activation or activation,  # Use ff_activation if provided
            norm=ff_norm,
            dropout=ff_dropout,
        )

        x = ff_block(x)

        # --- Residual Connection ---
        # For the first block, add a residual connection from the input layer
        if i == 0:
            if x.shape[-1] != input_layer.shape[-1]:
                residual_proj = Dense(x.shape[-1], use_bias=False)(input_layer)
            else:
                residual_proj = input_layer

            x = Add()([x, residual_proj])

            if norm == 'layer_norm':
                x = LayerNormalization()(x)

            if dropout > 0:
                x = Dropout(dropout)(x)

            # Update skip_connection
            skip_connection = x
        else:
            # Residual connections between blocks
            if skipped_blocks > 0 and (i % skipped_blocks == 0):
                if x.shape[-1] != skip_connection.shape[-1]:
                    residual_proj = Dense(x.shape[-1], use_bias=False)(skip_connection)
                else:
                    residual_proj = skip_connection

                x = Add()([x, residual_proj])

                if norm == 'layer_norm':
                    x = LayerNormalization()(x)

                if dropout > 0:
                    x = Dropout(dropout)(x)

                # Update skip_connection
                skip_connection = x

    # --- Final Representation ---
    final_repr = x

    # Optional residual connection to the final representation
    if skip_repr:
        if final_repr.shape[-1] != skip_connection.shape[-1]:
            residual_proj = Dense(final_repr.shape[-1], use_bias=False)(skip_connection)
        else:
            residual_proj = skip_connection

        final_repr = Add(name='repr_layer')([final_repr, residual_proj])

    if norm == 'layer_norm':
        final_repr = LayerNormalization()(final_repr)

    if dropout > 0:
        final_repr = Dropout(dropout)(final_repr)

    # --- PDS Normalization ---
    if pds:
        final_repr_output = NormalizeLayer(name='normalize_layer')(final_repr)
    else:
        final_repr_output = final_repr

    # --- Output Layer ---
    if output_dim > 0:
        output_layer = Dense(output_dim, activation=activation, name='forecast_head')(final_repr_output)
        model_output = {
            'repr': final_repr_output,
            'output': output_layer,
            'attention_scores': last_attention_scores
        }
    else:
        model_output = {
            'repr': final_repr_output,
            'attention_scores': last_attention_scores
        }

    # --- Model Creation ---
    if sam_rho > 0.0:
        model = SAMModel(inputs=input_layer, outputs=model_output, rho=sam_rho, name=name)
    else:
        model = Model(inputs=input_layer, outputs=model_output, name=name)

    return model
