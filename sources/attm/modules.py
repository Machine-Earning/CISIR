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
            dropout: float = 0.0,
            norm: str = None,
            skipped_layers: int = 0  # 0 means no residual, >0 means residual is active
    ):
        """
        AttentionBlock initializes a customizable attention-based neural network block.

        Parameters:
        input_dim (int): Dimensionality of input data.
        hidden_units (List[int]): List of hidden layer units.
        output_dim (int): Output dimensionality.
        hidden_activation (str): Activation function for hidden layers. Default is 'tanh'.
        dropout (float): Dropout rate. Default is 0.0 (no dropout).
        norm (str): Type of normalization ('batch_norm', 'layer_norm', or None). Default is None.
        skipped_layers (int): Number of layers to skip for residual connections. Default is 0 (no residual).
        """
        super(AttentionBlock, self).__init__()
        self.input_dim = input_dim
        self.hidden_units = hidden_units
        self.output_dim = output_dim
        self.hidden_activation = hidden_activation
        self.dropout = dropout
        self.norm = norm
        self.skipped_layers = skipped_layers
        self.dense_layers: List[Dense] = []
        self.dropout_layers: List[Dropout] = []
        self.norm_layers: List[Layer] = []
        self.projection_layers: Dict[int, Dense] = {}

        for i, units in enumerate(hidden_units):
            self.dense_layers.append(Dense(units, activation=None))
            if self.dropout > 0.0:
                self.dropout_layers.append(Dropout(self.dropout))
            if self.norm == 'batch_norm':
                self.norm_layers.append(BatchNormalization())
            elif self.norm == 'layer_norm':
                self.norm_layers.append(LayerNormalization())

        self.output_layer = Dense(output_dim)

    def build(self, input_shape: tf.TensorShape) -> None:
        for i, units in enumerate(self.hidden_units):
            if self.skipped_layers > 0 and i % self.skipped_layers == 0 and i > 0:
                self.projection_layers[i] = Dense(units, use_bias=False)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        x = inputs
        residual_layer = None

        for i, dense in enumerate(self.dense_layers):
            if self.skipped_layers > 0 and i % self.skipped_layers == 0 and i > 0:
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

            if self.dropout > 0.0:
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
                 attn_hidden_units: Optional[List[int]] = None,
                 attn_hidden_activation: str = 'leaky_relu',
                 attn_dropout: float = 0,
                 attn_norm: Optional[str] = None,
                 attn_skipped_layers: int = 2,
                 output_dim: int = 1,
                 output_activation: Optional[str] = 'leaky_relu',
                 dropout: float = 0.0,
                 norm: Optional[str] = None,
                 a: float = 1):
        """
        Initialize the TanhAttentiveBlock.

        Args:
            attn_hidden_units (Optional[List[int]]): List of integers for hidden layer units in attention mechanism.
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
        self.attn_hidden_units = attn_hidden_units or [3]
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
            dropout=self.attn_dropout,
            norm=self.attn_norm,
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

        # Pass the weighted inputs through the final dense layer
        output = self.dense_layer(weighted_inputs)

        # Optionally apply normalization - should be after the dense layer
        if self.norm_layer is not None:
            output = self.norm_layer(output)

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




def create_attentive_model_dict(
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
            attn_hidden_units=attn_hidden_units,
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
        attn_hidden_units=attn_hidden_units,
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
            attn_hidden_units=attn_hidden_units,
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

    if sam_rho > 0.0:
        model = SAMModel(inputs=input_layer, outputs=model_output, rho=sam_rho, name=name)
    else:
        model = Model(inputs=input_layer, outputs=model_output, name=name)

    return model


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
            attn_hidden_units=attn_hidden_units,
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
        attn_hidden_units=attn_hidden_units,
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
            attn_hidden_units=attn_hidden_units,
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
            attn_hidden_units=attn_hiddens,
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
        attn_hidden_units=attn_hiddens,
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


def create_attentive_model2_dict(
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
    The final output layer is a Dense layer instead of a TanhAttentiveBlock.

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
    - activation: Activation function to use in the blocks. Default is LeakyReLU.
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
            attn_hidden_units=attn_hidden_units,
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
        attn_hidden_units=attn_hidden_units,
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
        # Replace the TanhAttentiveBlock with a simple Dense layer for the output
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

    if sam_rho > 0.0:
        model = SAMModel(inputs=input_layer, outputs=model_output, rho=sam_rho, name=name)
    else:
        model = Model(inputs=input_layer, outputs=model_output, name=name)

    return model

# class FeedForwardBlock(Layer):
#     """
#     A custom layer that applies a feed-forward network with optional residual connections,
#     dropout, and normalization.

#     Attributes:
#         hidden_units (List[int]): A list of integers representing the number of units
#                                   in each hidden layer of the feed-forward network.
#         hidden_activation (str): The activation function to use in the hidden layers.
#         dropout (float): The dropout rate to use in the feed-forward network.
#         norm (str): The type of normalization to use ('batch_norm' or 'layer_norm').
#         skipped_layers (int): The number of layers between residual connections.
#         output_dim (int): The dimensionality of the output.
#         output_activation (str): The activation function to use in the output layer.
#         residual (bool): Whether to include residual connections within the feed-forward network.
#     """

#     def __init__(self,
#                  hidden_units: Optional[List[int]] = None,
#                  hidden_activation: str = 'leaky_relu',
#                  dropout: float = 0.0,
#                  norm: Optional[str] = None,
#                  skipped_layers: int = 2,
#                  output_dim: Optional[int] = None,
#                  output_activation: Optional[str] = None,
#                  **kwargs):
#         """
#         Initialize the FeedForwardBlock.

#         Args:
#             hidden_units (Optional[List[int]]): List of integers for hidden layer units.
#             hidden_activation (str): Activation function to use in hidden layers.
#             dropout (float, optional): The dropout rate to use in the feed-forward layers. Defaults to 0.0.
#             norm (Optional[str], optional): The type of normalization to use ('batch_norm' or 'layer_norm'). Defaults to None.
#             skipped_layers (int, optional): The number of layers between residual connections. Defaults to 2.
#             output_dim (Optional[int], optional): The dimensionality of the output. If None, output dimension equals input dimension.
#             output_activation (Optional[str], optional): Activation function for the output layer. Defaults to None.
#         """
#         super(FeedForwardBlock, self).__init__(**kwargs)
#         self.residual_projection = None
#         self.output_activation_layer = None
#         self.output_norm = None
#         self.output_dense = None
#         self.hidden_units = hidden_units or [64, 64]
#         self.hidden_activation = hidden_activation
#         self.dropout = dropout
#         self.norm = norm
#         self.skipped_layers = skipped_layers
#         self.output_dim = output_dim
#         self.output_activation = output_activation
#         self.layers_list = []
#         self.residual = self.skipped_layers > 0

#     def build(self, input_shape):
#         """
#         Build the feed-forward network layers.

#         Args:
#             input_shape (tf.TensorShape): The shape of the input tensor.
#         """
#         previous_units = input_shape[-1]
#         residual_input_dim = previous_units

#         # Build hidden layers
#         for idx, units in enumerate(self.hidden_units):
#             dense_layer = Dense(units)
#             self.layers_list.append(dense_layer)

#             if self.norm == 'batch_norm':
#                 norm_layer = BatchNormalization()
#                 self.layers_list.append(norm_layer)
#             elif self.norm == 'layer_norm':
#                 norm_layer = LayerNormalization()
#                 self.layers_list.append(norm_layer)

#             if self.hidden_activation == 'leaky_relu':
#                 activation_layer = LeakyReLU()
#             else:
#                 activation_layer = Activation(self.hidden_activation)
#             self.layers_list.append(activation_layer)

#             if self.dropout > 0.0:
#                 dropout_layer = Dropout(self.dropout)
#                 self.layers_list.append(dropout_layer)

#             # Implement residual connections within the feed-forward network
#             if self.residual and self.skipped_layers > 0 and (idx + 1) % self.skipped_layers == 0:
#                 # Save the index to add a residual connection later
#                 self.layers_list.append('residual')

#         # Build output layer
#         output_units = self.output_dim if self.output_dim is not None else input_shape[-1]
#         self.output_dense = Dense(output_units)

#         if self.norm == 'batch_norm':
#             self.output_norm = BatchNormalization()
#         elif self.norm == 'layer_norm':
#             self.output_norm = LayerNormalization()
#         else:
#             self.output_norm = None

#         if self.output_activation == 'leaky_relu':
#             self.output_activation_layer = LeakyReLU()
#         elif self.output_activation:
#             self.output_activation_layer = Activation(self.output_activation)
#         else:
#             self.output_activation_layer = None

#         # Prepare for residual connections
#         if self.residual and residual_input_dim != output_units:
#             self.residual_projection = Dense(output_units, use_bias=False)
#         else:
#             self.residual_projection = None

#         super(FeedForwardBlock, self).build(input_shape)

#     def call(self, inputs, **kwargs):
#         """
#         Perform the forward pass of the FeedForwardBlock.

#         Args:
#             inputs (tf.Tensor): The input tensor.

#         Returns:
#             tf.Tensor: The output tensor after applying the feed-forward network.
#         """
#         x = inputs
#         residual_connection = inputs if self.residual else None
#         skip_connection = None

#         for layer in self.layers_list:
#             if layer == 'residual' and self.residual:
#                 if skip_connection is not None:
#                     x = Add()([x, skip_connection])
#                 skip_connection = x
#             else:
#                 x = layer(x)

#         # Apply residual connection at the end if specified
#         if self.residual and skip_connection is not None:
#             if self.residual_projection is not None:
#                 residual_connection = self.residual_projection(residual_connection)
#             x = Add()([x, residual_connection])

#         # Output layer
#         x = self.output_dense(x)

#         if self.output_norm is not None:
#             x = self.output_norm(x)

#         if self.output_activation_layer is not None:
#             x = self.output_activation_layer(x)

#         return x

# def create_attentive_model3_dict(
#         input_dim: int = 25,
#         output_dim: int = 1,
#         hidden_blocks=None,
#         repr_dim: int = 128,
#         dropout: float = 0.0,
#         activation='leaky_relu',
#         pds: bool = False,
#         norm: str = None,
#         sam_rho: float = 0.05,
#         name: str = 'attentive_mlp_ff',
#         # Attention block parameters
#         attn_hidden_units: Optional[List[int]] = None,
#         attn_hidden_activation: str = 'tanh',
#         attn_dropout: float = -1,
#         attn_norm: Optional[str] = None,
#         attn_skipped_layers: int = 2,
#         # Feed-forward network parameters
#         ff_hidden_units: Optional[List[int]] = None,
#         ff_hidden_activation: str = 'leaky_relu',
#         ff_dropout: float = 0.0,
#         ff_norm: Optional[str] = None,
#         ff_skipped_layers: int = 2,
# ) -> Model:
#     """
#     Create a model where each TanhAttentiveBlock is followed by a feed-forward network block,
#     with residual connections after each block, similar to the Transformer architecture.
#     The final output layer is a Dense layer.

#     Parameters:
#     - input_dim (int): Number of features in the input data.
#     - output_dim (int): Dimension of the output layer.
#     - hidden_blocks (List[int]): Output dimensions of each block.
#     - attn_hidden_units (List[int]): Units in each hidden layer of the attention block.
#     - attn_hidden_activation (str): Activation function for the attention block's hidden layers.
#     - attn_dropout (float): Dropout rate for the attention layers.
#     - attn_norm (str): Normalization type for the attention layers ('batch_norm' or 'layer_norm').
#     - attn_skipped_layers (int): Number of layers between residual connections in the attention block.
#     - repr_dim (int): Number of features in the final representation vector.
#     - dropout (float): Dropout rate for the model.
#     - activation: Activation function to use in the blocks.
#     - pds (bool): If True, use a NormalizeLayer after the representation layer.
#     - norm (str): Normalization type ('batch_norm' or 'layer_norm').
#     - sam_rho (float): Size of the neighborhood for perturbation in SAM.
#     - name (str): Name of the model.
#     - ff_hidden_units (List[int]): Units in each layer of the feed-forward network.
#     - ff_hidden_activation (str): Activation function for the feed-forward network.
#     - ff_dropout (float): Dropout rate for the feed-forward network.
#     - ff_norm (str): Normalization type for the feed-forward network.
#     - ff_skipped_layers (int): Number of layers between residual connections in the feed-forward network.


#     Returns:
#     - Model: A Keras model instance.
#     """

#     if hidden_blocks is None:
#         hidden_blocks = [50, 50]

#     input_layer = Input(shape=(input_dim,))
#     x = input_layer

#     last_attention_scores = None  # To store the last attention scores if needed

#     for i, block_output_dim in enumerate(hidden_blocks):

#         # Save input for residual connection
#         attn_input = x

#         # Attention block
#         attn_block = TanhAttentiveBlock(
#             attn_hidden_units=attn_hidden_units,
#             attn_hidden_activation=attn_hidden_activation,
#             attn_dropout=attn_dropout,
#             attn_norm=attn_norm,
#             attn_skipped_layers=attn_skipped_layers,
#             output_dim=int(x.shape[-1]),
#             norm=norm,
#             dropout=dropout,
#             output_activation=activation
#         )

#         attn_output_dict = attn_block(x)
#         attn_output = attn_output_dict['output']
#         last_attention_scores = attn_output_dict['attention_scores']

#         # Apply residual connection after attention block
#         if attn_output.shape[-1] != attn_input.shape[-1]:
#             attn_input = Dense(attn_output.shape[-1], use_bias=False)(attn_input)
#         x = Add()([attn_output, attn_input])

#         # Apply normalization after residual addition
#         if norm == 'batch_norm':
#             x = BatchNormalization()(x)
#         elif norm == 'layer_norm':
#             x = LayerNormalization()(x)

#         # Save input for residual connection
#         ff_input = x

#         # Feed-forward block
#         ff_block = FeedForwardBlock(
#             hidden_units=ff_hidden_units,
#             hidden_activation=ff_hidden_activation,
#             dropout=ff_dropout,
#             norm=ff_norm,
#             skipped_layers=ff_skipped_layers,
#             output_dim=block_output_dim,
#             output_activation=activation
#         )
#         ff_output = ff_block(x)

#         # Apply residual connection after feed-forward block
#         if ff_output.shape[-1] != ff_input.shape[-1]:
#             ff_input = Dense(ff_output.shape[-1], use_bias=False)(ff_input)
#         x = Add()([ff_output, ff_input])

#         # Apply normalization after residual addition
#         if norm == 'batch_norm':
#             x = BatchNormalization()(x)
#         elif norm == 'layer_norm':
#             x = LayerNormalization()(x)

#     # Final representation layer
#     x = Dense(repr_dim)(x)

#     # Apply normalization if specified
#     if norm == 'batch_norm':
#         x = BatchNormalization()(x)
#     elif norm == 'layer_norm':
#         x = LayerNormalization()(x)

#     # Apply activation
#     if activation == 'leaky_relu':
#         x = LeakyReLU()(x)
#     else:
#         x = Activation(activation)(x)

#     # Apply dropout if specified
#     if dropout > 0.0:
#         x = Dropout(dropout)(x)

#     final_repr = x  # Final representation

#     if pds:
#         final_repr_output = NormalizeLayer(name='normalize_layer')(final_repr)
#     else:
#         final_repr_output = final_repr

#     # Final output layer (simple Dense layer)
#     if output_dim > 0:
#         output_layer = Dense(output_dim, activation=activation, name='forecast_head')(final_repr_output)
#         model_output = {
#             'repr': final_repr_output,
#             'output': output_layer,
#             'attention_scores': last_attention_scores
#         }
#     else:
#         model_output = {
#             'repr': final_repr_output,
#             'attention_scores': last_attention_scores
#         }

#     if sam_rho > 0.0:
#         model = SAMModel(inputs=input_layer, outputs=model_output, rho=sam_rho, name=name)
#     else:
#         model = Model(inputs=input_layer, outputs=model_output, name=name)

#     return model
