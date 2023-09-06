import tensorflow as tf
from sert.models._custom_layers import TabularEncoder, SequentialEncoder
from keras_nlp.layers import TransformerEncoder

# _SERT & _SERNN have one dense hidden layer before the final layer as opposed to SERNN & SERT.


class _SERT(tf.keras.Model):
    """
    SERT (Sparse Encoder Representations from Transformers) model for sets.

    Parameters:
    - num_var: Number of variables.
    - emb_dim: Embedding dimension.
    - num_head: Number of attention heads for the Transformer Encoder.
    - ffn_dim: Dimension of the feed-forward network in the Transformer Encoder.
    - num_repeat: Number of repetitions for the Transformer Encoder block.
    - num_out: Number of output units (nodes) in the final Dense layer.
                For regression, this is typically 1. But it can be more than 1 for multi-target regression.
                For classification, it's equal to the number of classes.
    - dropout (optional): Dropout rate for the Transformer Encoder. Defaults to 0.
    - task (optional): The task type. Defaults to 'regression'. Can be 'classification' or 'regression'.
    """

    def __init__(self, num_var, emb_dim, num_head, ffn_dim, num_repeat, num_out, dropout=0, task='regression', **kwargs):
        super(_SERT, self).__init__(**kwargs)

        # Initial Tabular Encoder to process input data.
        self.input_encoder = TabularEncoder(num_var, emb_dim)
        # Sequence of Transformer Encoder blocks.
        self.transformer_encoders = [TransformerEncoder(num_heads=num_head,
                                                        intermediate_dim=ffn_dim,
                                                        dropout=dropout)
                                     for _ in range(num_repeat)]
        # Flattening layer to reshape the output of the Transformer Encoders.
        self.flatten = tf.keras.layers.Flatten()
        # Placeholder for dynamically computed dense layer.
        self.hidden_layer = None
        # Depending on the specified task, define the final dense layer.
        if task == 'regression':
            self.final_layer = tf.keras.layers.Dense(num_out)
        elif task == 'classification':
            self.final_layer = tf.keras.layers.Dense(
                num_out, activation='softmax')
        else:
            raise ValueError(
                "Invalid task. Choose 'regression' or 'classification'.")

        self.task = task

    def call(self, x):
        """
        Forward pass for the SERT model.

        Parameters:
        - x: A list containing input tensors.

        Returns:
        - x: Processed output tensor after passing through the model layers.
        """

        # Pass the input through the Tabular Encoder.
        x, padding_mask = self.input_encoder(x)
        # Sequentially pass the output through the Transformer Encoder blocks.
        for encoder in self.transformer_encoders:
            x = encoder(x, padding_mask=padding_mask)
        # Flatten the output.
        x = self.flatten(x)
        # If the hidden layer hasn't been defined, compute its size and initialize it.
        if self.hidden_layer is None:
            dense_size = int(x.shape[-1] ** 0.5)
            self.hidden_layer = tf.keras.layers.Dense(dense_size)
        # Pass the flattened embeddings through the hidden layer.
        x = self.hidden_layer(x)
        # Pass through the final Dense layer.
        x = self.final_layer(x)

        return x


class _TimeSERT(tf.keras.Model):
    """
    TimeSERT (Timeseries Sparse Encoder Representations from Transformers) model for sets with timestamps.

    Parameters:
    - num_var: Number of variables.
    - emb_dim: Embedding dimension.
    - num_head: Number of attention heads for the Transformer Encoder.
    - ffn_dim: Dimension of the feed-forward network in the Transformer Encoder.
    - num_repeat: Number of repetitions for the Transformer Encoder block.
    - num_out: Number of output units (nodes) in the final Dense layer.
                For regression, this is typically 1. But it can be more than 1 for multi-target regression.
                For classification, it's equal to the number of classes.
    - dropout (optional): Dropout rate for the Transformer Encoder. Defaults to 0.
    - task (optional): The task type. Defaults to 'regression'. Can be 'classification' or 'regression'.
    """

    def __init__(self, num_var, emb_dim, num_head, ffn_dim, num_repeat, num_out, dropout=0, task='regression', **kwargs):
        super(_TimeSERT, self).__init__(**kwargs)

        # Initial Tabular Encoder to process input data.
        self.input_encoder = SequentialEncoder(num_var, emb_dim)
        # Sequence of Transformer Encoder blocks.
        self.transformer_encoders = [TransformerEncoder(num_heads=num_head,
                                                        intermediate_dim=ffn_dim,
                                                        dropout=dropout)
                                     for _ in range(num_repeat)]
        # Flattening layer to reshape the output of the Transformer Encoders.
        self.flatten = tf.keras.layers.Flatten()
        # Placeholder for dynamically computed dense layer.
        self.hidden_layer = None
        # Depending on the specified task, define the final dense layer.
        if task == 'regression':
            self.final_layer = tf.keras.layers.Dense(num_out)
        elif task == 'classification':
            self.final_layer = tf.keras.layers.Dense(
                num_out, activation='softmax')
        else:
            raise ValueError(
                "Invalid task. Choose 'regression' or 'classification'.")

        self.task = task

    def call(self, x):
        """
        Forward pass for the SERT model.

        Parameters:
        - x: A list containing input tensors.

        Returns:
        - x: Processed output tensor after passing through the model layers.
        """

        # Pass the input through the Tabular Encoder.
        x, padding_mask = self.input_encoder(x)
        # Sequentially pass the output through the Transformer Encoder blocks.
        for encoder in self.transformer_encoders:
            x = encoder(x, padding_mask=padding_mask)
        # Flatten the output.
        x = self.flatten(x)
        # If the hidden layer hasn't been defined, compute its size and initialize it.
        if self.hidden_layer is None:
            dense_size = int(x.shape[-1] ** 0.5)
            self.hidden_layer = tf.keras.layers.Dense(dense_size)
        # Pass the flattened embeddings through the hidden layer.
        x = self.hidden_layer(x)
        # Pass through the final Dense layer.
        x = self.final_layer(x)

        return x
