import tensorflow as tf
from sert.models._custom_layers import TabularEncoder, SequentialEncoder


class SERNN(tf.keras.Model):
    """
    SERNN (Sparse Encoder Representations from Neural Nets) model for sets.

    Parameters:
    - num_var: Number of variables.
    - emb_dim: Embedding dimension.
    - num_out: Number of output nodes.
               For regression, this is typically 1. But it can be more than 1 for multi-target regression.
               For classification, it's equal to the number of classes.
    - task (str): Specifies the type of prediction task ('regression' or 'classification').
    """

    def __init__(self, num_var, emb_dim, num_out, task='regression', **kwargs):
        super(SERNN, self).__init__(**kwargs)

        # Encoder to encode input data into dense representations.
        self.input_encoder = TabularEncoder(num_var, emb_dim)
        # Flatten layer to flatten embeddings.
        self.flatten = tf.keras.layers.Flatten()
        # Placeholder for dynamically computed dense layer.
        self.hidden_layer = None
        # Define the final layer based on the specified task.
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
        Forward pass for the SERNN model.

        Parameters:
        - x: Input data consisting of continuous values, categorical indices, and a mask.

        Returns:
        - output: Model's predictions based on the input data.
        """

        # Encode the input data.
        x, padding_mask = self.input_encoder(x)
        # Apply the padding mask.
        x = tf.multiply(x, tf.cast(tf.expand_dims(
            padding_mask, axis=-1), tf.float32))
        # Flatten the embeddings.
        x = self.flatten(x)
        # If the hidden layer hasn't been defined, compute its size and initialize it.
        if self.hidden_layer is None:
            dense_size = int(x.shape[-1] ** 0.5)
            self.hidden_layer = tf.keras.layers.Dense(dense_size)
        # Pass the flattened embeddings through the hidden layer.
        x = self.hidden_layer(x)
        # Compute the output.
        x = self.final_layer(x)

        return x


class TimeSERNN(tf.keras.Model):
    """
    TimeSERNN (Timeseries Sparse Encoder Representations from Neural Nets) model for sets with timestamps.

    Parameters:
    - num_var: Number of variables.
    - emb_dim: Embedding dimension.
    - num_out: Number of output nodes.
               For regression, this is typically 1. But it can be more than 1 for multi-target regression.
               For classification, it's equal to the number of classes.
    - task (str): Specifies the type of prediction task ('regression' or 'classification').
    """

    def __init__(self, num_var, emb_dim, num_out, task='regression', **kwargs):
        super(TimeSERNN, self).__init__(**kwargs)

        # Encoder to encode input data into dense representations.
        self.input_encoder = SequentialEncoder(num_var, emb_dim)
        # Flatten layer to flatten embeddings.
        self.flatten = tf.keras.layers.Flatten()
        # Placeholder for dynamically computed dense layer.
        self.hidden_layer = None
        # Define the final layer based on the specified task.
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
        Forward pass for the SERNN model.

        Parameters:
        - x: Input data consisting of continuous values, categorical indices, and a mask.

        Returns:
        - output: Model's predictions based on the input data.
        """

        # Encode the input data.
        x, padding_mask = self.input_encoder(x)
        # Apply the padding mask.
        x = tf.multiply(x, tf.cast(tf.expand_dims(
            padding_mask, axis=-1), tf.float32))
        # Flatten the embeddings.
        x = self.flatten(x)
        # If the hidden layer hasn't been defined, compute its size and initialize it.
        if self.hidden_layer is None:
            dense_size = int(x.shape[-1] ** 0.5)
            self.hidden_layer = tf.keras.layers.Dense(dense_size)
        # Pass the flattened embeddings through the hidden layer.
        x = self.hidden_layer(x)
        # Compute the output.
        x = self.final_layer(x)

        return x
