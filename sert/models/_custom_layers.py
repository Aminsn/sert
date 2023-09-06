"""
Continuous Value Embedding (CVE) and Tabular/Sequential Encoder Layers

This module provides implementations of the Continuous Value Embedding (CVE) layer, along with TabularEncoder and SequentialEncoder layers for encoding non-sequential tabular and sequential data, respectively. 

Classes:
- CVE: Implements the Continuous Value Embedding (CVE) layer.
- TabularEncoder: A layer for encoding tabular set input data (variable name, value).
- SequentialEncoder: A layer for encoding sequential data, considering both temporal and non-temporal information.

Usage Example:
import tensorflow as tf

# Create a Continuous Value Embedding (CVE) layer
cve_layer = CVE(hid_units=128, output_dim=64)

# Create a TabularEncoder layer
tabular_encoder = TabularEncoder(num_var=10, emb_dim=32)

# Create a SequentialEncoder layer
sequential_encoder = SequentialEncoder(num_var=20, emb_dim=64)

# Example usage of the layers
inputs = [value_tensor, var_id_tensor, category_mask_tensor]
tabular_output, tabular_padding_mask = tabular_encoder(inputs)

inputs = [time_tensor, value_tensor, var_id_tensor, category_mask_tensor]
sequential_output, sequential_padding_mask = sequential_encoder(inputs)

MIT License:

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


Author: Amin Shoari Nejad <amin.shoarinejad@gmail.com>
"""

import tensorflow as tf
import numpy as np


class CVE(tf.keras.layers.Layer):
    """
    This class follows the architecture of the Continuous Value Embedding (CVE) layer in the paper [STraTS](https://arxiv.org/abs/2107.14293).

    Parameters:
    - hid_units: number of hidden units.
    - output_dim: output dimension.
    """

    def __init__(self, hid_units, output_dim, **kwargs):
        self.hid_units = hid_units
        self.output_dim = output_dim
        super(CVE, self).__init__(**kwargs)

    def build(self, input_shape):
        """
        This method initializes the weights of the layer.
        """

        self.W1 = self.add_weight(name='CVE_W1',
                                  shape=(1, self.hid_units),
                                  initializer='glorot_uniform',
                                  trainable=True)
        self.b1 = self.add_weight(name='CVE_b1',
                                  shape=(self.hid_units,),
                                  initializer='zeros',
                                  trainable=True)
        self.W2 = self.add_weight(name='CVE_W2',
                                  shape=(self.hid_units, self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        super(CVE, self).build(input_shape)

    def call(self, x):
        """
        This method contains the logic for the layer's forward pass.
        """
        x = tf.expand_dims(x, axis=-1)
        x = tf.linalg.matmul(tf.math.tanh(tf.nn.bias_add(
            tf.linalg.matmul(x, self.W1), self.b1)), self.W2)
        return x

    def compute_output_shape(self, input_shape):
        """
        Computes the output shape of the layer.
        """
        return (input_shape[0], self.output_dim)


class TabularEncoder(tf.keras.layers.Layer):
    """
    A layer to encode tabular set input data (variable name, value).

    Parameters:
    - num_var: Number of unique variables.
    - emb_dim: Embedding dimension.
    """

    def __init__(self, num_var, emb_dim, **kwargs):
        super(TabularEncoder, self).__init__(**kwargs)

        # Continuous Value Encoder for encoding continuous values.
        self.cve = CVE(int(np.sqrt(emb_dim)), emb_dim)
        # Embedding layer for encoding variables names.
        self.embedding = tf.keras.layers.Embedding(num_var + 1, emb_dim)
        # Add layer to combine the embeddings of values and  variables names.
        self.add = tf.keras.layers.Add()
        # Lambda layer to create a mask for categorical variables (Their augmented 0 values should be masked).
        self.make_mask = tf.keras.layers.Lambda(
            lambda x: tf.clip_by_value(x, 0, 1))

    def call(self, inputs):
        """
        Forward pass for the Tabular Encoder.

        Parameters:
        - inputs: A list containing three elements:
            * value: Continuous values.
            * var_id: Indices of the variable names.
            * category_mask: A mask for categorical variables.

        Returns:
        - sum_emb: Combined embedding for continuous values and variables names.
        - padding_mask: A mask indicating padding.
        """
        # Casting inputs to float32 type.
        inputs = [tf.cast(tensor, tf.float32) for tensor in inputs]
        # Extracting individual tensors from inputs.
        value, var_id, category_mask = inputs
        # Obtaining embeddings for event values.
        values_emb = self.cve(value)
        # Casting the type and expanding dimensions of the categorical masks.
        category_mask = tf.cast(tf.expand_dims(
            category_mask, axis=-1), tf.float32)
        # Element-wise multiplication of embeddings with the category mask.
        values_emb = values_emb * category_mask
        # Obtain embeddings for variables ids.
        var_id_emb = self.embedding(var_id)
        # Combine the embeddings of event values and ids.
        sum_emb = self.add([values_emb, var_id_emb])
        # Generate a mask indicating where padding occurs.
        padding_mask = self.make_mask(var_id)

        return sum_emb, padding_mask


class SequentialEncoder(tf.keras.layers.Layer):
    """
    A layer to encode sequential data, considering both temporal and non temporal information
    (combining continuous values, time and other values, and variables names embeddings).

    Parameters:
    - num_var: Number of features.
    - emb_dim: Embedding dimension.
    """

    def __init__(self, num_var, emb_dim, **kwargs):
        super(SequentialEncoder, self).__init__(**kwargs)

        # Continuous Value Encoder for encoding continuous values.
        self.cve_value = CVE(int(np.sqrt(emb_dim)), emb_dim)
        # Continuous Value Encoder for encoding time values.
        self.cve_time = CVE(int(np.sqrt(emb_dim)), emb_dim)
        # Embedding layer for encoding variables names.
        self.embedding = tf.keras.layers.Embedding(num_var + 1, emb_dim)
        # Add layer to combine the embeddings of continuous, variables names, and time variables.
        self.add = tf.keras.layers.Add()
        # Lambda layer to create a mask for categorical variables (Their augmented 0 values should be masked).
        self.make_mask = tf.keras.layers.Lambda(
            lambda x: tf.clip_by_value(x, 0, 1))

    def call(self, inputs):
        """
        Forward pass for the Sequential Encoder.

        Parameters:
        - inputs: A list containing four elements:
            * time: Time of the event.
            * value: value of the event.
            * var_id: Index of the event.
            * category_mask: A mask for categorical variables.

        Returns:
        - sum_emb: Combined embedding for temporal, continuous, and id values.
        - padding_mask: A mask indicating padding.
        """

        # Casting inputs to float32 type.
        inputs = [tf.cast(tensor, tf.float32) for tensor in inputs]
        # Extracting individual tensors from inputs.
        time, value, var_id, category_mask = inputs
        # Obtaining embeddings for time and event values.
        time_emb = self.cve_time(time)
        values_emb = self.cve_value(value)
        # Casting the type and expanding dimensions of the categorical masks.
        category_mask = tf.cast(tf.expand_dims(
            category_mask, axis=-1), tf.float32)
        # Element-wise multiplication of embeddings with the category mask.
        values_emb = values_emb * category_mask
        # Obtain embeddings for variables ids.
        var_id_emb = self.embedding(var_id)
        # Combine the embeddings of the event time, value, and id.
        sum_emb = self.add([time_emb, values_emb, var_id_emb])
        # Generate a mask indicating where padding occurs.
        padding_mask = self.make_mask(var_id)

        return sum_emb, padding_mask
