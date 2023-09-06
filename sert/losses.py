"""
losses.py

This module provides custom loss functions for use with TensorFlow/Keras models.

Functions:
    - WeightedCrossentropy: Custom loss function to apply weights to different classes
      in a categorical crossentropy setting.
    - MaskedMSE: Custom mean squared error loss function where certain values
      can be masked and excluded from contributing to the loss.

Author:
    - Amin Shoari Nejad <amin.shoarinejad@gmail.com>

"""

import tensorflow as tf


class WeightedCrossentropy(tf.keras.losses.Loss):
    """
    This class implements a weighted categorical crossentropy loss.

    Args:
    - class_weights (list or array): Weights for each class. Its length should
      be equal to the number of classes.

    Methods:
    - call(y_true, y_pred): Computes the loss between true and predicted labels.

    Example:
        loss_fn = WeightedCrossentropy(class_weights=[1, 2, 3])
        loss = loss_fn(y_true, y_pred)
    """

    def __init__(self, class_weights, name='weighted_categorical_crossentropy'):
        super(WeightedCrossentropy, self).__init__(name=name)
        self.class_weights = class_weights

    def call(self, y_true, y_pred):
        """
        Computes the weighted categorical crossentropy loss between true and predicted labels.

        Args:
        - y_true (Tensor): True labels.
        - y_pred (Tensor): Predicted labels.

        Returns:
        - loss (Tensor): Weighted categorical crossentropy loss value.
        """

        class_idx = tf.argmax(y_true, axis=-1)
        # Ensure indices are int32
        class_idx = tf.cast(class_idx, dtype=tf.int32)
        sample_weights = tf.gather(self.class_weights, class_idx)
        cce = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
        return tf.reduce_mean(tf.cast(sample_weights, dtype=tf.float32) * cce, axis=-1)


class MaskedMSE(tf.keras.losses.Loss):
    """
    This class implements a masked mean squared error (MSE) loss.

    Methods:
    - call(y_true, y_pred): Computes the masked MSE between true and predicted labels.

    Example:
        loss_fn = MaskedMSE()
        loss = loss_fn(y_true, y_pred)
    """

    def __init__(self, name='masked_mean_squared_error'):
        super(MaskedMSE, self).__init__(name=name)

    def call(self, y_true, y_pred):
        """
        Computes the masked mean squared error loss between true and predicted labels.

        Args:
        - y_true (Tensor): True labels combined with a mask in the last dimension.
        - y_pred (Tensor): Predicted labels.

        Returns:
        - loss (Tensor): Masked mean squared error loss value.
        """

        y_true = tf.cast(y_true, tf.float32)
        y_true, mask_array = tf.unstack(y_true, axis=-1)
        squared_diffs = tf.square(y_true - y_pred)
        squared_diffs = tf.multiply(squared_diffs, mask_array)
        loss_sum = tf.reduce_sum(squared_diffs)
        mask_sum = tf.reduce_sum(mask_array)
        return loss_sum / mask_sum
