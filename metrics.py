from argparse import ArgumentError
from email.mime import message
import tensorflow as tf
# from metrics import DiceCoefficient

import numpy as np

class DiceCoefficient(tf.keras.metrics.Metric):

    def __init__(self, name='dice_coeff',**kwargs):
        super(DiceCoefficient, self).__init__(name=name, **kwargs)
        self.target_class_ids = []

    def update_state(self, y_true:tf.Tensor, y_pred:tf.Tensor, sample_weight=None):
        """
        Calculates dice metric for one batch of image-masks.
        """
        y_true = tf.convert_to_tensor(y_true)
        y_pred = tf.convert_to_tensor(y_pred)

        y_true = tf.math.argmax(y_true, axis=-1)
        y_pred = tf.math.argmax(y_pred, axis=-1)

        if y_true.shape.ndims>1:
            y_true = tf.reshape(y_true, [-1])
        if y_pred.shape.ndims>1:
            y_pred = tf.reshape(y_pred, [-1])

        self.confmat = tf.math.confusion_matrix(y_true, y_pred)

    def result(self):
        sum_over_row = tf.cast(tf.reduce_sum(self.total_cm, axis=0), dtype=self._dtype)
        sum_over_col = tf.cast(tf.reduce_sum(self.total_cm, axis=1), dtype=self._dtype)
        true_positives = tf.cast(tf.linalg.tensor_diag_part(self.total_cm), dtype=self._dtype)

        denominator = sum_over_col + sum_over_row

        true_positives = tf.gather(true_positives, self.target_class_ids)
        denominator = tf.gather(denominator, self.target_class_ids)