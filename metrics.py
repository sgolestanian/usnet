from argparse import ArgumentError
from email.mime import message
import tensorflow as tf
import numpy as np


def count_true_positives(label, prediction):
    """
    This function counts true positive values.
    """
    true_positive = tf.math.logical_and(prediction, label)
    true_positive = tf.cast(true_positive, tf.int32)
    return tf.math.reduce_sum(true_positive)

def count_true_negatives(label, prediction):
    """
    This function counts true positive values.
    """
    true_negative = tf.math.logical_and(tf.math.logical_not(prediction), tf.math.logical_not(label))
    true_negative = tf.cast(true_negative, tf.int32)
    return tf.math.reduce_sum(true_negative)

def count_false_positives(label, prediction):
    """
    This function counts true positive values.
    """
    false_positive = tf.math.logical_and(tf.math.logical_not(prediction), label)
    false_positive = tf.cast(false_positive, tf.int32)
    return tf.math.reduce_sum(false_positive)

def count_false_negatives(label, prediction):
    """
    This function counts true positive values.
    """
    false_negative = tf.math.logical_and(prediction, tf.math.logical_not(label))
    false_negative = tf.cast(false_negative, tf.int32)
    return tf.math.reduce_sum(false_negative)



class DiceCoefficient(tf.keras.metrics.Metric):

    def __init__(self, name='dice_coeff',**kwargs):
        super(DiceCoefficient, self).__init__(name=name, **kwargs)
        self.dices = []

    def update_state(self, y_true:tf.Tensor, y_pred:tf.Tensor, sample_weight=None):
        """
        Calculates dice metric for one batch of image-masks.
        """
        print(tf.shape(y_true))
        print(tf.shape(y_pred))
        for y_true_instance, y_pred_instance in zip(y_true, y_pred):
            self.dices.append(self.calculate_dice_image(y_true=y_true_instance, y_pred=y_pred_instance))
        
    def calculate_dice_image(self, y_true:tf.Tensor, y_pred:tf.Tensor) -> float:
        """
        Calculates dice metric for a single image-mask set.
        """
        num_classes = tf.shape(y_pred)[-1]
        dice_per_class = np.zeros((num_classes,))
        for class_id in range(num_classes):
            dice_per_class[class_id] = self.calculate_dice_class(y_true=y_true, y_pred=y_pred, class_id=class_id)
            print(dice_per_class[class_id])

        return tf.math.reduce_mean(dice_per_class)

    def calculate_dice_class(self, y_true:tf.Tensor, y_pred:tf.Tensor, class_id:int) -> float:
        """
        Calculates dice metric for one class of a single instance.
        """
        y_pred = tf.math.equal(y_pred, class_id)
        y_pred = tf.cast(y_pred, dtype=tf.bool)
        y_true = tf.math.equal(y_true, class_id)
        y_true = tf.cast(y_true, dtype=tf.bool)

        print(tf.shape(y_pred))
        print(tf.shape(y_true))

        tp = count_true_positives(label=y_true, prediction=y_pred)
        tn = count_true_negatives(label=y_true, prediction=y_pred)
        fp = count_false_positives(label=y_true, prediction=y_pred)
        fn = count_false_negatives(label=y_true, prediction=y_pred)

        return (2*tp)/(fn + 2*tp + fp)
        
    def reset_state(self):
        self.dices = []
        

    def result(self):
        if len(self.dices) == 0:
            return 0
        return tf.math.reduce_mean(self.dices)