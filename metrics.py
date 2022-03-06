from argparse import ArgumentError
from email.mime import message
import tensorflow as tf


def count_true_positive(label, prediction):
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

    def __init__(self, class_id=0, name='dice_coeff',**kwargs):
        super(DiceCoefficient, self).__init__(name=name, **kwargs)
        self.sub_metrics = [
                            tf.keras.metrics.Precision(name='precision'),
                            tf.keras.metrics.Recall(name='recall')
        ]
        self.class_id = class_id
        self.true_positive = tf.Variable(0)
        self.false_positive = tf.Variable(0)
        self.true_negative = tf.Variable(0)
        self.false_negative = tf.Variable(0)
        self.dice_values = tf.Variable(0)
        self.batch_dices = []

    def update_state(self, y_true, y_pred, sample_weight=None):
        pred_mask = tf.argmax(y_pred, axis=-1)
        pred_mask = tf.math.equal(pred_mask, self.class_id)
        label_mask = tf.argmax(y_true, axis=-1)
        label_mask = tf.math.equal(label_mask, self.class_id)
        
        self.true_positive.assign_add(count_true_positive(label=label_mask, prediction=pred_mask))
        self.false_positive.assign_add(count_false_positives(label=label_mask, prediction=pred_mask))
        self.true_negative.assign_add(count_true_negatives(label=label_mask, prediction=pred_mask))
        self.false_negative.assign_add(count_false_negatives(label=label_mask, prediction=pred_mask))
        

    def calculate_dice(self):
        return (2*self.true_positive)/(2*self.true_positive + self.false_negative + self.false_positive)
        

    def reset_state(self):
        self.true_positive.assign(0)
        self.false_positive.assign(0)
        self.true_negative.assign(0)
        self.false_negative.assign(0)
        

    def result(self):
        return self.calculate_dice()