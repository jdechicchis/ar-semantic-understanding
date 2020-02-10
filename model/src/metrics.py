"""
Custom evaluation metrics for model.
"""

import tensorflow as tf
import numpy as np
from tensorflow.python.keras import backend as K

def weighted_categorical_crossentropy(class_weights):
    """
    Calculate weighted categorical crossentropy.
    """
    class_weights = K.variable(np.array(class_weights))

    def apply(y_true, y_pred):
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        # Clip to prevent NaN's and Inf's
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        # Calculate
        loss = y_true * K.log(y_pred) * class_weights
        loss = -K.sum(loss, -1)
        return loss
    return apply

def f1_score():
    """
    Calculate f1-score = 2 * ((precision * recall) / (precision + recall)).
    """
    precision = tf.keras.metrics.Precision()
    recall = tf.keras.metrics.Recall()

    def custom_f1_score(y_true, y_pred):
        p = precision(y_true, y_pred)
        r = recall(y_true, y_pred)
        return 2 * ((p * r) / (p + r + K.epsilon()))

    return custom_f1_score

def balanced_accuracy():
    """
    Calculate balanced_accuracy = (TPR + TNR) / 2.
    """
    true_positives_metric = tf.keras.metrics.TruePositives()
    false_positives_metric = tf.keras.metrics.FalsePositives()
    true_negatives_metric = tf.keras.metrics.TrueNegatives()
    false_negatives_metric = tf.keras.metrics.FalseNegatives()
    def custom_balanced_accuracy(y_true, y_pred):
        true_positives = true_positives_metric(y_true, y_pred)
        false_positives = false_positives_metric(y_true, y_pred)
        true_negatives = true_negatives_metric(y_true, y_pred)
        false_negatives = false_negatives_metric(y_true, y_pred)

        true_positive_rate = true_positives / (true_positives + false_negatives)
        true_negative_rate = true_negatives / (false_positives + true_negatives)

        return (true_positive_rate + true_negative_rate) / 2

    return custom_balanced_accuracy

def iou(y_true, y_pred):
    """
    intersection = np.logical_and(target, prediction)
    union = np.logical_or(target, prediction)
    iou_score = np.sum(intersection) / np.sum(union)
    """
    intersection = K.round(K.clip(y_true * y_pred, 0, 1))
    union = K.round(K.clip(y_true + y_pred, 0, 1))
    return K.sum(intersection) / K.sum(union)
