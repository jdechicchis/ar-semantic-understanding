"""
Custom evaluation metrics for model.
"""

import tensorflow as tf
import numpy as np
from tensorflow.python.keras import backend as K

def categorical_crossentropy_with_sample_weights(sample_weights):
    loss = tf.keras.losses.CategoricalCrossentropy()
    def apply(y_true, y_pred):
        return loss(y_true, y_pred, sample_weight=sample_weights)
    return apply

def categorical_accuracy_with_sample_weights(sample_weights):
    metric = tf.keras.metrics.CategoricalAccuracy()
    def apply(y_true, y_pred):
        return metric(y_true, y_pred, sample_weight=sample_weights)
    return apply

def custom_f1_score(y_true, y_pred):
    """
    Calculate f1-score.
    """
    return 2 * (K.sum(y_true * y_pred)+ K.epsilon()) / (K.sum(y_true) + K.sum(y_pred) + K.epsilon())

def weighted_categorical_crossentropy(class_weights):
    """
    Calculate weighted categorical crossentropy.
    """
    class_weights = K.variable(np.array(class_weights))

    def apply(y_true, y_pred):
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        # clip to prevent NaN's and Inf's
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        # calc
        loss = y_true * K.log(y_pred) * class_weights
        loss = -K.sum(loss, -1)
        return loss
    return apply

def calculate_recall(y_true, y_pred):
    """
    Calculate the recall (i.e. true positive rate).
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def calculate_true_negative_rate(y_true, y_pred):
    """
    Calculate the true negative rate.
    """
    #TODO: implement
    #print(y_true.shape)
    true_negatives = K.sum(K.round(-1 * K.clip(y_true * y_pred, 0, 1) + 1))
    possible_negatives = (BATCH_SIZE * HEIGHT * WIDTH * NUM_CLASSES) -  K.sum(K.round(K.clip(y_true, 0, 1)))
    true_negative_rate = true_negatives / (possible_negatives + K.epsilon())
    #print("\n\ntrue_negatives: {}\npossible_negatives: {}\ntrue_negative_rate: {}\n\n".format(true_negatives, possible_negatives, true_negative_rate))
    return true_negative_rate

def calculate_precision(y_true, y_pred):
    """
    Calculate precision.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_score(y_true, y_pred):
    """
    Calculate f1-score.
    """
    precision = calculate_precision(y_true, y_pred)
    recall = calculate_recall(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))

#@tf.function
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

        #recall = tf.keras.metrics.Recall(y_true, y_pred)

    #assert true_positive_rate == recall, "TPR and recall must match!"

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