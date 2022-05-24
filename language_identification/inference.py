"""
Language identification for the three languages: German, English, Spanish.

This module inference.py encapsulates the inference functions, this means:
input a text sample in one of the languages
and the inference functions will take care of the steps needed to process the text
sample and then to apply the pre-trained model to get back the most probable
language and its probability.
"""

import tensorflow as tf
import os
import numpy as np
import pandas as pd
from sklearn import preprocessing

import language_identification.constants as constants

# data directory to the training data, from where the vectorizer will be built
DATA_DIRECTORY = "data"

# model directory from where the model is read
MODEL_DIRECTORY = "model"


def load_model(data_source, filename_bare: str):
    """
    Loads a H5 model file from the file system. This was built during training with training.py --train-model

    Load a KERAS model (H5py format) from a file in the MODEL DIRECTORY, built during training with
    training.py --train-model.
    Take care: you will need the vectorizer for the text (on same training data)
    and the label encoder to match the particular model file!

    :param data_source: subdirectory within the model directory
    :param filename_bare: pure filename of a model file within the model directory
    :return: the saved model in KERAS representation
    """
    # check for file
    path = os.path.join(MODEL_DIRECTORY, data_source, filename_bare)
    if os.path.exists(path) and filename_bare != '' and filename_bare.endswith(".h5"):
        # load the model
        saved_model = tf.keras.models.load_model(path)
        return saved_model
    else:
        return None


def create_vectorize_layer(data_source, filename_bare: str):
    """
    Creates the vectorize layer that is needed for the input text to be converted to numbers.

    Extract the features from a text sample. This means to apply a word => integer
    mapping on the words of the text sample in exactly the same manner as a training
    set of training words was transformed from words (strings) to integers.
    For this a vectorize layer is created with a standard strategy on word processing.

    :param data_source: data_source: subdirectory within the data directory
    :param filename_bare: a list of text samples in either of the three languages
    :return: adapted vectorization layer, this means a layer trained to the data
    """
    # check for file
    path = os.path.join(DATA_DIRECTORY, data_source, filename_bare)
    if os.path.exists(path) and filename_bare != '':
        # load training data in to Pandas data frame
        train_df = pd.read_csv(path)
        # filter training_data to the languages of choice
        train_df = train_df.loc[train_df.labels.isin(list(constants.lang_labels.keys()))]

        # Prepare a general text vectorizer that
        # turns all text to lower case and that strips all punctuation,
        # can work up up to max_features different words (see constants)
        # returns an integer for each word
        # processes the first sequence_length words from each text sample - (see constants)
        vectorize_layer = tf.keras.layers.TextVectorization(
            standardize="lower_and_strip_punctuation",
            max_tokens=constants.max_features,
            output_mode='int',
            output_sequence_length=constants.sequence_length
        )
        # PRE ADAPT the vectorizer by adapting to the most frequent words
        # in the training dataframe's column "text".
        # The "adapt" method will chop each training sample to sequence_length,
        # then do a word absolute frequency analysis, take the max_tokens most
        # frequent words and assign integers, all other words will turn to
        # a predefined integer in KERAS with meaning "not a word from the set"
        vectorize_layer.adapt(train_df["text"].to_list())
        return vectorize_layer
    else:
        return None


def detect_language(text:str, vectorize_layer, model):
    """
    Runs the language detection model on a text input using the vectorize layer and model.

    :param text: a text samples in either of the languages
    :param vectorize_layer: an adapted vectorize layer from KERAS
    :param model: a KERAS model trained before
    :return: tuple of most probable language (as ISO code "de", "en", "es")
            and probability of the most probably language after applying a softmax function on the three
            probabilities for the triplet probabilities (text is spanish, text is english and text is german).
    """

    def softmax(x):
        """
        Compute softmax values for each set of scores in x. This scales the probabilities from the
        result classes of the classifier to a nice distribution between 0 and 1.

        See: https://en.wikipedia.org/wiki/Softmax_function
        Source of implementation: https://www.delftstack.com/de/howto/numpy/numpy-softmax
        I changed the names to maxm, summ, e_xm, to express they are matrices.
        And to not overload Python internal names for max and sum.

        :param x: must be a numpy array
        """
        maxm = np.max(x, axis=1, keepdims=True)  # returns max of each row and keeps same dims
        e_xm = np.exp(x - maxm)  # subtracts each row with its max value
        summ = np.sum(e_xm, axis=1, keepdims=True)  # returns sum of each row and keeps same dims
        return e_xm / summ

    # APPLY the text vectorizer
    # send the text sample from the user as a list with one item through the text vectorizer
    text_in_vectorized_representation = vectorize_layer([text])

    # do the actual classification, will return probabilities for each of the three languages
    logits = model.predict(text_in_vectorized_representation)
    # now equalize probabilities to a softmax function
    probits = softmax(logits)

    # get best probability of the three, after softmax
    probability = np.max(probits, axis=1)[0]

    # and choose the best prediction
    idx_predictions = np.argmax(probits, axis=1)

    # needed to reverse the output from the classifier, which is an integer for the class to a language name
    le = preprocessing.LabelEncoder()
    le.fit(list(constants.lang_labels.keys()))

    # back transformation from a class number like 0, 1, 2 to a class name like de, en, es
    language = list(le.inverse_transform(idx_predictions))[0]

    return language, probability
