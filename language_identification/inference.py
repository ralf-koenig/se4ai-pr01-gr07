"""
Language identification for the three languages: German, English, Spanish.

This module encapsulates the inference functions, this means:
input a text sample in one of the three languages
and the inference functions will take care of the steps needed to
process the text sample and then to apply the pre-trained model
to get back the most probable language and its probability.
"""

import os.path

import numpy as np
import tensorflow as tf
from sklearn import preprocessing

# common constants that must match between training and inference
import language_identification.constants as constants


def load_model(data_source, filename_bare):
    """
    Load a KERAS model (H5py format) from a file in the MODEL DIRECTORY.
    Take care: you will need the vectorizer for the text (on same training data)
    and the label encoder to match the particular model file!

    :param filename_bare: the pure filename within the model directory
    :return: the saved model in KERAS representation
    """
    filename = os.path.join(constants.MODEL_DIRECTORY, data_source, filename_bare)
    saved_model = tf.keras.models.load_model(filename)
    return saved_model


def extract_features(train_df, input_text_sample):
    """
    Extract the features from a text sample. This means to apply a word => integer
    mapping on the words of the text sample in exactly the same manner as a training
    set of training words was transformed from words (strings) to integers.
    For this a vectorize layer is created with a standard strategy on word processing.

    :param train_df: a panda DataFrame with the Training data, needs the text samples in a column "text"
    :param input_text_sample: a list of text samples in either of the three languages
    :return: text in vectorized representation (as a list of integers)
    """
    # Prepare a general text vectorizer that
    # turns all text to lower case and that strips all punctuation,
    # can work up up to max_features different words (see constants)
    # returns an integer for each word
    # processes the first sequence_length words from each text sample - (see constants)
    vectorize_layer = tf.keras.layers.TextVectorization(
        standardize="lower_and_strip_punctuation",
        max_tokens=constants.max_features,
        output_mode='int',
        output_sequence_length=constants.sequence_length)

    # PRE TRAIN the vectorizer by using the adapting to the most frequent words
    # in the training dataframe's column "text".
    # The "adapt" method will chop each training sample to sequence_length,
    # then do a word absolute frequency analysis, take the max_tokens most
    # frequent words and assign integers, all other words will turn to
    # a predefined integer in KERAS with meaning "not a word from the set"
    vectorize_layer.adapt(train_df["text"].to_list())

    # now process the text_sample that was input by the user

    # make it a list, if it was one string only
    if not isinstance(input_text_sample, list):
        input_text_sample = [input_text_sample]

    # now APPLY the text vectorizer
    # send the text samples from the user through the text vectorizer
    text_in_vectorized_representation = vectorize_layer(input_text_sample)
    return text_in_vectorized_representation


def infer_from_model(model, text_in_vectorized_representation):
    """
    Runs the language detection model on a text_in_vectorized_representation and model.

    :param model: a panda DataFrame with the Training data, needs the text samples in a column "text"
    :param text_in_vectorized_representation: a list of text samples in either of the three languages
    :return: tuple of most probable language (as ISO code "de", "en", "es")
            and probability of the most probably language after applying a softmax function on the three
            probabilities for the triplet probabilities (text is spanish, text is english and text is german)
    """

    def softmax(x):
        """
        Compute softmax values for each set of scores in x.
        x must be a numpy array
        See: https://en.wikipedia.org/wiki/Softmax_function
        Source of implementation: https://www.delftstack.com/de/howto/numpy/numpy-softmax
        I changed the names to maxm, summ, e_xm, to express they are matrices.
        And to not overload Python internal names for max and sum.
        """
        maxm = np.max(x, axis=1, keepdims=True)  # returns max of each row and keeps same dims
        e_xm = np.exp(x - maxm)  # subtracts each row with its max value
        summ = np.sum(e_xm, axis=1, keepdims=True)  # returns sum of each row and keeps same dims
        return e_xm / summ

    # set up a simple Label encoder
    le = preprocessing.LabelEncoder()
    # that will translate between ["de", "en", "es"] <=> [0, 1, 2]
    le.fit(list(constants.lang_labels.keys()))  # 0=de, 1=en, 2=es, label encoder does alphabetic sorting

    # do the actual classification, will return probabilities for each of the three languages
    logits = model.predict(text_in_vectorized_representation)
    # now equalize probabilities of the three outputs using a softmax function
    probits = softmax(logits)
    # and choose the best prediction, still returning class number
    idx_predictions = np.argmax(probits, axis=1)

    # turn the class number 0, 1, 2 to a class name like de, en, es
    language = list(le.inverse_transform(idx_predictions))[0]

    # get best probability of the three, after softmax
    probability = np.max(probits, axis=1)[0]

    return language, probability
