"""
Language identification for the three languages: German, English, Spanish.

This module encapsulates common constants for model training and inference pipeline.
They need to match, so that the model works.
"""

import tensorflow as tf
import os.path

# specify directories relative to the "language_identification" folder
DATA_DIRECTORY = os.path.join("..", "data")
MODEL_DIRECTORY = os.path.join("..", "model")

# Limit for the text vectorization step, process only top 10k most frequent words
max_features = 10 * 1000

# Limit for the text vectorization step, process only the first 50 words of each text sample
sequence_length = 50

# Constants for using batches in the process of building KERAS datasets to speed things up quite a bit
batch_size = 32
AUTOTUNE = tf.data.AUTOTUNE

# Select only "es", "en" and "de" as languages
lang_list = ["es", "en", "de"]
