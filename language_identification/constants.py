"""
Language identification for the three languages: German, English, Spanish.

This module encapsulates common constants for model training and inference pipeline.
They need to match, so that the system works.
"""

import os.path

#############################################################################
# LANGUAGES
#############################################################################
# Language selection:  "es", "en" and "de"
# influences scraping.py to scrape articles from Wikipedia of only these languages
# influences filter on the Huggingface data set while training
# influences Label Encoder in training to build the classes
# influences Label Encoder in inference to inverse the classes
# influences User Interface in App
lang_labels = {"es": "Spanish", "en": "English", "de": "German"}

#############################################################################
# DIRECTORIES
#############################################################################
# specify directories relative to the "language_identification" folder

# specify directory where all data files will be put in subdirectories: downloaded data, scraping results
DATA_DIRECTORY = os.path.join("..", "data")

# specify directory where all model files will be put in subdirectories
MODEL_DIRECTORY = os.path.join("..", "model")

# directory where user feedback from the database will by put by
# "python training.py --feedback <filename.csv>"
FEEDBACK_DIRECTORY = os.path.join("..", "data", "feedback")

#############################################################################
# PARAMETERS ON TRAINING AND INFERENCE
#############################################################################

# Limit for the text vectorization step, process only top 10k most frequent words
max_features = 10 * 1000

# Limit for the text vectorization step, process only the first 50 words of each text sample
sequence_length = 50

# Constants for using batches in the process of building KERAS datasets to speed things up quite a bit
batch_size = 32
