"""
Language identification for the three languages: German, English, Spanish.

This module encapsulates the model building functions, this means
applying supervised learning on a classification problem:
start with labeled text samples. "label" is the true language of the sample,
expressed as ISO code like en, es, de.

The model building functions will take care of the steps needed to
process the text samples and to train a proper KERAS model for
this classifier.

Needs environment variable DATABASE_URL to read user feedback from database.
For training a model, this environment variable is not needed.
"""

import os
import pandas as pd
import requests
import os.path
from sklearn import preprocessing
import sys
import psycopg2
import csv
import argparse

# suppress tensorflow CUDA warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

# append the "language_identification" folder to PYTHONPATH
# where Python searches for packages
sys.path.append('..')
# this way you can start training.py with
# python training.py -h
# inside the language_identification folder. "-h" will list all options.

# common constants that must match between training and inference
import language_identification.constants as constants


def acquire_text_sample_data_huggingface():
    """
    Get nicely uniform text samples from Huggingface.
    See: https://huggingface.co/datasets/papluca/language-identification
    """

    def download(url, target_filename):
        """Emulate wget: download a CSV file from a URL to a file in the DATA DIRECTORY."""
        filename = os.path.join(constants.DATA_DIRECTORY, target_filename)
        if not os.path.exists(filename):
            with open(filename, "wb") as f:
                f.write(requests.get(url).content)

    download("https://huggingface.co/datasets/papluca/language-identification/resolve/main/train.csv", "train.csv")
    download("https://huggingface.co/datasets/papluca/language-identification/resolve/main/valid.csv", "valid.csv")
    download("https://huggingface.co/datasets/papluca/language-identification/resolve/main/test.csv", "test.csv")


def acquire_user_feedback_from_database(to_csv_file):
    """
    Gets user feedback from the database and put to data/feedback.
    """

    # DATABASE_URL is defined at https://dashboard.heroku.com/apps/se4ai-pr01-gr07/settings
    # under Config Vars
    # For local execution on a developer machine, add an environment variable
    # DATABASE_URL to the "Run Configuration" of your IDE or in your shell.
    # The database URL with credentials is submitted to you via e-mail.
    print("Connecting to database ...")
    try:
        conn = psycopg2.connect(os.environ['DATABASE_URL'])
    except KeyError:
        print("\n[ERROR] Set environmment variable DATABASE_URL to a Postgres URI before using this routine.")
        sys.exit(1)

    # Open cursor to perform database operation
    cur = conn.cursor()
    postgres_select_query = '''SELECT language_suggested_by_user, 
                                      text_from_user_input 
                               FROM language_identification.feedback'''
    cur.execute(postgres_select_query)
    print("Fetching feedback records ...")
    db_feedback_records = cur.fetchall()

    to_csv_file = os.path.join(constants.FEEDBACK_DIRECTORY, to_csv_file)

    with open(to_csv_file, 'w', encoding='utf-8', newline='') as csv_file:
        file_writer = csv.writer(csv_file, delimiter=',')
        file_writer.writerow(['labels', 'text'])
        for record in db_feedback_records:
            file_writer.writerow([record[0], record[1]])

    print()
    print(f"{len(db_feedback_records)} feedback records written to {to_csv_file}")

    # Close communications with database
    cur.close()
    conn.close()


def preprocess_data(data_source):
    """
    Load data from CSV into Panda Dataframes and filter to the languages of interest.
    Pandas Dataframe of label (like "de", "es", "en") and text (text sample as string)

    :return: train_df: Pandas Dataframe of training data
             val_df:   Pandas Dataframe of validation data
             test_df:  Pandas Dataframe of of test data
    """

    train_df = pd.read_csv(os.path.join(constants.DATA_DIRECTORY, data_source, "train.csv"))
    val_df = pd.read_csv(os.path.join(constants.DATA_DIRECTORY, data_source, "valid.csv"))
    test_df = pd.read_csv(os.path.join(constants.DATA_DIRECTORY, data_source, "test.csv"))

    # Keep only "en", "es" and "de", drop all other text samples
    train_df = train_df.loc[train_df.labels.isin(list(constants.lang_labels.keys()))]
    val_df = val_df.loc[val_df.labels.isin(list(constants.lang_labels.keys()))]
    test_df = test_df.loc[test_df.labels.isin(list(constants.lang_labels.keys()))]

    return train_df, val_df, test_df


def extract_features(train_df, val_df, test_df):
    """
    Extract the
    :param train_df: Pandas Dataframe of training data
    :param val_df:   Pandas Dataframe of validation data
    :param test_df:  Pandas Dataframe of test data
    :return: train_ds: Tensorflow data set with labels and texts vectorized to features
             val_ds:   Tensorflow data set with labels and texts vectorized to features
             test_ds:  Tensorflow data set with labels and texts vectorized to features
    """

    ###################################################################
    # Step 1: Vectorize language labels:
    #       en, de, es => integers
    #       1a) do so for a Label encoder
    #       1b) then apply the Label encoder on the three data sets
    ###################################################################

    # Create dictionary for encoding labels
    # This is done using LabelEncoder from scikit-learn
    # It's quite useful when the number if classes is high
    le = preprocessing.LabelEncoder()
    le.fit(list(constants.lang_labels.keys()))
    num_classes = len(le.classes_)

    # BEWARE: "pop" will remove the 'labels' attribute from the data frame
    train_labels = tf.keras.utils.to_categorical(le.transform(train_df.pop('labels')), num_classes=num_classes)
    val_labels = tf.keras.utils.to_categorical(le.transform(val_df.pop('labels')), num_classes=num_classes)
    test_labels = tf.keras.utils.to_categorical(le.transform(test_df.pop('labels')), num_classes=num_classes)

    # then build Tensorflow datasets using tf.data
    raw_train_ds = tf.data.Dataset.from_tensor_slices((train_df["text"].to_list(), train_labels))  # X, y
    raw_val_ds = tf.data.Dataset.from_tensor_slices((val_df["text"].to_list(), val_labels))
    raw_test_ds = tf.data.Dataset.from_tensor_slices((test_df["text"].to_list(), test_labels))

    ###################################################################
    # Step 2: Vectorize words in the text samples
    #           => one integer for each word (limits on length and word count apply)
    ###################################################################

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
    vectorize_layer.adapt(train_df["text"].to_list())  # vectorize layer is fitted to the training data

    # Transform the "text" attribute in the Tensorflow datasets
    # from strings to integers using the vectorize_layer
    # do so for all three sets
    train_ds = raw_train_ds.map(lambda x, y: (vectorize_layer(x), y))  # returns vectorize_layer(text), label
    val_ds = raw_val_ds.map(lambda x, y: (vectorize_layer(x), y))
    test_ds = raw_test_ds.map(lambda x, y: (vectorize_layer(x), y))

    return train_ds, val_ds, test_ds


def train_model(train_ds, val_ds):
    """
    Model specification from a network of sequential layers, training, and validation.
    Before training and validation, data sets are batched and prefetched to speed things up.

    :param train_ds: Tensoflow data set of training data
    :param val_ds: Tensoflow data set of validation data
    :return: model in KERAS form, which is compiled, trained and validated
    """

    #######################################################################
    # Neural network architecture is a sequence of layers
    #######################################################################
    # Our simple architecture is defined as follows:
    # * `Embedding` layer for building a more dense and compact representation for each of the top
    #           10.000 most frequent words
    # * `Dropout` layer for improving regularization
    # * `GlobalAveragePooling1D` for returning a fixed-length output vector for each example. It averages over
    #       the sequence dimension, allowing the model to handle variable size inputs (less than 50) in a simple way.
    # * `Dropout` layer for improving regularization
    # * `Dense` layer for the logits of each class (es, en, de)

    embedding_dim = 16
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(constants.max_features + 1, embedding_dim),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(3)])

    # Specify loss, optimizer and metrics
    model.compile(loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=0.1),
                  optimizer='adam',
                  metrics=['accuracy'])

    # Applying cache techniques on training and validation data sets
    # to improve training time (training data set) and inference time (validation data set).
    # It allows tensorflow to prepare the data while it trains the model
    train_ds = train_ds.batch(batch_size=constants.batch_size)
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.batch(batch_size=constants.batch_size)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

    # Finally do training in a number of epochs
    epochs = 10
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs)
    return model


def evaluate_model(model, test_ds):
    """
    Evaluate the model data now from the *testing* set to get loss and accuracy.
    Also uses batching and prefetching to speed things up.

    :param model: KERAS model to use, which was trained before
    :param test_ds: test dataset
    :return: loss
             accuracy
    """

    # Applying cache techniques for improving inference and training time
    # It allows tensorflow to prepare the data while it trains the model
    test_ds = test_ds.batch(batch_size=constants.batch_size)
    test_ds = test_ds.prefetch(tf.data.AUTOTUNE)
    # now use KERAS function to
    loss, accuracy = model.evaluate(test_ds)
    return loss, accuracy


def save_model_to_model_directory(model, data_source, filename):
    # Save the model
    if not os.path.isdir(os.path.join(constants.MODEL_DIRECTORY, data_source)):
        os.mkdir(constants.MODEL_DIRECTORY)

    # ATTENTION: Here we are only saving the tf model.
    # Neither the Vectorize layer nor the LabelEncoder were saved.
    # they are taken care of separately, by having the *inference*
    # pipeline rebuild the Vectorize layer and LabelEncoder
    # based on common constants in language_identification.constants
    save_result = model.save(
        filepath=os.path.join(constants.MODEL_DIRECTORY, data_source, filename),
        overwrite=False
    )
    return save_result


def training_pipeline(data_source):
    train_df, val_df, test_df = preprocess_data(data_source)
    train_ds, val_ds, test_ds = extract_features(train_df, val_df, test_df)
    model = train_model(train_ds, val_ds)
    loss, accuracy = evaluate_model(model, test_ds)
    print("Loss: ", loss)
    print("Accuracy: ", accuracy)
    save_model_to_model_directory(model, data_source, "simple_mlp_novectorize.h5")


def main():
    parser = argparse.ArgumentParser(description='Language Identification - Feedback Download and Training Pipeline')

    group = parser.add_mutually_exclusive_group()
    group.add_argument("-f", "--feedback", help="add <filename.csv> to write feedback from DB to a local CSV file")
    group.add_argument("-t", "--train-model", help="train a model for one particular data source",
                       choices=['huggingface', 'wikipedia'])
    args = parser.parse_args()

    if getattr(args, 'feedback') is not None:
        filename = getattr(args, 'feedback')
        acquire_user_feedback_from_database(filename)

    if getattr(args, 'train_model') is not None:
        data_source = getattr(args, 'train_model')
        training_pipeline(data_source)

    if getattr(args, 'feedback') is None and getattr(args, 'train_model') is None:
        parser.print_help(sys.stderr)


if __name__ == "__main__":
    main()
