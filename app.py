"""
Language identification for the three languages: German, English, Spanish.

This module app.py encapsulates a graphical user interface in a web application served by the Python package streamlit.
"""

import os.path
import streamlit as st
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn import preprocessing
import os
import psycopg2

# define two constants
lang_labels = {"es": "Spanish", "en": "English", "de": "German"}
lang_list = ["es", "en", "de"]


# TODO: change to use inference pipeline here from the language_identification package
# import language_identification.inference

# allow_output_mutation=True so to suppress the warning:
# "CachedObjectMutationWarning: Return value of load_model() was mutated between runs."
@st.cache(allow_output_mutation=True)
def load_model(filename: str):
    """
    Loads a H5 model file from the file system. This was built during training.
    """
    if filename == '':
        return None
    else:
        # Load the model
        model_directory = "model/huggingface"
        path = os.path.join(model_directory, filename)

        try:
            saved_model = tf.keras.models.load_model(path)
            return saved_model
        except OSError:
            st.error("Could not open/read file:", path)


# cannot be cached due to object type, would need hash method for caching in streamlit
def create_vectorize_layer():
    """
    Creates the vectorize layer that is needed for the input text to be converted to numbers.
    """
    max_features = 10000  # only top 10K most frequent words will be used in the vectorization of the text
    sequence_length = 50  # up to 50 word will be considered from each text sample

    # create a layer that processes words to integer numbers
    vectorize_layer = tf.keras.layers.TextVectorization(
        standardize="lower_and_strip_punctuation",
        max_tokens=max_features,
        output_mode='int',
        output_sequence_length=sequence_length)

    data_directory = "data//huggingface"
    # load training data in to Pandas data frame
    train_df = pd.read_csv(os.path.join(data_directory, "train.csv"))

    # filter training_data to the languages of choice
    train_df = train_df.loc[train_df.labels.isin(lang_list)]

    # create mappings: word to integer, on the same training data,
    # but it will now be used to vectorize the text, input by the user
    # then to be forwarded to the inference model
    vectorize_layer.adapt(train_df["text"].to_list())
    return vectorize_layer


def _detect_language(model, vectorize_layer):
    """
    Detects the actual language of the input text.
    The input text itself is read from the session state.
    Output is returned via session state variables as well.
    """

    # softmax to make a nice distribution between 0 and 1
    def softmax(x):
        """Compute softmax values for each set of scores in x."""
        maxm = np.max(x, axis=1, keepdims=True)  # returns max of each row and keeps same dims
        e_xm = np.exp(x - maxm)  # subtracts each row with its max value
        summ = np.sum(e_xm, axis=1, keepdims=True)  # returns sum of each row and keeps same dims
        return e_xm / summ

    text = st.session_state.text_input
    text_vectorized = vectorize_layer([text])

    # do the actual classification, will return probabilities for each of the three languages
    logits = model.predict(text_vectorized)
    # now equalize probabilities to a softmax function
    probits = softmax(logits)
    # and choose the best prediction
    idx_predictions = np.argmax(probits, axis=1)

    # needed to reverse the output from the classifier, which is an integer for the class to a language name
    le = preprocessing.LabelEncoder()
    le.fit(lang_list)

    # get back from a class number like 0, 1, 2 to a class name like de, en, es
    language = list(le.inverse_transform(idx_predictions))[0]

    # get best probability of the three, after softmax
    probability = np.max(probits, axis=1)[0]

    st.session_state.language = language
    st.session_state.probability = probability

    # turn UI state model to next state
    st.session_state.ui_state = "render_result"


def submit_feedback():
    """
    Insert a record into a table in a database that collects user feedback.
    """

    # DATABASE_URL is defined at https://dashboard.heroku.com/apps/se4ai-pr01-gr07/settings
    # under Config Vars
    conn = psycopg2.connect(os.environ['DATABASE_URL'])

    # Open cursor to perform database operation
    cur = conn.cursor()
    postgres_insert_query = '''
        INSERT INTO language_identification.feedback(text_from_user_input, 
                                                    language_by_classifier, 
                                                    probability_by_classifier, 
                                                    language_suggested_by_user)
        VALUES (%s, %s, %s, %s )
    '''
    record_to_insert = (
        st.session_state.text_input,
        st.session_state.language,
        float(st.session_state.probability),
        st.session_state.lang_hint)
    cur.execute(postgres_insert_query, record_to_insert)
    conn.commit()

    # Close communications with database
    cur.close()
    conn.close()

    # This will insert something like: "My text", "de", "0.23232", "en".
    # So the classifier regarded this "My text" as German with 0.23 probability, but the user considers it English
    # A record can also support the classifier like "muchas gracias", "es", "0.394", "es"

    # turn UI state model to next state
    st.session_state.ui_state = "render_feedback"


def get_input():
    # load pickled language model
    language_model = load_model("simple_mlp_novectorize.h5")
    # create the necessary vectorize layer for the text input
    vectorize_layer = create_vectorize_layer()

    st.text_area(
        "Please enter sample text to detect language (use 20-50 words for best results):",
        placeholder=None,
        disabled=False,
        key="text_input"
    )

    st.button(
        label='Detect language',
        on_click=_detect_language,
        args=(language_model, vectorize_layer)
    )


def render_result():
    st.text_area("Your entered text", value=st.session_state.text_input, disabled=True, key="text_input")

    st.write(
        f"""The classifier considers this text to be in {lang_labels[st.session_state.language]}. 
        It is {round(float(st.session_state.probability) * 100, 1)}% sure.""")

    st.subheader("Did it get it right?")

    selected_language = st.selectbox('What is the real language? Please select in any case.',
                                     ('Spanish', 'English', 'German'),
                                     index=list(lang_labels.keys()).index(st.session_state.language))

    st.session_state.lang_hint = list(lang_labels.keys())[list(lang_labels.values()).index(selected_language)]

    st.write(
        f"""You classify this text as {lang_labels[st.session_state.lang_hint]}.""")
    if st.session_state.language == st.session_state.lang_hint:
        st.write(f"""Looks like the classifier got it right. What a smart fellow!""")
    else:
        st.write(f"""Oh dear, there is still a long way to go for better training data and AI.""")

    st.button('Submit feedback',
              on_click=submit_feedback)


def render_feedback():
    st.subheader("Your feedback")

    # use for debugging only
    # st.write(
    #     st.session_state.text_input,
    #     st.session_state.language,
    #     st.session_state.probability,
    #     st.session_state.lang_hint
    # )

    st.success("Thank you for your feedback. This will improve the classifier.")
    st.session_state.clear()
    st.button('Start all over!')


def main():
    # The next two UI elements are common for all screens
    st.title('SE4AI - Language Identification - Group 07')

    with st.sidebar:
        """
        This is a simple streamlit application that guesses the language of the text passed out of German, 
        English, Spanish.
        """

    # initialize session state variables
    if "text_input" not in st.session_state:
        st.session_state.text_input = ""
    if "language" not in st.session_state:
        st.session_state.language = ""
    if "probability" not in st.session_state:
        st.session_state.probability = ""
    if "lang_hint" not in st.session_state:
        st.session_state.lang_hint = ""
    if "ui_state" not in st.session_state:
        st.session_state.ui_state = "get_input"

    ################################################################################################
    # Use a very simple state machine model for the UI: get_input -> render_result -> render_feedback
    ################################################################################################

    if st.session_state.ui_state == "get_input":
        get_input()
    elif st.session_state.ui_state == "render_result":
        render_result()
    elif st.session_state.ui_state == "render_feedback":
        render_feedback()


if __name__ == "__main__":
    main()
