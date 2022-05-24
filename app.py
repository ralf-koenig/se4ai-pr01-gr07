"""
Language identification for the three languages: German, English, Spanish.

This module app.py encapsulates a graphical user interface in a web application
served by the Python package streamlit.
"""

import streamlit as st
import os
import psycopg2

# common constants that must match between training and inference
import language_identification.constants as constants
import language_identification.inference as inference

# select "huggingface" (as in lab course)
# or select "wikipedia" (from our scraping data) for the model
# this will then be used as a sub folder in DATA_DIRECTORY and MODEL_DIRECTORY
DATA_SOURCE = "wikipedia"


# allow_output_mutation=True so to suppress the warning:
# "CachedObjectMutationWarning: Return value of load_model() was mutated between runs."
@st.cache(allow_output_mutation=True)
def load_model(data_source, filename_bare: str):
    model = inference.load_model(data_source, filename_bare)
    return model


def _detect_language(vectorize_layer, model):
    """
    Read text_input from session state, vectorization layer and model from arguments
    and run language detection. inference.opy does the heavy lifting.

    :param vectorize_layer: a KERAS TextVectorization layer
    :param model:  a KERAS model trained by training.py --train_model
    :return: language (most probable language as considered by the classifier)
            and probability via session state
    """
    text = st.session_state.text_input
    language, probability = inference.detect_language(text, vectorize_layer, model)

    st.session_state.language = language
    st.session_state.probability = probability

    # turn UI state model to next state
    st.session_state.ui_state = "render_result"


def _submit_feedback():
    """
    Insert a feedback record into a table in a database. With training.py --feedback
    this feedback can be queried from the database and put into data/feedback
    for inspection and selection and eventually using feedback for retraining the
    model later on this additional data.
    """

    # DATABASE_URL is defined at https://dashboard.heroku.com/apps/se4ai-pr01-gr07/settings
    # under Config Vars
    # For local execution on a developer machine, add an environment variable
    # DATABASE_URL to the "Run Configuration" of your IDE
    # The database URL with credentials is submitted to you via e-mail.
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
        str(st.session_state.text_input),
        str(st.session_state.language),
        float(st.session_state.probability),
        str(st.session_state.lang_hint))
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


def _get_input():
    """
    First state in the UI state model:
    Load language model and vectorization layer from the inference functions.
    Provide text box for user to input text sample.
    Provide button to start language detection.

    Wait for on_click event on button to proceed, then start .
    """
    language_model = load_model(DATA_SOURCE, "simple_mlp_novectorize.h5")
    vectorize_layer = inference.create_vectorize_layer(DATA_SOURCE, "train.csv")

    st.text_area(
        label="Please enter sample text to detect language (use 20-50 words for best results, max 1000 characters):",
        max_chars=1000,
        placeholder=None,
        disabled=False,
        key="text_input"
    )

    st.button(
        label='Detect language',
        on_click=_detect_language,
        args=(vectorize_layer, language_model)
    )


def _render_result():
    """
    Second state in UI model: Provide results of running language detection to user.
    Also provides a selection box for the user to give his/her opinion on classification quality
    and the "true" language. Waits for the user to click a button to send this feedback.
    """
    st.text_area("Your entered text", value=st.session_state.text_input, disabled=True, key="text_input")

    st.write(
        f"""The classifier considers this text to be in {constants.lang_labels[st.session_state.language]}. 
        It is {round(float(st.session_state.probability) * 100, 1)}% sure.""")

    st.subheader("Did it get it right?")

    # preselect the suggested language by the classifier also for user feedback
    selected_language = st.selectbox('What is the real language? Please select in any case.',
                                     ('Spanish', 'English', 'German'),
                                     index=list(constants.lang_labels.keys()).index(st.session_state.language))

    # turn German => "de", English => "en", Spanish => "es"
    st.session_state.lang_hint = list(constants.lang_labels.keys())[
        list(constants.lang_labels.values()).index(selected_language)]

    st.write(
        f"""You classify this text as {constants.lang_labels[st.session_state.lang_hint]}.""")
    if st.session_state.language == st.session_state.lang_hint:
        st.write(f"""Looks like the classifier got it right. What a smart fellow!""")
    else:
        st.write(f"""Oh dear, there is still a long way to go for better training data and AI.""")

    st.button('Submit feedback',
              on_click=_submit_feedback)


def _render_feedback():
    """
    Just say thank you. And wait for button click to restart with newly initialized session.
    """
    st.subheader("Your feedback")
    st.success("Thank you for your feedback. This will improve the classifier.")
    st.session_state.clear()
    st.button('Start all over!')


def main():
    """
    Main Loop of streamlit, where it acts as a one-page application.
    Uses a simple state machine model for the UI: get_input -> render_result -> render_feedback.
    Uses session variables to handle state.
    """

    # common UI elements for all screens
    st.title('SE4AI - Language Identification - Group 07')

    st.write(f"Classifier is working on processed source data from: {DATA_SOURCE}.")

    with st.sidebar:
        """This is a simple streamlit application that guesses the language of the text passed 
        out of German, English, Spanish."""

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

    # simple three-state state machine
    if st.session_state.ui_state == "get_input":
        _get_input()
    elif st.session_state.ui_state == "render_result":
        _render_result()
    elif st.session_state.ui_state == "render_feedback":
        _render_feedback()


if __name__ == "__main__":
    main()
