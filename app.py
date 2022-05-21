import streamlit as st
import os.path
import sys
import tensorflow as tf
import numpy as np
from sklearn import preprocessing
import pandas as pd


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
        model_directory = "model"
        path = os.path.join(model_directory, filename)

        try:
            saved_model = tf.keras.models.load_model(path)
            return saved_model
        except OSError:
            print("Could not open/read file:", path)
            sys.exit()


# cannot be cached due to object type, would need hash method for caching in streamlit
def create_vectorize_layer():
    """
    Creates the vectorize layer that is needed for the input text to be converted to numbers.
    """
    max_features = 10000  # top 10K most frequent words
    sequence_length = 50  # We defined it in the previous data exploration section

    vectorize_layer = tf.keras.layers.TextVectorization(
        standardize="lower_and_strip_punctuation",
        max_tokens=max_features,
        output_mode='int',
        output_sequence_length=sequence_length)

    data_directory = "data"

    train_df = pd.read_csv(os.path.join(data_directory, "train.csv"))
    vectorize_layer.adapt(train_df["text"].to_list())  # vectorize layer is fitted to the training data
    return vectorize_layer


def _detect_language(model, vectorize_layer):
    """
    Detects the actual language of the input text.
    The input text itself is read from the session state.
    Output is returned via session state as well.
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

    logits = model.predict(text_vectorized)
    probits = softmax(logits)
    idx_predictions = np.argmax(probits, axis=1)

    le = preprocessing.LabelEncoder()
    lang_list = ["es", "en", "de"]
    le.fit(lang_list)

    probability = np.max(probits, axis=1)[0]
    language = list(le.inverse_transform(idx_predictions))[0]

    st.session_state.language = language
    st.session_state.probability = probability
    st.session_state.ui_state = "render_result"


def submit_feedback():
    # insert record into database

    # INSERT INTO lang_ident_feedback VALUES ( text =  st.session_state.text_input,
    #          predicted_language = st.session_state.language,
    #          probability = st.session_state.probability,
    #          language_hint = st.session_state.lang_hint )

    # This will insert something like: "My text", "de", "0.23232", "en".

    st.session_state.ui_state = "render_feedback"


def main():
    lang_labels = {"es": "Spanish", "en": "English", "de": "German"}

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
        # load pickled language model
        language_model = load_model("simple_mlp_novectorize.h5")
        # create the necessary vectorize layer for the text input
        vectorize_layer = create_vectorize_layer()

        st.text_area(
            "Please enter sample text to detect language:",
            placeholder=None,
            disabled=False,
            key="text_input"
        )

        st.button(
            label='Detect language',
            on_click=_detect_language,
            args=(language_model, vectorize_layer)
        )

    elif st.session_state.ui_state == "render_result":
        st.text_area("Your entered text", value=st.session_state.text_input, disabled=True, key="text_input")

        st.write(
            f"""The classifier considers this text to be in {lang_labels[st.session_state.language]}. 
            It is {round(float(st.session_state.probability) * 100, 1)}% sure.""")

        st.subheader("Did I get it right?")

        selected_language = st.selectbox('What is the real language? Please select in any case.',
                                         ('Spanish', 'English', 'German'),
                                         index=list(lang_labels.keys()).index(st.session_state.language))

        st.session_state.lang_hint = list(lang_labels.keys())[list(lang_labels.values()).index(selected_language)]

        st.write(
            f"""You classify this text as {lang_labels[st.session_state.lang_hint]}.""")
        if st.session_state.language == st.session_state.lang_hint:
            st.write(f"""Looks like, the classifier got it right. What a smart fellow!""")
        else:
            st.write(f"""Oh dear, there is still a long way to go for AI.""")

        st.button('Submit feedback',
                  on_click=submit_feedback)

    elif st.session_state.ui_state == "render_feedback":
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


if __name__ == "__main__":
    main()
