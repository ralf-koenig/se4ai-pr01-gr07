import streamlit as st


@st.cache
def load_model(path: str) -> str:
    '''
    TODO: implement
    '''
    if path == '':
        return None
    else:
        return ""


def _detect_language(model, text: str):
    '''
    TODO: insert language model magic here
    '''
    language = "Gibberish"
    st.write("This text appears to be the language: {}.".format(language))


if __name__ == "__main__":

    # TODO: Load pickled language model here
    language_model = load_model("")
    # st.balloons()
    with st.sidebar:
        """
        This is a simple streamlit application that guesses the language of the text passed.
        """

    text_input = st.text_area(
        "Please enter sample text to detect language: ",
        value="",
        placeholder=None,
        disabled=False,
    )

    button = st.button(
        "Detect language",
        on_click=_detect_language,
        args=(
            language_model,
            text_input,
        ),
    )
