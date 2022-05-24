"""
Language identification for the three languages: German, English, Spanish.

This module test_app.py encapsulates some basic testing of the inference pipeline.
"""

from language_identification.inference import *

import pytest

# Common test samples for both classifiers: huggingface and wikipedia
text_sample_de = "Heute ist ein wunderschöner Tag!"
text_sample_en = "Reading will help you to improve your understanding of the language and build your vocabulary."
text_sample_es = "Voy al parque a las cinco de la tarde, cuando termino los deberes de la escuela."


def test_load_model():
    # adding comment
    # test whether method returns None for empty string input
    assert load_model('', '') is None
    assert load_model('', 'test') is None
    assert load_model('huggingface', 'simple_mlp_novectorize.h5') is not None
    assert load_model('wikipedia', 'simple_mlp_novectorize.h5') is not None


def test_create_vectorize_layer():
    assert create_vectorize_layer('', '') is None
    assert create_vectorize_layer('', 'test') is None
    assert create_vectorize_layer('huggingface', 'train.csv') is not None
    assert create_vectorize_layer('wikipedia', 'train.csv') is not None


def test_detect_language_huggingface():
    vectorize_layer = create_vectorize_layer('huggingface', 'train.csv')
    model = load_model('huggingface', 'simple_mlp_novectorize.h5')

    print()

    # Test German on Huggingface data
    language, probability = detect_language(text_sample_de, vectorize_layer, model)
    print(text_sample_de, "\n", language, probability)
    assert language == 'de'
    assert probability > 0.4

    # Test English on Huggingface data
    language, probability = detect_language(text_sample_en, vectorize_layer, model)
    print(text_sample_en, "\n", language, probability)
    assert language == 'en'
    assert probability > 0.4

    # Test Spanish on Huggingface data
    language, probability = detect_language(text_sample_es, vectorize_layer, model)
    print(text_sample_es, "\n", language, probability)
    assert language == 'es'
    assert probability > 0.4


def test_detect_language_wikipedia():
    vectorize_layer = create_vectorize_layer('wikipedia', 'train.csv')
    model = load_model('wikipedia', 'simple_mlp_novectorize.h5')

    print()

    # Test German on Wikipedia data
    language, probability = detect_language(text_sample_de, vectorize_layer, model)
    print(text_sample_de, "\n", language, probability)
    assert language == 'de'
    assert probability > 0.4

    # Test English on Wikipedia data
    language, probability = detect_language(text_sample_en, vectorize_layer, model)
    print(text_sample_en, "\n", language, probability)
    assert language == 'en'
    assert probability > 0.4

    # Test Spanish on Wikipedia data
    language, probability = detect_language(text_sample_es, vectorize_layer, model)
    print(text_sample_es, "\n", language, probability)
    assert language == 'es'
    assert probability > 0.4
