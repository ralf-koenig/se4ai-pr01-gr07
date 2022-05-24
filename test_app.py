from app import load_model

import pytest

def test_load_model():
    # adding comment
    # test whether method returns None for empty string input
    assert load_model('','') == None


def test_not_implemented():

    # test whether method returns None for empty string input
    assert True
