# TODO(anguelos) handle tokenizers that require large files
import pytest
import ddp_tkn

from unittest.mock import mock_open, patch


known_tokenizers = ddp_tkn.Tokenizer.get_known_children()

def test_string_tokenization():
    t = ddp_tkn.TokenizerAsciiChar()

@pytest.mark.parametrize("cls", known_tokenizers)
def test_string_instatiation(cls):
    t = cls()


@pytest.mark.parametrize("cls", known_tokenizers)
def test_string_tokenization(cls):
    t = cls()
    test_str = "This is a test string."
    token_strs = t.create_string_tokens(test_str)
    assert all([isinstance(s, str) for s in token_strs])
    

@pytest.mark.parametrize("cls", known_tokenizers)
def test_string_tokenization(cls):
    t = cls()
    test_str = "This is a test string."
    token_dict = t.create_dictionary_tokens(test_str)
    assert all([isinstance(t[0], int) and isinstance(t[1], int) and isinstance(t[2], int) for t in token_dict])

@pytest.mark.parametrize("cls", known_tokenizers)
def test_tokenization_reproducibility(cls):
    t = cls()
    test_str = "This is a test string."
    ID = cls.ID()
    assert ddp_tkn.tokenize(test_str, ID) == t.tokenize(test_str)

def test_list_tokenizers():
    assert set(ddp_tkn.list_tokenizers()) == set([cls.ID() for cls in known_tokenizers])
    assert set(ddp_tkn.list_tokenizers()) == {"AsciiChar", "Glove"}