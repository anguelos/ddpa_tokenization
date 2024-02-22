=======================================
ddpa_tokenization Package Documentation
=======================================

.. image:: https://img.shields.io/pypi/v/ddpa_tokenization.svg
    :target: https://pypi.python.org/pypi/ddpa_tokenization

Introduction
------------

The `ddpa_tokenization` package is a Python library that provides tokenization functionality for natural language processing tasks. It offers various tokenization algorithms and utilities to preprocess text data.

Features
--------

- Reprocible tokenization

Installation
------------

You can install `ddpa_tokenization` using pip:

.. code-block:: bash

    pip install ddp_tokenization

Usage
-----

To use `ddpa_tokenization`, you need to import the necessary modules and functions:

.. code-block:: python

    from ddp_tokenization import tokenize

    text = "This is a sample sentence. Another sentence follows."
    words = word_tokenize(text)
    sentences = sentence_tokenize(text)

    print(words)
    print(sentences)


Testing
-------

You will need to install the `pytest` and `pytest-cov` packages to run the tests.
you can install them with the following command:

.. code-block:: bash

pip install pytest pytest-cov


You can run the tests for `ddpa_tokenization` using the following command:

.. code-block:: bash

PYTHONPATH="./src/" pytest test --cov='./src'



Contributing
------------

If you would like to contribute to the development of `ddpa_tokenization`, please follow the guidelines in the CONTRIBUTING.md file.

License
-------

`ddpa_tokenization` is licensed under the MIT License. See the LICENSE file for more details.
