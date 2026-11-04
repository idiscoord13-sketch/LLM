import pytest


@pytest.fixture(scope="session")
def tok():
    from myllm.tiny_models import tiny_tokenizer

    return tiny_tokenizer()
