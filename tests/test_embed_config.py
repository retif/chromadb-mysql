"""Embedding config — pure, no DB/torch."""

from chromadb_mysql_backend import _embed


def test_default_model():
    assert _embed.model_from_env() == "all_minilm_l12_v2"


def test_model_override(monkeypatch):
    monkeypatch.setenv(_embed.ENV_MODEL, "multilingual-e5-small")
    assert _embed.model_from_env() == "multilingual-e5-small"


def test_default_dim():
    assert _embed.DEFAULT_DIM == 384


def test_default_mode_is_db():
    assert _embed.mode_from_env() == "db"


def test_mode_override(monkeypatch):
    monkeypatch.setenv(_embed.ENV_MODE, "client")
    assert _embed.mode_from_env() == "client"


def test_vector_to_sql():
    assert _embed.vector_to_sql([1.0, 2.5]) == "[1.0,2.5]"
