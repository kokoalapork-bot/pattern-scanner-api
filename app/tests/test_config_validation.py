import importlib
import os
import sys

import pytest


def reload_config_with_env(monkeypatch, **env):
    for key in [
        "COINGECKO_AUTH_MODE",
        "COINGECKO_API_KEY",
        "COINGECKO_BASE_URL",
    ]:
        monkeypatch.delenv(key, raising=False)

    for key, value in env.items():
        monkeypatch.setenv(key, value)

    if "app.config" in sys.modules:
        importlib.reload(sys.modules["app.config"])
        return sys.modules["app.config"]

    import app.config
    importlib.reload(app.config)
    return app.config


def test_demo_config_valid(monkeypatch):
    config = reload_config_with_env(
        monkeypatch,
        COINGECKO_AUTH_MODE="demo",
        COINGECKO_API_KEY="demo-key",
        COINGECKO_BASE_URL="https://api.coingecko.com/api/v3",
    )
    assert config.settings.coingecko_auth_mode == "demo"
    assert config.settings.coingecko_header_name == "x-cg-demo-api-key"


def test_pro_config_valid(monkeypatch):
    config = reload_config_with_env(
        monkeypatch,
        COINGECKO_AUTH_MODE="pro",
        COINGECKO_API_KEY="pro-key",
        COINGECKO_BASE_URL="https://pro-api.coingecko.com/api/v3",
    )
    assert config.settings.coingecko_auth_mode == "pro"
    assert config.settings.coingecko_header_name == "x-cg-pro-api-key"


def test_demo_cannot_use_pro_base(monkeypatch):
    with pytest.raises(Exception):
        reload_config_with_env(
            monkeypatch,
            COINGECKO_AUTH_MODE="demo",
            COINGECKO_API_KEY="demo-key",
            COINGECKO_BASE_URL="https://pro-api.coingecko.com/api/v3",
        )


def test_pro_cannot_use_public_base(monkeypatch):
    with pytest.raises(Exception):
        reload_config_with_env(
            monkeypatch,
            COINGECKO_AUTH_MODE="pro",
            COINGECKO_API_KEY="pro-key",
            COINGECKO_BASE_URL="https://api.coingecko.com/api/v3",
        )


def test_missing_key_fails(monkeypatch):
    with pytest.raises(Exception):
        reload_config_with_env(
            monkeypatch,
            COINGECKO_AUTH_MODE="demo",
            COINGECKO_API_KEY="",
            COINGECKO_BASE_URL="https://api.coingecko.com/api/v3",
        )


def test_whitespace_key_fails(monkeypatch):
    with pytest.raises(Exception):
        reload_config_with_env(
            monkeypatch,
            COINGECKO_AUTH_MODE="demo",
            COINGECKO_API_KEY="   ",
            COINGECKO_BASE_URL="https://api.coingecko.com/api/v3",
        )
