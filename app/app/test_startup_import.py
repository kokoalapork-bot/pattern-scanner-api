import os


def test_app_main_import_smoke() -> None:
    os.environ.setdefault("COINGECKO_AUTH_MODE", "demo")
    os.environ.setdefault("COINGECKO_API_KEY", "test-demo-key")
    os.environ.setdefault("COINGECKO_BASE_URL", "https://api.coingecko.com/api/v3")

    import app.main  # noqa: F401
