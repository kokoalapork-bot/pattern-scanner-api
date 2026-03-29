
from app.main import app


def test_app_import():
    assert app.title == "Crypto Pattern Scanner API"
