
from app.config import get_settings


def test_settings_defaults():
    settings = get_settings()
    assert settings.default_min_age_days >= 1
    assert settings.default_max_age_days >= settings.default_min_age_days
