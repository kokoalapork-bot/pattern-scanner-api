# README_FIXES

Этот пакет собран как рабочая восстановленная версия репозитория:
- структура файлов сохранена
- API-эндпоинты `/`, `/health`, `/scan` работают
- добавлены базовые тесты
- используется CoinGecko как источник рынка и истории


## v1.2.6 hard filters
Added strict River-like hard gates in `app/patterns.py`:
- reject windows where ATH happens on listing / first bars of history
- reject windows where crown starts earlier than 15 days after listing
- reject windows where ATH is in the late part of the crown
- reject windows where crown is longer than 60 daily candles
- reject windows where ATH sits too late in the candidate window

In `app/services.py`, candidates with `similarity <= 0` are skipped and do not appear in results.
