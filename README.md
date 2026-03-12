# Crypto Pattern Scanner API

FastAPI-сервис для поиска криптоактивов, похожих на паттерн:

**корона слева -> сброс -> длинная полка -> правый шпиль -> возврат в полку**

## Что делает
- берет список монет с CoinGecko
- фильтрует по возрасту 14-450 дней
- исключает стейблы и tokenized stocks/ETF
- грузит дневную историю цены
- считает similarity score
- возвращает top-N совпадений

## Ограничения
Это MVP. Источник данных по умолчанию — CoinGecko.
Для более точной версии позже лучше подключить более точный market-data provider или биржевые API.

## Локальный запуск

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Swagger UI:
`http://localhost:8000/docs`

## Пример запроса

```bash
curl -X POST "http://localhost:8000/scan" \
  -H "Content-Type: application/json" \
  -d '{
    "pattern_name": "crown_shelf_right_spike",
    "min_age_days": 14,
    "max_age_days": 450,
    "top_k": 10,
    "max_coins_to_evaluate": 150,
    "vs_currency": "usd",
    "include_notes": true
  }'
```

## Деплой

### Render
Build command:
`pip install -r requirements.txt`

Start command:
`uvicorn app.main:app --host 0.0.0.0 --port $PORT`

### Railway
Start command:
`uvicorn app.main:app --host 0.0.0.0 --port $PORT`

## GPT Actions
После деплоя открой:
`https://YOUR-DOMAIN/openapi.json`

И импортируй схему в Custom GPT -> Actions.
