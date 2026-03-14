# Pattern Scanner Fixed (v1.1.0)

Что исправлено:
- исправлен синтаксический обрыв в `services.py`, из-за которого сервис не собирался локально;
- дефолт режима поиска переведён в `pre_breakout_only`, чтобы лучше ловить ранние базы RIVER/SIREN-стиля;
- расширен диапазон для `max_coins_to_evaluate` и `market_batch_size` до 1000;
- добавлен `universe_target_count` (по умолчанию 1000), чтобы сканировать батчами по 30 внутри выборки 1000;
- снижено количество лишних запросов к CoinGecko: число страниц для авто-скана теперь считается от нужного окна, а не всегда тянет весь рынок;
- добавлены rate-limit-friendly паузы, ретраи и уважение `Retry-After`;
- смягчён `position_bonus`, чтобы ранние паттерны в первых 14–90 днях не резались слишком агрессивно;
- добавлены более короткие окна: 14/18/21/24/28/.../100 дней;
- паттерн-скоринг стал терпимее к pre-breakout/base-first сетапам с сильной полкой и мягкой короной.

Рекомендованный запрос:
```json
{
  "pattern_name": "crown_shelf_right_spike",
  "min_age_days": 14,
  "max_age_days": 90,
  "top_k": 30,
  "max_coins_to_evaluate": 30,
  "market_batch_size": 30,
  "universe_target_count": 1000,
  "market_offset": 0,
  "stage_mode": "pre_breakout_only",
  "include_notes": true,
  "return_pre_filter_candidates": true,
  "compact_response": false
}
```

Важно:
- для реального запуска нужен рабочий `COINGECKO_API_KEY`;
- live-тест против CoinGecko я здесь не выполнял, поэтому честно не подтверждаю реальный runtime-результат API, только синтаксис, импорт и логику.
