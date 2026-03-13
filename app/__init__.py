# pattern_scanner_patch

Готовый патч для `/scan`:

- принимает `market_offset` и `market_batch_size`
- умеет `compact_response=true`, чтобы action не падал на огромном payload
- сохраняет batching метаданные в response
- добавляет мягкий rescue для финального gate в кейсах уровня `stakestone`
- batching строится по `coingecko_id`

## Что менять у себя

### Вариант 1 — заменить route-файл
1. Возьми `pattern_scanner_patch/router.py`
2. Подключи `router` в свой FastAPI app
3. Замени `run_scan_service()` на свой реальный вызов сканера

### Вариант 2 — встроить кусками
Перенеси из `router.py`:
- `ScanRequest`
- `soften_final_gate`
- `postprocess_scan_payload`
- `compact_scan_response`
- `build_market_universe`
- `scan(req: ScanRequest)`

## Единственное место, которое обязательно надо подцепить

В `router.py` есть:

```python
def run_scan_service(req: ScanRequest) -> Dict[str, Any]:
    raise NotImplementedError(...)
```

Замени на что-то вроде:

```python
def run_scan_service(req: ScanRequest) -> Dict[str, Any]:
    return scan_service.scan(req.model_dump())
```

или

```python
def run_scan_service(req: ScanRequest) -> Dict[str, Any]:
    return run_pattern_scan(**req.model_dump())
```

## Что тестировать после деплоя

### A
```json
{
  "pattern_name": "crown_shelf_right_spike",
  "min_age_days": 14,
  "max_age_days": 90,
  "top_k": 5,
  "max_coins_to_evaluate": 20,
  "vs_currency": "usd",
  "include_notes": false,
  "market_offset": 0,
  "market_batch_size": 20
}
```

### B
```json
{
  "pattern_name": "crown_shelf_right_spike",
  "min_age_days": 14,
  "max_age_days": 90,
  "top_k": 5,
  "max_coins_to_evaluate": 20,
  "vs_currency": "usd",
  "include_notes": false,
  "market_offset": 20,
  "market_batch_size": 20
}
```

### C
```json
{
  "pattern_name": "crown_shelf_right_spike",
  "min_age_days": 14,
  "max_age_days": 90,
  "top_k": 10,
  "max_coins_to_evaluate": 10,
  "vs_currency": "usd",
  "include_notes": true,
  "coingecko_ids": ["river", "siren-2", "stakestone"]
}
```

## Ожидания
- A/B больше не падают из-за unknown kwargs
- A/B дают разные `market_batch_ids`
- обычный market scan не раздувает payload так сильно
- `stakestone` больше не режется так тупо, если он проходит rescue-условия
