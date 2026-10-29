# Job Extraction Service

Кейс 1 курсу AI Engineering: сервіс витягує структуровані дані з «брудних» текстів вакансій українського ринку праці за схемою `JobPosting`, порівнює провайдерську (Claude) і локальну (Ollama) моделі та обґрунтовує вибір для продакшну цифрами. Звіт: [REPORT.md](REPORT.md).

## Швидкий старт

Потрібно: **Python 3.11+**, ключ Anthropic API, [Ollama](https://ollama.com/download) з моделлю `qwen2.5:3b`.

```bash
python3 -m venv .venv && source .venv/bin/activate
make install                     # pip install -r requirements.txt
cp .env.example .env             # впишіть ANTHROPIC_API_KEY
ollama pull qwen2.5:3b           # ~2 ГБ; Ollama має бути запущена (застосунок або `ollama serve`)
make check                       # перевірка ключа і Ollama

make run                         # = python main.py: усі прогони -> метрики -> таблиці в REPORT.md
```

- `make run` проганяє всі 34 тексти через матрицю з `config.yaml`. Найдовша частина — локальна модель (~30–40 хв на MacBook M1 Pro). Відповіді кешуються в `results/cache/`, тож повторний запуск миттєвий і безкоштовний.
- `make quick` робить те саме лише на golden set (10 текстів), це кілька хвилин.
- `make report` перераховує метрики й таблиці з уже збережених `results/runs/` без жодного виклику моделей.
- Вартість повного прогону Claude: менше $0.10 для Haiku 5.5; Haiku 4.5 приблизно в 10 разів дорожча.

Інші команди: `make regress` (регресія промптів v1 → v2 на golden set), `make golden` (перевірка розмітки), `python -m jobex.extract_one data/raw/job_15.txt --model qwen2.5-3b-local --sampling t0` (одна модель на одному тексті).

## Як це працює

```
data/raw/*.txt → prompts/vN (system + user) → модель (config.yaml) зі structured output
   → Pydantic-валідація → ok | 1 спроба виправлення з текстом помилок → repaired | failed
   → results/runs/<модель>__<промпт>__<семплінг>.jsonl (модель, версія промпту, семплінг, токени, час, вартість)
   → метрики на golden set → results/metrics.{json,csv}, results/golden_diff.md, таблиці в REPORT.md
```

- **Схема** (`jobex/schema.py`): 12 полів, 8 enum, вкладений `salary`, списки об'єктів `skills` і `languages`. Усі поля обов'язкові: «немає даних» = явне `null`/`[]`/`unknown`.
- **Моделі** (`jobex/clients/`): спільний інтерфейс `LLMClient`.
	- Claude через офіційний SDK з нативною JSON Schema (`output_config.format`) і fallback на tool use.
	- Ollama через `/api/chat` з `format` = JSON Schema.
	- Ретраї (експоненційний backoff + jitter на таймаут/429/5xx/529) і таймаути — спільні.
- **Конфіг** (`config.yaml`): моделі з цінами, профілі семплінгу (`t0`, `t07_p09`), матриця експерименту. Claude Haiku 5.5 не приймає `temperature`/`top_p`, тому для неї один прогін `api_default`, а порівняння семплінгу — на Haiku 4.5 і qwen.
- **Промпти** (`prompts/`): версії `v1`, `v2` з `meta.yaml` і [CHANGELOG](prompts/CHANGELOG.md).
- **Golden set** (`data/golden.jsonl`): 10 розмічених вакансій, правила — [data/LABELING.md](data/LABELING.md).

## Структура

| Шлях | Що там |
|---|---|
| `main.py`, `Makefile` | запуск: `run`, `eval`, `report`, `regress`, `check` |
| `config.yaml` | моделі, ціни, семплінг, матриця експерименту |
| `jobex/schema.py` | Pydantic-схема `JobPosting` |
| `jobex/clients/` | `base.py` (інтерфейс + ретраї), `anthropic_client.py`, `ollama_client.py` |
| `jobex/pipeline.py` | прогін, валідація, repair, логування, кеш |
| `jobex/evaluate.py`, `jobex/report.py` | метрики і таблиці звіту |
| `prompts/` | версії промптів + CHANGELOG |
| `data/raw/` | 34 вакансії (Djinni, Telegram), PII замінено на `[REMOVED]`; джерела — `data/sources.csv` |
| `data/golden.jsonl`, `data/LABELING.md` | golden set і правила розмітки |
| `results/runs/` | виходи моделей (JSONL, по рядку на документ) |
| `results/metrics.*`, `results/golden_diff.md` | метрики і розбіжності з golden set |
| `REPORT.md` | звіт-порівняння і вибір для продакшну |

## Ключі

Ключ API лише в `.env` (у `.gitignore`); у репозиторії — `.env.example`.
