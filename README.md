# Job Extraction Service

Кейс 1 курсу AI Engineering: сервіс витягує структуровані дані з «брудних» текстів вакансій українського ринку праці за схемою `JobPosting`, порівнює провайдерську (Claude) і локальну (Ollama) моделі та обґрунтовує вибір для продакшну цифрами. Звіт: [REPORT.md](REPORT.md).

**Результат:** у продакшн обрано **Claude Haiku 5.5 + промпт v2**. Показники на golden set:
- точність по полях 0.985;
- 100% валідних відповідей;
- p95 6.5 с;
- $1.03 за 1000 документів.

Для порівняння: локальна qwen2.5:3b дає 0.75, Claude Haiku 4.5 — 0.935 при цілих $7.2 за 1000 документів. Деталі в [REPORT.md](REPORT.md).

## Де що шукати (вимоги ДЗ → файли)

| Вимога ДЗ | Де в репозиторії |
|---|---|
| ≥30 «брудних» текстів без PII, 10 розмічених вручну | `data/raw/` (34 вакансії), `data/sources.csv`, `data/golden.jsonl`, правила розмітки — `data/LABELING.md` |
| **Крок 1.** Схема (≥6 полів, enum, вкладений об'єкт/список) | `jobex/schema.py`: 12 полів, 8 enum, `salary`, `skills[]`, `languages[]` |
| Baseline-промпт | `prompts/v1/` |
| Провайдерська модель зі structured output | `jobex/clients/anthropic_client.py` (нативна JSON Schema + fallback на tool use) |
| Retry і timeout | `jobex/clients/base.py` (`with_retries`: backoff + jitter), таймаути в `config.yaml` |
| Прогнати всі тексти, зберегти результати | `jobex/pipeline.py` → `results/runs/*.jsonl` |
| **Крок 2.** Локальна модель через спільний інтерфейс | `jobex/clients/ollama_client.py`, інтерфейс `LLMClient` у `jobex/clients/base.py` |
| temperature / top_p у конфігу, ≥2 конфігурації | `config.yaml` → `sampling` (`t0`, `t07_p09`) |
| Точність по полях, % валідних, p50/p95, $/1000 | `jobex/evaluate.py` → `results/metrics.{json,csv}`, таблиці в `REPORT.md` |
| **Крок 3.** Промпти з версіями та changelog | `prompts/v1/`, `prompts/v2/` (`meta.yaml`), `prompts/CHANGELOG.md` |
| Валідація, 1 спроба виправлення, далі `failed` | `jobex/pipeline.py` (`extract_one`), шаблон `prompts/vN/repair.md` |
| Лог моделі, версії промпту, семплінгу | кожен рядок `results/runs/*.jsonl`: `model`, `prompt_version`, `prompt_sha`, `sampling`, `sampling_sent` |
| **Крок 4.** Таблиця модель × промпт × семплінг | `REPORT.md` → «Результати» |
| 3–5 характерних помилок | `REPORT.md` → «Розбір характерних помилок»; усі розбіжності — `results/golden_diff.md` |
| Вибір для продакшну + умови зміни | `REPORT.md` → «Вибір для продакшну» |
| Stretch: третя модель / регресія / кеш | Claude Haiku 4.5 у `config.yaml`; `make regress`; `results/cache/` (`jobex/pipeline.py`) |
| Запуск однією командою, ключі не в коді | `make run` (= `python main.py`), `.env.example`, `.env` у `.gitignore` |

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
| `data/corpus/` | корпус 150 SE-вакансій для наступних ДЗ (див. нижче) |

## Дані для наступних ДЗ

Для наступних етапів курсу (RAG-пошук по вакансіях, агент порівняння вакансій) уже зібрано окремий корпус в одному домені — **Software Engineering**: [`data/corpus/djinni_se_raw.jsonl`](data/corpus/README.md).

- **Обсяг і джерело:** 150 реальних вакансій з Djinni, жовтень 2026.
- **Стеки:** Python, Java, JavaScript, Node.js, PHP, .NET, Golang, C++, iOS, Android, React Native, Data Engineer, Data Science (по 10), а також Rust, Flutter, Ruby, Scala.
- **Поля кожної вакансії:** `id`, `url`, `keyword` (стек), `title`, `published`, повний `text`.
- **Мова описів:** ~2/3 англійською, ~1/3 українською. Персональних даних не виявлено.
- **Дані ДЗ1 заморожені:** `data/raw`, golden set і `results` не змінювались, тож результати ДЗ1 відтворювані як є.
- **План використання:**
	1. Прогнати витяг (Claude Haiku 5.5 + v2) по корпусу і отримати структуровану базу вакансій.
	2. Побудувати пошук із фільтрами (стек, рівень, формат, зарплата) і RAG з посиланнями на конкретні вакансії.

## Ключі

Ключ API лише в `.env` (у `.gitignore`); у репозиторії — `.env.example`.
