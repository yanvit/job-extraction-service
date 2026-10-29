PY ?= python3

.PHONY: install check run quick eval report regress golden

install:            ## встановити залежності
	$(PY) -m pip install -r requirements.txt

check:              ## перевірити ключ API і Ollama
	$(PY) main.py check

run:                ## повний прогін: усі моделі × промпти × семплінг на всіх 34 текстах -> eval -> REPORT.md
	$(PY) main.py all

quick:              ## швидко: лише golden set (10 текстів)
	$(PY) main.py all --docs golden

eval:
	$(PY) main.py eval

report:             ## перегенерувати таблиці REPORT.md і results/golden_diff.md з наявних results/runs
	$(PY) main.py eval && $(PY) main.py report

regress:            ## регресія промптів v1 -> v2 на golden set (модель: MODEL=...)
	$(PY) main.py regress --model $(or $(MODEL),claude-haiku-5-5) --baseline v1 --candidate v2

golden:             ## перевірити golden set і згенерувати data/golden_review.md
	$(PY) -m jobex.golden check && $(PY) -m jobex.golden review
