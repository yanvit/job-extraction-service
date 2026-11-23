# Evals RAG — 29 питань

Згенеровано `make evals`. Кеш LLM: 69 з кешу, 0 нових викликів; 21 с.

| Метрика | Значення | Поріг |
|---|---|---|
| «не знаю» там, де треба (усі питання) | 1.0 | ≥ 0.9 |
| відповів на питання без відповіді | 0.0 | ≤ 0.15 |
| відмовився, хоча відповідь є | 0.0 |  |
| правильне джерело в цитатах | 1.0 | ≥ 0.85 |
| ключові факти у відповіді | 1.0 | ≥ 0.8 |
| без типової помилки пастки | 1.0 | ≥ 1.0 |
| LLM-суддя: правильність | 0.955 | ≥ 0.8 |
| LLM-суддя: правильність на пастках | 0.8 |  |
| faithfulness: підтверджені твердження | 0.959 | ≥ 0.9 |
| faithfulness: відповіді без жодного непідтвердженого | 0.909 |  |
| retrieval recall@5 (hybrid+rerank) | 1.0 | ≥ 0.9 |

Вартість відповідей: **$0.55 на 1000 питань** (відмови на порозі reranker-а безкоштовні: 4 з 29). Затримка p50 1566 мс, p95 10771 мс. Суддя за цей прогін: $0.421.

**Статус: ✅ усі пороги пройдено**

## По питаннях

| id | тип | статус | джерело | факти | суддя | faithfulness | цитовані документи |
|---|---|---|---|---|---|---|---|
| t01 | answerable | answered | ✅ | 1.00 | correct | 3/3 | kpmg-diia-city-2022, law-diia-city, unba-diia-city-resident |
| t02 | answerable | answered | ✅ | 1.00 | correct | 1/2 | kpmg-diia-city-2022, law-diia-city, unba-diia-city-resident |
| t03 | answerable | answered | ✅ | 1.00 | correct | 2/2 | devua-diia-city-how-to, kpmg-diia-city-2022, pwc-diia-city-2021, unba-diia-city-resident |
| t04 | answerable | answered | ✅ | 1.00 | correct | 1/1 | prokadry-gig-fop |
| t05 | answerable | answered | ✅ | 1.00 | correct | 4/4 | unba-diia-city-resident |
| t06 | answerable | answered | ✅ | 1.00 | correct | 2/2 | devua-diia-city-how-to, prokadry-gig-fop |
| t07 | answerable | answered | ✅ | 1.00 | correct | 2/2 | buhplatforma-fop3-2026, monobank-fop3-2026 |
| t08 | answerable | answered | ✅ | 1.00 | correct | 2/2 | monobank-fop3-2026 |
| t09 | answerable | answered | ✅ | 1.00 | correct | 2/2 | dou-salary-summer-2026 |
| t10 | answerable | answered | ✅ | 1.00 | correct | 3/3 | law-diia-city, unba-diia-city-resident |
| t11 | answerable | answered | ✅ | 1.00 | correct | 1/2 | law-diia-city |
| t12 | answerable | answered | ✅ | 1.00 | correct | 1/1 | djinni-854780 |
| t13 | answerable | answered | ✅ | 1.00 | correct | 4/4 | djinni-690808, djinni-853382, djinni-854141, djinni-855072 |
| t14 | answerable | answered | ✅ | 1.00 | correct | 1/1 | djinni-848477 |
| t15 | answerable | answered | ✅ | 1.00 | correct | 2/2 | dou-salary-summer-2026, dou-salary-winter-2026 |
| t16 | answerable | answered | ✅ | 1.00 | correct | 3/3 | buhplatforma-fop3-2026, monobank-fop3-2026 |
| t17 | answerable | answered | ✅ | 1.00 | correct | 2/2 | dou-salary-winter-2026, forbes-median-salary-2026 |
| t18 | trap | answered | ✅ | 1.00 | correct | 2/2 | kpmg-diia-city-2022, pwc-diia-city-2021, unba-diia-city-resident |
| t19 | trap | answered | ✅ | 1.00 | correct | 2/2 | dou-salary-summer-2026 |
| t20 | trap | answered | ✅ | 1.00 | correct | 1/1 | dou-salary-winter-2026 |
| t21 | trap | partial | ✅ | 1.00 | incorrect | 2/2 | dou-salary-winter-2026 |
| t22 | trap | answered | ✅ | 1.00 | correct | 4/4 | prokadry-gig-fop, pwc-diia-city-2021 |
| n01 | no_answer | not_found (model) | — | — | — | — | — |
| n02 | no_answer | not_found (model) | — | — | — | — | — |
| n03 | no_answer | not_found (retrieval) | — | — | — | — | — |
| n04 | no_answer | not_found (retrieval) | — | — | — | — | — |
| n05 | no_answer | not_found (retrieval) | — | — | — | — | — |
| n06 | no_answer | not_found (model) | — | — | — | — | — |
| n07 | no_answer | not_found (retrieval) | — | — | — | — | — |
