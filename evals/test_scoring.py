"""Тести детермінованих scoring-функцій: python -m pytest evals/test_scoring.py -q"""
from evals.scoring import contains, deterministic, norm


def test_norm_numbers():
    assert norm("10 091 049 грн") == "10091049 грн"
    assert norm("5 %") == "5%"
    assert norm("1,5%") == "1.5%"
    assert norm("дев’яти") == "дев'яти"


def test_contains_boundaries():
    assert contains("ставка 5% ПДФО", "5%")
    assert not contains("ставка 15% єдиного", "5%")          # «5%» не всередині «15%»
    assert not contains("12000 євро", "1200")                # число не є префіксом більшого
    assert contains("Резидентство не гарантує бронювання", "не гарант")   # основа слова
    assert not contains("Google Cloud", "Go")
    assert contains("Ruby, Go та Kotlin", "Go")


def _res(status, claims, sources):
    return {"status": status, "claims": claims,
            "sources": [{"sid": sid, "chunk": {"doc_id": d}} for sid, d in sources]}


def test_deterministic_answerable():
    item = {"category": "answerable", "key_facts": [["4400"], ["Ruby"]], "sources": ["dou-salary-winter-2026"],
            "must_not": ["4500"]}
    r = _res("answered", [{"text": "Ruby on Rails — $4400 взимку", "sources": ["S1"]}],
             [("S1", "dou-salary-winter-2026"), ("S2", "dou-salary-summer-2026")])
    d = deterministic(item, r)
    assert d == {"abstain_ok": True, "key_facts": 1.0, "key_facts_missed": [], "source_hit": True,
                 "must_not_ok": True, "must_not_found": []}


def test_source_hit_counts_only_cited():
    item = {"category": "answerable", "key_facts": [], "sources": ["dou-salary-winter-2026"]}
    r = _res("answered", [{"text": "x", "sources": ["S2"]}],
             [("S1", "dou-salary-winter-2026"), ("S2", "dou-salary-summer-2026")])
    assert deterministic(item, r)["source_hit"] is False    # правильне джерело було в контексті, але не процитоване


def test_no_answer_abstain():
    item = {"category": "no_answer", "sources": []}
    assert deterministic(item, _res("not_found", [], []))["abstain_ok"] is True
    assert deterministic(item, _res("answered", [{"text": "вигадка", "sources": ["S1"]}], [("S1", "x")]))["abstain_ok"] is False
