"""BM25 (Okapi) без зовнішніх залежностей — щоб було видно, як працює лексичний пошук.

score(q, d) = Σ_t∈q  IDF(t) · tf(t,d)·(k1+1) / (tf(t,d) + k1·(1 − b + b·|d|/avgdl))
IDF(t) = ln(1 + (N − df(t) + 0.5) / (df(t) + 0.5))
"""
from __future__ import annotations

import math
import re
from collections import Counter

TOKEN = re.compile(r"[\w#+.]+", re.UNICODE)       # «c#», «c++», «node.js» лишаються одним токеном
STOP = set("і й та в у на з до за що як не це для від по про а або the a an of to in and or for with on is are".split())


def tokenize(text: str) -> list[str]:
    toks = [t.strip(".").lower() for t in TOKEN.findall(text)]
    return [t for t in toks if t and t not in STOP]


class BM25:
    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tfs = [Counter(tokenize(d)) for d in docs]
        self.lens = [sum(tf.values()) for tf in self.tfs]
        self.avgdl = sum(self.lens) / max(1, len(self.lens))
        df: Counter = Counter()
        for tf in self.tfs:
            df.update(tf.keys())
        n = len(docs)
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    def scores(self, query: str) -> list[float]:
        q = tokenize(query)
        out = []
        for tf, dl in zip(self.tfs, self.lens):
            s = 0.0
            for t in q:
                f = tf.get(t)
                if f:
                    s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
            out.append(s)
        return out
