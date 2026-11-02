"""Chunking: документи з data/processed/documents.jsonl → чанки з метаданими.

    python -m rag.chunking              # structure-aware → data/processed/chunks.jsonl + chunks_manifest.json
    python -m rag.chunking --compare    # + порівняння з фіксованим розміром → data/processed/chunking_comparison.md

Основна стратегія — structure-aware (по розділах):
  1. документ ділимо на розділи за заголовками (шлях заголовків «H1 > H2 > H3» — це `section`);
  2. блоки розділу (абзаци, пункти списку, рядки таблиць) пакуємо в чанк до TARGET символів,
     НЕ розриваючи блок; блок, довший за MAX, ріжемо по реченнях;
  3. замалі розділи (< MIN, напр. «Ми пропонуємо:» з двома пунктами) зливаємо з наступним розділом
     того самого документа, щоб не було «порожніх» чанків;
  4. до тексту для ембеддингу додаємо контекстний заголовок: «назва документа › розділ» (contextual header).
Базова лінія для порівняння — фіксоване вікно FIXED_SIZE символів з перекриттям FIXED_OVERLAP по всьому тексту.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"

# ~3–4 символи на токен для змішаного UA/EN тексту → 1200 символів ≈ 300–400 токенів:
# влазить у ліміт 512 токенів мультимовних embedding-моделей (e5 / bge-m3 small) з запасом на заголовок.
TARGET, MAX, MIN = 1200, 1400, 300
FIXED_SIZE, FIXED_OVERLAP = 1000, 200
SENT_END = re.compile(r"(?<=[.!?…])\s+")


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str                       # текст чанка (для відповіді й цитати)
    embed_text: str                 # контекстний заголовок + текст (для ембеддингу / BM25)
    title: str
    section: str                    # «H1 > H2 > H3» або "" для фіксованих чанків
    source: str
    format: str
    doc_type: str
    lang: str
    version: str
    license: str
    doc_sha256: str
    position: int                   # порядковий номер чанка в документі
    pages: list[int] = field(default_factory=list)
    meta: dict = field(default_factory=dict)
    strategy: str = "structure"
    n_chars: int = 0
    approx_tokens: int = 0
    chunk_sha256: str = ""
    cut_inside_block: bool = False   # межа чанка розрізала абзац / пункт списку


def approx_tokens(s: str) -> int:
    return max(1, round(len(s) / 3.5))


def split_long(text: str, limit: int) -> list[str]:
    """Блок довший за limit → шматки по реченнях (а якщо речення гігантське — по словах)."""
    parts, cur = [], ""
    for sent in SENT_END.split(text):
        if len(sent) > limit:                     # одне величезне «речення» (таблиця, перелік через ;)
            words, sent_parts, w = sent.split(" "), [], ""
            for x in words:
                if len(w) + len(x) + 1 > limit and w:
                    sent_parts.append(w)
                    w = ""
                w = f"{w} {x}".strip()
            sent_parts.append(w)
            for sp in sent_parts:
                if cur:
                    parts.append(cur)
                cur = sp
            continue
        if cur and len(cur) + len(sent) + 1 > limit:
            parts.append(cur)
            cur = sent
        else:
            cur = f"{cur} {sent}".strip()
    if cur:
        parts.append(cur)
    return parts


def sections(doc: dict) -> list[tuple[str, list[dict]]]:
    """[(шлях заголовків, блоки розділу)] у порядку документа."""
    path: dict[int, str] = {}
    out: list[tuple[str, list[dict]]] = []
    cur: list[dict] = []
    cur_path = ""
    for b in doc["blocks"]:
        if b["type"] == "heading":
            if cur:
                out.append((cur_path, cur))
                cur = []
            lvl = b["level"] or 2
            path = {k: v for k, v in path.items() if k < lvl}
            path[lvl] = b["text"]
            cur_path = " > ".join(path[k] for k in sorted(path))
        else:
            cur.append(b)
    if cur:
        out.append((cur_path, cur))
    return out


def _render(block: dict) -> str:
    return f"- {block['text']}" if block["type"] == "list_item" else block["text"]


SPLIT_MARK = "\u2063"   # невидимий маркер «блок розрізано всередині» (для метрики), прибирається перед записом


def structure_chunks(doc: dict) -> list[tuple[str, str, list[int]]]:
    """→ [(section, text, pages)]."""
    secs = sections(doc)
    # 3) злити замалі розділи з наступним
    merged: list[tuple[str, list[dict]]] = []
    carry_path, carry = "", []
    for path, blocks in secs:
        if carry:
            # злиття замалого розділу з наступним: у метаданих лишаємо обидва останні заголовки
            if carry_path and path:
                path = f"{carry_path.split(' > ')[-1]} | {path}"
            else:
                path = path or carry_path
            blocks = carry + blocks
            carry_path, carry = "", []
        size = sum(len(b["text"]) for b in blocks)
        if size < MIN:
            carry_path, carry = path, blocks
        else:
            merged.append((path, blocks))
    if carry:
        if merged:
            p, bl = merged[-1]
            merged[-1] = (p, bl + carry)
        else:
            merged.append((carry_path, carry))

    out: list[tuple[str, str, list[int]]] = []
    for path, blocks in merged:
        buf: list[str] = []
        pages: set[int] = set()

        def emit():
            if buf:
                out.append((path, "\n".join(buf), sorted(pages)))
            buf.clear()
            pages.clear()

        for b in blocks:
            pieces = split_long(_render(b), MAX) if len(b["text"]) > MAX else [_render(b)]
            if len(pieces) > 1:
                pieces = [p + SPLIT_MARK for p in pieces[:-1]] + pieces[-1:]
            for piece in pieces:
                if buf and sum(len(x) + 1 for x in buf) + len(piece) > TARGET:
                    emit()
                buf.append(piece)
                if b.get("page"):
                    pages.add(b["page"])
        emit()
    return out


def fixed_chunks(doc: dict) -> list[tuple[str, str, list[int]]]:
    text = "\n".join(_render(b) for b in doc["blocks"])
    out, i = [], 0
    while i < len(text):
        out.append(("", text[i:i + FIXED_SIZE], []))
        if i + FIXED_SIZE >= len(text):
            break
        i += FIXED_SIZE - FIXED_OVERLAP
    return out


def build(docs: list[dict], strategy: str) -> list[Chunk]:
    chunks: list[Chunk] = []
    for d in docs:
        parts = structure_chunks(d) if strategy == "structure" else fixed_chunks(d)
        for i, (section, text, pages) in enumerate(parts):
            cut_inside = text.endswith(SPLIT_MARK) or (strategy == "fixed" and not _at_block_boundary(d, i, len(text)))
            text = text.replace(SPLIT_MARK, "")
            if len(section) > 200:                       # дуже довгий шлях заголовків — лишаємо хвіст
                section = "…" + section[-199:]
            header = d["title"] + (f" › {section}" if section else "")
            if d["doc_type"] == "vacancy" and d.get("meta", {}).get("stack"):
                header += f" [вакансія, стек: {d['meta']['stack']}]"
            embed_text = f"{header}\n{text}"
            chunks.append(Chunk(
                chunk_id=f"{d['doc_id']}#{strategy[0]}{i:03d}", doc_id=d["doc_id"], text=text, embed_text=embed_text,
                title=d["title"], section=section, source=d["source"], format=d["format"], doc_type=d["doc_type"],
                lang=d["lang"], version=d["version"], license=d["license"], doc_sha256=d["content_sha256"],
                position=i, pages=pages, meta=d.get("meta", {}), strategy=strategy, n_chars=len(text),
                approx_tokens=approx_tokens(embed_text), cut_inside_block=cut_inside,
                chunk_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]))
    return chunks


# ----------------------------------------------------------------------------- порівняння стратегій

def _block_ends(doc: dict) -> set[int]:
    ends, pos = set(), 0
    for b in doc["blocks"]:
        pos += len(_render(b)) + 1
        ends.add(pos - 1)
    return ends


def _at_block_boundary(doc: dict, position: int, n_chars: int) -> bool:
    end = (FIXED_SIZE - FIXED_OVERLAP) * position + n_chars
    ends = doc.setdefault("_ends", _block_ends(doc))
    return end in ends or end - 1 in ends or end + 1 in ends


def _section_spans(doc: dict) -> list[tuple[int, int]]:
    """Межі розділів у «плоскому» тексті документа (як його ріже fixed_chunks)."""
    spans, pos, start = [], 0, 0
    for b in doc["blocks"]:
        line = _render(b)
        if b["type"] == "heading" and pos > start:
            spans.append((start, pos))
            start = pos
        pos += len(line) + 1
    spans.append((start, pos))
    return spans


def quality(chunks: list[Chunk], docs: dict[str, dict]) -> dict:
    sizes = [c.n_chars for c in chunks]
    starts_mid = sum(1 for c in chunks if c.text[:1].islower() or c.text[:1] in ",;)")
    cut = sum(1 for c in chunks if c.cut_inside_block)
    crosses = 0
    if chunks and chunks[0].strategy == "fixed":
        for c in chunks:                         # чанк захоплює текст двох і більше розділів?
            spans = _section_spans(docs[c.doc_id])
            start = (FIXED_SIZE - FIXED_OVERLAP) * c.position
            end = start + c.n_chars
            crosses += sum(1 for a, b in spans if a < end and b > start) > 1
    tiny = sum(1 for s in sizes if s < 200)
    return {
        "chunks": len(chunks), "median_chars": int(statistics.median(sizes)), "p95_chars": sorted(sizes)[int(.95 * len(sizes))],
        "max_tokens_approx": max(c.approx_tokens for c in chunks), "tiny_lt_200": tiny,
        "start_mid_sentence_pct": round(100 * starts_mid / len(chunks), 1),
        "cut_inside_block_pct": round(100 * cut / len(chunks), 1),
        "cross_section_pct": round(100 * crosses / len(chunks), 1),
        "with_section_metadata_pct": round(100 * sum(1 for c in chunks if c.section) / len(chunks), 1),
    }


def comparison_md(docs: list[dict], s_chunks: list[Chunk], f_chunks: list[Chunk]) -> str:
    by_id = {d["doc_id"]: d for d in docs}
    qs, qf = quality(s_chunks, by_id), quality(f_chunks, by_id)
    rows = [("Чанків", "chunks"), ("Медіана, символів", "median_chars"), ("p95, символів", "p95_chars"),
            ("Макс. токенів (≈)", "max_tokens_approx"), ("Крихітних (< 200 симв.)", "tiny_lt_200"),
            ("Починається посеред речення, %", "start_mid_sentence_pct"),
            ("Межа чанка розрізає абзац / пункт, %", "cut_inside_block_pct"),
            ("Захоплює ≥2 розділи, %", "cross_section_pct"), ("Має розділ у метаданих, %", "with_section_metadata_pct")]
    out = ["# Порівняння chunking-стратегій", "",
           f"Structure-aware (TARGET={TARGET}, MAX={MAX}, MIN={MIN} символів) vs фіксоване вікно "
           f"{FIXED_SIZE} символів з перекриттям {FIXED_OVERLAP}. Корпус: {len(docs)} документів.", "",
           "| Метрика | Structure-aware | Fixed |", "|---|---|---|"]
    out += [f"| {name} | {qs[k]} | {qf[k]} |" for name, k in rows]
    out += ["", "_Structure-aware чанки за побудовою не перетинають межу розділу (крім злиття замалих розділів)._", "",
            "## Приклади на окремих документах", ""]
    examples = ["law-diia-city", "dou-salary-winter-2026", "kpmg-diia-city-2022",
                next(d["doc_id"] for d in docs if d["doc_type"] == "vacancy")]
    for doc_id in examples:
        s = [c for c in s_chunks if c.doc_id == doc_id]
        f = [c for c in f_chunks if c.doc_id == doc_id]
        out += [f"### {doc_id}", "", f"- structure-aware: {len(s)} чанків; fixed: {len(f)} чанків", ""]
        for label, cs in (("structure-aware", s), ("fixed", f)):
            if len(cs) > 1:
                c = cs[1]
                head = c.text[:160].replace("\n", " ⏎ ")
                tail = c.text[-80:].replace("\n", " ⏎ ")
                out += [f"**{label}, чанк #1** (розділ: `{c.section or '—'}`)", "",
                        f"> {head} … {tail}", ""]
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--compare", action="store_true")
    args = ap.parse_args()
    docs = [json.loads(l) for l in (PROCESSED / "documents.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    chunks = build(docs, "structure")
    with (PROCESSED / "chunks.jsonl").open("w", encoding="utf-8") as fh:
        for c in chunks:
            fh.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")
    per_doc: dict[str, int] = {}
    for c in chunks:
        per_doc[c.doc_id] = per_doc.get(c.doc_id, 0) + 1
    by_id = {d["doc_id"]: d for d in docs}
    manifest = {"strategy": "structure", "params": {"target": TARGET, "max": MAX, "min": MIN},
                "documents": len(docs), "chunks": len(chunks), "quality": quality(chunks, by_id),
                "per_document": [{"doc_id": k, "chunks": v, "doc_sha256": by_id[k]["content_sha256"][:16]}
                                 for k, v in per_doc.items()]}
    (PROCESSED / "chunks_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"{len(chunks)} chunks from {len(docs)} documents → data/processed/chunks.jsonl")
    print("quality:", manifest["quality"])
    if args.compare:
        fixed = build(docs, "fixed")
        with (PROCESSED / "chunks_fixed.jsonl").open("w", encoding="utf-8") as fh:
            for c in fixed:
                fh.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")
        (PROCESSED / "chunking_comparison.md").write_text(comparison_md(docs, chunks, fixed), encoding="utf-8")
        print("fixed:", quality(fixed, by_id))
        print("comparison → data/processed/chunking_comparison.md")


if __name__ == "__main__":
    main()
