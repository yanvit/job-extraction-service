"""Ingestion: сирі документи (HTML / PDF / Markdown) → очищені структуровані документи.

    python -m rag.ingest        # → data/processed/documents.jsonl + data/processed/ingest_report.json

Документ = метадані + список блоків. Блок = {"type": heading|paragraph|list_item|table_row, "text", "level"?, "page"?}.
Структуру (заголовки, списки, таблиці) зберігаємо, бо на ній будується chunking (rag/chunking.py).
"""
from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml
from bs4 import BeautifulSoup, NavigableString, Tag

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "data" / "sources" / "manifest.yaml"
DOCS_DIR = ROOT / "data" / "docs"
VACANCIES_DIR = ROOT / "data" / "corpus" / "vacancies"
OUT_DIR = ROOT / "data" / "processed"


@dataclass
class Block:
    type: str                 # heading | paragraph | list_item | table_row
    text: str
    level: int | None = None  # для heading: 1..4
    page: int | None = None   # для PDF


@dataclass
class Document:
    doc_id: str
    title: str
    source: str
    format: str               # html | pdf | md
    doc_type: str             # law | guide | article | report | vacancy
    lang: str
    version: str
    license: str
    blocks: list[Block] = field(default_factory=list)
    content_sha256: str = ""

    def text(self) -> str:
        return "\n".join(b.text for b in self.blocks)


# ----------------------------------------------------------------------------- cleaning

_INVISIBLE = dict.fromkeys(map(ord, "​‌‍⁠﻿­"), None)   # zero-width, soft hyphen
BOILERPLATE = re.compile(
    r"^(поділитись:?|поділитися:?|скачати|читайте також.*|підписуйтесь.*|share|опубліковано .*р\.|"
    r"monoбізнес кіт|реклама|коментарі.*|\d+ коментар.*|усі статті за темою|статті за темою|"
    r"поширити|купити|категорія|новини|дата|\d+ хв читання|©.*)$",
    re.IGNORECASE,
)


def clean_text(s: str) -> str:
    s = unicodedata.normalize("NFC", s).translate(_INVISIBLE)   # не NFKC: він перетворює «№» на «No»
    s = s.replace(" ", " ")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r" *\n *", "\n", s)
    return s.strip()


def clean_blocks(blocks: list[Block], drop_patterns: list[str] | None = None) -> list[Block]:
    """Нормалізація тексту, прибирання boilerplate і (опційно) специфічних для сайту рядків (manifest: drop_patterns)."""
    extra = [re.compile(p, re.IGNORECASE) for p in (drop_patterns or [])]
    out: list[Block] = []
    for b in blocks:
        b.text = clean_text(b.text)
        if not b.text or BOILERPLATE.match(b.text) or any(x.search(b.text) for x in extra):
            continue
        if out and out[-1].text == b.text and out[-1].type == b.type:   # підряд однакові блоки (дубль у верстці)
            continue
        out.append(b)
    return out


# ----------------------------------------------------------------------------- HTML

DROP_TAGS = {"script", "style", "noscript", "nav", "header", "footer", "aside", "form", "button", "svg",
             "iframe", "figure", "img", "picture", "video", "select", "input"}
INLINE_TAGS = {"a", "b", "strong", "i", "em", "span", "code", "sup", "sub", "u", "small", "mark", "abbr", "time", "label"}
HEADINGS = {"h1": 1, "h2": 2, "h3": 3, "h4": 4, "h5": 4, "h6": 4}


def _inline_text(el: Tag) -> str:
    return el.get_text(" ", strip=True)


def html_blocks(root: Tag) -> list[Block]:
    """Обхід DOM у порядку документа: блокові теги → блоки, решта контейнерів — рекурсія."""
    out: list[Block] = []
    buf: list[str] = []

    def flush():
        txt = " ".join(x for x in buf if x.strip())
        if txt.strip():
            out.append(Block("paragraph", txt))
        buf.clear()

    def walk(node: Tag):
        for ch in node.children:
            if isinstance(ch, NavigableString):
                buf.append(str(ch))
                continue
            if not isinstance(ch, Tag) or ch.name in DROP_TAGS:
                continue
            if ch.name == "br":
                flush()
            elif ch.name in INLINE_TAGS:
                buf.append(_inline_text(ch))
            elif ch.name in HEADINGS:
                flush()
                out.append(Block("heading", _inline_text(ch), level=HEADINGS[ch.name]))
            elif ch.name == "p":
                flush()
                out.append(Block("paragraph", _inline_text(ch)))
            elif ch.name in ("ul", "ol"):
                flush()
                for li in ch.find_all("li", recursive=False):
                    out.append(Block("list_item", _inline_text(li)))
            elif ch.name == "table":
                flush()
                for tr in ch.find_all("tr"):
                    cells = [c.get_text(" ", strip=True) for c in tr.find_all(["th", "td"])]
                    if any(cells):
                        out.append(Block("table_row", "| " + " | ".join(cells) + " |"))
            else:
                flush()
                walk(ch)
        flush()

    walk(root)
    return out


def parse_html(path: Path, selector: str | None) -> list[Block]:
    soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="ignore"), "lxml")
    root = soup.select_one(selector) if selector else None
    if root is None:  # fallback: найбільший article/main
        cands = soup.find_all(["article", "main"]) or [soup.body or soup]
        root = max(cands, key=lambda e: len(e.get_text()))
    return html_blocks(root)


LAW_SECTION = re.compile(r"^Розділ [IVXLC]+\.?", re.IGNORECASE)
LAW_ARTICLE = re.compile(r"^Стаття \d+(-\d+)?\.")
EDITORIAL_NOTE = re.compile(r"^\{.*\}$", re.S)


def postprocess_law(blocks: list[Block]) -> list[Block]:
    """Законодавчий текст: «Розділ …» → h2, «Стаття N. …» → h3; редакційні примітки {…} прибираємо."""
    out = []
    for b in blocks:
        t = b.text.strip()
        if EDITORIAL_NOTE.match(t):
            continue
        if LAW_SECTION.match(t):
            out.append(Block("heading", t, level=2))
        elif LAW_ARTICLE.match(t):
            out.append(Block("heading", t, level=3))
        else:
            out.append(b)
    return out


# ----------------------------------------------------------------------------- PDF

def parse_pdf(path: Path) -> list[Block]:
    """PDF (у нас — презентаційні документи): сторінка = слайд/розділ.

    1) прибрати рядки, що повторюються на ≥50% сторінок (колонтитули, номери);
    2) склеїти рядки в абзаци: розрив лише після .!?: або перед маркером списку;
    3) перенос «сло-\\nво» склеюється без пробілу; перший короткий рядок сторінки — заголовок.
    """
    from pypdf import PdfReader

    pages = [(p.extract_text() or "") for p in PdfReader(str(path)).pages]
    page_lines = [[clean_text(l) for l in t.splitlines() if clean_text(l)] for t in pages]
    counts: dict[str, int] = {}
    for lines in page_lines:
        for l in set(lines):
            counts[l] = counts.get(l, 0) + 1
    n = max(1, len(page_lines))
    repeated = {l for l, c in counts.items() if n >= 3 and c / n >= 0.5}

    blocks: list[Block] = []
    for pno, lines in enumerate(page_lines, 1):
        lines = [l for l in lines if l not in repeated and not re.fullmatch(r"\d{1,3}", l)]
        if not lines:
            continue
        if len(lines[0]) <= 80:
            title, lines = lines[0], lines[1:]
            # заголовок слайда часто розбитий на 2 рядки: «… для» + «набуття статусу …»
            if lines and not re.search(r"[.!?:]$", title) and lines[0][:1].islower() and len(lines[0]) <= 80:
                title, lines = title + " " + lines[0], lines[1:]
            blocks.append(Block("heading", title, level=2, page=pno))
        para = ""
        for l in lines:
            bullet = re.match(r"^([•▪●\-–]|\d+[.)])\s+", l)
            if bullet and para:
                blocks.append(Block("paragraph", para, page=pno))
                para = ""
            if para.endswith("-") and l[:1].islower():
                para = para + l
            elif para and not re.search(r"[.!?:;]$", para):
                para = para + " " + l
            else:
                if para:
                    blocks.append(Block("paragraph", para, page=pno))
                para = l
        if para:
            blocks.append(Block("paragraph", para, page=pno))
    return blocks


# ----------------------------------------------------------------------------- Markdown (вакансії)

def parse_markdown(text: str) -> tuple[dict, list[Block]]:
    meta: dict = {}
    if text.startswith("---"):
        _, fm, text = text.split("---", 2)
        meta = yaml.safe_load(fm) or {}
    blocks: list[Block] = []
    para: list[str] = []

    def flush():
        if para:
            blocks.append(Block("paragraph", " ".join(para)))
            para.clear()

    for line in text.splitlines():
        s = line.strip()
        m = re.match(r"^(#{1,4})\s+(.*)", s)
        if not s:
            flush()
        elif m:
            flush()
            blocks.append(Block("heading", m.group(2), level=len(m.group(1))))
        elif s.startswith("- "):
            flush()
            blocks.append(Block("list_item", s[2:]))
        else:
            para.append(s)
    flush()
    return meta, blocks


# ----------------------------------------------------------------------------- pipeline

def content_hash(doc: Document) -> str:
    norm = re.sub(r"\s+", " ", doc.text().lower()).strip()
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()


def shingles(text: str, k: int = 5) -> set[str]:
    w = re.findall(r"\w+", text.lower())
    return {" ".join(w[i:i + k]) for i in range(max(0, len(w) - k + 1))}


def load_documents() -> tuple[list[Document], list[str]]:
    docs, missing = [], []
    for d in yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))["documents"]:
        path = DOCS_DIR / f"{d['id']}.{d['format']}"
        if not path.exists():
            missing.append(d["id"])
            continue
        blocks = parse_pdf(path) if d["format"] == "pdf" else parse_html(path, d.get("content_selector"))
        if d["doc_type"] == "law":
            blocks = postprocess_law(blocks)
        docs.append(Document(d["id"], d["title"], d["url"], d["format"], d["doc_type"], d["lang"],
                             str(d["version"]), d["license"], clean_blocks(blocks, d.get("drop_patterns"))))
    for p in sorted(VACANCIES_DIR.glob("*.md")):
        meta, blocks = parse_markdown(p.read_text(encoding="utf-8"))
        lang = "uk" if len(re.findall(r"[а-яіїєґ]", p.read_text(encoding="utf-8"), re.I)) > 500 else "en"
        docs.append(Document(meta.get("id", p.stem), meta.get("title", p.stem), meta.get("source", ""), "md",
                             "vacancy", lang, str(meta.get("published", "")), "djinni-public-posting",
                             clean_blocks(blocks)))
    return docs, missing


def deduplicate(docs: list[Document], near_threshold: float = 0.9) -> tuple[list[Document], list[dict], list[dict]]:
    """Точні дублікати (однаковий хеш нормалізованого тексту) прибираємо. Майже-дублікати (Jaccard ≥ 0.9
    по 5-словних шинглах): дві вакансії → прибираємо пізнішу (репост); довідкові документи лише позначаємо —
    це можуть бути різні версії одного звіту, які потрібні для пасток у тестсеті."""
    seen: dict[str, str] = {}
    kept, exact = [], []
    for d in docs:
        d.content_sha256 = content_hash(d)
        if d.content_sha256 in seen:
            exact.append({"dropped": d.doc_id, "duplicate_of": seen[d.content_sha256]})
            continue
        seen[d.content_sha256] = d.doc_id
        kept.append(d)
    sh = {d.doc_id: shingles(d.text()) for d in kept}
    by_id = {d.doc_id: d for d in kept}
    near, drop = [], set()
    ids = list(sh)
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            if not sh[a] or not sh[b] or a in drop or b in drop:
                continue
            j = len(sh[a] & sh[b]) / len(sh[a] | sh[b])
            if j >= near_threshold:
                # повторно опублікована вакансія — шум, прибираємо; довідкові документи лише позначаємо
                both_vacancies = by_id[a].doc_type == by_id[b].doc_type == "vacancy"
                near.append({"a": a, "b": b, "jaccard": round(j, 3), "action": f"dropped {b}" if both_vacancies else "kept both"})
                if both_vacancies:
                    drop.add(b)
    return [d for d in kept if d.doc_id not in drop], exact, near


def main() -> None:
    docs, missing = load_documents()
    docs, exact, near = deduplicate(docs)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUT_DIR / "documents.jsonl").open("w", encoding="utf-8") as fh:
        for d in docs:
            fh.write(json.dumps(asdict(d), ensure_ascii=False) + "\n")
    by_type: dict[str, int] = {}
    for d in docs:
        by_type[f"{d.doc_type}/{d.format}"] = by_type.get(f"{d.doc_type}/{d.format}", 0) + 1
    report = {
        "documents": len(docs), "by_type_format": by_type, "missing_files": missing,
        "exact_duplicates_dropped": exact, "near_duplicates": near,
        "per_document": [{"doc_id": d.doc_id, "format": d.format, "blocks": len(d.blocks),
                          "chars": len(d.text()), "sha256": d.content_sha256[:16]} for d in docs],
    }
    (OUT_DIR / "ingest_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"{len(docs)} documents → data/processed/documents.jsonl  {by_type}")
    print(f"exact duplicates dropped: {len(exact)}, near-duplicates: {near}, missing files: {missing}")


if __name__ == "__main__":
    main()
