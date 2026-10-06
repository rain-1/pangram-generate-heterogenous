from __future__ import annotations

import hashlib
import json
import os
import random
import re
import tempfile
import unicodedata
from pathlib import Path
from typing import Any, Iterable

VERSION = "1.0"
WORD = re.compile(r"\w+(?:['’\-]\w+)*", re.UNICODE)
SPLITS = ("train", "validation", "test")


def words(text: str) -> list[str]:
    return WORD.findall(text)


def digest(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalized_hash(text: str) -> str:
    return text_hash(" ".join(unicodedata.normalize("NFKC", text).casefold().split()))


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        raise ValueError("expected a JSON object")
                except (ValueError, TypeError) as error:
                    raise ValueError(f"{path}:{number}: {error}") from error
                rows.append(row)
    return rows


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".tmp-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    atomic_text(path, "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def sentence_spans(text: str) -> list[tuple[int, int]]:
    """Conservative English-oriented heuristic; offsets always refer to original text.

    Blank lines are boundaries. Common titles, initials and decimal points aren't.
    Use paragraph mode or pre-segmented source documents for other languages.
    """
    boundaries = []
    abbreviations = {"mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc", "e.g", "i.e"}
    for match in re.finditer(r'''[.!?]+["'”’\)\]]*(?=\s|$)|\n[ \t\r]*\n''', text):
        end = match.end()
        if match.group().startswith("."):
            token = re.search(r"([\w.]+)\.$", text[:match.start() + 1])
            author_abbreviation = re.search(r"\bet\s+al\.$", text[:match.start() + 1], re.I)
            if author_abbreviation or (token and (token[1].lower() in abbreviations or len(token[1]) == 1)):
                continue
        boundaries.append(end)
    boundaries.append(len(text))
    result = []
    start = 0
    for end in sorted(set(boundaries)):
        raw = text[start:end]
        left = start + len(raw) - len(raw.lstrip())
        right = end - len(raw) + len(raw.rstrip())
        if left < right:
            result.append((left, right))
        start = end
    return result


def paragraph_spans(text: str) -> list[tuple[int, int]]:
    result = []
    start = 0
    for end in [m.start() for m in re.finditer(r"\n[ \t\r]*\n", text)] + [len(text)]:
        raw = text[start:end]
        left = start + len(raw) - len(raw.lstrip())
        right = end - len(raw) + len(raw.rstrip())
        if left < right:
            result.append((left, right))
        start = end
    return result


def select_blocks(text: str, settings: dict, seed: str) -> list[dict]:
    rng = random.Random(seed)
    units = (sentence_spans if settings["unit"] == "sentence" else paragraph_spans)(text)
    def eligible(start: int, end: int) -> bool:
        if settings.get("require_prose_paragraphs"):
            fragment = text[start:end]
            if any(not re.search(r'''[.!?]["'”’\)\]]*$''', fragment[a:b]) for a, b in paragraph_spans(fragment)):
                return False
        return len(sentence_spans(text[start:end])) > settings["summary_max_sentences"]

    selected: list[tuple[int, int, int]] = []
    if settings["mode"] == "alternating":
        sizes = [n for n in settings["block_sizes"] if len(units) >= 2 * n]
        if not sizes:
            return []
        n = rng.choice(sizes)
        # Keep the first N units human; the final partial chunk remains human.
        for i in range(n, len(units) - n + 1, 2 * n):
            start, end = units[i][0], units[i + n - 1][1]
            if eligible(start, end):
                selected.append((start, end, n))
    else:
        candidates = []
        for n in settings["block_sizes"]:
            for i in range(len(units) - n + 1):
                start, end = units[i][0], units[i + n - 1][1]
                if eligible(start, end):
                    candidates.append((start, end, n))
        rng.shuffle(candidates)
        target = rng.randint(settings["min_blocks"], settings["max_blocks"])
        total = 0
        for start, end, n in candidates:
            if total + end - start > len(text) * settings["max_replaced_fraction"]:
                continue
            if any(start < b and end > a for a, b, _ in selected):
                continue
            selected.append((start, end, n))
            total += end - start
            if len(selected) == target:
                break
        if len(selected) < settings["min_blocks"]:
            return []
    return [
        {"start": a, "end": b, "units": n,
         "paragraphs": len(paragraph_spans(text[a:b])),
         "sentences": len(sentence_spans(text[a:b])), "characters": b - a}
        for a, b, n in sorted(selected)
    ]


def copy_fraction(generated: str, original: str, n: int) -> float:
    """Fraction of generated words covered by exact normalized source n-grams."""
    generated_words = [w.casefold() for w in words(generated)]
    source_words = [w.casefold() for w in words(original)]
    if len(generated_words) < n:
        return float(bool(generated_words) and generated.strip() == original.strip())
    source = {tuple(source_words[i:i + n]) for i in range(len(source_words) - n + 1)}
    covered = set()
    for i in range(len(generated_words) - n + 1):
        if tuple(generated_words[i:i + n]) in source:
            covered.update(range(i, i + n))
    return len(covered) / len(generated_words)


def validate_partition(text: str, spans: list[dict]) -> None:
    if not isinstance(text, str) or not text or not isinstance(spans, list) or not spans:
        raise ValueError("text and spans must be nonempty")
    cursor = 0
    for span in spans:
        start, end = span.get("start"), span.get("end")
        if type(start) is not int or type(end) is not int or start != cursor or not start < end <= len(text):
            raise ValueError(f"spans must partition [0, {len(text)}) without gaps or overlaps")
        if span.get("label") not in ("human", "ai"):
            raise ValueError("span label must be human or ai")
        if "text" in span and span["text"] != text[start:end]:
            raise ValueError("span text does not match its offsets")
        cursor = end
    if cursor != len(text):
        raise ValueError("spans do not cover the document")


def valid_source_origin(source: dict, allow_candidates: bool = False) -> bool:
    if source.get("human_verified") is True:
        return True
    return (allow_candidates and source.get("human_verified") is False
            and source.get("authorship") == "human_candidate"
            and isinstance(source.get("human_origin_basis"), dict)
            and bool(source["human_origin_basis"]))


def validate_record(record: dict) -> None:
    if not isinstance(record.get("id"), str) or not record["id"]:
        raise ValueError("record id must be a nonempty string")
    if record.get("schema_version") != VERSION:
        raise ValueError("unsupported schema_version")
    if record.get("offset_unit") != "unicode_codepoint" or record.get("end_exclusive") is not True:
        raise ValueError("expected Unicode codepoint, half-open offsets")
    if record.get("split") not in SPLITS:
        raise ValueError("invalid split")
    text = record["text"]
    spans = record["spans"]
    validate_partition(text, spans)
    if record.get("text_sha256") != text_hash(text):
        raise ValueError("text hash mismatch")
    source = record["source"]
    original = source["text"]
    for field in ("id", "group_id", "reference", "provenance"):
        if not isinstance(source.get(field), str) or not source[field]:
            raise ValueError(f"source {field} must be a nonempty string")
    if source.get("normalized_sha256") != normalized_hash(original):
        raise ValueError("normalized source hash mismatch")
    allow_candidates = record.get("construction", {}).get("source_origin_policy") == "documented_candidates"
    if source.get("sha256") != text_hash(original) or not valid_source_origin(source, allow_candidates):
        raise ValueError("invalid human source provenance")
    if source.get("human_verified") is False and "candidate" not in record.get("label_definition", {}).get("human", ""):
        raise ValueError("candidate source labels must disclose uncertain source origin")
    replacements = {item["id"]: item for item in record["replacements"]}
    if len(replacements) != len(record["replacements"]):
        raise ValueError("duplicate replacement ids")
    cursor = 0
    seen = []
    for span in spans:
        start, end = span["source_start"], span["source_end"]
        if type(start) is not int or type(end) is not int or start != cursor or not start < end <= len(original):
            raise ValueError("source spans must partition the original document")
        fragment = text[span["start"]:span["end"]]
        if span["label"] == "human":
            if fragment != original[start:end] or span.get("author") != {"type": "non_ai", "id": source.get("author_id")} or span.get("replacement_id") is not None:
                raise ValueError("human span must be an exact, unattributed source slice")
        else:
            replacement = replacements.get(span.get("replacement_id"))
            author = span.get("author")
            if not isinstance(author, dict) or author.get("type") != "ai" or not author.get("requested_model"):
                raise ValueError("AI span needs generator provenance")
            if replacement is None or replacement["source_start"] != start or replacement["source_end"] != end:
                raise ValueError("AI span replacement/source mismatch")
            if fragment != replacement["generated_text"] or author != {"type": "ai", **replacement["generation"]["provenance"]}:
                raise ValueError("AI span generation mismatch")
            if replacement["generation"]["text"] != fragment or replacement["summarization"]["text"] != replacement["summary"]:
                raise ValueError("replacement completion text mismatch")
            for completion in (replacement["generation"], replacement["summarization"]):
                if text_hash(completion["prompt"]) != completion["provenance"]["prompt_sha256"]:
                    raise ValueError("replacement prompt hash mismatch")
            if not replacement.get("summary"):
                raise ValueError("replacement needs a summary")
            seen.append(span["replacement_id"])
        cursor = end
    if cursor != len(original) or len(seen) != len(replacements) or len(set(seen)) != len(seen):
        raise ValueError("incomplete source or replacement coverage")
    fraction = sum(s["end"] - s["start"] for s in spans if s["label"] == "ai") / len(text)
    if type(record.get("ai_fraction_chars")) not in (int, float) or not 0 <= record["ai_fraction_chars"] <= 1 or abs(record["ai_fraction_chars"] - fraction) > 1e-12:
        raise ValueError("AI fraction mismatch")


def assemble(source: dict, replacements: list[dict], record_id: str, split: str, metadata: dict) -> dict:
    original = source["text"]
    pieces, spans = [], []
    source_cursor = final_cursor = 0

    def append(fragment: str, start: int, end: int, replacement: dict | None = None) -> None:
        nonlocal final_cursor
        if not fragment:
            return
        spans.append({
            "start": final_cursor, "end": final_cursor + len(fragment),
            "label": "ai" if replacement else "human",
            "source_start": start, "source_end": end,
            "author": {"type": "ai", **replacement["generation"]["provenance"]} if replacement else {"type": "non_ai", "id": source.get("author_id")},
            "replacement_id": replacement["id"] if replacement else None,
        })
        pieces.append(fragment)
        final_cursor += len(fragment)

    for replacement in sorted(replacements, key=lambda r: r["source_start"]):
        start, end = replacement["source_start"], replacement["source_end"]
        if not source_cursor <= start < end <= len(original):
            raise ValueError("replacement ranges overlap or fall outside the source")
        append(original[source_cursor:start], source_cursor, start)
        append(replacement["generated_text"], start, end, replacement)
        source_cursor = end
    append(original[source_cursor:], source_cursor, len(original))
    text = "".join(pieces)
    record = {
        "schema_version": VERSION, "id": record_id, "split": split,
        "offset_unit": "unicode_codepoint", "end_exclusive": True,
        "label_definition": {"human": ("retained human-origin candidate source; non-AI label is conditional on documentary provenance"
                                          if source.get("human_verified") is False else "retained verified non-AI source"),
                             "ai": "LLM-emitted replacement prose"},
        "text": text, "text_sha256": text_hash(text), "spans": spans,
        "source": {**source, "sha256": text_hash(original)},
        "replacements": replacements, "construction": metadata,
        "ai_fraction_chars": sum(s["end"] - s["start"] for s in spans if s["label"] == "ai") / len(text),
    }
    validate_record(record)
    return record


def token_labels(spans: list[dict], offsets: list[tuple[int, int]]) -> list[int]:
    """Majority character overlap: human=0, AI=1, special/empty/tie=-100.

    Pass a fast tokenizer's offset_mapping for the FINAL document. Offsets must
    use Unicode codepoints. Ties crossing a provenance boundary are ignored.
    """
    result = []
    cursor = 0
    previous = 0
    for start, end in offsets:
        if start == end:
            result.append(-100)
            continue
        if start < previous or not 0 <= start < end <= spans[-1]["end"]:
            raise ValueError("token offsets must be ordered and within the document")
        previous = start
        while spans[cursor]["end"] <= start:
            cursor += 1
        counts = [0, 0]
        for span in spans[cursor:]:
            if span["start"] >= end:
                break
            counts[span["label"] == "ai"] += max(0, min(end, span["end"]) - max(start, span["start"]))
        result.append(-100 if counts[0] == counts[1] else int(counts[1] > counts[0]))
    return result
