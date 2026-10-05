from __future__ import annotations

from html import escape
from pathlib import Path
from urllib.parse import urlsplit

from .core import atomic_text
from .pipeline import validate_dataset


STYLE = """
:root{color-scheme:light;--ink:#20332b;--muted:#627168;--line:#dbe3dc;
 --human:#17623a;--human-bg:#e6f4e9;--ai:#974210;--ai-bg:#fff0df}
*{box-sizing:border-box}body{margin:0;background:#f6f7f2;color:var(--ink);
 font:16px/1.6 system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
main{max-width:1550px;margin:0 auto;padding:48px 40px 60px}
.eyebrow{font-size:12px;letter-spacing:.15em;text-transform:uppercase;font-weight:700;color:var(--muted)}
h1{font-size:clamp(28px,4vw,46px);line-height:1.18;letter-spacing:-.035em;margin:12px 0}
.intro{max-width:820px;color:var(--muted);margin:0 0 22px}
.legend{display:flex;gap:12px;flex-wrap:wrap;margin-bottom:30px}
.tag{padding:6px 13px;border-radius:8px;font-size:13px;font-weight:650}
.human{color:var(--human);background:var(--human-bg)}.ai{color:var(--ai);background:var(--ai-bg)}
.cards{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:24px;align-items:start}
.card{min-width:0;background:#fff;border:1px solid var(--line);border-radius:14px;overflow:hidden}
.card-head{padding:24px 28px;border-bottom:1px solid var(--line);background:#fafcf9}
h2{margin:0;font-size:24px;letter-spacing:-.025em}.model{font-size:13px;color:var(--muted);margin-top:5px;overflow-wrap:anywhere}
.counts{display:flex;gap:18px;flex-wrap:wrap;margin-top:16px;font-size:13px}
.counts strong{font-size:17px;display:block;color:var(--ink)}
.document{padding:26px 28px 30px;font:17px/1.85 Georgia,"Times New Roman",serif;
 white-space:pre-wrap;overflow-wrap:anywhere}
.fragment{box-decoration-break:clone;-webkit-box-decoration-break:clone;border-radius:2px}
.card details{border-top:1px solid var(--line);padding:16px 28px;color:var(--muted);font-size:14px}
summary{cursor:pointer;font-weight:600}summary:focus-visible{outline:2px solid var(--human);outline-offset:5px}
.brief{color:var(--ink);font:15px/1.7 system-ui,sans-serif;margin:14px 0 4px}
.original{margin-top:30px;background:white;border:1px solid var(--line);border-radius:12px;padding:18px 24px}
.original .document{padding:22px 0 4px;max-width:850px}.original .source-meta{font-size:13px;color:var(--muted);margin-top:12px}
a{color:var(--human)}footer{font-size:13px;color:var(--muted);margin-top:28px}
@media(max-width:1200px){.cards{grid-template-columns:1fr}.card{max-width:900px}.original{max-width:900px}}
@media(max-width:600px){main{padding:25px 15px 36px}.card-head{padding:20px}.document{padding:20px;font-size:16px}.card details{padding:15px 20px}.original{padding:16px 20px}}
@media print{body{background:white}main{padding:0}.cards{grid-template-columns:1fr}.card{break-inside:avoid;margin-bottom:20px}.human,.ai{print-color-adjust:exact;-webkit-print-color-adjust:exact}}
"""


def colored_text(record: dict) -> str:
    parts = []
    for span in record["spans"]:
        author = span["author"]
        label = span["label"]
        identity = (author.get("reported_model") or author["requested_model"]) if label == "ai" else (author.get("id") or "unknown source author")
        tooltip = escape(f"{'AI' if label == 'ai' else 'Human (non-AI)'} · {identity}", quote=True)
        fragment = escape(record["text"][span["start"]:span["end"]])
        parts.append(f'<span class="fragment {label}" data-start="{span["start"]}" data-end="{span["end"]}" title="{tooltip}">{fragment}</span>')
    return "".join(parts)


def render_html(records: list[dict], output: Path, limit: int = 10) -> dict:
    """Standalone, escaped HTML; colors come from construction labels."""
    validate_dataset(records)
    if limit < 1:
        raise ValueError("HTML document limit must be positive")
    mixed = [r for r in records if any(s["label"] == "ai" for s in r["spans"])][:limit]
    if not mixed:
        raise ValueError("no AI-containing records to display")
    cards = []
    sources = {}
    for record in mixed:
        ai_spans = [s for s in record["spans"] if s["label"] == "ai"]
        authors = [s["author"] for s in ai_spans]
        models = list(dict.fromkeys(a.get("reported_model") or a["requested_model"] for a in authors))
        kinds = {a["kind"] for a in authors}
        name = "GPT" if kinds == {"codex"} else "Claude" if kinds == {"claude"} else " / ".join(models)
        model_line = " · ".join(models)
        if any(a.get("reported_model") is None for a in authors):
            model_line += " · requested model"
        paragraphs = sum(r.get("source_paragraphs", 0) for r in record["replacements"])
        paragraph_caption = f"{paragraphs} original paragraphs replaced" if paragraphs else f"{len(ai_spans)} source blocks replaced"
        briefs = "".join(f'<p class="brief">{escape(r["summary"])}</p>' for r in record["replacements"])
        record_id = escape(record["id"], quote=True)
        source = record['source']
        title = escape(source.get('title') or source['id'])
        reference = source['reference']
        if urlsplit(reference).scheme in ('http', 'https'):
            title = f'<a href="{escape(reference, quote=True)}" target="_blank" rel="noopener noreferrer">{title}</a>'
        year = source.get('human_origin_basis', {}).get('publication_year')
        source_line = f'<div class="model">{title}' + (f' · {escape(str(year))}' if year else '') + '</div>'
        cards.append(
            f'<article class="card" data-record-id="{record_id}">'
            f'<header class="card-head"><h2>{escape(name)}</h2><div class="model">{escape(model_line)}</div>'
            + source_line +
            f'<div class="counts"><div><strong>{record["ai_fraction_chars"]:.0%}</strong>AI characters</div>'
            f'<div><strong>{len(ai_spans)}</strong>replacement block{("s" if len(ai_spans) != 1 else "")}</div>'
            f'<div><strong>{paragraphs or "—"}</strong>original paragraphs replaced</div></div></header>'
            f'<div class="document" aria-label="{escape(name, quote=True)} document with colored authorship spans">{colored_text(record)}</div>'
            f'<details><summary>Read the content description · {escape(paragraph_caption)}</summary>{briefs}</details></article>'
        )
        sources.setdefault(record["source"]["sha256"], record["source"])
    originals = []
    for source in sources.values():
        reference = source["reference"]
        if urlsplit(reference).scheme in ("http", "https"):
            reference_html = f'<a href="{escape(reference, quote=True)}" target="_blank" rel="noopener noreferrer">Source reference</a>'
        else:
            reference_html = escape(reference)
        originals.append(
            '<details class="original"><summary>Read the original non-AI document</summary>'
            f'<div class="source-meta">{escape(source.get("author_id") or "Unknown author")} · {reference_html}</div>'
            f'<div class="document"><span class="fragment human">{escape(source["text"])}</span></div></details>'
        )
    page = (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        '<title>Human and AI · Document comparison</title>'
        f'<style>{STYLE}</style></head><body><main>'
        '<div class="eyebrow">Heterogeneous text dataset · Example</div>'
        '<h1>Human and AI, in the same document</h1>'
        '<p class="intro">Blocks of original paragraphs were condensed into a few sentences, '
        'then expanded by their assigned model and spliced back into the source text. '
        'Read the complete resulting documents below.</p>'
        '<div class="legend" aria-label="Authorship legend"><span class="tag human">Green · Human (non-AI)</span>'
        '<span class="tag ai">Orange · AI generated</span></div>'
        + ('<p class="intro">Source-origin caveat: green text is retained from documented human-origin candidates, '
           'not independently certified human wording. AI spans have exact generation provenance.</p>'
           if any(r['source'].get('human_verified') is False for r in mixed) else '')
        +
        f'<p class="intro">Showing {len(mixed)} mixed documents from {len(sources)} distinct sources.</p>'
        f'<section class="cards" aria-label="Generated document comparison">{"".join(cards)}</section>'
        + "".join(originals)
        + '<footer>Colors follow the exact construction spans. They are known labels, not detector predictions. '
        'Hover over a passage to see its author. This file opens offline in a browser.</footer></main></body></html>'
    )
    atomic_text(output, page)
    return {"html": str(output.resolve()), "documents": len(mixed), "sources": len(sources)}
