# Heterogeneous AI Spans

![Dataset overview](assets/overview.svg)

**{{TOTAL}} mixed documents · {{TOTAL}} matched human controls · {{SPAN_TOTAL}} AI replacement spans · {{TOTAL}} distinct source passages**

An English dataset for locating AI-written sections inside otherwise human-origin documents, with exact character spans and the identity of the model that wrote each replacement. The labels are construction provenance: they record which text was retained and which text was generated, rather than the judgment of another AI detector.

Human means **non-AI**. Retained source text has documented historical provenance and remains explicitly marked as a *human-origin candidate*, not independently certified wording. Every source has `human_verified=false`; the richer JSONL also retains `authorship="human_candidate"` and `binary_target=null`.

## Start here

```python
from datasets import load_dataset

data = load_dataset("{{REPO_ID}}", "mixed")
row = data["train"][0]
for span in row["spans"]:
    fragment = row["text"][span["start"]:span["end"]]
    print(span["label"], span["reported_model"] or span["requested_model"], fragment[:120])
```

If this repository is private, sign in with `hf auth login` or provide your existing read token to the library. Do not place tokens in source files.

- **Training / evaluation:** use the Parquet configurations below.
- **Full provenance:** [records/all.jsonl](records/all.jsonl) includes original source metadata, briefs, exact prompts, model responses, quality measurements and author metadata.
- **Visual inspection:** download [the complete ZIP](heterogeneous-ai-spans-v{{VERSION}}.zip), extract it, and open `review/index.html`. The review works offline, with model, cohort and split filters; retained human text is green and AI text is orange.
- **Format details:** [schema guide](schema/README.md) and [full-record JSON Schema](schema/full-record.schema.json).
- **Verification:** [release manifest](manifest/release.json), [split counts](manifest/splits.json), [quality reports](audits), and [SHA-256 checksums](SHA256SUMS.txt).

## What's included

{{WRITER_HEADER}}
{{WRITER_TABLE}}

These are the identities recorded by the generation backends. Requested and reported identities remain separate. Opus 3 used a visibly selected browser model; its API checkpoint, revision and sampling parameters were not reported. It must not be silently equated with a particular dated API checkpoint. {{GENERATION_BACKENDS}}

Each source passage is assigned to one writer; different models did not rewrite the same source passage. Some long parent books or reports contribute multiple disjoint excerpts. The original batch includes fiction, encyclopedia/reference text, and professional economic reports. The ML subset contains narrative excerpts, **not full papers**. Its current-Opus examples had already completed before the decision to focus future Opus runs on stories.

### Version 1.1.0 expansion

The added batch contains 100 new documents each from Haiku, Sonnet and current Opus, plus 300 matched controls, with no additional Opus 3 generation. Opus uses historical-fiction passages only. Haiku and Sonnet each receive 15 Gutenberg passages, 35 Standard Ebooks passages, 25 WikiText passages and 25 Beige Book passages. New excerpt IDs, normalized texts and parent ranges are checked against the previous release. Existing book/author/report split assignments are inherited; the older 1,400 document IDs and text are retained.

The redundant **`source_text_sha256` Parquet column has been removed from every configuration**, including the previous data. Compute it from `source_text` when needed. The full provenance JSONL retains the recorded upstream hashes. The previous v1.0.0 Hub commit remains available as version history.

{{GPT_EXPANSION_SECTION}}

## Hugging Face configurations and splits

`mixed` is the default. All configurations use the same documented Parquet schema and preserve their original split assignments.

| Configuration | Train | Validation | Test | Total |
|---|---:|---:|---:|---:|
{{SPLIT_TABLE}}

```python
controls = load_dataset("{{REPO_ID}}", "human_controls")
papers = load_dataset("{{REPO_ID}}", "ml_papers_mixed")
original = load_dataset("{{REPO_ID}}", "original_mixed")
expansion = load_dataset("{{REPO_ID}}", "expansion_mixed")
{{ADDITIONAL_CONFIG_EXAMPLES}}
everything = load_dataset("{{REPO_ID}}", "all")
```

For a fixed snapshot, add `revision="v{{VERSION}}"` to `load_dataset`. The original release remains available at `revision="v1.0.0"`; its schema included the now-removed redundant source-hash column.

**The configurations overlap; do not concatenate `all` with the others.** Each control shares the original source with one mixed document and is intentionally in the same split. Source IDs, group IDs and normalized duplicate texts are isolated across train/validation/test across all cohorts. JMLR split groups are paper-level; historical fiction uses the recorded book/author grouping. The source pair is a useful unit for analysis; treating paired controls and mixed documents as independent observations can inflate confidence intervals.

## Source provenance

| Collection | Source passages | Evidence |
|---|---:|---|
| Project Gutenberg | {{COUNT_gutenberg_selected}} | Named historical works and retained raw-source hashes; current ebook revision dates are not independently established. |
| Standard Ebooks | {{COUNT_standardebooks}} | Named historical works and pinned repository commits from before 2023. |
| WikiText-2 raw | {{COUNT_wikitext2_raw}} | The 2016 Wikipedia-derived release, retained raw Parquet/article hashes and mechanical punctuation restoration. Individual article revision IDs are unavailable. |
| Federal Reserve Beige Book | {{COUNT_beigebook}} | Dated official narratives and saved source hashes; capture-version dates are not independently certified. |
| JMLR, 2000–2014 | {{COUNT_jmlr_pre2015}} | Publication year agrees between the journal index and PDF; original PDF/layout/index hashes, authors and paragraph locations are retained. |

The ML sample was selected with a seeded shuffle from JMLR volumes 1–15. {{ML_EXTRACTION_PARAGRAPHS}} Selection does not join separate eligible runs across rejected paragraphs. Extraction used Poppler rather than a language model, with recorded line joining, dehyphenation and normalization. Heuristic paragraphs can include a figure caption joined to nearby narrative, and inline mathematical notation loses some visual layout; source coordinates and the visual-review findings retain these artifacts. Selection therefore favors PDFs with usable narrative extraction; it is not a representative sample of all ML papers. Human-only source material was drawn from the upstream documented collections, not accepted on the strength of a detector score.

The historical-fiction/reference/finance source collection was pinned to upstream repository revision `b618ca42b85fcb4cd8ddcc60cf18ed957262cd6a`. The generation plans and selection details are recorded in [provenance](provenance); original source excerpts and metadata are in [sources/originals.jsonl](sources/originals.jsonl).

## How mixed documents were made

1. Choose whole blocks of 3, 4 or 6 source paragraphs using a seeded random plan, retaining surrounding human text. A document may contain one or two replacement blocks.
2. Condense each selected block into **2–3 complete sentences**, using Haiku. This is a multi-paragraph content brief, not a short list of keywords.
3. Give the assigned writer the brief plus bounded neighboring context, excluding the selected original blocks. Ask it to expand the brief into document prose.
4. Splice the generated text into the original source and construct an exact, contiguous span partition with retained-source and replacement coordinates.
5. Save the full prompt/response provenance and an unchanged control from the same source.

Successful generations were cached. Failed quality attempts were retried; the exact accepted prompt is saved. Corrective prompts for memorized fiction could change fictional names/details or use modern wording. Model-reported metadata and unreported fields are kept distinct. Subscription/UI generation does not establish deterministic replay or disclose every sampling setting.

## Quality checks

- All {{RECORD_TOTAL}} release records pass the full JSON Schema and exact reconstruction/span checks. Text and prompt content are unchanged by packaging.
- All {{SPAN_TOTAL}} AI replacement blocks passed the configured source-copy checks, including checks against the full own-parent text rather than only the excerpt. The check uses normalized exact 8-gram coverage with a 15% ceiling. This is **not an Internet-wide plagiarism check** and cannot prove that every generated sentence is novel.
- Final audits report no missing planned documents or outstanding prose-review flags.
- Five outputs per participating modern writer were reviewed in each cohort. {{PDF_REVIEW_DESCRIPTION}} The full dataset was not manually reviewed sentence by sentence.
- Two cached Opus 3 passages that substantially copied other chapters of their source books were quarantined and replaced. The final export contains their accepted replacements; the quality history is retained in the original-cohort quarantine report.
- Factual/causal fidelity, exact output length and exact paragraph count are **not** acceptance requirements. The objective is usable, correctly attributed text for a detector; rewritten paper passages are not reliable accounts of the original research.

## Span and author semantics

`start` and `end` index the resulting `text` using **Unicode code points** and half-open intervals: `text[start:end]`. They are not byte, token or JavaScript UTF-16 offsets. The spans cover every character without gaps or overlaps, including separators. The offline review converts to code points before slicing.

`source_start` / `source_end` refer to the retained original `source_text`. For a human span, its text is exactly the corresponding source slice. For an AI span, that range identifies the original block being replaced and may have a different length. AI `reported_model` and `requested_model` identify the writer; retained human spans carry the source author where known. Unknown human authors remain null.

The Parquet view intentionally provides a stable, compact author representation. The authoritative full record in JSONL contains the richer author object, full usage metadata, generation and summarization provenance, prompts, source hashes and replacement history. Join the two views by `id`. See the [schema guide](schema/README.md) for an example.

## Files and reproducibility

```text
data/<configuration>/{train,validation,test}.parquet
records/{all,mixed,human-controls}.jsonl
sources/originals.jsonl
schema/{README.md,full-record.schema.json,arrow-schema.txt}
audits/<cohort>/
provenance/                    # portable selection/configuration and copy-policy exports
manifest/                      # release, splits, upstream hashes, metadata transformations
review/index.html              # self-contained offline review of all {{RECORD_TOTAL}} records
examples/load_and_inspect.py
reproduction/                  # generator, collection/audit scripts, tests and configuration
SHA256SUMS.txt
heterogeneous-ai-spans-v{{VERSION}}.zip
```

The source text, generated text, exact spans, text hashes and prompts are preserved. Local absolute file paths are converted to portable upstream artifact hints; references that pointed at local corpus files are replaced with the recorded upstream public URL. Private Claude chat URLs, including URL-valued request IDs, are replaced with stable SHA-256 references. These changes are documented in the release manifest. Frozen original plan IDs refer to the **original local plans**; the portable planning exports are clearly labeled representations, not byte-identical plan files.

Full original PDFs/books, raw browser responses/session state, local logs, credentials and machine-specific runtime files are not included. Full-parent reference hashes and check outcomes remain available. The bundled scripts explain regeneration, but access to the upstream sources and compatible model subscriptions/backends is required; the historic browser model may not remain available.

## Intended use and limitations

Suitable for span-level AI/non-AI detection research, mixed-authorship segmentation, matched-pair experiments and exploratory model attribution. Labels describe the construction process and depend on the documented source-origin evidence. This pilot is synthetic, small, English-only and limited to {{WRITER_COUNT}} recorded writers; its performance is not a claim about real-world detector reliability or unseen models. Prompt style, source domains, extraction artifacts and replacement boundaries may become shortcuts. Do not use a classifier trained only on this collection to make consequential claims about an individual's authorship.

## Rights and citation

The collection combines different source terms. **No single permissive content license is asserted.** Attribution and the per-source rights descriptions remain in every full record and in the Parquet `source_license` field. In particular, WikiText's recorded license-version discrepancy and historical JMLR paper-specific license uncertainty remain unresolved. See [LICENSE.md](LICENSE.md) before reuse or redistribution; the packaging does not override upstream rights or service terms.

```bibtex
@misc{open_text_detector_heterogeneous_ai_spans_2026,
  title = {Heterogeneous AI Spans: Human/AI Mixed Documents},
  author = {{Open Text Detector}},
  year = {2026},
  url = {https://huggingface.co/datasets/{{REPO_ID}}},
  note = {Version {{VERSION}}; {{TOTAL}} mixed documents and {{TOTAL}} matched controls}
}
```
