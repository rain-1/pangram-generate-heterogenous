# Record and span format

The release has two representations joined by the same document `id`:

1. **Parquet training/viewer records** have the stable schema in `arrow-schema.txt`. These expose text, original source text, source/model identities and spans directly to `datasets.load_dataset`.
2. **Full JSONL records** conform to `full-record.schema.json` and retain exact briefs/prompts, response text, nested author provenance, quality checks and source metadata.

## Coordinates

All offsets count **Unicode code points**, begin at zero and have an exclusive end. Python's ordinary string slicing gives the labeled fragment. An emoji may count as more than one UTF-16 code unit in JavaScript; use `Array.from(text).slice(start,end).join("")` there. Combining marks remain separate code points. Do not normalize, trim or change line endings before applying stored spans.

Every resulting document has a contiguous, ordered partition: first span starts at 0, last ends at `len(text)`, and each span begins where the preceding span ended. Human spans use `label="human"`, meaning retained non-AI source-candidate text. AI spans use `label="ai"`.

## Parquet columns

| Columns | Meaning |
|---|---|
| `id`, `split`, `kind`, `cohort` | Document identifier, preserved split, `mixed`/`human_control`, {{COHORT_CONFIGS}}. |
| `text`, `text_sha256` | Resulting exact document and SHA-256 of UTF-8 bytes. |
| `source_text` | Original excerpt before replacement. Its hash can be computed when needed. |
| `source_id`, `group_id`, `source_dataset` | Original excerpt, split-isolation group and source collection. |
| `source_author`, `source_title`, `source_reference`, `source_license` | Attribution, recorded title, upstream URL and rights description. |
| `human_verified` | False throughout: historical evidence is retained without certifying all wording. |
| `generator_backend`, `requested_model`, `reported_model`, `model_identity_status` | The writer of the mixed document; null for unchanged controls. |
| `offset_unit`, `end_exclusive` | Always `unicode_codepoint` and true. |
| `ai_fraction_chars` | Proportion of resulting code points assigned to AI. |
| `spans` | List of span objects, each including a stable compact author representation. |

Each Parquet span contains:

Version 1.1.0 removes the redundant `source_text_sha256` column from every Parquet configuration. To compute a source hash, use `hashlib.sha256(row["source_text"].encode("utf-8")).hexdigest()`. Original upstream/source hashes remain in the rich JSONL provenance.

```json
{
  "start": 120,
  "end": 540,
  "label": "ai",
  "source_start": 120,
  "source_end": 700,
  "replacement_id": "example-replacement-id",
  "author_type": "ai",
  "author_id": null,
  "backend": "haiku",
  "requested_model": "claude-haiku-4-5-20251001",
  "reported_model": "claude-haiku-4-5-20251001",
  "model_identity_status": "reported"
}
```

This is an illustrative object, not a particular real document. For human spans, `author_type="non_ai"`, `author_id` identifies the recorded source author or is null, model fields are null, `model_identity_status="source_candidate"`, and `replacement_id` is null.

An AI span's source coordinates identify the replaced original block, not a word-by-word alignment. Source and output lengths need not match. Result spans include adjacent whitespace assigned by the construction; they partition the complete string.

## Full JSONL structure

```text
schema_version, id, split, offset_unit, end_exclusive
label_definition
text, text_sha256, ai_fraction_chars
spans[]
  start, end, label, source_start, source_end, replacement_id
  author: {type, id?} or rich AI provenance object
source
  id, group_id, reference, text, sha256, normalized_sha256
  author_id, dataset, genre, title, license, human_origin_basis
  human_verified=false, authorship=human_candidate, binary_target=null
  upstream_metadata, preprocessing?, paragraph_indices?, parent_character_range?
replacements[]
  id, source_start, source_end, source_units, source_paragraphs, ...
  summary, summarization: {text, prompt, provenance, quality, ...}
  generated_text, generation: {text, prompt, provenance, quality, ...}
construction
  kind, plan_id, seed, prompt_version, selection_unit, ...
```

The rich AI author/provenance includes backend kind, requested/reported identity, revision/provider where reported, finish reason, generation time, exact prompt hash, request reference, sampling, usage and metadata. Unreported fields remain null/empty. Codex writers retain `reported_model=null` and `model_identity_status="requested_only"` when the CLI does not report the served identity; `requested_model` still identifies the explicitly requested writer. The selected browser Opus 3 identity is `ui_label_only`, without a claimed exact API checkpoint. Private browser URL values are replaced with stable SHA-256 references; local paths become explicit upstream artifact hints.

`source.text` is the canonical excerpt used for reconstruction. `source_start/source_end` always index that string, not a whole PDF or entire book. Where available, `parent_character_range`, raw hashes and PDF paragraph/page locations identify the earlier extraction stage. Human spans reproduce the source slice exactly; replacing the recorded source blocks with `generated_text` reconstructs the resulting document exactly.

## Loading full metadata

```python
import json
from huggingface_hub import hf_hub_download

path = hf_hub_download("{{REPO_ID}}", "records/all.jsonl", repo_type="dataset")
with open(path, encoding="utf-8") as handle:
    record = json.loads(next(handle))
for span in record["spans"]:
    print(span["label"], span["author"], record["text"][span["start"]:span["end"]][:80])
```

The flat Parquet spans use explicit author columns; they do not themselves conform to the full-record JSON Schema. Validate the full JSONL when using that schema. Use `examples/load_and_inspect.py` for a dependency-free integrity check of the downloaded full records.
