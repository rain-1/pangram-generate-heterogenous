# Generate heterogeneous human/AI documents

For separate MAGPIE-style synthetic user prompts and multi-turn follow-ups, see
[the prompt generator](promptgen/README.md). It runs as `python -m promptgen`
and uses its own configuration and dataset format.

Generate a dataset for **span annotation of non-AI versus AI text**, with the
author of each span recorded for later generator attribution. Start with verified
non-AI documents, select passages, compress each passage into a short content
brief, generate new prose from the brief, and splice it into the original.

The Python 3.11+ CLI has no runtime dependencies. It supports Codex and Claude
Code subscriptions through their installed CLIs, OpenRouter, and an
OpenAI-compatible server such as vLLM on your GPU machine.

## Connection to Pangram 4

[Pangram 4, §2.2](https://arxiv.org/html/2607.27183v1#S2.SS2) uses topic briefs
to generate synthetic mirrors and rejects excessive copying. Its heterogeneous
evaluation in [§5.5.2](https://arxiv.org/html/2607.27183v1#S5.SS5.SSS2) replaces
alternating chunks of 1, 4, 8, 12, 16, or 20 sentences. This project applies the
brief bottleneck to blocks of several paragraphs, condensing each into a few
sentences before expanding it again. It also supports random-block sampling.
It implements dataset construction, not the detector architecture.

Our label is **construction provenance of the final wording**: retained source
text is `human` (meaning non-AI), and inserted LLM output is `ai`. The underlying
ideas can remain human. Pangram's three-class task also includes AI-assisted
editing; this binary dataset does not represent that editing task. Exact spans
come from the splice operation, so no similarity-based label inference is needed.

## Input

One JSON object per line:

```json
{
  "id": "collection:document-001",
  "group_id": "collection:author-or-book-001",
  "reference": "file:///corpus/document-001.txt",
  "author_id": "author-001",
  "text": "The complete original document goes here...",
  "human_verified": true,
  "provenance": "Describe the evidence that this source is non-AI",
  "language": "en",
  "genre": "essay",
  "license": "Your actual source license"
}
```

Required fields: `id`, `reference`, `text`, `human_verified=true`, `provenance`.
The verification field is your assertion, not an automatic determination.
For a research pilot using documented human-origin candidates, explicitly set
`dataset.allow_human_candidates=true`. Such inputs must retain
`human_verified=false`, `authorship="human_candidate"`, and a nonempty
`human_origin_basis` object alongside the usual source reference and provenance.
The output preserves those fields and declares the retained-source label as
conditional on documentary provenance. This option does not certify candidate
wording or change an upstream null authorship target into a confirmed label.
The default still requires verified non-AI sources.
Use evidence such as original historical publication, pre-LLM archival records,
or a documented writing process. Record a source's actual permissions and license.
`author_id` is optional; an unknown non-AI author is represented as `null`.
An optional `generator_names` list assigns a source to specific configured
backends, for example `["haiku"]`. Without it, each generator receives every
source. These assignments are frozen into the plan and respected during
generation and assembly, allowing disjoint source sets for a model comparison.
Other metadata is retained, including dates, source domain, or language background.

`group_id` defaults to the source ID. Set it to the author, book, conversation,
article family, or another leakage boundary when multiple documents are related.
An optional `split` explicitly sets `train`, `validation`, or `test`.
Shared groups and normalized exact duplicates are joined transitively before
splitting. Conflicting explicit splits are rejected. Near-duplicate documents
still require upstream deduplication or shared group IDs.

## Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
cp config.example.toml config.toml
# Edit model IDs, endpoints, backend list and dataset settings in config.toml.

heterogeneous plan --input data/human.jsonl --config config.toml --out runs/pilot
heterogeneous run --input data/human.jsonl --config config.toml --out runs/pilot --jobs 4
heterogeneous validate runs/pilot/dataset.jsonl
```

You can use `python -m heterogeneous` in place of `heterogeneous` without installing.
`plan` performs no model calls and reports the planned counts before retries.
Start with a small corpus; generation uses your subscriptions/API budget.
Remove unwanted `[[generators]]` tables or use `--backend NAME`.

A small public-domain input and a subscription-only configuration are included:

```bash
python -m heterogeneous run --input examples/human.jsonl \
  --config examples/config.pilot.toml --out runs/example-v2 --jobs 2
```

[examples/mixed-document.json](examples/mixed-document.json) is an actual Codex
output from this pilot, with a non-AI source from the final chapter of Jane Austen's
*Pride and Prejudice*. The source is taken verbatim from
[Project Gutenberg](https://www.gutenberg.org/files/1342/1342-0.txt).
The pilot configuration pins `gpt-6-luna` and `claude-haiku-4-5-20251001`; adjust
them if your account does not offer these IDs. This one-source example demonstrates
the format and integrations; it cannot provide a meaningful train/test benchmark.

Separate stages support inspecting briefs, retrying failed work, and generating
on different machines:

```bash
heterogeneous summarize --out runs/pilot --jobs 2
heterogeneous generate --out runs/pilot --backend codex --jobs 1
heterogeneous generate --out runs/pilot --backend claude --jobs 1
heterogeneous generate --out runs/pilot --backend local --jobs 8
heterogeneous generate --out runs/pilot --backend openrouter --jobs 4
heterogeneous assemble --out runs/pilot
```

The directory contains the frozen `plan.json`, per-call caches, `dataset.jsonl`,
`train.jsonl`, `validation.jsonl`, `test.jsonl`, and `manifest.json`.
Assembling a selected backend produces suffixed files such as `dataset-local.jsonl`.
Do not concatenate these blindly: each includes the same human controls.
Use the combined assembly after copying all backend caches into one run directory.
If the local server is on another machine, transfer the plan and summary cache and
run the local generation stage there, then copy back `cache/generation/local/`.

Successful calls are written atomically and reused on reruns. A changed input,
model configuration, or dataset setting requires a new run directory. Selection
and split assignment are seeded; LLM output itself is not guaranteed deterministic.
Run one process per backend per directory. Per-backend concurrency limits apply
inside a process. A crash after an accepted server call but before its cache write
can require a duplicate paid request; provider calls are not exactly-once.

The current prompt version is 3: paragraph blocks and sentence-based descriptions,
with explicit closing-event fidelity and whole neighbouring paragraphs for continuity.
The earlier word-based settings (`summary_min_words`, `summary_max_words`,
`summary_ratio`, `min_block_words`, and word-length ratios) are no longer accepted.
Use the updated config and a fresh run directory; old cached prompts cannot be
used for a new run. Existing dataset records retain their original audit trail.

`--allow-partial` on assembly explicitly permits missing variants and lists them
in the manifest. Default assembly fails when anything is missing. It never makes
model calls or relabels failed replacements as human.

## Backends

**Codex subscription:** sign in with `codex login`. The adapter checks for ChatGPT
login, removes API-key overrides from its child environment, and calls
`codex exec` with an explicit model, ephemeral session, read-only sandbox, and
final-response capture. It ignores user configuration to avoid loading a different
model provider. See official [authentication](https://developers.openai.com/codex/auth/)
and [non-interactive documentation](https://developers.openai.com/codex/noninteractive/).
Codex event output may not report the actual served model; in that case
`reported_model=null` and the requested identity remains explicit.

**Claude Code subscription:** sign in through `claude` or `claude auth login`.
The adapter checks for claude.ai login and uses print mode, JSON output, tools and
MCP disabled, no persistent session, and an empty temporary working directory.
API-key/cloud-provider overrides are removed from its child environment. Do not
add `--bare`: that mode skips subscription OAuth. See
[authentication](https://code.claude.com/docs/en/authentication) and
[programmatic use](https://code.claude.com/docs/en/headless).
`modelUsage` supplies the reported identity when it contains exactly one model.
Otherwise retain the map and use `reported_model=null`.

Use exact model IDs where available. Aliases can change over time. CLI backends
retain their coding-agent prompt scaffolding; record `kind` when comparing them
with plain chat-completion outputs. They do not provide the same sampling controls
as HTTP backends. `cli_args` allows Codex `-c` / `--config` overrides for
`model_reasoning_effort`, `model_reasoning_summary` and `model_verbosity`, or
Claude `--effort`. Model, output and authentication overrides are rejected.

**OpenRouter:** set `OPENROUTER_API_KEY`. The adapter sends chat completions and
records the response model, usage, fingerprint and routing metadata when returned.
It requests no cross-model fallback. You may pin a provider through `extra_body`.
See [OpenRouter's API documentation](https://openrouter.ai/docs/api/api-reference/chat/send-chat-completion-request).
Provider identities and aliases should be kept distinct from model families.

**4×A100 / open weights:** run an OpenAI-compatible vLLM server on the machine.
For a model requiring multiple cards, a starting command is:

```bash
vllm serve /models/your-pinned-checkpoint \
  --served-model-name your-model-id \
  --tensor-parallel-size 4 \
  --dtype bfloat16 \
  --max-model-len 8192 \
  --generation-config vllm \
  --host 127.0.0.1 --port 8000
```

Point `base_url` at `http://127.0.0.1:8000/v1`. Record the weights revision in
`revision` and use the same served model name in the config. Through an SSH tunnel
the endpoint can also be used from your laptop. Choose the model and batch/context
limits based on whether your cards have 40 GB or 80 GB and on KV-cache requirements.
For a model that fits on one GPU, replicas can provide better throughput than
four-way tensor parallelism. See vLLM's
[serving](https://docs.vllm.ai/en/stable/serving/online_serving/) and
[parallelism](https://docs.vllm.ai/en/stable/serving/parallelism_scaling/) documentation.
Reasoning models can need a larger token budget or a model-specific chat-template
option to disable thinking. Truncated/non-prose completions are rejected.

## JSON contract

[dataset.schema.json](dataset.schema.json) is the machine-readable contract.
Every record has the final `text`, a complete `spans` partition, an embedded
original `source` with `reference`, and a `replacements` audit trail.

| Field | Meaning |
| --- | --- |
| `spans[].start`, `end` | Half-open offsets into final `text` |
| `spans[].label` | `human` (non-AI) or `ai` |
| `spans[].author.type` | `non_ai` or `ai` |
| `spans[].author.id` | Source author ID, or null; non-AI spans only |
| `spans[].author.requested_model`, `reported_model` | Requested and backend-reported AI model identities |
| `spans[].source_start`, `source_end` | Original source slice retained or replaced |
| `spans[].replacement_id` | Link to the replacement's summary and generation; null for non-AI |
| `replacements[]` | Brief, prompts, generated passage, authors, settings, usage and quality checks |
| `source.reference`, `sha256` | Original location and exact UTF-8 content hash |
| `construction.plan_id` | Frozen run configuration and selection identity |
| `ai_fraction_chars` | Fraction of final characters assigned to AI |

Offsets count **Unicode codepoints**, not UTF-8 bytes or JavaScript UTF-16 units.
In Python, `text[start:end]` selects the exact span. In JavaScript use
`Array.from(text).slice(start, end).join("")`. Do not normalize or reformat text
after computing offsets. Every character, including whitespace, is covered once.
Source separators retain non-AI provenance. AI-internal whitespace belongs to AI.
Generated leading/trailing whitespace is stripped before insertion; the stored
generation text is the inserted text. Adjacent AI replacements remain separate
when they have distinct audit IDs, even if their labels and model are identical.

`source_start:source_end` is a replacement map, not a word-level alignment.
Hashes, partition coverage, exact retention of human slices, AI authors, replacement
coverage, and split isolation are checked by `validate`. JSON Schema describes
types; relationships involving actual text require the runtime validator.

## Sampling, quality and evaluation

Default sampling chooses 1–3 non-overlapping blocks of variable size while retaining
non-AI source text. Each block contains 3, 4, or 6 consecutive paragraphs by default.
The source replacement fraction is capped in random mode;
the final AI fraction is measured after generation and can differ because lengths
change. `mode="alternating"` keeps the first N units human and alternates complete
N-unit replacement chunks; short final remainders remain human. The `min_blocks`,
`max_blocks` and fraction cap apply only to random mode. Ineligible short chunks
are skipped, so inspect the actual spans rather than assuming strict alternation.
For a fixed-resolution benchmark set `block_sizes=[N]` and one variant per source.
`unit="sentence"` is still available for sentence-chunk experiments, but a block
must contain more sentences than the configured maximum description length.

Descriptions contain 2–3 concise, complete sentences in one paragraph by default.
They capture the important entities, facts, relationships and progression across
the whole selected block. They must be shorter than the source block; the measured
character compression ratio is recorded for inspection, without a hard ratio target.
All generators receive the same description and selection for a given
variant. The original selected passage is never included in the generation prompt.
`context_chars=0` enforces the strict brief bottleneck; positive values pass retained
neighbouring text, excluding every selected block. Store and evaluate this condition
separately because it supplies additional human wording to the generator.
The budget is soft: the nearest complete paragraph is included even if it exceeds
`context_chars`, so a sentence is not cut in the middle.

Generation expands the description into the original number of paragraphs,
separated by blank lines, developing its ideas with supporting detail and transitions.
There are no word-count targets or limits. Paragraph count controls the replacement
structure; comparable detail is a prompt instruction rather than an exact length guarantee.

Quality gates check description sentence count and punctuation, compression,
replacement paragraph count, code fences, empty/NUL output,
and the generated-word coverage of source-matching n-grams. Checks use the entire
source document. There are bounded quality retries and HTTP/transient retries;
authentication and CLI quota failures are retained for manual recovery. These
checks do not verify factual accuracy, semantic continuity or subtle copying.
Inspect a pilot for topic fidelity, seams, formatting shortcuts and refusal text.
Sentence segmentation is an English-oriented heuristic; use paragraph mode or
curated input segmentation for other languages and unusual punctuation. The copying
check still uses regex word n-grams; that check needs adaptation for unsegmented CJK
text. Word n-grams are only a copying metric, not the units of summary or generation.

Keep fully non-AI controls for false-positive measurement, and include varied
genres, dates, lengths, languages and language backgrounds in the actual corpus.
Stratify held-out evaluation by block size, final AI fraction, genre, model and
backend. Reserve unseen model families and newer models to measure generalization.
Synthetic splices alone do not establish performance on real collaborative writing;
a manually verified real-world holdout is a separate dataset.

Predictions should be JSONL containing `id`, `text_sha256`, and a full partition
of predicted `spans` with `start`, `end`, `label`:

```bash
heterogeneous evaluate --gold runs/pilot/test.jsonl \
  --predictions predictions.jsonl --boundary-tolerance 20
```

This reports character precision/recall/F1/accuracy, one-to-one boundary matches
within the configured codepoint tolerance, document AI-fraction MAE, and the
fraction of fully non-AI controls with any predicted AI. These are explicitly
character metrics, not Pangram's token metrics. Undefined metrics are null.
Token labels can be derived later using `heterogeneous.core.token_labels` with
a fast tokenizer's offset mapping; special tokens and ties are ignored.

Train using final text as input and spans/authors as supervision. Never feed the
source document, brief, prompt, authors or provenance metadata into the detector.
These contain the answer. Generator attribution is preserved here but no attribution
classifier or detector training is implemented.

## Tests

```bash
python -m unittest discover -s tests -v
```

Tests use synthetic fixtures solely to test software, never as verified training
documents. They cover Unicode splicing, source reconstruction, split isolation,
resume/missing-result behavior, quality checks and backend response parsing.
The small example was also generated live through both subscription CLIs.
The HTTP adapters are tested against a local mock server; actual OpenRouter and
GPU-server access require your API key/server and have not been exercised here.

## Colored HTML preview

Export a standalone HTML comparison with retained non-AI text in green and generated
text in orange. It includes the original document and each replacement's content
description in expandable sections. Colors use the stored construction spans.

```bash
python -m heterogeneous html runs/example-v2/dataset.jsonl \
  --output examples/comparison.html
```

Open the file directly in any browser. No server, external assets, or network
connection is required. The exporter validates the source JSONL and escapes its
text before rendering. It displays up to 10 mixed records by default; change that
with `--limit`.

## Documentary-source pilot and verification

`examples/config.heterogeneous-100x4.toml` assigns one writing model per source:
Haiku, Sonnet, Opus, or browser-only Opus 3. `scripts/prepare_span_pilot.py`
freezes 400 disjoint passages across four dated source collections, retaining
the upstream candidate-authorship caveat and original source hashes. Regional
Federal Reserve reports are kept within their district boundaries.

`scripts/run_span_pilot.py smoke --run RUN` generates a small quality probe.
`verify-five --run RUN` completes five documents per current CLI writer. Successful
calls are cached. `scripts/review_span_pilot.py` records independent model reviews;
these are fallible triage, not human ground truth. The direct read-through notes
and exact-integrity checks can be rendered with `scripts/build_span_verification.py`.
Install the `verification` optional dependency, or run that script with
`uv run --no-project --with jsonschema python` and `PYTHONPATH=.`.

For Opus 3, `scripts/browser_span_bridge.py --run RUN` serves a loopback HTML
form that displays queued prompts and accepts browser-visible responses with
their chat URLs. It never calls private claude.ai APIs. Use a fresh chat with
Opus 3 visibly selected for each request; the model picker may reset on a new
chat. Responses are checked before insertion, and rejected attempts are saved
with corrective feedback for retries. UI attribution records the label `Opus 3`,
not an unreported API checkpoint or sampling settings.

The full `all` stage requires a `quality-gate.json` decision of `proceed` for the
same frozen plan. The detector task-fitness review checks exact labels, intact
retained text, generation provenance and usable prose. Factual drift and length
differences are descriptive notes, not reasons to discard detector examples.
Paragraph counts remain prompt targets, but are not a hard rejection rule when
the matching detector-task gate declares `paragraph_count_policy: soft_target`.
The first 15-document verification is saved under
`runs/heterogeneous-pilot-20261005-v2/verification-15.html`, with separate usable,
review-needed, and rejected JSONL files.

Audit the current assembled dataset with
`PYTHONPATH=. python scripts/audit_span_pilot.py --run RUN` (requires `jsonschema`).
This checks schema, exact retained text and spans, cache/prompt integrity, distinct
source assignments, split isolation and model attribution. It records length ratios
and possible refusal/introductory-text flags without treating factual fidelity as
a detector quality gate. Add `--require-complete` to require every planned mixed
document: 100 per writer for the four-writer pilot. The report is
`detector-quality-audit.json`.

## Historical machine-learning papers

`runs/ml-papers-20261005` freezes 300 different JMLR papers published in
2000–2014, assigned 100 each to Haiku, Sonnet and current Opus. Each input is
a continuous excerpt of at least six recovered narrative paragraphs, rather
than a full paper with equations and bibliography. Raw PDFs, journal indexes,
layout XML, hashes, author names and paragraph page/bounding-box locations are
retained. Publication years must agree between the journal index and PDF.
No model is used for extraction; the preprocessing records line joining,
dehyphenation and normalization. Historical evidence is strong, while the
explicit candidate-authorship status and historical-license caveat remain.

`scripts/prepare_ml_papers.py --run RUN --config CONFIG --count 300` prepares
the source corpus and frozen plan. It requires the `sources` optional
dependencies and Poppler's `pdftotext`. `source-check.json` records integrity
checks, and `source-visual-review.json` records the five rendered PDF-page
checks. The frozen `verification-selection.json` selects five documents per
writer for the generation probe.

```bash
PYTHONPATH=. python scripts/run_ml_paper_pilot.py --run RUN --scope probe
# Review model-probe-review.html and record the decision in model-probe-audit.json.
PYTHONPATH=. python scripts/run_ml_paper_pilot.py --run RUN --scope all --jobs 20
PYTHONPATH=. python scripts/run_ml_paper_pilot.py --run RUN --scope assemble
PYTHONPATH=. python scripts/audit_span_pilot.py --run RUN --require-complete
```

The full stage requires a matching successful source review and model-probe
decision. Cached successes are reused on resumption. Outputs include
`mixed.jsonl`, `human-controls.jsonl`, complete span/source/provenance records
in `dataset.jsonl`, and all completed mixed documents in `review.html`.
The 15-document probe passed detector-task review; factual drift and output
length remain nonblocking. Paper IDs define split groups, so a paper and its
unchanged control cannot cross train/validation/test boundaries.

Browser request scheduling may use independent fresh chats with reservations
to avoid duplicate work. Exceptional additional quality attempts are recorded
in `browser-extra-attempts.json` against the frozen plan; they do not relax copy
checks or overwrite successful generations. Stronger retry prompts can change
fictional details to avoid reconstructing memorized books, and the exact prompt
and rejected attempts remain in provenance.

The optional `parent-copy-policy.json` pins full parent references without
sending them to the model. It catches a memorized paragraph from a different
chapter that an excerpt-only check would miss, with Unicode/apostrophe
normalization and the same 15% copy threshold. Books and reports use the saved
complete parent text, WikiText uses the hashed raw release and original article
row bounds, and JMLR uses the recovered full paper body. Build/audit these
references with `scripts/audit_parent_copy.py --run RUN` (`pyarrow` is required
for WikiText). `--quarantine` preserves flagged cache files and records retry
selections; it never overwrites those earlier outputs. Regenerate the affected
blocks, assemble again and rerun the audit. `parent-copy-audit.json` records the
coverage and each block's score; it is an own-parent check, not an Internet-wide
plagiarism search.

The completed handoff is `runs/heterogeneous-20261005-combined/index.html`:
700 mixed documents from distinct sources and 700 unchanged controls. Haiku,
Sonnet and current Opus each contribute 200 mixed documents (100 original
sources and 100 papers), with 100 additional browser Opus 3 documents. Both
cohorts passed complete span/provenance and full-parent copy audits. The combined
export checks source/group/normalized-duplicate split isolation across cohorts
and preserves existing assignments. Rebuild it with:

```bash
PYTHONPATH=. python scripts/build_dataset_handoff.py \
  --runs runs/heterogeneous-pilot-20261005-v2 runs/ml-papers-20261005 \
  --out runs/heterogeneous-20261005-combined
```

The public dataset is
[open-text-detector/heterogeneous-ai-spans](https://huggingface.co/datasets/open-text-detector/heterogeneous-ai-spans).
Its v1.0.0 tag preserves the initial release. The next expansion freezes 300
unused excerpts (100 each Haiku, Sonnet and current Opus), with Opus restricted
to historical fiction. Prior source IDs, normalized text and parent ranges are
excluded, and existing book/author/report split assignments are inherited.
The probe has five documents per writer, and the full run requires a matching
successful `model-probe-audit.json`. Successful model calls are cached.

```bash
PYTHONPATH=. uv run --no-project --with requests --with beautifulsoup4 \
  python scripts/prepare_span_expansion.py --run runs/heterogeneous-expansion-20261005
PYTHONPATH=. uv run --no-project --with pyarrow \
  python scripts/audit_parent_copy.py --run runs/heterogeneous-expansion-20261005
PYTHONPATH=. python scripts/run_span_expansion.py \
  --run runs/heterogeneous-expansion-20261005 --scope probe
# Review the probe and record model-probe-audit.json before the full stage.
PYTHONPATH=. python scripts/run_span_expansion.py \
  --run runs/heterogeneous-expansion-20261005 --scope all --jobs 20 --writer-jobs 4
PYTHONPATH=. uv run --no-project --with pyarrow \
  python scripts/audit_parent_copy.py --run runs/heterogeneous-expansion-20261005
PYTHONPATH=. uv run --no-project --with jsonschema \
  python scripts/audit_span_pilot.py --run runs/heterogeneous-expansion-20261005 --require-complete
```

The packaging script accepts `--combined` and `--version` for subsequent
releases. Version 1.1.0 removes the redundant `source_text_sha256` Parquet
column throughout; it is computable from `source_text`. The richer JSONL
retains upstream/source hash provenance. The new `expansion_mixed` configuration
keeps new documents distinguishable from the original and paper batches.

```bash
PYTHONPATH=. python scripts/build_dataset_handoff.py \
  --runs runs/heterogeneous-pilot-20261005-v2 runs/ml-papers-20261005 \
         runs/heterogeneous-expansion-20261005 \
  --out runs/heterogeneous-20261005-expanded
PYTHONPATH=. uv run --no-project --with pyarrow --with jsonschema --with pyyaml \
  python scripts/package_hf_dataset.py --combined runs/heterogeneous-20261005-expanded \
  --out releases/heterogeneous-ai-spans-v1.1.0 --version 1.1.0 \
  --previous-release releases/heterogeneous-ai-spans-v1.0.0
```

`scripts/publish_hf_dataset.py` verifies checksums and local loading, guards
the expected Hub parent commit, uploads an authorized release, creates its
version tag, and verifies every remote file hash and configuration. Its
`--receipt` belongs outside the release folder. Runtime model logs and
credentials are not uploaded.

## GPT general and paper expansion

The next cohorts use explicitly requested `gpt-6.1-sol` and `gpt-6-luna`
through Codex subscription authentication: 100 general documents and 100
pre-2015 ML-paper excerpts per writer, with distinct sources and matched
controls. The requested IDs and low reasoning effort are frozen in the
configuration. Codex does not report the served identity in these outputs,
so `reported_model` stays null and `model_identity_status` is `requested_only`.
The summarizer remains Haiku for consistency with the earlier cohorts.

```bash
PYTHONPATH=. python scripts/prepare_span_expansion.py \
  --run runs/gpt-general-20261005 \
  --combined runs/heterogeneous-20261005-expanded --seed 2026100503 \
  --config examples/config.gpt-expansion.json \
  --quotas examples/quotas.gpt-expansion.json
PYTHONPATH=. uv run --no-project --with beautifulsoup4 --with requests \
  python scripts/prepare_ml_papers.py --run runs/gpt-ml-papers-20261005 \
  --config examples/config.gpt-ml.toml --count 200 --minimum-paragraphs 5 \
  --seed 2026100504 --exclude runs/heterogeneous-20261005-expanded/dataset.jsonl \
  --source-cache runs/ml-papers-20261005/sources/jmlr/raw
```

General-document quotas reflect the remaining unused eligible pool and are
identical for both writers. The paper expansion excludes all previously used
papers, requires five continuous eligible paragraphs, and allows replacement
of up to 65% of the original characters. The same 3/4/6-paragraph blocks,
2–3-sentence briefs and 15% normalized 8-gram copy ceiling apply. Extraction
retains recorded PDF paragraph locations; inline math is flattened and a
figure caption can be joined to nearby narrative by the heuristic recovery.

For each cohort, prepare full-parent references, choose five probe variants
per writer in `verification-selection.json`, and run `run_span_expansion.py`
with `--scope probe`. Papers also require a matching visual source review.
Read the probe and save its decision in `model-probe-audit.json` before
`--scope all`. Final assembly and audits use the same commands as the earlier
expansion. The runner checks source hashes, model-probe approval and complete
parent-reference coverage before bulk generation.

Version 1.2.0 combines all five cohorts, preserving every previous full record.
It adds `gpt_general_mixed` and `gpt_ml_papers_mixed`; `ml_papers_mixed` includes
both the earlier and GPT paper cohorts. All Parquet configurations continue
to omit `source_text_sha256`.

```bash
PYTHONPATH=. python scripts/build_dataset_handoff.py \
  --runs runs/heterogeneous-pilot-20261005-v2 runs/ml-papers-20261005 \
         runs/heterogeneous-expansion-20261005 \
         runs/gpt-general-20261005 runs/gpt-ml-papers-20261005 \
  --out runs/heterogeneous-20261005-with-gpt
PYTHONPATH=. uv run --no-project --with pyarrow --with jsonschema --with pyyaml \
  python scripts/package_hf_dataset.py --combined runs/heterogeneous-20261005-with-gpt \
  --out releases/heterogeneous-ai-spans-v1.2.0 --version 1.2.0 \
  --previous-release releases/heterogeneous-ai-spans-v1.1.0
```
