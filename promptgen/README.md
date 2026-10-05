# Synthetic user prompts

This is a separate generator from `heterogeneous`. It generates synthetic user
messages, optionally paired with assistant replies and later user follow-ups.
Every generated message is attributed to an AI model, including messages whose
conversation role is `user`. This output is not a corpus of human-written text.

**MAGPIE** samples an instruction-tuned model as a raw causal language model at
the beginning of a user message. The tokenizer renders its own chat template
with an open final user message, and the generator sends those exact token IDs
to vLLM's `/v1/completions` endpoint. The model directly predicts the tokens
occupying the user's role. This uses the instruct checkpoint outside its usual
assistant-response inference pattern. There is no prompt asking the assistant
to invent a user message.

The [MAGPIE repository](https://github.com/magpie-align/magpie) also describes
extending complete exchanges into multi-turn conversations using this method.

## Install and serve a checkpoint

From the project directory:

```bash
pip install -e '.[prompts]'
```

Install vLLM in an environment appropriate for your GPU machine. The client only
loads the tokenizer; it can run on another machine. For example, to serve the
Qwen checkpoint used to verify template rendering:

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct \
  --tensor-parallel-size 4 \
  --dtype bfloat16 \
  --max-model-len 8192 \
  --generation-config vllm \
  --host 127.0.0.1 --port 8000
```

Use a pinned checkpoint revision for a frozen experiment. Set the config's
`model.revision` to identify the served weights and `model.tokenizer_revision`
to the tokenizer's commit hash; launch vLLM with the same revision. If using
`--served-model-name`, `model.model` is that alias and `model.tokenizer` is the
actual checkpoint path or Hugging Face repository. The tokenizer must match
the server's token vocabulary. The server's reported model name does not prove
which checkpoint revision is loaded.

To access a remote machine without exposing its inference service publicly:

```bash
ssh -L 8000:127.0.0.1:8000 your-gpu-machine
```

The client targets vLLM's completion extensions, including token-ID prompts and
explicit stop-token IDs. Ordinary hosted chat endpoints do not generally expose
the raw user-prefix behavior needed for MAGPIE. This command does not use Codex
or Claude subscriptions. See [vLLM serving documentation](https://docs.vllm.ai/en/stable/serving/online_serving/)
and [completion request fields](https://docs.vllm.ai/en/stable/api/vllm/entrypoints/openai/completion/protocol/).

## Generate first user messages

```bash
python -m promptgen plan \
  --config examples/config.magpie.toml --out runs/prompts

python -m promptgen run \
  --config examples/config.magpie.toml --out runs/prompts --jobs 16
```

`plan` loads the tokenizer and may download its files, but performs no model
inference. It prints the first inference prefix and saves `first-prefix.txt`.
The real Qwen tokenizer produced this ending:

```text
<|im_start|>user
```

There is no user content or closing token after that header. The model predicts
the user content next. Templates may supply their own default system message;
Qwen's does. See [Hugging Face chat template documentation](https://huggingface.co/docs/transformers/chat_templating)
for `continue_final_message` and why adding special tokens a second time is wrong.

`count` is the number of candidate conversations, not a promise of that many
unique accepted prompts. Rejections and duplicates can reduce output. The
manifest reports actual counts. `user_max_tokens` and `response_max_tokens`
are inference caps; there is no minimum word count or forced short prompt.

## Generate a follow-up after a complete turn

Either set `generation.user_turns = 2` to generate
`user → assistant → user` from scratch, or start from completed exchanges:

```bash
python -m promptgen run \
  --config examples/config.magpie.toml \
  --input examples/conversations.jsonl \
  --out runs/followups --jobs 16
```

Input is JSONL with an `id`, optional `reference`, and `messages` containing
`role`, `content` and optional `author`. Each seed must have alternating user
and assistant messages and end in an assistant reply; a leading system message
is allowed. Seed authorship is preserved when supplied and recorded as unknown
otherwise. The example seed is explicitly marked as invented demonstration
text. `count` candidates cycle through seeds in file order. `user_turns` counts
new user messages beyond the seed. Each intermediate assistant reply is
generated before the next user message. Set `emit_final_response = true` to
also answer the final user message.

All steps use the configured open-weight model. For MAGPIE the follow-up prefix
contains the completed conversation, then another open user header. It does not
ask the assistant to predict what a user would say.

For example, the Qwen prefix ends like this (bracketed text represents the
actual completed messages):

```text
<|im_start|>user
[first user message]<|im_end|>
<|im_start|>assistant
[assistant reply]<|im_end|>
<|im_start|>user
```

Sampling begins immediately after the final newline. It stops at the next
control token before another role begins.

`system_prompt` conditions MAGPIE and is explicitly marked in output; leave it
empty for the plain method.

## Files, quality checks and resuming

- `plan.json`: configuration, original seed messages,
  tokenizer vocabulary/template fingerprints and candidate IDs.
- `conversations.jsonl`: accepted complete conversations with per-message
  authorship and full generation records.
- `prompts.jsonl`: one row per generated user message, with its preceding
  context, conversation/seed IDs and author/model details.
- `rejected.jsonl`: exhausted candidates and context-aware duplicate candidates.
- `cache/`: each completed step and every attempted completion, including
  truncated or malformed output. These are audit records, not accepted data.
- `manifest.json`: candidate, accepted conversation, prompt and rejection counts.

Generation records preserve the actual inference text, input token IDs and
hashes, requested/reported models, declared revision, sampling seed/settings,
finish/stop reasons, usage and request ID. Output content is trimmed at its
edges; the raw completion remains in the generation record. Configuration and
cached files never include API-key values; authentication is read from the
environment variable named by `model.api_key_env`.

For user-prefix generation, declared special tokens, added control tokens and
known native role delimiters are stop tokens. Some checkpoints omit real chat
delimiters from their special-token metadata. The resolved stop IDs and control
strings are saved in the tokenizer identity. For assistant generation, the
tokenizer's EOS and configured additional stop IDs are used. Models with plain
text turn delimiters need `model.stop_strings`; these apply to user sampling.
`model.response_stop_strings` separately controls assistant generation.
Control tokens remain visible during decoding so role leakage can be detected.
Empty output, unfinished/truncated output, control-token leakage and leading
role labels are rejected with bounded retries. These are structural checks;
answerability, naturalness, topic compliance and semantic diversity still need
evaluation. Text-only instruction models are the initial target; structured
reasoning, tool calls and multimodal message formats are not handled.

The default `model.encoder = "jinja"` uses the published chat template.
`deepseek_v4` and `inkling` use the installed vLLM native text encoders on the
GPU host and fingerprint their source. Inkling prompts retain native token IDs;
their decoded text is diagnostic and is never re-encoded for inference.
`model.trust_remote_code` defaults to false; enable it only for checkpoints
whose published tokenizer requires it. These adapters do not generate media
or tool calls.

Deduplication compares normalized user text within the same preceding context.
It removes entire candidate conversations when a user message duplicates an
earlier one in that context. It does not detect semantic paraphrases. Split
seeded conversations by their original source/group, keeping all descendants
together; this generator does not assign train/test splits for you.

Resume without repeating successful calls:

```bash
python -m promptgen resume --out runs/prompts --jobs 16
```

A changed plan or tokenizer cannot reuse the existing run directory. Exhausted
steps remain rejected on resume. Use a fresh directory for a changed experiment.
Sampling seeds are recorded, but GPU/server differences can still change results.

Plan/schema version 1.1 supports direct user-prefix sampling only. Earlier plans
from the initial two-method implementation must be recreated in a fresh directory.

## Detached four-GPU model sweep

`examples/sweep.json` selects eight labs/checkpoints with just one Qwen model.
Each model receives 100 candidates in each of four conditions: first user or
follow-up user, at temperature 0.7 or 1.0. The 50 synthetic completed exchanges
in `examples/magpie-seeds.jsonl` are shared across models. This gives 3,200
candidates, with one attempt each so retries cannot conceal failure rates.
Structural acceptance does not establish that a prompt is natural or useful;
review the raw completions as well as accepted prompts.

Run the supervisor on the GPU host, alongside vLLM:

```bash
python -m promptgen.sweep \
  --manifest examples/sweep.json \
  --seeds examples/magpie-seeds.jsonl \
  --out /data/magpie-sweep/results
```

It serves one checkpoint at a time using all four GPUs, pins its repository
revision, and records the native first/follow-up prefixes. Each model directory
contains `server.log`, `server-command.json`, `revision.json`, `prefixes.json`,
and a run directory for each condition. `status.json` tracks progress and
distinguishes setup/runtime errors from sampled-text rejections. Failed models
do not prevent subsequent models from running. Restarting the same command
resumes unfinished candidates and skips completed models. An exclusive output
lock prevents two supervisors from running the same experiment concurrently.

`promptgen/bootstrap.py` can install an isolated vLLM environment, run the
project checks, and launch this sweep. Start it detached on the GPU host:

```bash
nohup python3 promptgen/bootstrap.py \
  --project "$PWD" \
  --runtime /home/user/.cache/magpie-sweep \
  --out /data/magpie-sweep/results \
  </dev/null >/data/magpie-sweep/job.log 2>&1 &
```

Create `/data/magpie-sweep` first. For gated checkpoints, authenticate on the
host or pass `--token-file` pointing to a private file readable only by your
account. Keep credentials outside the project and results directories.
The bootstrap uses the example sweep manifest and seeds in the project.
It limits CPU worker threads and selects vLLM's native sampler, avoiding a
FlashInfer sampling-kernel compilation dependency on a local CUDA toolkit.

The deployed HF experiment uses `/data/magpie-sweep-20261003` on
`open-text-detector/training`. Its project, results and job log are under that
directory. Check it from any computer with your HF SSH access:

```bash
ssh open-text-detector-training@ssh.hf.space \
  'cat /data/magpie-sweep-20261003/results/status.json'
```

Both the sampler and inference server run remotely without an SSH tunnel.
Closing the terminal or rebooting your local computer leaves them running.
Stopping, rebuilding or restarting the HF Space stops its processes; the
results under `/data` are checkpoints for resuming the same experiment.

The next batch uses `examples/sweep.round2.json` and runs under
`/data/magpie-sweep-20261003-round2`. It tests Seed-OSS, ERNIE, Kimi Linear,
MiniMax, Ling, MiMo, DeepSeek and Inkling, with the same four conditions and
shared seeds. Each model first gets three ordinary assistant questions;
`assistant-controls/` saves their complete outputs and provenance. The simple
expected-substring checks help diagnosis and do not certify serving quality.
DeepSeek and Inkling are experimental on A100; startup errors remain separate
from user-sampling outcomes. MiniMax M2.7 has a non-commercial license.

```bash
ssh open-text-detector-training@ssh.hf.space \
  'cat /data/magpie-sweep-20261003-round2/results/status.json'
```

`examples/sweep.round3.pilot.json` tests Granite 4.2-30B, Ling 3.0 Flash in
BF16, Mistral Small 4 and Nemotron 3 Super in BF16. It uses ten attempts per
condition: twenty first-user and twenty follow-up attempts per model across
the two temperatures. Ten seed exchanges spread across the original fixture
collection are saved in `examples/magpie-seeds.pilot.jsonl`. Assistant controls
remain separate. The pilot runs at `/data/magpie-sweep-20261003-round3-pilot`.

The manifest supports `max_model_len` and `gpu_memory_utilization` for short
smoke tests. Per-model `ignore_patterns` excludes duplicate checkpoint formats.
`server_pythonpath` can select an isolated dependency overlay for the server;
the path is recorded in `server-runtime-override.json`. The Mistral pilot uses
Transformers 5.16.0 because the installed 5.17.0 removed a Pixtral symbol that
vLLM 0.30.0 imports. The rest of the runtime is unchanged.
