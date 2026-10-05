from __future__ import annotations

import inspect
import re
from heterogeneous.core import digest, text_hash


def load_tokenizer(model: dict):
    try:
        from transformers import AutoTokenizer, TokenizersBackend
    except ImportError as error:
        raise ValueError("install tokenizer support with: pip install -e '.[prompts]'") from error
    options = {"revision": model["tokenizer_revision"] or None, "trust_remote_code": model["trust_remote_code"]}
    backend = TokenizersBackend if model["encoder"] != "jinja" else AutoTokenizer
    tokenizer = backend.from_pretrained(model["tokenizer"], **options)
    if type(tokenizer).__name__ == "MistralCommonBackend":
        # Use the checkpoint's published Jinja template and fast tokenizer.
        # MistralCommonBackend has no get_chat_template implementation and its
        # text decoding is not intended for round-tripping control tokens.
        tokenizer = AutoTokenizer.from_pretrained(model["tokenizer"],
            mistral_format=False, fix_mistral_regex=True, **options)
    return tokenizer


class TokenPrompt(str):
    """Readable diagnostic text whose authoritative encoding is native IDs."""
    def __new__(cls, text, ids):
        value = super().__new__(cls, text)
        value.ids = list(ids)
        return value


class Templates:
    def __init__(self, tokenizer, model: dict):
        self.tokenizer = tokenizer
        self.encoder = model["encoder"]
        if self.encoder == "jinja":
            template = tokenizer.get_chat_template()
            if not template:
                raise ValueError("tokenizer must have an explicit chat template")
        elif self.encoder == "deepseek_v4":
            from vllm.tokenizers import deepseek_v4_encoding
            self.native = deepseek_v4_encoding
            template = inspect.getsource(self.native)
        else:
            from vllm.renderers import inkling_encoding
            self.native = inkling_encoding
            template = inspect.getsource(self.native)
        vocab = tokenizer.get_vocab()
        added = getattr(tokenizer, "get_added_vocab", lambda: {})()
        controls = {s for s in added if re.fullmatch(r"<[^\s]+>", s)}
        controls.update(s for s in vocab if s in {
            "<|im_start|>", "<|im_end|>", "<think>", "</think>",
            "<|start|>", "<|end|>", "<|message|>", "<|channel|>",
            "<|start_header_id|>", "<|end_header_id|>",
            "[gMASK]", "[sop]", "<|user|>", "<|assistant|>",
            "]~b]", "[e~[", "<seed:bos>", "<seed:eos>"})
        self.special_tokens = sorted(set(tokenizer.all_special_tokens) | controls | set(model["stop_strings"]))
        # Stop at any control token during user synthesis. In particular this
        # catches a premature assistant header even if no user EOS precedes it.
        self.user_stops = sorted(set(tokenizer.all_special_ids + model["stop_token_ids"] +
                                     [vocab[s] for s in controls]))
        eos = tokenizer.eos_token_id
        self.response_stops = sorted(set(([eos] if isinstance(eos, int) else (eos or [])) + model["stop_token_ids"]))
        self.identity = {"name": model["tokenizer"], "requested_revision": model["tokenizer_revision"] or None,
                         "resolved_commit": tokenizer.init_kwargs.get("_commit_hash"),
                         "encoder": self.encoder, "chat_template_sha256": text_hash(template), "vocabulary_sha256": digest(vocab),
                         "user_control_strings": self.special_tokens,
                         "user_stop_token_ids": self.user_stops, "response_stop_token_ids": self.response_stops}

    def render(self, messages: list[dict], user: bool) -> str:
        if self.encoder == "deepseek_v4":
            if user:
                marker = "__MAGPIE_OPEN_USER_SENTINEL__"
                rendered = self.native.encode_messages(messages + [{"role": "user", "content": marker}], thinking_mode="chat")
                suffix = self.native.ASSISTANT_SP_TOKEN + self.native.thinking_end_token
                if not rendered.endswith(marker + suffix):
                    raise ValueError("DeepSeek encoder did not preserve the user boundary")
                return rendered[:-len(marker + suffix)]
            return self.native.encode_messages(messages, thinking_mode="chat")
        if self.encoder == "inkling":
            native, tokenizer = self.native, self.tokenizer
            vocab = tokenizer.get_vocab()
            class Adapter:
                def encode_text(self, text):
                    return tokenizer.encode(text, add_special_tokens=False)
                def encode_special(self, token):
                    for spelling in native.SPECIAL_TOKEN_SPELLINGS.get(token, (token,)):
                        if spelling in vocab:
                            return vocab[spelling]
                    raise ValueError(f"Inkling token absent: {token}")
            adapter = Adapter()
            ids = native.render_inkling_messages(messages, adapter, add_generation_prompt=not user)
            if user:
                ids += [adapter.encode_special(native.MESSAGE_USER), adapter.encode_special(native.CONTENT_TEXT)]
            return TokenPrompt(tokenizer.decode(ids, skip_special_tokens=False), ids)
        if user:
            # Recent Transformers strips header whitespace when the final
            # content is empty. Render real placeholder content, then remove
            # only that content so every byte of the native header survives.
            marker = "__MAGPIE_OPEN_USER_SENTINEL__"
            messages = messages + [{"role": "user", "content": marker}]
        rendered = self.tokenizer.apply_chat_template(messages, tokenize=False,
            add_generation_prompt=not user, continue_final_message=user)
        if user:
            if not rendered.endswith(marker):
                raise ValueError("chat template did not preserve the user-content boundary")
            return rendered[:-len(marker)]
        return rendered

    def user_prompt(self, messages: list[dict]) -> str:
        return self.render(messages, user=True)

    def token_ids(self, text: str) -> list[int]:
        if isinstance(text, TokenPrompt):
            return text.ids.copy()
        return self.tokenizer.encode(text, add_special_tokens=False)
