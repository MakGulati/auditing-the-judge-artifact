"""
LLM backend abstraction.

Two implementations:
  APIBackend   – calls any OpenAI-compatible HTTP endpoint (Mistral AI, vLLM server, Ollama)
  VLLMBackend  – runs inference in-process via vLLM's AsyncLLMEngine (GGUF or HF safetensors)
"""
from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class Completion:
    """A generated reply plus whether the token budget cut it off.

    Truncation used to be invisible: `chat_complete` returned only the text, so a
    reply that ran out of budget mid-sentence was indistinguishable from one the
    model chose to end. On MATH that silently relabelled a quarter of the dataset as
    "wrong answer" (the final answer is the last thing written, so a cut-off solution
    has none), and on GSM8K it was worse — the answer parser's last-number fallback
    invented a plausible wrong answer instead of failing.

    ``truncated is None`` means the backend could not report it, which is different
    from "not truncated" and must not be recorded as such.
    """
    text: str
    truncated: bool | None = None


class LLMBackend(ABC):
    @abstractmethod
    async def chat_complete(
        self,
        messages: list[dict],
        temperature: float = 0.0,
        max_tokens: int = 1024,
        top_p: float = 1.0,
    ) -> str:
        """Return assistant message content for the given chat messages."""
        ...

    async def complete(
        self,
        messages: list[dict],
        temperature: float = 0.0,
        max_tokens: int = 1024,
        top_p: float = 1.0,
    ) -> Completion:
        """Like `chat_complete`, but also reports whether the budget truncated it.

        The default cannot know, so it reports ``None``. Every real backend overrides
        this; the fallback exists so a test double implementing only `chat_complete`
        keeps working and is honestly recorded as "unknown".
        """
        # By keyword, not position: every other caller in the codebase passes these by
        # name, and a subclass (or a test double) is free to declare them keyword-only.
        text = await self.chat_complete(messages, temperature=temperature,
                                        max_tokens=max_tokens, top_p=top_p)
        return Completion(text=text, truncated=None)

    async def aclose(self) -> None:
        """Release resources. Override if the backend holds GPU memory."""
        pass


# ── API backend ──────────────────────────────────────────────────────────────

class APIBackend(LLMBackend):
    """Wraps an AsyncOpenAI client with a semaphore for rate-limiting."""

    def __init__(self, client, model: str, semaphore: asyncio.Semaphore) -> None:
        self._client = client
        self._model = model
        self._sem = semaphore

    async def chat_complete(
        self,
        messages: list[dict],
        temperature: float = 0.0,
        max_tokens: int = 1024,
        top_p: float = 1.0,
    ) -> str:
        return (await self.complete(messages, temperature, max_tokens, top_p)).text

    async def complete(
        self,
        messages: list[dict],
        temperature: float = 0.0,
        max_tokens: int = 1024,
        top_p: float = 1.0,
    ) -> Completion:
        async with self._sem:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
            )
        choice = response.choices[0]
        reason = getattr(choice, "finish_reason", None)
        return Completion(
            text=choice.message.content or "",
            # Only "length" proves the budget cut it off. An unrecognised or absent
            # reason is unknown, not False.
            truncated=(reason == "length") if reason is not None else None,
        )


# ── vLLM backend ─────────────────────────────────────────────────────────────

class VLLMBackend(LLMBackend):
    """
    Runs GGUF (or HF safetensors) inference in-process using vLLM's AsyncLLMEngine.

    Parameters
    ----------
    model       HuggingFace repo ID that contains the GGUF file(s).
    gguf_file   Filename of the specific .gguf file inside the repo
                (e.g. 'Ministral-3-8B-Instruct-2512-Q4_K_M.gguf').
                If omitted, the model is treated as a standard HF safetensors repo.
    tokenizer   HF repo ID for the tokenizer. Defaults to `model` (works when the
                GGUF repo also hosts tokenizer.json / tokenizer_config.json).
    gpu_memory_utilization
                Fraction of GPU VRAM for the KV cache (default 0.85).
    max_model_len
                Override the model's context window (tokens). Useful to cap memory.
    """

    def __init__(
        self,
        model: str,
        gguf_file: str | None = None,
        tokenizer: str | None = None,
        gpu_memory_utilization: float = 0.85,
        max_model_len: int | None = None,
        max_num_seqs: int | None = None,
        quantization: str | None = None,
        kv_cache_dtype: str | None = None,
        mistral_format: bool = False,
        assistant_prefill: str = "",
    ) -> None:
        from vllm import AsyncEngineArgs, AsyncLLMEngine

        if mistral_format and gguf_file:
            raise ValueError("--mistral_format is for native safetensors repos, not GGUF")
        if mistral_format and assistant_prefill:
            raise ValueError("--assistant_prefill needs the HF chat-template path, "
                             "not --mistral_format")
        # e.g. "\n</think>\n\n" closes the think block that reasoning templates
        # (DeepSeek-R1 distills) force open, disabling chain-of-thought.
        self._prefill = assistant_prefill

        model_path = model
        is_gguf = False

        if gguf_file:
            from huggingface_hub import hf_hub_download
            print(f"Downloading {gguf_file} from {model} …")
            model_path = hf_hub_download(repo_id=model, filename=gguf_file)
            print(f"Saved to {model_path}")
            is_gguf = True

        tokenizer_id = tokenizer or model

        # GGUF forces its own quant; otherwise honor an explicit --quantization (e.g. "fp8":
        # online weight quant that ~halves weight VRAM, freeing KV cache for a much larger
        # context — e.g. DeepSeek-R1-Distill-Qwen-14B goes 4k -> 32k on a 32 GB card).
        quant = "gguf" if is_gguf else quantization

        # max_num_seqs caps the engine's max concurrent sequences AND the sampler
        # warmup batch. Lower it (e.g. 16) for a big model on a small card: a ~28 GB
        # model on a 32 GB GPU OOMs during the default 128-request sampler warmup even
        # though weights + KV fit. None = vLLM default.
        engine_kwargs = dict(
            model=model_path,
            tokenizer=tokenizer_id,
            load_format="gguf" if is_gguf else "auto",
            quantization=quant,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            dtype="auto",
        )
        if max_num_seqs is not None:
            engine_kwargs["max_num_seqs"] = max_num_seqs
        if kv_cache_dtype is not None:
            engine_kwargs["kv_cache_dtype"] = kv_cache_dtype   # e.g. "fp8" ~halves KV -> more ctx

        # Native mistral checkpoints (e.g. Ministral-3-8B-2512: text_config
        # model_type 'ministral3') can't be parsed by the transformers config/
        # tokenizer that vLLM<=0.10 ships. Their repos carry params.json +
        # tekken.json + consolidated.safetensors, so load via vLLM's mistral
        # format and template prompts with mistral_common instead of an HF
        # tokenizer (prompts go to the engine pre-tokenized).
        self._mtok = None
        if mistral_format:
            engine_kwargs["config_format"] = "mistral"
            engine_kwargs["load_format"] = "mistral"
            engine_kwargs["tokenizer_mode"] = "mistral"
            from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
            self._mtok = MistralTokenizer.from_hf_hub(tokenizer_id)
        engine_args = AsyncEngineArgs(**engine_kwargs)
        self._engine = AsyncLLMEngine.from_engine_args(engine_args)
        if not mistral_format:
            from transformers import AutoTokenizer
            self._tok = AutoTokenizer.from_pretrained(tokenizer_id)

    async def chat_complete(
        self,
        messages: list[dict],
        temperature: float = 0.0,
        max_tokens: int = 1024,
        top_p: float = 1.0,
    ) -> str:
        return (await self.complete(messages, temperature, max_tokens, top_p)).text

    async def complete(
        self,
        messages: list[dict],
        temperature: float = 0.0,
        max_tokens: int = 1024,
        top_p: float = 1.0,
    ) -> Completion:
        from vllm import SamplingParams
        from vllm.utils import random_uuid

        if self._mtok is not None:
            from mistral_common.protocol.instruct.request import ChatCompletionRequest
            toks = self._mtok.encode_chat_completion(
                ChatCompletionRequest(messages=messages)).tokens
            prompt = {"prompt_token_ids": toks}
        else:
            prompt = self._tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            ) + self._prefill
        params = SamplingParams(
            temperature=temperature,
            # top_p is ignored when temperature=0; set 1.0 to avoid vLLM warning
            top_p=top_p if temperature > 0 else 1.0,
            max_tokens=max_tokens,
        )
        text, reason = "", None
        async for out in self._engine.generate(prompt, params, random_uuid()):
            if out.finished:
                text = out.outputs[0].text
                reason = getattr(out.outputs[0], "finish_reason", None)
        return Completion(
            text=text,
            truncated=(reason == "length") if reason is not None else None,
        )

    async def aclose(self) -> None:
        # V0 AsyncLLMEngine exposes shutdown_background_loop(); V1 AsyncLLM uses shutdown().
        for name in ("shutdown_background_loop", "shutdown"):
            fn = getattr(self._engine, name, None)
            if fn is not None:
                fn()
                return


# ── transformers backend ─────────────────────────────────────────────────────

class TransformersBackend(LLMBackend):
    """In-process generation via plain `transformers`, for models vLLM can't load.

    Needed for the new **mistral3** multimodal checkpoints (e.g.
    Ministral-3-8B-Instruct-2512): their GGUF architecture isn't supported by the
    transformers/vLLM GGUF loader, and the pinned vLLM 0.6 predates mistral3
    safetensors support. This backend loads the FP8 weights and dequantizes to
    bf16 — the exact load path `extract_hidden_rich.py` uses — so generation and
    activation extraction share identical weights.

    Concurrent `chat_complete` calls are coalesced by a micro-batcher (grouped by
    sampling params) into batched `model.generate` calls, recovering most of the
    throughput a serial HF loop would lose.
    """

    def __init__(
        self,
        model: str,
        gpu_memory_utilization: float = 0.85,  # unused; kept for build_backend parity
        max_batch: int = 16,
        collect_window: float = 0.02,
        gen_max_tokens: int = 1024,
    ) -> None:
        import torch
        from transformers import (
            AutoConfig, AutoModelForImageTextToText, AutoTokenizer, FineGrainedFP8Config,
        )

        # mistral3 FP8 dequant references this dtype on some transformers builds
        torch.float8_e8m0fnu = getattr(torch, "float8_e8m0fnu", torch.float8_e4m3fn)
        self._torch = torch

        # The default --model points at the GGUF repo (no safetensors/tokenizer.json);
        # transformers needs the base checkpoint, so drop a trailing "-GGUF".
        if model.endswith("-GGUF"):
            model = model[: -len("-GGUF")]

        # AutoTokenizer reads tokenizer_config.json -> proper eos/pad/bos + chat_template
        # (a bare PreTrainedTokenizerFast(tokenizer_file=...) has no special tokens, which
        # breaks padding and stop conditions during generation).
        self._tok = AutoTokenizer.from_pretrained(model)
        if self._tok.pad_token is None:
            self._tok.pad_token = self._tok.eos_token
        self._tok.padding_side = "left"

        # FP8 checkpoints (e.g. Ministral-3-8B-2512) must be dequantized to bf16 to run on
        # GPUs without native FP8 (Ampere); bf16 checkpoints (e.g. Gemma3-12b) load as-is.
        # Detect from the model config so the same backend serves both.
        cfg = AutoConfig.from_pretrained(model)
        qc = getattr(cfg, "quantization_config", None)
        qmethod = (qc.get("quant_method") if isinstance(qc, dict)
                   else getattr(qc, "quant_method", None)) if qc else None
        load_kwargs = dict(dtype=torch.bfloat16, device_map="cuda")
        if qmethod == "fp8":
            load_kwargs["quantization_config"] = FineGrainedFP8Config(dequantize=True)
            print(f"Loading {model} (FP8 -> bf16 dequantize) …", flush=True)
        else:
            print(f"Loading {model} (bf16) …", flush=True)
        self._model = AutoModelForImageTextToText.from_pretrained(model, **load_kwargs).eval()
        self._eos = self._tok.eos_token_id
        self._pad = self._tok.pad_token_id

        self._max_batch = max_batch
        self._collect_window = collect_window
        # Ceiling on max_new_tokens. HF batched generate() runs the whole batch until
        # the *slowest* sequence finishes, so a low cap limits the head-of-line tax of
        # one non-terminating CoT — but it must match the vLLM backend's budget
        # (solve_greedy asks for 1024). A lower value here truncates solutions, which
        # turns into unparseable answers and a wrong-rate that differs from the vLLM
        # path for reasons unrelated to the model. Keep the two backends in sync.
        self._gen_max_tokens = gen_max_tokens
        self._truncation_warned = False
        self._queue: asyncio.Queue = asyncio.Queue()
        self._worker = asyncio.ensure_future(self._batch_loop())

    async def chat_complete(
        self,
        messages: list[dict],
        temperature: float = 0.0,
        max_tokens: int = 1024,
        top_p: float = 1.0,
    ) -> str:
        return (await self.complete(messages, temperature, max_tokens, top_p)).text

    async def complete(
        self,
        messages: list[dict],
        temperature: float = 0.0,
        max_tokens: int = 1024,
        top_p: float = 1.0,
    ) -> Completion:
        if int(max_tokens) > self._gen_max_tokens and not self._truncation_warned:
            import sys
            print(f"[WARN] TransformersBackend caps max_tokens at {self._gen_max_tokens}; "
                  f"a request for {int(max_tokens)} will be truncated. Truncated solutions "
                  f"parse to None and are labelled wrong.", file=sys.stderr, flush=True)
            self._truncation_warned = True
        max_tokens = min(int(max_tokens), self._gen_max_tokens)
        loop = asyncio.get_event_loop()
        fut: asyncio.Future = loop.create_future()
        # bucket key groups requests that can share one generate() call
        key = (temperature > 0, round(float(temperature), 3),
               round(float(top_p), 3) if temperature > 0 else 1.0, int(max_tokens))
        await self._queue.put((messages, key, fut))
        return await fut

    async def _batch_loop(self) -> None:
        while True:
            prompt, key, fut = await self._queue.get()
            batch = [(prompt, fut)]
            # let a few more requests pile up so we can batch them
            await asyncio.sleep(self._collect_window)
            buckets = {key: list(batch)}
            while not self._queue.empty() and sum(len(v) for v in buckets.values()) < self._max_batch:
                p2, k2, f2 = self._queue.get_nowait()
                buckets.setdefault(k2, []).append((p2, f2))
            for k, items in buckets.items():
                try:
                    await asyncio.get_event_loop().run_in_executor(
                        None, self._run_batch, k, items
                    )
                except Exception as exc:  # noqa: BLE001 — propagate to each waiter
                    for _, f in items:
                        if not f.done():
                            f.set_exception(exc)

    def _run_batch(self, key, items) -> None:
        torch = self._torch
        do_sample, temperature, top_p, max_tokens = key
        messages_batch = [m for m, _ in items]
        enc = self._tok.apply_chat_template(
            messages_batch,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            padding=True,
            return_dict=True,
        )
        ids = enc["input_ids"].to("cuda")
        attn = enc["attention_mask"].to("cuda")
        gen_kwargs = dict(
            input_ids=ids, attention_mask=attn, max_new_tokens=max_tokens,
            pad_token_id=self._pad, do_sample=do_sample,
        )
        if do_sample:
            gen_kwargs.update(temperature=temperature, top_p=top_p)
        with torch.no_grad():
            out = self._model.generate(**gen_kwargs)
        new = out[:, ids.shape[1]:]
        texts = self._tok.batch_decode(new, skip_special_tokens=True)
        # `generate` pads every row to the longest in the batch, so row length is not
        # the reply length. A reply that stopped on its own contains an EOS; one the
        # budget cut off does not — that absence IS the truncation signal.
        eos_ids = {i for i in (self._tok.eos_token_id, self._pad) if i is not None}
        stopped = [bool(eos_ids & set(row.tolist())) for row in new]
        for (_, f), txt, ended in zip(items, texts, stopped):
            if not f.done():
                f.get_loop().call_soon_threadsafe(
                    f.set_result, Completion(text=txt, truncated=not ended))

    async def aclose(self) -> None:
        self._worker.cancel()
