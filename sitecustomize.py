from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any


TRACE_CONTEXT_WINDOW = int(os.environ.get("EAGLE3_TRACE_CONTEXT_WINDOW", "16"))
PLACEHOLDER_TOKEN_ID = -1
_VLLM_STEP_COUNTS: dict[str, int] = {}
_SGLANG_STEP_COUNTS: dict[str, int] = {}


@lru_cache(maxsize=1)
def _load_tokenizer() -> Any:
    tokenizer_path = os.environ.get("EAGLE3_TRACE_TOKENIZER_PATH")
    if not tokenizer_path:
        return None
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)


def _decode_token_pieces(tokenizer: Any, token_ids: list[int]) -> list[str]:
    if tokenizer is None:
        return [str(token_id) for token_id in token_ids]
    pieces = []
    for token_id in token_ids:
        try:
            pieces.append(
                tokenizer.decode(
                    [int(token_id)],
                    clean_up_tokenization_spaces=False,
                    skip_special_tokens=False,
                )
            )
        except Exception:
            pieces.append(f"<decode_error:{token_id}>")
    return pieces


def _decode_token_text(tokenizer: Any, token_ids: list[int]) -> str:
    if tokenizer is None or not token_ids:
        return ""
    try:
        return tokenizer.decode(
            token_ids,
            clean_up_tokenization_spaces=False,
            skip_special_tokens=False,
        )
    except Exception:
        return ""


def _append_trace_events(events: list[dict[str, Any]]) -> None:
    trace_path = os.environ.get("EAGLE3_TRACE_PATH")
    if not trace_path or not events:
        return
    path = Path(trace_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for event in events:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")


def _build_trace_event(
    *,
    rid: str,
    backend: str,
    prefix_token_ids: list[int],
    draft_chunk_token_ids: list[int],
    accept_len: int,
    committed_token_ids: list[int],
    step_index: int,
) -> dict[str, Any]:
    tokenizer = _load_tokenizer()
    prefix_tail = prefix_token_ids[-TRACE_CONTEXT_WINDOW:]
    accepted_token_ids = draft_chunk_token_ids[:accept_len]
    rejected_candidate_token_ids = (
        draft_chunk_token_ids[accept_len:] if accept_len < len(draft_chunk_token_ids) else []
    )
    replacement_token_id = (
        committed_token_ids[accept_len] if len(committed_token_ids) > accept_len else None
    )
    return {
        "rid": rid,
        "backend": backend,
        "step_index": step_index,
        "accept_len": accept_len,
        "prefix_token_ids": prefix_tail,
        "prefix_tokens": _decode_token_pieces(tokenizer, prefix_tail),
        "prefix_text": _decode_token_text(tokenizer, prefix_tail),
        "draft_chunk_token_ids": draft_chunk_token_ids,
        "draft_chunk_tokens": _decode_token_pieces(tokenizer, draft_chunk_token_ids),
        "accepted_token_ids": accepted_token_ids,
        "accepted_tokens": _decode_token_pieces(tokenizer, accepted_token_ids),
        "rejected_candidate_token_ids": rejected_candidate_token_ids,
        "rejected_candidate_tokens": _decode_token_pieces(tokenizer, rejected_candidate_token_ids),
        "committed_token_ids": committed_token_ids,
        "committed_tokens": _decode_token_pieces(tokenizer, committed_token_ids),
        "committed_text": _decode_token_text(tokenizer, committed_token_ids),
        "replacement_token_id": replacement_token_id,
        "replacement_token": (
            _decode_token_pieces(tokenizer, [replacement_token_id])[0]
            if replacement_token_id is not None
            else None
        ),
    }


def _build_vllm_trace_events(
    model_runner: Any,
    spec_decode_metadata: Any,
    sampled_token_ids: Any,
) -> list[dict[str, Any]]:
    tokenizer = _load_tokenizer()
    req_ids = list(model_runner.input_batch.req_ids[: len(spec_decode_metadata.num_draft_tokens)])
    sampled_rows = sampled_token_ids.tolist()
    flat_draft_token_ids = spec_decode_metadata.draft_token_ids.tolist()
    events = []
    offset = 0

    for row_index, req_id in enumerate(req_ids):
        req_state = model_runner.requests.get(req_id)
        num_draft_tokens = int(spec_decode_metadata.num_draft_tokens[row_index])
        draft_chunk = [int(token_id) for token_id in flat_draft_token_ids[offset : offset + num_draft_tokens]]
        offset += num_draft_tokens

        if req_state is None:
            prefix_ids: list[int] = []
        else:
            prefix_ids = [int(token_id) for token_id in (req_state.prompt_token_ids or [])]
            prefix_ids.extend(
                int(token_id) for token_id in req_state.output_token_ids if int(token_id) >= 0
            )
        prefix_tail = prefix_ids[-TRACE_CONTEXT_WINDOW:]

        sampled_row = [int(token_id) for token_id in sampled_rows[row_index]]
        committed_token_ids = [token_id for token_id in sampled_row if token_id != PLACEHOLDER_TOKEN_ID]

        accept_len = 0
        for pos, draft_token_id in enumerate(draft_chunk):
            if pos < len(committed_token_ids) and committed_token_ids[pos] == draft_token_id:
                accept_len += 1
            else:
                break

        accepted_token_ids = draft_chunk[:accept_len]
        rejected_candidate_token_ids = draft_chunk[accept_len:] if accept_len < len(draft_chunk) else []
        replacement_token_id = (
            committed_token_ids[accept_len] if len(committed_token_ids) > accept_len else None
        )
        step_index = _VLLM_STEP_COUNTS.get(req_id, 0) + 1
        _VLLM_STEP_COUNTS[req_id] = step_index
        events.append(
            {
                "rid": req_id,
                "backend": "angelslim_vllm",
                "step_index": step_index,
                "accept_len": accept_len,
                "prefix_token_ids": prefix_tail,
                "prefix_tokens": _decode_token_pieces(tokenizer, prefix_tail),
                "prefix_text": _decode_token_text(tokenizer, prefix_tail),
                "draft_chunk_token_ids": draft_chunk,
                "draft_chunk_tokens": _decode_token_pieces(tokenizer, draft_chunk),
                "accepted_token_ids": accepted_token_ids,
                "accepted_tokens": _decode_token_pieces(tokenizer, accepted_token_ids),
                "rejected_candidate_token_ids": rejected_candidate_token_ids,
                "rejected_candidate_tokens": _decode_token_pieces(tokenizer, rejected_candidate_token_ids),
                "committed_token_ids": committed_token_ids,
                "committed_tokens": _decode_token_pieces(tokenizer, committed_token_ids),
                "committed_text": _decode_token_text(tokenizer, committed_token_ids),
                "replacement_token_id": replacement_token_id,
                "replacement_token": (
                    _decode_token_pieces(tokenizer, [replacement_token_id])[0]
                    if replacement_token_id is not None
                    else None
                ),
                "internal_request_id": req_id,
            }
        )

    return events


def install_vllm_trace_patch() -> None:
    if os.environ.get("EAGLE3_TRACE_BACKEND") != "vllm":
        return
    try:
        from vllm.v1.worker import gpu_model_runner as gmr
    except Exception:
        return

    if getattr(gmr.GPUModelRunner, "_eagle3_trace_patched", False):
        return

    original_sample = gmr.GPUModelRunner._sample

    def traced_sample(self: Any, logits: Any, spec_decode_metadata: Any) -> Any:
        if spec_decode_metadata is None:
            return original_sample(self, logits, spec_decode_metadata)

        sampling_metadata = self.input_batch.sampling_metadata
        sampler_output = self.rejection_sampler(
            spec_decode_metadata,
            None,
            logits,
            sampling_metadata,
        )
        try:
            _append_trace_events(
                _build_vllm_trace_events(self, spec_decode_metadata, sampler_output.sampled_token_ids)
            )
        except Exception:
            pass
        self._update_states_after_model_execute(sampler_output.sampled_token_ids)
        return sampler_output

    gmr.GPUModelRunner._sample = traced_sample
    gmr.GPUModelRunner._eagle3_trace_patched = True


def install_sglang_trace_patch() -> None:
    if os.environ.get("EAGLE3_TRACE_BACKEND") != "sglang":
        return
    try:
        from sglang.srt.entrypoints import engine as engine_module
        from sglang.srt.managers import scheduler as scheduler_module
        from sglang.srt.managers import scheduler_output_processor_mixin as sopm
    except Exception:
        return

    def apply_scheduler_patch() -> None:
        if getattr(sopm.SchedulerOutputProcessorMixin, "_eagle3_trace_patched", False):
            return

        original = sopm.SchedulerOutputProcessorMixin._resolve_spec_overlap_token_ids

        def request_prefix_token_ids(req: Any) -> list[int]:
            prefix: list[int] = []
            for attr in ("origin_input_ids", "input_ids", "prompt_token_ids"):
                value = getattr(req, attr, None)
                if value is None:
                    continue
                try:
                    items = value.tolist() if hasattr(value, "tolist") else list(value)
                except TypeError:
                    items = []
                if items:
                    prefix.extend(int(token_id) for token_id in items if int(token_id) >= 0)
                    break
            for attr in ("output_ids", "decoded_ids", "output_token_ids"):
                value = getattr(req, attr, None)
                if value is None:
                    continue
                try:
                    items = value.tolist() if hasattr(value, "tolist") else list(value)
                except TypeError:
                    items = []
                if items:
                    prefix.extend(int(token_id) for token_id in items if int(token_id) >= 0)
                    break
            return prefix

        def wrapped(self: Any, result: Any, batch: Any) -> Any:
            predict_tokens = original(self, result, batch)
            trace_events: list[dict[str, Any]] = []
            try:
                next_token_ids = result.next_token_ids.tolist()
                accept_lens = result.accept_lens.tolist()
                stride = self.draft_worker.speculative_num_draft_tokens
                for i, req in enumerate(batch.reqs):
                    rid = req.rid
                    committed = [int(token_id) for token_id in predict_tokens[i]]
                    if not committed:
                        continue
                    draft_and_bonus = next_token_ids[i * stride : i * stride + accept_lens[i]]
                    if not draft_and_bonus:
                        continue
                    accept_len = max(0, int(accept_lens[i]) - 1)
                    draft_chunk = [int(token_id) for token_id in draft_and_bonus[:-1]]
                    step_index = _SGLANG_STEP_COUNTS.get(rid, 0) + 1
                    _SGLANG_STEP_COUNTS[rid] = step_index
                    trace_events.append(
                        _build_trace_event(
                            rid=rid,
                            backend="specforge_sglang",
                            prefix_token_ids=request_prefix_token_ids(req),
                            draft_chunk_token_ids=draft_chunk,
                            accept_len=accept_len,
                            committed_token_ids=committed,
                            step_index=step_index,
                        )
                    )
            except Exception:
                trace_events = []
            _append_trace_events(trace_events)
            return predict_tokens

        sopm.SchedulerOutputProcessorMixin._resolve_spec_overlap_token_ids = wrapped
        sopm.SchedulerOutputProcessorMixin._eagle3_trace_patched = True

    engine_module.Engine.run_scheduler_process_func = staticmethod(run_sglang_scheduler_process_with_trace)
    apply_scheduler_patch()


def run_sglang_scheduler_process_with_trace(*args: Any, **kwargs: Any) -> Any:
    from sglang.srt.managers.scheduler import run_scheduler_process

    install_sglang_trace_patch()
    return run_scheduler_process(*args, **kwargs)


install_vllm_trace_patch()
install_sglang_trace_patch()
