from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from eval.config_loader import RUN_EVAL_CONFIG

TRACE_CONTEXT_WINDOW = int(
    os.environ.get(
        "EAGLE3_TRACE_CONTEXT_WINDOW",
        RUN_EVAL_CONFIG.get("trace_context_window", 16),
    )
)
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


def _decode_single_token(tokenizer: Any, token_id: int | None) -> str | None:
    if token_id is None:
        return None
    return _decode_token_pieces(tokenizer, [token_id])[0]


def _flatten_items(value: Any) -> list[Any]:
    if value is None:
        return []
    if hasattr(value, "reshape") and hasattr(value, "tolist"):
        try:
            return value.reshape(-1).tolist()
        except Exception:
            pass
    if isinstance(value, (list, tuple)):
        flat: list[Any] = []
        for item in value:
            if isinstance(item, (list, tuple)):
                flat.extend(_flatten_items(item))
            else:
                flat.append(item)
        return flat
    try:
        return list(value)
    except TypeError:
        return [value]


def _to_int_list(value: Any) -> list[int]:
    items = _flatten_items(value)
    return [int(token_id) for token_id in items if int(token_id) >= 0]


def _to_raw_int_list(value: Any) -> list[int]:
    items = _flatten_items(value)
    return [int(token_id) for token_id in items]


def _request_prefix_token_ids(req: Any) -> list[int]:
    prefix: list[int] = []
    for attr in ("origin_input_ids", "input_ids", "prompt_token_ids"):
        items = _to_int_list(getattr(req, attr, None))
        if items:
            prefix.extend(items)
            break
    for attr in ("output_ids", "decoded_ids", "output_token_ids"):
        items = _to_int_list(getattr(req, attr, None))
        if items:
            prefix.extend(items)
            break
    return prefix


def _build_tree_relationships(
    next_token: list[int],
    next_sibling: list[int],
) -> tuple[list[int | None], list[list[int]], list[int]]:
    num_nodes = min(len(next_token), len(next_sibling))
    parent_indices: list[int | None] = [None] * num_nodes
    child_indices: list[list[int]] = [[] for _ in range(num_nodes)]

    for parent_idx in range(num_nodes):
        child_idx = next_token[parent_idx]
        visited: set[int] = set()
        while 0 <= child_idx < num_nodes and child_idx not in visited:
            child_indices[parent_idx].append(child_idx)
            if parent_indices[child_idx] is None:
                parent_indices[child_idx] = parent_idx
            visited.add(child_idx)
            child_idx = next_sibling[child_idx]

    root_indices = [idx for idx, parent_idx in enumerate(parent_indices) if parent_idx is None]
    return parent_indices, child_indices, root_indices


def _build_sglang_tree_event(
    *,
    rid: str,
    prefix_token_ids: list[int],
    step_index: int,
    tree_token_ids: list[int],
    tree_positions: list[int],
    retrieve_index: list[int],
    retrieve_next_token: list[int],
    retrieve_next_sibling: list[int],
    accepted_local_indices: list[int],
    committed_token_ids: list[int],
    accept_len: int,
) -> dict[str, Any]:
    tokenizer = _load_tokenizer()
    prefix_tail = prefix_token_ids[-TRACE_CONTEXT_WINDOW:]
    tree_tokens = _decode_token_pieces(tokenizer, tree_token_ids)
    draft_path_node_indices = accepted_local_indices[1:]
    accepted_path_token_ids = committed_token_ids[:accept_len]
    accepted_path_tokens = _decode_token_pieces(tokenizer, accepted_path_token_ids)
    bonus_token_id = committed_token_ids[accept_len] if len(committed_token_ids) > accept_len else None
    frontier_node_index = accepted_local_indices[-1] if accepted_local_indices else None

    parent_indices, child_indices, root_indices = _build_tree_relationships(
        retrieve_next_token,
        retrieve_next_sibling,
    )

    if frontier_node_index is not None and 0 <= frontier_node_index < len(child_indices):
        frontier_child_node_indices = child_indices[frontier_node_index]
    else:
        frontier_child_node_indices = root_indices
    frontier_child_token_ids = [
        tree_token_ids[idx]
        for idx in frontier_child_node_indices
        if 0 <= idx < len(tree_token_ids)
    ]
    frontier_child_tokens = _decode_token_pieces(tokenizer, frontier_child_token_ids)

    tree_nodes = []
    for node_index, token_id in enumerate(tree_token_ids):
        parent_index = parent_indices[node_index]
        children = child_indices[node_index]
        tree_nodes.append(
            {
                "node_index": node_index,
                "token_id": token_id,
                "token": tree_tokens[node_index],
                "position": tree_positions[node_index] if node_index < len(tree_positions) else None,
                "parent_node_index": parent_index,
                "child_node_indices": children,
            }
        )

    return {
        "rid": rid,
        "backend": "specforge_sglang",
        "step_index": step_index,
        "accept_len": accept_len,
        "context_len": len(prefix_token_ids),
        "num_draft_tokens": max(0, len(tree_token_ids) - 1),
        "prefix_token_ids": prefix_tail,
        "prefix_tokens": _decode_token_pieces(tokenizer, prefix_tail),
        "prefix_text": _decode_token_text(tokenizer, prefix_tail),
        # Keep legacy top-level fields for existing summaries.
        "draft_chunk_token_ids": tree_token_ids[1:],
        "draft_chunk_tokens": _decode_token_pieces(tokenizer, tree_token_ids[1:]),
        "accepted_token_ids": accepted_path_token_ids,
        "accepted_tokens": accepted_path_tokens,
        "rejected_candidate_token_ids": frontier_child_token_ids,
        "rejected_candidate_tokens": frontier_child_tokens,
        "committed_token_ids": committed_token_ids,
        "committed_tokens": _decode_token_pieces(tokenizer, committed_token_ids),
        "committed_text": _decode_token_text(tokenizer, committed_token_ids),
        "replacement_token_id": bonus_token_id,
        "replacement_token": _decode_single_token(tokenizer, bonus_token_id),
        "tree": {
            "anchor_token_id": tree_token_ids[0] if tree_token_ids else None,
            "anchor_token": _decode_single_token(tokenizer, tree_token_ids[0] if tree_token_ids else None),
            "draft_token_ids": tree_token_ids,
            "draft_tokens": tree_tokens,
            "positions": tree_positions,
            "retrieve_index": retrieve_index,
            "retrieve_next_token": retrieve_next_token,
            "retrieve_next_sibling": retrieve_next_sibling,
            "parent_node_indices": parent_indices,
            "root_node_indices": root_indices,
            "nodes": tree_nodes,
        },
        "verify": {
            "accepted_tree_node_indices": accepted_local_indices,
            "accepted_draft_node_indices": draft_path_node_indices,
            "accepted_draft_token_ids": accepted_path_token_ids,
            "accepted_draft_tokens": accepted_path_tokens,
            "frontier_node_index": frontier_node_index,
            "frontier_child_node_indices": frontier_child_node_indices,
            "frontier_child_token_ids": frontier_child_token_ids,
            "frontier_child_tokens": frontier_child_tokens,
            "bonus_token_id": bonus_token_id,
            "bonus_token": _decode_single_token(tokenizer, bonus_token_id),
            "committed_token_ids": committed_token_ids,
            "committed_tokens": _decode_token_pieces(tokenizer, committed_token_ids),
            "first_divergence": {
                "frontier_node_index": frontier_node_index,
                "draft_child_node_indices": frontier_child_node_indices,
                "draft_child_token_ids": frontier_child_token_ids,
                "draft_child_tokens": frontier_child_tokens,
                "target_token_id": bonus_token_id,
                "target_token": _decode_single_token(tokenizer, bonus_token_id),
            },
        },
    }


def _build_sglang_v1_trace_events_from_verify(
    batch: Any,
    spec_info: Any,
    verify_output: Any,
) -> list[dict[str, Any]]:
    if (
        spec_info is None
        or getattr(spec_info, "draft_token", None) is None
        or getattr(spec_info, "draft_token_num", None) is None
        or verify_output is None
        or getattr(verify_output, "accept_length_per_req_cpu", None) is None
        or getattr(verify_output, "verified_id", None) is None
    ):
        return []

    draft_token_num = int(spec_info.draft_token_num or 0)
    if draft_token_num <= 0:
        return []

    draft_tokens = _to_int_list(spec_info.draft_token)
    positions = _to_raw_int_list(getattr(spec_info, "positions", None))
    retrieve_index_rows = _to_raw_int_list(getattr(spec_info, "retrive_index", None))
    retrieve_next_token_rows = _to_raw_int_list(getattr(spec_info, "retrive_next_token", None))
    retrieve_next_sibling_rows = _to_raw_int_list(getattr(spec_info, "retrive_next_sibling", None))
    accept_lens = [max(0, int(x)) for x in verify_output.accept_length_per_req_cpu]
    verified_ids = _to_int_list(verify_output.verified_id)
    accepted_indices = _to_int_list(getattr(verify_output, "accepted_indices", None))
    reqs = list(getattr(batch, "reqs", []))
    if len(accept_lens) != len(reqs):
        return []

    events: list[dict[str, Any]] = []
    verified_offset = 0
    accepted_offset = 0
    for i, req in enumerate(reqs):
        draft_start = i * draft_token_num
        draft_end = draft_start + draft_token_num
        tree_token_ids = draft_tokens[draft_start:draft_end]
        tree_positions = positions[draft_start:draft_end]
        retrieve_index = retrieve_index_rows[draft_start:draft_end]
        retrieve_next_token = retrieve_next_token_rows[draft_start:draft_end]
        retrieve_next_sibling = retrieve_next_sibling_rows[draft_start:draft_end]
        if not tree_token_ids:
            continue

        accept_len = min(accept_lens[i], max(0, len(tree_token_ids) - 1))
        committed_len = min(len(verified_ids) - verified_offset, accept_len + 1)
        if committed_len <= 0:
            continue
        committed_token_ids = verified_ids[verified_offset : verified_offset + committed_len]
        verified_offset += committed_len
        accepted_count = min(len(accepted_indices) - accepted_offset, accept_len + 1)
        local_accepted_indices = [
            accepted_indices[accepted_offset + j] - draft_start
            for j in range(accepted_count)
        ]
        accepted_offset += accepted_count

        current_prefix = _request_prefix_token_ids(req)
        prefix_token_ids = (
            current_prefix[:-committed_len] if committed_len <= len(current_prefix) else []
        )

        rid = str(getattr(req, "rid", i))
        step_index = _SGLANG_STEP_COUNTS.get(rid, 0) + 1
        _SGLANG_STEP_COUNTS[rid] = step_index
        events.append(
            _build_sglang_tree_event(
                rid=rid,
                prefix_token_ids=prefix_token_ids,
                step_index=step_index,
                tree_token_ids=tree_token_ids,
                tree_positions=tree_positions,
                retrieve_index=retrieve_index,
                retrieve_next_token=retrieve_next_token,
                retrieve_next_sibling=retrieve_next_sibling,
                accepted_local_indices=local_accepted_indices,
                committed_token_ids=committed_token_ids,
                accept_len=accept_len,
            )
        )

    return events


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
                "context_len": len(prefix_ids),
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
        from sglang.srt.speculative import eagle_worker as eagle_worker_module
    except Exception:
        return

    def apply_scheduler_patch() -> None:
        if getattr(sopm.SchedulerOutputProcessorMixin, "_eagle3_trace_patched", False):
            return

        original_overlap = sopm.SchedulerOutputProcessorMixin._resolve_spec_overlap_token_ids

        def wrapped_overlap(self: Any, result: Any, batch: Any) -> Any:
            predict_tokens = original_overlap(self, result, batch)
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
                            prefix_token_ids=_request_prefix_token_ids(req),
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

        sopm.SchedulerOutputProcessorMixin._resolve_spec_overlap_token_ids = wrapped_overlap
        sopm.SchedulerOutputProcessorMixin._eagle3_trace_patched = True

    def apply_eagle_worker_patch() -> None:
        if getattr(eagle_worker_module.EAGLEWorker, "_eagle3_trace_patched", False):
            return

        original_verify = eagle_worker_module.EAGLEWorker.verify

        def wrapped_verify(self: Any, batch: Any, spec_info: Any) -> Any:
            output = original_verify(self, batch, spec_info)
            trace_events: list[dict[str, Any]] = []
            if int(getattr(self, "tp_rank", 0)) == 0:
                try:
                    _, verify_output, _, _ = output
                    trace_events = _build_sglang_v1_trace_events_from_verify(
                        batch,
                        spec_info,
                        verify_output,
                    )
                except Exception:
                    trace_events = []
            _append_trace_events(trace_events)
            return output

        eagle_worker_module.EAGLEWorker.verify = wrapped_verify
        eagle_worker_module.EAGLEWorker._eagle3_trace_patched = True

    engine_module.Engine.run_scheduler_process_func = staticmethod(run_sglang_scheduler_process_with_trace)
    apply_scheduler_patch()
    apply_eagle_worker_patch()


def run_sglang_scheduler_process_with_trace(*args: Any, **kwargs: Any) -> Any:
    from sglang.srt.managers.scheduler import run_scheduler_process

    install_sglang_trace_patch()
    return run_scheduler_process(*args, **kwargs)


install_vllm_trace_patch()
install_sglang_trace_patch()
