#!/usr/bin/env python
from __future__ import annotations

import argparse
import atexit
import json
import os
import socket
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
SPECFORGE_ROOT = ROOT / "third_party" / "SpecForge"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SPECFORGE_ROOT) not in sys.path:
    sys.path.insert(0, str(SPECFORGE_ROOT))

from eval.prepare_data import DATASET_NAMES
from eval.prompt_templates import build_prompt_messages
from run_eval import (
    BACKEND_SPECFORGE_NATIVE,
    MTBENCH_TURN1_HISTORY_RESERVE,
    adjusted_sampling_params,
    append_jsonl,
    build_trace_event,
    build_trace_stats,
    json_dump,
    read_jsonl,
    render_chat_prompt,
    safe_name,
    score_sample,
    summarise_dataset_results,
)
from specforge.distributed import destroy_distributed, init_distributed
from specforge.modeling.auto import AutoEagle3DraftModel
from specforge.modeling.target import get_eagle3_target_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SpecForge-native sliding-window Eagle3 evaluation."
    )
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--target-model-path", required=True)
    parser.add_argument("--draft-model-path", required=True)
    parser.add_argument("--sample-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", choices=DATASET_NAMES, required=True)
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--draft-steps", type=int, default=None)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260507)
    parser.add_argument("--gpu-ids", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def cleanup_distributed() -> None:
    try:
        if dist.is_initialized():
            destroy_distributed()
    except Exception:
        pass


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def init_runtime(
    args: argparse.Namespace,
    *,
    rank: int,
    world_size: int,
    master_port: int,
) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)
    torch.manual_seed(args.seed)
    init_distributed(timeout=10, tp_size=args.tp_size, sp_ulysses_size=1, sp_ring_size=1)
    atexit.register(cleanup_distributed)


def sample_next_token(logits: torch.Tensor) -> int:
    return int(logits.argmax(dim=-1).item())


def load_training_args(draft_model_path: str) -> argparse.Namespace:
    state_path = Path(draft_model_path) / "training_state.pt"
    if not state_path.exists():
        raise FileNotFoundError(
            f"Missing training state at {state_path}. "
            "SpecForge native evaluation requires the saved training args."
        )
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    train_args = state.get("args")
    if train_args is None:
        raise KeyError(f"Missing args in training state: {state_path}")
    return train_args


class SpecForgeNativeRunner:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        current_device = torch.cuda.current_device()
        self.target_device = torch.device(f"cuda:{current_device}")
        self.draft_device = self.target_device
        self.train_args = load_training_args(args.draft_model_path)
        self.target_backend = getattr(self.train_args, "target_model_backend", "sglang")
        self.attention_backend = getattr(self.train_args, "attention_backend", "sdpa")
        self.draft_steps = args.draft_steps or getattr(self.train_args, "ttt_length", 1)

        self.tokenizer = AutoTokenizer.from_pretrained(
            args.target_model_path,
            trust_remote_code=args.trust_remote_code,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if self.target_backend != "sglang":
            raise ValueError(
                f"SpecForge native evaluation currently requires an sglang target backend, "
                f"but checkpoint recorded {self.target_backend!r}."
            )

        self.target = get_eagle3_target_model(
            args.target_model_path,
            backend=self.target_backend,
            torch_dtype=torch.bfloat16,
            trust_remote_code=args.trust_remote_code,
        )
        self.target.set_aux_hidden_states_layers()

        self.draft = AutoEagle3DraftModel.from_pretrained(
            args.draft_model_path,
            attention_backend=self.attention_backend,
            torch_dtype=torch.bfloat16,
            trust_remote_code=args.trust_remote_code,
        )
        self.draft.load_embedding(args.target_model_path)
        self.draft = self.draft.to(self.draft_device).eval()
        self.draft_sliding_window = self.draft.get_sliding_window()

        self.stop_token_ids = {
            token_id
            for token_id in [self.tokenizer.eos_token_id]
            if token_id is not None
        }

    def draft_to_target_token_id(self, draft_token_id: int) -> int:
        target_token_id = draft_token_id + int(self.draft.d2t[draft_token_id].item())
        if target_token_id < 0 or target_token_id >= len(self.tokenizer):
            raise ValueError(
                f"Draft token {draft_token_id} mapped to invalid target token {target_token_id}"
            )
        return target_token_id

    @torch.no_grad()
    def target_forward(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        attention_mask = torch.ones_like(input_ids, device=input_ids.device)
        loss_mask = torch.ones_like(input_ids, device=input_ids.device)
        _, logits_list, aux_hidden_states_list, _ = self.target.extend(
            input_ids=input_ids,
            attention_mask=attention_mask,
            loss_mask=loss_mask,
            return_last_hidden_states=False,
            return_logits=True,
        )
        hidden_states = torch.cat(
            [hidden_states.unsqueeze(0) for hidden_states in aux_hidden_states_list], dim=0
        )
        logits = torch.cat([logits.unsqueeze(0) for logits in logits_list], dim=0)
        return hidden_states, logits

    @torch.no_grad()
    def draft_propose(self, input_ids: torch.Tensor, hidden_states: torch.Tensor) -> list[int]:
        draft_input_ids = input_ids.to(self.draft_device)
        draft_hidden_states = self.draft.project_hidden_states(hidden_states.to(self.draft_device))
        cache_hidden = [[], []]
        proposals: list[int] = []

        for _ in range(self.draft_steps):
            attention_mask = torch.ones_like(draft_input_ids, device=self.draft_device)
            decoder_mask = self.draft.prepare_decoder_attention_mask(
                attention_mask,
                draft_hidden_states,
                batch_size=1,
                seq_length=draft_input_ids.shape[1],
                past_key_values_length=0,
            )
            position_ids = torch.arange(
                draft_input_ids.shape[1],
                device=self.draft_device,
            ).unsqueeze(0)
            input_embeds = self.draft.embed_input_ids(draft_input_ids)
            draft_hidden_states = self.draft.backbone(
                input_embeds=input_embeds,
                hidden_states=draft_hidden_states,
                cache_hidden=cache_hidden,
                attention_mask=decoder_mask,
                position_ids=position_ids,
                past_key_values=None,
                use_cache=True,
            )
            draft_logits = self.draft.compute_logits(draft_hidden_states)
            draft_token_id = sample_next_token(draft_logits[:, -1, :])
            target_token_id = self.draft_to_target_token_id(draft_token_id)
            proposals.append(target_token_id)

            next_token = torch.tensor(
                [[target_token_id]],
                device=self.draft_device,
                dtype=draft_input_ids.dtype,
            )
            draft_input_ids = torch.cat([draft_input_ids[:, 1:], next_token], dim=1)

            if target_token_id in self.stop_token_ids:
                break

        return proposals

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int,
        *,
        rid: str,
        turn_index: int | None = None,
    ) -> tuple[str, dict[str, Any]]:
        encoded = self.tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=False,
        )
        prompt_ids = encoded["input_ids"].to(self.target_device)
        current_ids = prompt_ids
        accept_history: list[int] = []
        trace_events: list[dict[str, Any]] = []
        step_index = 0

        while current_ids.shape[1] - prompt_ids.shape[1] < max_new_tokens:
            hidden_states, _ = self.target_forward(current_ids)
            proposals = self.draft_propose(current_ids, hidden_states)
            if not proposals:
                break

            proposal_tensor = torch.tensor(
                [proposals], device=self.target_device, dtype=current_ids.dtype
            )
            verify_ids = torch.cat([current_ids, proposal_tensor], dim=1)
            _, verify_logits = self.target_forward(verify_ids)
            verify_preds = verify_logits.argmax(dim=-1)[0]

            base_idx = current_ids.shape[1] - 1
            accept_len = 0
            for idx, proposal in enumerate(proposals):
                if int(verify_preds[base_idx + idx].item()) != proposal:
                    break
                accept_len += 1

            next_token = int(verify_preds[base_idx + accept_len].item())
            appended = proposals[:accept_len] + [next_token]
            appended_tensor = torch.tensor(
                [appended], device=self.target_device, dtype=current_ids.dtype
            )
            step_index += 1
            trace_events.append(
                build_trace_event(
                    rid=rid,
                    backend=BACKEND_SPECFORGE_NATIVE,
                    tokenizer=self.tokenizer,
                    prefix_token_ids=current_ids[0].tolist(),
                    draft_chunk_token_ids=[int(token_id) for token_id in proposals],
                    accept_len=accept_len,
                    committed_token_ids=[int(token_id) for token_id in appended],
                    step_index=step_index,
                    turn_index=turn_index,
                )
            )
            current_ids = torch.cat([current_ids, appended_tensor], dim=1)
            accept_history.append(accept_len)

            if next_token in self.stop_token_ids:
                break

        generated_ids = current_ids[0, prompt_ids.shape[1] :].tolist()
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        trace_meta = build_trace_stats(trace_events)
        meta = {
            **trace_meta,
            "accept_history": accept_history,
            "trace_events": trace_events,
        }
        return text, meta


def evaluate_sample(
    runner: SpecForgeNativeRunner,
    dataset_name: str,
    sample: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rid = f"{dataset_name}::{safe_name(str(sample['sample_id']))}"
    trace_events: list[dict[str, Any]] = []

    if dataset_name == "mtbench":
        messages = [{"role": "system", "content": "You are a helpful assistant."}]
        turns = []
        for turn_index, user_turn in enumerate(sample["turns"]):
            messages.append({"role": "user", "content": user_turn})
            prompt = render_chat_prompt(runner.tokenizer, messages)
            sampling_params, prompt_tokens = adjusted_sampling_params(
                runner.tokenizer,
                prompt,
                "mtbench",
                reserve_tokens=MTBENCH_TURN1_HISTORY_RESERVE if turn_index == 0 else 0,
            )
            text, meta = runner.generate(
                prompt,
                int(sampling_params["max_new_tokens"]),
                rid=rid,
                turn_index=turn_index + 1,
            )
            turns.append(
                {
                    "turn_index": turn_index + 1,
                    "text": text,
                    "token_ids": [],
                    "meta_info": {
                        "prompt_tokens": prompt_tokens,
                        "max_new_tokens": sampling_params["max_new_tokens"],
                        **meta,
                    },
                }
            )
            trace_events.extend(meta["trace_events"])
            messages.append({"role": "assistant", "content": text})
        response = {
            "rid": rid,
            "text": turns[-1]["text"] if turns else "",
            "token_ids": [],
            "meta_info": turns[-1]["meta_info"] if turns else {},
            "turns": turns,
        }
        score = {"metric_name": "generation_only", "score": None}
        return {
            "response": response,
            "score": score,
            "trace_events": trace_events,
        }, trace_events

    messages = build_prompt_messages(dataset_name, sample)
    prompt = render_chat_prompt(runner.tokenizer, messages)
    sampling_params, prompt_tokens = adjusted_sampling_params(
        runner.tokenizer,
        prompt,
        dataset_name,
    )
    text, meta = runner.generate(
        prompt,
        int(sampling_params["max_new_tokens"]),
        rid=rid,
    )
    response = {
        "rid": rid,
        "text": text,
        "token_ids": [],
        "meta_info": {
            "prompt_tokens": prompt_tokens,
            "max_new_tokens": sampling_params["max_new_tokens"],
            **meta,
        },
    }
    score = score_sample(dataset_name, sample, response)
    trace_events.extend(meta["trace_events"])
    return {"response": response, "score": score, "trace_events": trace_events}, trace_events


def run_worker(
    rank: int,
    world_size: int,
    master_port: int,
    args: argparse.Namespace,
) -> None:
    init_runtime(args, rank=rank, world_size=world_size, master_port=master_port)
    runner = SpecForgeNativeRunner(args)
    is_writer = rank == 0

    model_log_dir = args.output_dir
    combined_path = model_log_dir / "combined_results.jsonl"
    if is_writer:
        model_log_dir.mkdir(parents=True, exist_ok=True)
        if combined_path.exists():
            combined_path.unlink()
    dist.barrier()

    summaries = []
    for dataset_name in args.datasets:
        dataset_rows = read_jsonl(args.sample_dir / f"{dataset_name}.jsonl")
        if args.sample_size is not None:
            dataset_rows = dataset_rows[: args.sample_size]
        result_rows: list[dict[str, Any]] = []
        trace_events: list[dict[str, Any]] = []
        dataset_dir = model_log_dir / dataset_name
        results_path = dataset_dir / "results.jsonl"
        if is_writer:
            dataset_dir.mkdir(parents=True, exist_ok=True)
            if results_path.exists():
                results_path.unlink()
        dist.barrier()

        started = time.time()
        for sample in dataset_rows:
            payload, sample_trace_events = evaluate_sample(runner, dataset_name, sample)
            if not is_writer:
                continue
            response = payload["response"]
            score = payload["score"]
            record = {
                "model_key": args.model_key,
                "model_name": args.model_name,
                "backend": "specforge_native",
                "dataset": dataset_name,
                "sample_id": sample["sample_id"],
                "rid": response["rid"],
                "text": response.get("text", ""),
                "meta_info": response["meta_info"],
                "spec_accept_length": response["meta_info"].get("spec_accept_length"),
                "spec_accept_rate": response["meta_info"].get("spec_accept_rate"),
                "spec_verify_ct": response["meta_info"].get("spec_verify_ct"),
                **score,
            }
            if dataset_name == "mtbench":
                record["turns"] = response["turns"]
            append_jsonl(results_path, record)
            append_jsonl(combined_path, record)
            result_rows.append(record)
            trace_events.extend(sample_trace_events)

        if is_writer:
            summary = summarise_dataset_results(
                model_key=args.model_key,
                dataset_name=dataset_name,
                results=result_rows,
                trace_events=trace_events,
            )
            summary["wall_time_sec"] = time.time() - started
            json_dump(dataset_dir / "summary.json", summary)
            summaries.append(summary)

            trace_out = dataset_dir / "spec_trace.jsonl"
            with trace_out.open("w", encoding="utf-8") as f:
                for event in trace_events:
                    f.write(json.dumps(event, ensure_ascii=False) + "\n")
        dist.barrier()

    if is_writer:
        final_summary = {
            "model_key": args.model_key,
            "model_name": args.model_name,
            "backend": "specforge_native",
            "gpu_ids": args.gpu_ids,
            "speculative_config": {
                "method": "specforge_native_sliding_window",
                "draft_steps": runner.draft_steps,
                "draft_attention_backend": runner.attention_backend,
                "draft_sliding_window": runner.draft_sliding_window,
                "target_backend": runner.target_backend,
                "tensor_parallel_size": args.tp_size,
            },
            "datasets": summaries,
        }
        json_dump(model_log_dir / "model_summary.json", final_summary)
    dist.barrier()


def main() -> None:
    args = parse_args()
    if args.tp_size < 1:
        raise ValueError(f"tp_size must be >= 1, got {args.tp_size}")
    if args.tp_size > len(args.gpu_ids):
        raise ValueError(
            f"tp_size ({args.tp_size}) cannot exceed visible gpu count ({len(args.gpu_ids)})"
        )

    mp.spawn(
        run_worker,
        nprocs=args.tp_size,
        args=(args.tp_size, find_free_port(), args),
        join=True,
    )


if __name__ == "__main__":
    main()
