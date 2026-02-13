#!/usr/bin/env python3
"""Preprocess BFCL multi-turn data into JSONL for T3RL training."""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
from pathlib import Path
from typing import Any, Optional

from bfcl_eval.constants.category_mapping import MULTI_TURN_CATEGORY
from bfcl_eval.constants.default_prompts import (
    DEFAULT_USER_PROMPT_FOR_ADDITIONAL_FUNCTION_PROMPTING,
)
from bfcl_eval.constants.enums import ModelStyle
from bfcl_eval.constants.type_mappings import GORILLA_TO_OPENAPI
from bfcl_eval.model_handler.utils import convert_to_tool
from bfcl_eval.utils import load_dataset_entry, load_ground_truth_entry

# Localized prompt constants (stable for preprocessing; avoids upstream prompt API drift)
DEFAULT_SYSTEM_PROMPT_FOR_CHAT_MODEL_WO_TOOLS = """You are an expert in composing functions. You are given a question and a set of possible functions. Based on the question, you will need to make one or more function/tool calls to achieve the purpose.

Your response must always start with your step-by-step reasoning process enclosed in `<thinking></thinking>` XML tags. This is for you to outline your plan and justify your chosen actions.

After the thinking block, you will perform one of the following actions:

1.  **If tool calls are necessary:** For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags.
    Example of a function call:
    <tool_call>
    {"name": <function-name>, "arguments": <args-json-object>}
    </tool_call>

2.  **If no tool calls are necessary or possible:** Directly provide a user-facing response in plain text. This applies if none of the functions can be used, or if the given question lacks the parameters required by the function.

At each turn, you should try your best to complete the tasks requested by the user within the current turn. Continue to output functions to call until you have fulfilled the user's request to the best of your ability. Once you have no more functions to call, the system will consider the current turn complete and proceed to the next turn or task."""

RANDOM_SEED = 42
random.seed(RANDOM_SEED)


def _extract_question_content(turn: Any) -> str:
    if isinstance(turn, list) and turn and isinstance(turn[0], dict):
        content = turn[0].get("content")
        if isinstance(content, str):
            return content
    if isinstance(turn, str):
        return turn
    raise ValueError(f"Unsupported turn format: {turn}")


def convert_bfcl_to_slime_format(
    entry: dict[str, Any],
    ground_truth_entry: dict[str, Any],
    dataset_type: str,
) -> dict[str, Any]:
    id_str = entry.get("id", "")
    involved_classes = entry.get("involved_classes", [])
    if not isinstance(involved_classes, list):
        raise ValueError("involved_classes must be a list")

    functions = copy.deepcopy(entry.get("function", []))

    missed_function_docs_by_turn: dict[str, list[dict[str, Any]]] = {}
    if isinstance(entry.get("missed_function"), dict):
        for turn_idx, missed_docs in entry["missed_function"].items():
            if not isinstance(missed_docs, list):
                continue
            missed_function_docs_by_turn[str(turn_idx)] = [
                doc for doc in missed_docs if isinstance(doc, dict)
            ]

    tools = convert_to_tool(
        functions, GORILLA_TO_OPENAPI, ModelStyle.OPENAI_COMPLETIONS
    )

    question_turns = entry.get("question")
    if not isinstance(question_turns, list) or not question_turns:
        raise ValueError(f"question must be non-empty list for entry: {id_str}")

    processed_questions: list[str] = []
    first_question: Optional[str] = None
    miss_func_turn_index: Optional[int] = None
    if missed_function_docs_by_turn:
        miss_func_turn_index = int(next(iter(missed_function_docs_by_turn.keys())))

    for turn_index, turn in enumerate(question_turns):
        if miss_func_turn_index is not None and turn_index == miss_func_turn_index:
            missed_docs = missed_function_docs_by_turn.get(str(turn_index), [])
            additional_functions_json = json.dumps(missed_docs, ensure_ascii=False)
            content = DEFAULT_USER_PROMPT_FOR_ADDITIONAL_FUNCTION_PROMPTING.format(
                functions=additional_functions_json
            )
        else:
            content = _extract_question_content(turn)

        if turn_index == 0:
            first_question = content
        else:
            processed_questions.append(content)

    if first_question is None:
        raise ValueError(f"first question missing for entry: {id_str}")

    if ground_truth_entry.get("id") != id_str:
        raise ValueError(f"ground truth id mismatch: {id_str}")

    ground_truth = ground_truth_entry.get("ground_truth")
    if not isinstance(ground_truth, list):
        raise ValueError(f"ground_truth missing for entry: {id_str}")

    return {
        "prompt": [
            {
                "role": "system",
                "content": DEFAULT_SYSTEM_PROMPT_FOR_CHAT_MODEL_WO_TOOLS,
            },
            {"role": "user", "content": first_question},
        ],
        "metadata": {
            "tools": tools,
            "data_source": dataset_type,
            "ability": "multi_turn_function_calling",
            "split": "train",
            "index": id_str,
            "bfcl_task": {
                "id": id_str,
                "initial_config": entry.get("initial_config", {}),
                "involved_classes": involved_classes,
                "ground_truth": ground_truth,
                "processed_question": processed_questions,
                "question": question_turns,
            },
        },
    }


def generate_id_splits(
    num_ids: int,
    train_ratio: float = 0.5,
    val_ratio: float = 0.125,
) -> tuple[set[int], set[int], set[int]]:
    if train_ratio < 0 or val_ratio < 0 or train_ratio + val_ratio > 1.0:
        raise ValueError("train_ratio and val_ratio must be non-negative and sum <= 1")

    all_ids = list(range(num_ids))
    random.shuffle(all_ids)

    num_train = int(num_ids * train_ratio)
    num_val = int(num_ids * val_ratio)

    train_ids = set(all_ids[:num_train])
    remaining_ids_after_train = all_ids[num_train:]
    val_ids = set(remaining_ids_after_train[:num_val])
    test_ids = set(remaining_ids_after_train[num_val:])

    if len(train_ids) + len(val_ids) + len(test_ids) != num_ids:
        raise RuntimeError("split size mismatch")

    return train_ids, val_ids, test_ids


def _save_jsonl(entries: list[dict[str, Any]], file_path: Path) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8") as file_obj:
        for entry in entries:
            file_obj.write(json.dumps(entry, ensure_ascii=False) + "\n")


def process_all_bfcl_files(
    output_dir: str, split_data: bool = True
) -> list[dict[str, Any]]:
    categories = list(MULTI_TURN_CATEGORY)

    train_ids, val_ids, test_ids = set(), set(), set()
    if split_data:
        num_unique_ids = 200
        train_ids, val_ids, test_ids = generate_id_splits(
            num_unique_ids,
            train_ratio=0.5,
            val_ratio=0.125,
        )
        print(
            f"已生成可复现的ID划分方案: {len(train_ids)} 训练, {len(val_ids)} 验证, {len(test_ids)} 测试."
        )

    all_entries: list[dict[str, Any]] = []
    train_entries: list[dict[str, Any]] = []
    val_entries: list[dict[str, Any]] = []
    test_entries: list[dict[str, Any]] = []

    for category in categories:
        print(f"=== 正在处理 {category} ===")
        category_data = load_dataset_entry(category, include_prereq=False, include_language_specific_hint=False)
        ground_truth_data = load_ground_truth_entry(category)
        ground_truth_by_id = {item.get("id"): item for item in ground_truth_data}

        for entry in category_data:
            full_id = entry.get("id")
            if not full_id:
                print(f"警告: 发现一条没有ID的数据，已跳过: {entry}")
                continue

            ground_truth_entry = ground_truth_by_id.get(full_id)
            if ground_truth_entry is None:
                print(f"警告: 找不到ID为 {full_id} 的ground truth，已跳过")
                continue

            try:
                slime_entry = convert_bfcl_to_slime_format(
                    entry, ground_truth_entry, category
                )
            except Exception as exc:
                print(f"处理 {full_id} 失败: {exc}")
                continue

            all_entries.append(slime_entry)

            if not split_data:
                continue

            try:
                numeric_id = int(str(full_id).split("_")[-1])
            except (IndexError, ValueError):
                print(f"警告: 无法从ID '{full_id}' 中提取数字部分，跳过划分。")
                continue

            if numeric_id in train_ids:
                slime_entry["metadata"]["split"] = "train"
                slime_entry["metadata"]["bfcl_task"]["split"] = "train"
                train_entries.append(slime_entry)
            elif numeric_id in val_ids:
                slime_entry["metadata"]["split"] = "val"
                slime_entry["metadata"]["bfcl_task"]["split"] = "val"
                val_entries.append(slime_entry)
            elif numeric_id in test_ids:
                slime_entry["metadata"]["split"] = "test"
                slime_entry["metadata"]["bfcl_task"]["split"] = "test"
                test_entries.append(slime_entry)

        print(f"已转换并分配 {len(category_data)} 条来自 {category} 的样本")

    print(f"\n=== 总共收集样本: {len(all_entries)} ===")

    output_dir_path = Path(output_dir)
    if split_data:
        if train_entries:
            _save_jsonl(train_entries, output_dir_path / "bfcl_train.jsonl")
            print(f"已保存 {len(train_entries)} 条训练样本")

            train_base_entries = [
                entry
                for entry in train_entries
                if entry.get("metadata", {}).get("data_source") == "multi_turn_base"
            ]
            if train_base_entries:
                _save_jsonl(
                    train_base_entries, output_dir_path / "bfcl_train_base.jsonl"
                )
                print(f"已保存 {len(train_base_entries)} 条 multi_turn_base 训练样本")

        if val_entries:
            _save_jsonl(val_entries, output_dir_path / "bfcl_val.jsonl")
            print(f"已保存 {len(val_entries)} 条验证样本")

        if test_entries:
            _save_jsonl(test_entries, output_dir_path / "bfcl_test.jsonl")
            print(f"已保存 {len(test_entries)} 条测试样本")
    elif all_entries:
        _save_jsonl(all_entries, output_dir_path / "bfcl_all.jsonl")
        print(f"已保存 {len(all_entries)} 条样本")

    return all_entries


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert BFCL data to T3RL JSONL format with consistent ID-based splitting."
    )
    parser.add_argument(
        "--output_dir",
        default="./data/processed/bfcl/",
        help="Output directory for processed data",
    )
    parser.add_argument(
        "--no_split",
        action="store_true",
        help="Don't split data into train/val/test, save as single file",
    )

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    process_all_bfcl_files(args.output_dir, split_data=not args.no_split)
    print("\n处理完成。")


if __name__ == "__main__":
    main()
