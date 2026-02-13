"""BFCL execution environment integrated as canonical `t3rl` env module."""

from __future__ import annotations

import copy
import importlib
import inspect
import json
from typing import Any, Callable

from bfcl_eval.constants.executable_backend_config import (
    CLASS_FILE_PATH_MAPPING,
    STATELESS_CLASSES,
)
from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_checker import (
    response_checker,
    state_checker,
)
from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils import (
    is_empty_execute_response,
)


class BFCLEnv:
    """Isolated execution environment for BFCL multi-turn function calling."""

    def __init__(
        self,
        initial_config: dict[str, Any],
        involved_classes: list[str],
        ground_truth_calls: list[list[str]],
        long_context: bool = False,
    ):
        self.initial_config = initial_config
        self.involved_classes = involved_classes
        self.ground_truth_calls = ground_truth_calls
        self.long_context = long_context

        self.model_instances: dict[str, Any] = {}
        self.model_method_registry: dict[str, Callable] = {}

        self._gt_instances: dict[str, Any] = {}
        self._gt_method_registry: dict[str, Callable] = {}

        self.all_turn_results: list[Any] = []
        self.current_turn_results: list[Any] = []

        self._initialize_environments()

    def _initialize_environments(self) -> None:
        self._init_single_environment(
            instances_dict=self.model_instances,
            method_registry=self.model_method_registry,
        )
        self._init_single_environment(
            instances_dict=self._gt_instances,
            method_registry=self._gt_method_registry,
        )

    def _init_single_environment(
        self,
        instances_dict: dict[str, Any],
        method_registry: dict[str, Callable],
    ) -> None:
        for class_name in self.involved_classes:
            module_name = CLASS_FILE_PATH_MAPPING[class_name]
            module = importlib.import_module(module_name)
            class_ = getattr(module, class_name)

            instance = class_()
            if class_name not in STATELESS_CLASSES:
                class_initial_config = self.initial_config.get(class_name, {})
                instance._load_scenario(
                    copy.deepcopy(class_initial_config),
                    long_context=self.long_context,
                )

            instances_dict[class_name] = instance
            for method_name, method in inspect.getmembers(
                instance,
                predicate=inspect.ismethod,
            ):
                if not method_name.startswith("_"):
                    method_registry[method_name] = method

    def execute_calls(self, func_call_list: list[str]) -> list[str]:
        results: list[str] = []
        for func_call in func_call_list:
            try:
                result = self._execute_single_call(
                    func_call,
                    self.model_method_registry,
                )
                results.append(self._format_result(result))
            except Exception as exc:
                results.append(f"Error during execution: {exc}")

        self.current_turn_results.extend(results)
        return results

    def calculate_turn_score(self, turn_index: int) -> dict[str, Any]:
        gt_calls = self.ground_truth_calls[turn_index]

        if not gt_calls or is_empty_execute_response(gt_calls):
            self.current_turn_results = []
            return {
                "turn_score": -1.0,
                "valid": True,
                "is_irrelevance_turn": True,
            }

        self.all_turn_results.extend(self.current_turn_results)

        if not self.current_turn_results or is_empty_execute_response(
            self.current_turn_results
        ):
            self.current_turn_results = []
            return {
                "turn_score": 0.0,
                "valid": False,
                "is_irrelevance_turn": False,
                "error_type": "no_model_calls",
            }

        gt_results = self._execute_gt_calls(gt_calls)

        state_check_result = state_checker(self.model_instances, self._gt_instances)
        state_valid = state_check_result["valid"]

        response_check_result = response_checker(
            self.all_turn_results,
            gt_results,
            turn_index,
        )
        response_valid = response_check_result["valid"]

        turn_score = 1.0 if state_valid and response_valid else 0.0
        self.current_turn_results = []

        return {
            "turn_score": turn_score,
            "valid": turn_score == 1.0,
            "is_irrelevance_turn": False,
            "state_valid": state_valid,
            "response_valid": response_valid,
            "state_diff": state_check_result.get("details", {}) if not state_valid else {},
            "response_diff": (
                response_check_result.get("details", {})
                if not response_valid
                else {}
            ),
            "error_type": (
                "state_mismatch"
                if not state_valid
                else "response_mismatch"
                if not response_valid
                else None
            ),
        }

    def _execute_gt_calls(self, gt_calls: list[str]) -> list[str]:
        results: list[str] = []
        for func_call in gt_calls:
            try:
                result = self._execute_single_call(func_call, self._gt_method_registry)
                results.append(self._format_result(result))
            except Exception as exc:
                results.append(f"Error during execution: {exc}")

        return results

    def _execute_single_call(
        self,
        func_call: str,
        method_registry: dict[str, Callable],
    ) -> Any:
        safe_namespace = method_registry.copy()
        safe_builtins = {
            "len": len,
            "str": str,
            "int": int,
            "float": float,
            "bool": bool,
            "list": list,
            "dict": dict,
            "tuple": tuple,
            "set": set,
        }

        self._validate_func_call(func_call)
        return eval(func_call, {"__builtins__": safe_builtins}, safe_namespace)

    def _validate_func_call(self, func_call: str) -> None:
        dangerous_patterns = [
            "kill",
            "exit",
            "quit",
            "remove",
            "unlink",
            "popen",
            "Popen",
            "run",
        ]
        if func_call in dangerous_patterns:
            raise Exception(f"Function call {func_call} is not allowed.")

    def _format_result(self, result: Any) -> str:
        if isinstance(result, str):
            return result
        if isinstance(result, dict):
            try:
                return json.dumps(result)
            except Exception:
                return str(result)
        return str(result)

    def replay_historical_calls(self, func_call_list: list[str]) -> None:
        for func_call in func_call_list:
            try:
                self._execute_single_call(func_call, self.model_method_registry)
            except Exception:
                pass

            try:
                self._execute_single_call(func_call, self._gt_method_registry)
            except Exception:
                pass

    def reset_turn(self) -> None:
        self.current_turn_results = []

    def get_instances(self) -> dict[str, Any]:
        return self.model_instances

    def get_gt_instances(self) -> dict[str, Any]:
        return self._gt_instances

    def get_cumulative_results(self) -> list[Any]:
        return self.all_turn_results

    def __repr__(self) -> str:
        return (
            f"BFCLEnv(classes={self.involved_classes}, "
            f"turns={len(self.ground_truth_calls)}, "
            f"long_context={self.long_context})"
        )

