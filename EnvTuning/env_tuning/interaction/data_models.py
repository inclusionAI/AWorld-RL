# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Any
from enum import Enum


class ResponseType(Enum):
    """响应类型枚举"""
    ANSWER = "answer"
    TOOL_CALL = "tool_call"
    PARSE_ERROR = "parse_error"


@dataclass
class ResponseData:
    """解析后的响应数据"""
    content: str
    response_type: ResponseType
    is_valid: bool
    error_message: Optional[str] = None
    has_error: bool = False


@dataclass
class ExecutionResult:
    """执行结果数据"""
    execution_results: List[Any]
    new_instances: Dict[str, Any]
    has_error: bool
    should_continue: bool
    decoded_responses: Optional[List[Any]] = None


@dataclass
class InstanceState:
    """实例状态数据"""
    initial_config: Dict[str, Any]
    involved_classes: List[str]
    ground_truth: List[Any]
    processed_question: List[str]
    question: List[str]
    
    involved_instances: Dict[str, Any]
    total_turns: int

    current_turn_index: int = 0
    current_turn_attempt_counts: int = 0

    all_turn_model_execution_results: List[Any] = field(default_factory=list)
    single_turn_model_execution_results: List[Any] = field(default_factory=list)
    single_turn_model_response_decode_list: List[Any] = field(default_factory=list)

    def reset_single_turn_buffers(self) -> None:
        """在进入下一轮对话时调用，清空本轮缓存。"""
        self.single_turn_model_execution_results.clear()
        self.single_turn_model_response_decode_list.clear()
        self.current_turn_attempt_counts = 0

    def add_exec_results(self, results: List[Any]) -> None:
        """本轮执行完，把结果加入缓存。"""
        self.single_turn_model_execution_results.extend(results)

    def flush_exec_results_to_all(self) -> None:
        """
        把单轮执行结果累加到整体结果，然后清空本轮缓存。
        在进入下一轮、或对当前轮做评测时调用。
        """
        self.all_turn_model_execution_results.extend(self.single_turn_model_execution_results)
        self.single_turn_model_execution_results.clear()

    def __repr__(self) -> str:
        return f"InstanceState({asdict(self)})"