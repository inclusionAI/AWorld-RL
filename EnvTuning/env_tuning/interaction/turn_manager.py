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

from typing import Tuple, Dict, Any
from .data_models import InstanceState
from .score_calculator import ScoreCalculator


class TurnManager:
    """管理多轮对话逻辑"""
    
    def __init__(self, score_calculator: ScoreCalculator):
        self.score_calculator = score_calculator
    
    def advance_to_next_turn(self, state: InstanceState, entry_id: str) -> Tuple[bool, str, float, Dict[str, Any]]:
        """
        推进到下一轮
        
        Args:
            state: 实例状态
            entry_id: 条目ID
            
        Returns:
            Tuple[bool, str, float, Dict[str, Any]]: (是否终止, 响应内容, 评分, 额外数据)
        """
        # 刷新执行结果
        state.flush_exec_results_to_all()
        
        # 获取当前轮次的 ground truth
        prev_turn_idx = state.current_turn_index
        ground_truth_calls = self._get_ground_truth_calls(state, prev_turn_idx)
        
        # 更新轮次索引
        state.current_turn_index += 1
        
        # 计算评分
        score = self.score_calculator.calculate_turn_score(state, ground_truth_calls, entry_id)
        
        # 准备下一个问题
        should_terminate, next_question = self._prepare_next_question(state)
        
        # 重置单轮缓存
        state.reset_single_turn_buffers()
        
        return should_terminate, next_question, score, {}
    
    def _get_ground_truth_calls(self, state: InstanceState, turn_index: int) -> list:
        """
        获取指定轮次的 ground truth 调用
        
        Args:
            state: 实例状态
            turn_index: 轮次索引
            
        Returns:
            list: ground truth 调用列表
        """
        if turn_index < len(state.ground_truth):
            return state.ground_truth[turn_index]
        return []
    
    def _prepare_next_question(self, state: InstanceState) -> Tuple[bool, str]:
        """
        准备下一个问题
        
        Args:
            state: 实例状态
            
        Returns:
            Tuple[bool, str]: (是否终止, 下一个问题)
        """
        if state.processed_question:
            next_question = state.processed_question.pop(0)
            return False, next_question
        else:
            return True, ""
    
    def should_force_quit(self, state: InstanceState, max_step_limit: int) -> bool:
        """
        判断是否应该强制退出
        
        Args:
            state: 实例状态
            max_step_limit: 最大步数限制
            
        Returns:
            bool: 是否应该强制退出
        """
        return state.current_turn_attempt_counts > max_step_limit
    
    def is_sequence_complete(self, state: InstanceState) -> bool:
        """
        判断序列是否完成
        
        Args:
            state: 实例状态
            
        Returns:
            bool: 序列是否完成
        """
        return len(state.processed_question) == 0
    
    def get_current_turn_info(self, state: InstanceState) -> Dict[str, Any]:
        """
        获取当前轮次信息
        
        Args:
            state: 实例状态
            
        Returns:
            Dict[str, Any]: 轮次信息
        """
        return {
            "current_turn": state.current_turn_index,
            "total_turns": state.total_turns,
            "attempt_count": state.current_turn_attempt_counts,
            "remaining_questions": len(state.processed_question)
        }
    
    def reset_turn_counters(self, state: InstanceState) -> None:
        """
        重置轮次计数器
        
        Args:
            state: 实例状态
        """
        state.current_turn_attempt_counts = 0
        state.reset_single_turn_buffers()