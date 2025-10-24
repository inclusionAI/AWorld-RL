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

from typing import Dict, List, Any
from .data_models import InstanceState
from .utils import is_empty_execute_response
from bfcl_env.multi_turn_utils import execute_multi_turn_func_call
from bfcl_env.multi_turn_checker import state_checker, response_checker


class ScoreCalculator:
    """计算评分相关逻辑"""
    
    def calculate_turn_score(self, state: InstanceState, ground_truth_calls: List[Any], entry_id: str) -> float:
        """
        计算当前轮次评分
        
        Args:
            state: 实例状态
            ground_truth_calls: 标准答案调用列表
            entry_id: 条目ID
            
        Returns:
            float: 评分
        """
        # 无 ground truth 的情况
        if not ground_truth_calls:
            return -1.0
        
        # 模型没有响应的情况
        if not state.single_turn_model_response_decode_list or is_empty_execute_response(state.single_turn_model_response_decode_list):
            return 0.0
        
        # 执行 ground truth
        gt_exec_res, gt_instances = self._execute_ground_truth(
            ground_truth_calls, state, entry_id
        )
        
        # 检查状态一致性和响应一致性
        if not self._check_state_consistency(state.involved_instances, gt_instances):
            return 0.0
        elif not self._check_response_validity(
            state.all_turn_model_execution_results, 
            gt_exec_res, 
            state.current_turn_index
        ):
            return 0.0
        else:
            return 1.0
    
    def _execute_ground_truth(self, ground_truth_calls: List[Any], state: InstanceState, entry_id: str) -> tuple:
        """
        执行 ground truth 调用
        
        Args:
            ground_truth_calls: 标准答案调用列表
            state: 实例状态
            entry_id: 条目ID
            
        Returns:
            tuple: (执行结果, 实例字典)
        """
        return execute_multi_turn_func_call(
            func_call_list=ground_truth_calls,
            initial_config=state.initial_config,
            involved_classes=state.involved_classes,
            model_name=f"{id(state)}_ground_truth",
            test_entry_id=entry_id,
            long_context=("long_context" in entry_id or "composite" in entry_id),
            is_evaL_run=True,
        )
    
    def _check_state_consistency(self, model_instances: Dict[str, Any], gt_instances: Dict[str, Any]) -> bool:
        """
        检查状态一致性
        
        Args:
            model_instances: 模型实例字典
            gt_instances: 标准答案实例字典
            
        Returns:
            bool: 是否一致
        """
        try:
            result = state_checker(model_instances, gt_instances)
            return result.get("valid", False)
        except Exception as e:
            print(f"State consistency check failed: {e}")
            return False
    
    def _check_response_validity(self, model_results: List[Any], gt_results: List[Any], turn_index: int) -> bool:
        """
        检查响应有效性
        
        Args:
            model_results: 模型结果列表
            gt_results: 标准答案结果列表
            turn_index: 轮次索引
            
        Returns:
            bool: 是否有效
        """
        try:
            result = response_checker(model_results, gt_results, turn_index)
            return result.get("valid", False)
        except Exception as e:
            print(f"Response validity check failed: {e}")
            return False
    
    def calculate_overall_score(self, all_turn_scores: List[float]) -> float:
        """
        计算整体评分
        
        Args:
            all_turn_scores: 所有轮次评分列表
            
        Returns:
            float: 整体评分
        """
        if not all_turn_scores:
            return 0.0
        
        # 可以根据需要实现不同的评分策略
        # 这里使用平均分
        return sum(score for score in all_turn_scores if score >= 0) / len(all_turn_scores)
    
    def is_ground_truth_empty(self, ground_truth_calls: List[Any]) -> bool:
        """
        检查 ground truth 是否为空
        
        Args:
            ground_truth_calls: 标准答案调用列表
            
        Returns:
            bool: 是否为空
        """
        return not ground_truth_calls