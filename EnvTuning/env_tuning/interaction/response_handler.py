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
from .data_models import ResponseData, ResponseType
from .utils import parse_model_response


class ResponseHandler:
    """处理模型响应相关逻辑"""
    
    def parse_and_validate(self, messages: List[Dict[str, Any]]) -> ResponseData:
        """
        解析和验证响应
        
        Args:
            messages: 消息列表
            
        Returns:
            ResponseData: 解析后的响应数据
        """
        # 验证消息格式
        if not messages or messages[-1]["role"] != "assistant":
            return ResponseData(
                content="",
                response_type=ResponseType.PARSE_ERROR,
                is_valid=False,
                error_message="Invalid message format",
                has_error=True
            )
        
        last_message_response = messages[-1]["content"]
        if last_message_response is None:
            return ResponseData(
                content="",
                response_type=ResponseType.PARSE_ERROR,
                is_valid=False,
                error_message="Model raw responses should not be None!",
                has_error=True
            )
        
        # 解析模型响应
        try:
            content, msg_flag = parse_model_response(last_message_response)
            
            if msg_flag == "answer":
                return ResponseData(
                    content=content,
                    response_type=ResponseType.ANSWER,
                    is_valid=True
                )
            elif msg_flag == "tool_call":
                return ResponseData(
                    content=content,
                    response_type=ResponseType.TOOL_CALL,
                    is_valid=True
                )
            else:
                return ResponseData(
                    content=content,
                    response_type=ResponseType.PARSE_ERROR,
                    is_valid=False,
                    error_message=msg_flag,
                    has_error=True
                )
        except Exception as e:
            return ResponseData(
                content=last_message_response,
                response_type=ResponseType.PARSE_ERROR,
                is_valid=False,
                error_message=str(e),
                has_error=True
            )
    
    def validate_message_format(self, messages: List[Dict[str, Any]]) -> bool:
        """
        验证消息格式
        
        Args:
            messages: 消息列表
            
        Returns:
            bool: 是否格式正确
        """
        return (messages and 
                messages[-1]["role"] == "assistant" and 
                messages[-1]["content"] is not None)