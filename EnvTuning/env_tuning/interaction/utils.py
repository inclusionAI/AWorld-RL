import json
import re
from typing import Dict, List, Optional, Tuple, Any, Union
import ast

def parse_query_response_prompting(api_response: str) -> dict:
        #TODO parsing the future thinking tag in the api_response
        resp_arr = api_response.split('</think>')
        if len(resp_arr) > 1:
            response = resp_arr[-1].strip()
        else:
            # 没有think标志位则用整个输出
            # response = "response outputs too long or no </think> in response."
            response = api_response
            
            print("resp_arr", resp_arr)
        return {
            "model_responses": response
        }


def parse_model_response(
    response: str,
) -> Tuple[str, str]:
    """
    Parse LLM response that must follow one of the two formats
    (thinking+tool_call) or (thinking+answer).

    Returns
    -------
    content : Union[str, list]
        - If <tool_call> is present: the parsed JSON (list) inside the tag.
        - If <answer>    is present: the string inside the tag (stripped).
        - If format error: the original response.
    msg : str
        "answer" or "tool_all"  if success;
        error description (English) on failure.
    """

    # Keep original for error path
    raw = response
    response = response.strip()

    # 1. 检查重复的 <think></think> 标签对
    thinking_matches = re.findall(
        r"<think>([\s\S]*?)</think>", response, flags=re.DOTALL
    )
    if len(thinking_matches) == 0:
        return raw, "Error: Missing <think></think> tags"
    if len(thinking_matches) > 1:
        return (
            raw,
            "Error: Multiple <think></think> tag pairs found. Only one pair is allowed.",
        )

    # 2. 检查重复的 <tool_call></tool_call> 标签对
    tool_matches = re.findall(
        r"<tool_call>([\s\S]*?)</tool_call>", response, flags=re.DOTALL
    )
    if len(tool_matches) > 1:
        return (
            raw,
            "Error: Multiple <tool_call></tool_call> tag pairs found. Only one pair is allowed.",
        )

    # 3. 检查重复的 <answer></answer> 标签对
    answer_matches = re.findall(
        r"<answer>([\s\S]*?)</answer>", response, flags=re.DOTALL
    )
    if len(answer_matches) > 1:
        return (
            raw,
            "Error: Multiple <answer></answer> tag pairs found. Only one pair is allowed.",
        )

    # 4. 检测 <tool_call> 与 <answer> 的互斥性
    has_tool_call = len(tool_matches) > 0
    has_answer = len(answer_matches) > 0

    if has_tool_call and has_answer:
        return (
            raw,
            "Error: Response cannot contain both <tool_call> and <answer> tags",
        )
    if not has_tool_call and not has_answer:
        return (
            raw,
            "Error: Response must contain either <tool_call> or <answer> tags",
        )

    # 5. 检查是否有标签外多余文本（允许空白）
    cleaned = re.sub(r"<think>[\s\S]*?</think>", "", response, flags=re.DOTALL)
    if has_tool_call:
        cleaned = re.sub(
            r"<tool_call>[\s\S]*?</tool_call>", "", cleaned, flags=re.DOTALL
        )
    else:  # has_answer
        cleaned = re.sub(r"<answer>[\s\S]*?</answer>", "", cleaned, flags=re.DOTALL)

    def _has_only_whitespace(text: str) -> bool:
        """辅助：判断字符串去掉空白后是否为空"""
        return text.strip() == ""

    if not _has_only_whitespace(cleaned):
        return (
            raw,
            "Error: Response must not contain text outside the required XML tags",
        )

    # 6. 提取并返回内容
    if has_tool_call:
        tool_body = tool_matches[0].strip()
        # 尝试解析 JSON；仅判断能否解析，进一步检查另行处理
        try:
            obj = json.loads(tool_body)
            json_again = json.dumps(obj)
            return json_again, "tool_call"
        except json.JSONDecodeError as e:
            return raw, f"Error: Invalid JSON inside <tool_call>: {e}"

    # answer 情况
    answer_body = answer_matches[0].strip()

    return answer_body, "answer"



def is_empty_execute_response(input_list: list):
    if len(input_list) == 0:
        return True
    if len(input_list) == 1 and len(input_list[0]) == 0:
        return True
    return False


def resolve_ast_call(elem):
    # Handle nested attributes for deeply nested module paths
    func_parts = []
    func_part = elem.func
    while isinstance(func_part, ast.Attribute):
        func_parts.append(func_part.attr)
        func_part = func_part.value
    if isinstance(func_part, ast.Name):
        func_parts.append(func_part.id)
    func_name = ".".join(reversed(func_parts))
    args_dict = {}
    for arg in elem.keywords:
        output = resolve_ast_by_type(arg.value)
        args_dict[arg.arg] = output
    return {func_name: args_dict}


def resolve_ast_by_type(value):
    if isinstance(value, ast.Constant):
        if value.value is Ellipsis:
            output = "..."
        else:
            output = value.value
    elif isinstance(value, ast.UnaryOp):
        output = -value.operand.value
    elif isinstance(value, ast.List):
        output = [resolve_ast_by_type(v) for v in value.elts]
    elif isinstance(value, ast.Dict):
        output = {
            resolve_ast_by_type(k): resolve_ast_by_type(v)
            for k, v in zip(value.keys, value.values)
        }
    elif isinstance(
        value, ast.NameConstant
    ):  # Added this condition to handle boolean values
        output = value.value
    elif isinstance(
        value, ast.BinOp
    ):  # Added this condition to handle function calls as arguments
        output = eval(ast.unparse(value))
    elif isinstance(value, ast.Name):
        output = value.id
    elif isinstance(value, ast.Call):
        if len(value.keywords) == 0:
            output = ast.unparse(value)
        else:
            output = resolve_ast_call(value)
    elif isinstance(value, ast.Tuple):
        output = tuple(resolve_ast_by_type(v) for v in value.elts)
    elif isinstance(value, ast.Lambda):
        output = eval(ast.unparse(value.body[0].value))
    elif isinstance(value, ast.Ellipsis):
        output = "..."
    elif isinstance(value, ast.Subscript):
        try:
            output = ast.unparse(value.body[0].value)
        except:
            output = ast.unparse(value.value) + "[" + ast.unparse(value.slice) + "]"
    else:
        raise Exception(f"Unsupported AST type: {type(value)}")
    return output

def ast_parse(input_str, language="Python"):
    if language == "Python":
        cleaned_input = input_str.strip("[]'")
        parsed = ast.parse(cleaned_input, mode="eval")
        extracted = []
        if isinstance(parsed.body, ast.Call):
            extracted.append(resolve_ast_call(parsed.body))
        else:
            for elem in parsed.body.elts:
                assert isinstance(elem, ast.Call)
                extracted.append(resolve_ast_call(elem))
        return extracted
    else:
        raise NotImplementedError(f"Unsupported language: {language}. Only support Python language by default.")


def parse_nested_value(value):
    """
    Parse a potentially nested value from the AST output.

    Args:
        value: The value to parse, which could be a nested dictionary, which includes another function call, or a simple value.

    Returns:
        str: A string representation of the value, handling nested function calls and nested dictionary function arguments.
    """
    if isinstance(value, dict):
        # Check if the dictionary represents a function call (i.e., the value is another dictionary or complex structure)
        if all(isinstance(v, dict) for v in value.values()):
            func_name = list(value.keys())[0]
            args = value[func_name]
            args_str = ", ".join(f"{k}={parse_nested_value(v)}" for k, v in args.items())
            return f"{func_name}({args_str})"
        else:
            # If it's a simple dictionary, treat it as key-value pairs
            return (
                "{"
                + ", ".join(f"'{k}': {parse_nested_value(v)}" for k, v in value.items())
                + "}"
            )
    return repr(value)


def decoded_output_to_execution_list(decoded_output):
    """
    Convert decoded output to a list of executable function calls.

    Args:
        decoded_output (list): A list of dictionaries representing function calls.

    Returns:
        list: A list of strings, each representing an executable function call.
    """
    execution_list = []
    for function_call in decoded_output:
        for key, value in function_call.items():
            args_str = ", ".join(f"{k}={parse_nested_value(v)}" for k, v in value.items())
            execution_list.append(f"{key}({args_str})")
    return execution_list



def default_decode_execute_prompting(result):
    result = result.strip("`\n ")
    if not result.startswith("["):
        result = "[" + result
    if not result.endswith("]"):
        result = result + "]"
    decoded_output = ast_parse(result)
    return decoded_output_to_execution_list(decoded_output)


def _build_call_str(name: str, args: Dict[str, Any]) -> str:
    """将函数名和参数字典转成可读形式 'func(a=1, b=\"x\")'。"""
    args_str = ", ".join(f"{k}={repr(v)}" for k, v in args.items())
    return f"{name}({args_str})"


def parse_tool_calls(raw: str
                     ) -> str:
    """
    """
    try:
        calls: List[Dict[str, Any]] = json.loads(raw)
    except json.JSONDecodeError as e:
        return "[]"

    if not isinstance(calls, list):
        calls = [calls] 

    call_strings: List[str] = []
    for call in calls:
        # (1) 必须是 dict
        if not isinstance(call, dict):
            continue

        # (2) 取 name；缺失或类型不对则跳过
        name = call.get("name")
        if not isinstance(name, str) or not name.strip():
            continue
        name = name.strip()

        # (3) 取 arguments；缺失 / None / 非 dict → {}
        args = call.get("arguments")
        if not isinstance(args, dict):
            args = {}

        # (4) 拼接字符串
        call_strings.append(_build_call_str(name, args))

    # call_strings = [_build_call_str(c["name"], c["arguments"]) for c in calls]
    # formatted_calls = "[" + ", ".join(call_strings) + "]"
    
    return "[" + ", ".join(call_strings) + "]" if call_strings else "[]"

def has_execution_error(execution_results: list[str]) -> bool:
    """
    Return True if any result in `execution_results` indicates a failure.

    A failure string is produced by `execute_multi_turn_func_call` when an
    exception occurs, and always starts with the prefix:
        "Error during execution: "

    Parameters
    ----------
    execution_results : list[str]
        List returned by `execute_multi_turn_func_call`.

    Returns
    -------
    bool
        True  – at least one function call failed  
        False – all function calls succeeded
    """
    error_prefix = "Error during execution:"
    return any(
        isinstance(res, str) and res.startswith(error_prefix)
        for res in execution_results
    )


def check_execution_results(execution_results: List[Any]) -> Tuple[bool, List[Any]]:
    """
    检测 execution_results 中的失败项。

    Parameters
    ----------
    execution_results : List[Any]
        execute_multi_turn_func_call 返回的 execution_results。

    Returns
    -------
    has_error : bool
        只要存在一项失败则为 True，否则为 False。
    failed_items : List[Any]
        所有被判定为失败的条目（原样返回，便于后续排查）。
    """
    error_prefix = "Error during execution:"

    def is_failure(item: Any) -> bool:
        #easy tool call
        if isinstance(item, str) and item.startswith(error_prefix):
            return True

        #hard tool call
        if isinstance(item, str) and item.lstrip().startswith("{"):
            # 3a) 先尝试用 json 解析
            try:
                obj = json.loads(item)
                if isinstance(obj, dict) and "error" in obj:
                    return True
            except json.JSONDecodeError:
                if "'error':" in item or '"error":' in item:
                    return True

        return False

    failed_items = [item for item in execution_results if is_failure(item)]
    has_error = bool(failed_items)
    return has_error, failed_items


if __name__ == "__main__":
    st = """[authenticate_travel(client_id=\"discover3rID9537\", client_secret=\"K3yToSecrecy!\", refresh_token=\"updat3Mofresh\", grant_type=\"write\", user_first_name=\"James\", user_last_name=\"Montgomery\"), book_flight(access_token=\"886764\", card_id=\"card_8911\", travel_date=\"2024-01-01\", travel_from=\"RMS\", travel_to=\"OKD\", travel_class=\"first\", travel_cost=2700.0)]""" 
    print(default_decode_execute_prompting(st))
    raw_calls = '[{"name": "authenticate_travel", "arguments": {"client_id": "discover3rID9537", "client_secret": "K3yToSecrecy!"}}, {"name": "run_up_down"}]'
    formatted_tool_calls = parse_tool_calls(raw_calls)
    
    print(formatted_tool_calls)