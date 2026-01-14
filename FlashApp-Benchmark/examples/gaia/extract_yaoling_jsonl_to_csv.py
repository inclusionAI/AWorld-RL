import csv
import json
import os
import re
import sys
from typing import Optional


HTML_URL_TEMPLATE = "https://render.lingguangcontent.com/p/lingguang/{trace_id}/index.html"


def extract_api_requirement_block(design_doc: str) -> str:
    """
    抽取 design_doc 中 “API 技能需求” 小节的主要内容，作为「API技能需求」列。

    策略：
    - 找到包含“API 技能需求”的行
    - 向下收集几行，直到遇到下一个二级标题（以 '##' 开头）或达到行数上限
    - 多行合并为一行，用 ' | ' 连接，便于 CSV 查看
    """
    if not design_doc:
        return ""

    lines = design_doc.splitlines()
    result_lines: list[str] = []
    collecting = False

    for line in lines:
        if not collecting and "API 技能需求" in line:
            collecting = True
            continue

        if not collecting:
            continue

        stripped = line.strip()
        if stripped.startswith("##"):
            # 到下一个小节了
            break
        if not stripped:
            # 空行跳过
            continue

        # 过滤掉 markdown 代码块标记
        if stripped.startswith("```"):
            continue

        # 去掉 markdown 项目符号和加粗
        stripped = re.sub(r"^[\-\*\+]\s*", "", stripped)
        stripped = re.sub(r"\*\*(.*?)\*\*", r"\1", stripped)
        stripped = stripped.strip()
        if stripped:
            result_lines.append(stripped)

        # 防御：最多收集若干行，避免吃掉整个文档
        if len(result_lines) >= 10:
            break

    # 处理结果：去掉"所需 API 技能:"前缀
    processed_lines = []
    for line in result_lines:
        # 去掉"所需 API 技能:"或"所需API技能:"前缀（可能有空格变化）
        line = re.sub(r"^所需\s*API\s*技能\s*[:：]\s*", "", line, flags=re.IGNORECASE)
        line = line.strip()
        if line:
            processed_lines.append(line)

    result = " | ".join(processed_lines)
    
    # 如果结果是"无"或类似的表示"没有"的词，返回空字符串
    if result.strip() in {"无", "无。", "无.", "none", "None", "N/A", "n/a", ""}:
        return ""
    
    return result


def extract_api_list_from_design_doc(design_doc: str) -> str:
    """
    从 design_doc 文本中抽取“所需 API 技能”一行的内容，作为「所需API技能」列（api_list）。

    主要适配类似：
    - **所需 API 技能**: 无
    - **所需 API 技能**: AUDIOCONTEXT2, PLAYTTS
    也可能没有加粗或有其它标点形式。
    """
    if not design_doc:
        return ""

    # 只匹配“所需 API 技能: xxx”这一行
    pattern = re.compile(r"所需\s*API\s*技能[^:：]*[:：]\s*([^\r\n]*)")
    match = pattern.search(design_doc)
    if not match:
        return ""

    api_text = match.group(1).strip()

    # 常见“无”的几种写法视为没有 API
    if api_text in {"无", "无。", "无.", "none", "None", "N/A", "n/a"}:
        return ""

    return api_text


def build_html_url(trace_id: str) -> str:
    """
    根据 trace_id 拼接 html_url：
    将模板 https://render.lingguangcontent.com/p/lingguang/xxx/index.html 中的 xxx 替换为 trace_id。
    """
    if not trace_id:
        return ""
    return HTML_URL_TEMPLATE.format(trace_id=trace_id)


def extract_record(obj: dict) -> Optional[dict]:
    """
    从单行 JSON 对象中抽取所需字段：
    - session_id
    - query
    - api_list
    - html_url

    若缺少 query 或 trace_id，则返回 None。
    """
    query = obj.get("query")
    trace_id = obj.get("trace_id")
    if not query or not trace_id:
        return None

    # 文件中没有显式 session_id 字段时，用 trace_id 作为 session_id
    session_id = obj.get("session_id") or trace_id

    prd = obj.get("prd") or {}
    design_doc = prd.get("design_doc") or ""

    # API 技能需求小节整体内容
    api_requirement = extract_api_requirement_block(design_doc)
    # 所需 API 技能 一行的内容
    api_list = extract_api_list_from_design_doc(design_doc)

    # required_skills 字段通常是一个 list，转成逗号分隔字符串
    raw_required_skills = prd.get("required_skills") or []
    if isinstance(raw_required_skills, list):
        required_skills = ",".join(str(x) for x in raw_required_skills)
    else:
        required_skills = str(raw_required_skills)

    html_url = build_html_url(trace_id)

    return {
        "session_id": session_id,
        "query": query,
        "api_requirement": api_requirement,
        "api_list": api_list,
        "required_skills": required_skills,
        "html_url": html_url,
    }


def process_jsonl(jsonl_path: str) -> str:
    """
    读取指定 jsonl 文件，抽取信息并写入同名 csv。

    返回生成的 csv 路径。
    """
    if not os.path.isfile(jsonl_path):
        raise FileNotFoundError(f"JSONL file not found: {jsonl_path}")

    base, _ = os.path.splitext(jsonl_path)
    csv_path = base + ".csv"

    records = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                # 某些行若解析失败，则跳过并继续
                continue

            rec = extract_record(obj)
            if rec is not None:
                records.append(rec)

    # 写 CSV
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "session_id",
                "query",
                "api_requirement",  # API技能需求
                "api_list",         # 所需API技能
                "required_skills",  # required_skills 字段
                "html_url",
            ],
        )
        writer.writeheader()
        for rec in records:
            writer.writerow(rec)

    return csv_path


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("Usage: python extract_yaoling_jsonl_to_csv.py /path/to/yaoling_test20.jsonl")
        return 1

    jsonl_path = argv[1]
    try:
        csv_path = process_jsonl(jsonl_path)
    except Exception as e:
        print(f"Error: {e}")
        return 1

    print(f"Wrote CSV to: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))


