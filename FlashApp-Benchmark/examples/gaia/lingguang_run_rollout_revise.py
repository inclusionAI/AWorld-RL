from aworld.config.conf import AgentConfig, TaskConfig
from aworld.core.agent.base import Agent
from aworld.core.task import Task
from aworld.runner import Runners
from datasets import load_dataset
from pathlib import Path
from typing import Any, Dict
from dotenv import load_dotenv
import re
import logging
import traceback
import os
import json
import pandas as pd
import glob
import subprocess
import socket
import time


http_server_process = None
http_server_base_url = None


def find_free_port():
    """找到一个空闲的本地端口"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def start_http_server(root_dir: str) -> str:
    """
    在指定目录下启动一个简单的HTTP静态文件服务，返回base_url，例如:
    http://127.0.0.1:8000
    """
    global http_server_process, http_server_base_url

    # 如果已经启动过，就直接复用
    if http_server_base_url is not None and http_server_process is not None:
        return http_server_base_url

    port = find_free_port()
    # 使用 python -m http.server 提供静态服务
    http_server_process = subprocess.Popen(
        ["python", "-m", "http.server", str(port)],
        cwd=root_dir,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # 等待服务就绪（简单粗暴地sleep一小会）
    time.sleep(1)

    http_server_base_url = f"http://127.0.0.1:{port}"
    logging.info(f"Started fallback HTTP server at {http_server_base_url}, root_dir={root_dir}")
    return http_server_base_url


def setup_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(os.getenv("LOG_FILE_PATH", "run_revise_agent.log"), mode="a", encoding="utf-8")
    file_handler.setLevel(logging.INFO) 

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)


if __name__ == "__main__":
    load_dotenv()

    setup_logging()

    example_output = {
        "intention": {
            "score": 0.2,
            "reason": "网页标题和主要的Header内容与用户需求高度相关，但是用户要的抗衰老应用更需要是知识型指南的形式，而不是仅通过按钮和页面的变化给用户虚拟展示变老流程。"
        },
        "static": {
            "score": 0.6,
            "reason": "页面中绝大多数需要的UI元素都能被正确渲染并可见，例如导航栏、主要内容区、部分功能按钮都出现在DOM结构中且布局合理。但仍有部分次要模块（如用户头像组件或一些辅助按钮）缺失，或没有渲染完整，比如'个人中心'区不可见，推荐位图标没有显示。整体排版和结构基本符合预期，但仍有部分次要元素未加载或因异常未被渲染。"
        }
    }

    example_json_str = json.dumps(example_output, indent=2, ensure_ascii=False)

    # flashapp_release_base = "/Users/yuchengyue/AWorld_local/gaia_dataset/flashapp-dt1214"
    flashapp_release_base = "/Users/yuchengyue/AWorld_local/gaia_dataset/flashapp-user200"
    
    def read_code_file(file_path):
        """读取代码文件"""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            return f"[读取失败：{file_path}, 错误：{e}]"
    
    def detect_folder_type(folder_path):
        """
        检测文件夹类型：react_ts 或 html_js
        如果有 backup_1 且包含 .ts 文件，返回 'react_ts'
        否则返回 'html_js'
        """
        # 查找backup_x文件夹
        backup_folders = []
        for item in os.listdir(folder_path):
            item_path = os.path.join(folder_path, item)
            if os.path.isdir(item_path) and item.startswith('backup_'):
                backup_folders.append(item_path)
        
        if backup_folders:
            # 检查第一个backup文件夹中是否有.ts文件
            backup_path = sorted(backup_folders)[0]
            for root, dirs, files in os.walk(backup_path):
                for file in files:
                    if file.endswith('.ts') or file.endswith('.tsx'):
                        return 'react_ts', backup_path
        
        # 默认使用 html_js 模式
        return 'html_js', None
    
    # 提取数据
    user_queries = []
    html_urls = []
    flashapp_folders = []
    original_scores = []  # 存储原始打分结果
    user_feedbacks = []  # 存储用户反馈信息
    
    # 读取带打分的CSV文件
    csv_path = '/Users/yuchengyue/AWorld_local/runs/flashapp/线上query-用户反馈-200条-带打分.csv'
    
    df = None
    if os.path.exists(csv_path):
        logging.info("Reading CSV file with scores")
        df = pd.read_csv(csv_path)
        # df = df.iloc[:5]  # 测试用，只处理前5条
    else:
        logging.error(f"CSV file not found: {csv_path}")
        exit(1)
    
    # 过滤掉关键字段为空的行
    required_columns = ['rewritten_query', 'artifact_id', 'artifact_version_number']
    df_filtered = df.dropna(subset=required_columns)
    
    skipped_count = len(df) - len(df_filtered)
    if skipped_count > 0:
        logging.info(f"Filtered out {skipped_count} rows with missing data, {len(df_filtered)} rows remaining")
    
    # 统计不同模式的数量
    react_ts_count = 0
    html_js_count = 0
    
    for idx, row in df_filtered.iterrows():
        rewritten_query = row.get('rewritten_query', '')
        # 某些数据的格式可能为：
        # 1) "rewritten_query: xxx"
        # 2) "target_app_id: ...\nrewritten_query: xxx"
        # 3) 直接就是query，如 "做一个计算器"
        # 这里统一只保留真正的用户自然语言需求部分
        if isinstance(rewritten_query, str) and rewritten_query.strip():
            # 使用正则表达式匹配，确保 'rewritten_query:' 出现在行首或换行符后
            pattern = r'(?:^|\n)rewritten_query:\s*(.+)'
            match = re.search(pattern, rewritten_query, re.DOTALL)
            if match:
                rewritten_query = match.group(1).strip()
            # 如果没有匹配到，说明直接就是query，原样保留
        artifact_id = row.get('artifact_id', '')
        artifact_version_number = row.get('artifact_version_number', '')
        
        # 构建文件夹名称: {artifact_id}-{artifact_version_number}
        folder_name = f"{artifact_id}-{int(artifact_version_number)}"
        folder_path = os.path.join(flashapp_release_base, folder_name)
        
        # 检查文件夹是否存在
        if not os.path.exists(folder_path):
            logging.warning(f"Folder not found: {folder_path}, skipping...")
            continue
        # 自动检测文件夹类型
        folder_type, backup_path = detect_folder_type(folder_path)
        
        if folder_type == 'react_ts':
            # React/TypeScript 模式
            react_ts_count += 1
            
            # 查找index.html文件（在backup根目录）
            index_html_path = os.path.join(backup_path, 'index.html')
            if not os.path.exists(index_html_path):
                logging.warning(f"index.html not found in {backup_path}, skipping...")
                continue
            
            # React/Vite 项目的 index.html 通常需要通过 dev server 才能正常加载，
            # 直接用 file:// 打开往往是空白页。
            # 这里优先使用数据集中提供的线上地址 html_url，作为真正的 Target URL。
            html_url = row.get('html_url', '')
            if not isinstance(html_url, str) or not html_url:
                # 如果没有提供线上 html_url，则兜底：启动一个本地静态服务并通过 HTTP 访问
                base_url = start_http_server(flashapp_release_base)
                # 计算 index.html 相对 flashapp_release_base 的路径，用于拼接 URL
                rel_path = os.path.relpath(index_html_path, flashapp_release_base)
                # 将路径中的反斜杠替换为 URL 友好的斜杠
                rel_path = rel_path.replace(os.sep, "/")
                html_url = f"{base_url}/{rel_path}"
            
            # 只查找指定的5个文件
            required_files = {
                'index.html': os.path.join(backup_path, 'index.html'),
                'src/App.tsx': os.path.join(backup_path, 'src', 'App.tsx'),
                'src/index.css': os.path.join(backup_path, 'src', 'index.css'),
                'src/App.css': os.path.join(backup_path, 'src', 'App.css'),
                'src/base.js': os.path.join(backup_path, 'src', 'base.js'),  # 可选
            }
            
            # 收集存在的文件（base.js是可选的）
            code_files = []
            for file_key, file_path in required_files.items():
                if file_key == 'src/base.js':
                    # base.js是可选的，如果不存在就跳过
                    if os.path.exists(file_path):
                        code_files.append((file_key, file_path))
                else:
                    # 其他文件是必需的
                    if os.path.exists(file_path):
                        code_files.append((file_key, file_path))
                    else:
                        logging.warning(f"Required file not found: {file_path}, skipping this entry...")
                        break
            else:
                # 如果所有必需文件都存在，添加到列表
                user_queries.append(rewritten_query)
                html_urls.append(html_url)
                flashapp_folders.append({
                    'folder_path': folder_path,
                    'backup_path': backup_path,
                    'html_file_path': index_html_path,
                    'code_files': code_files,  # 存储为(file_key, file_path)元组列表
                    'mode': 'react_ts'
                })
                # 保存原始打分结果
                original_scores.append({
                    'intension_score': row.get('intension_score', ''),
                    'intension_reason': row.get('intension_reason', ''),
                    'static_score': row.get('static_score', ''),
                    'static_reason': row.get('static_reason', ''),
                })
                # 保存用户反馈信息
                user_feedbacks.append({
                    'is_like': row.get('is_like', ''),
                    'is_dislike': row.get('is_dislike', ''),
                    'prev_origin_query': row.get('prev_origin_query', ''),
                    'next_origin_query': row.get('next_origin_query', ''),
                })
        
        else:
            # HTML/JS 模式
            html_js_count += 1
            
            # 优先使用根目录下的 .release_dist
            release_dist_path = os.path.join(folder_path, ".release_dist")
            
            # 如果根目录下没有 .release_dist，则尝试在 backup_x 目录中查找
            if not os.path.exists(release_dist_path):
                candidate_release_dist = None
                for item in os.listdir(folder_path):
                    item_path = os.path.join(folder_path, item)
                    if os.path.isdir(item_path) and item.startswith("backup_"):
                        rd_path = os.path.join(item_path, ".release_dist")
                        if os.path.exists(rd_path):
                            candidate_release_dist = rd_path
                            break
                if candidate_release_dist:
                    release_dist_path = candidate_release_dist
                    logging.info(f"Using backup .release_dist at {release_dist_path} for folder {folder_path}")
                else:
                    logging.warning(f"No .release_dist found under {folder_path}, skipping...")
                    continue
            
            # 查找index.html文件（可能在.release_dist根目录或子目录）
            html_files = []
            for root, dirs, files in os.walk(release_dist_path):
                for file in files:
                    if file.endswith('.html'):
                        html_files.append(os.path.join(root, file))
            
            if not html_files:
                logging.warning(f"No HTML file found in {release_dist_path}, skipping...")
                continue
            
            # 使用第一个找到的HTML文件（通常是index.html）
            html_file_path = html_files[0]
            
            # 转换为file:// URL
            html_url = f"file://{html_file_path}"
            
            # 查找所有.js文件
            # 优先查找.release_dist目录下直接存在的.js文件
            js_files = []
            js_dir_path = os.path.join(release_dist_path, "js")
            
            # 首先查找.release_dist根目录下的.js文件
            for file in os.listdir(release_dist_path):
                if file.endswith('.js'):
                    js_files.append(os.path.join(release_dist_path, file))
            
            # 如果没有找到，则查找.release_dist/js文件夹下的.js文件
            if not js_files and os.path.exists(js_dir_path):
                for file in os.listdir(js_dir_path):
                    if file.endswith('.js'):
                        js_files.append(os.path.join(js_dir_path, file))
            
            # 查找CSS文件（如果.release_dist下有css文件夹）
            css_files = []
            css_dir_path = os.path.join(release_dist_path, "css")
            if os.path.exists(css_dir_path):
                for file in os.listdir(css_dir_path):
                    if file.endswith('.css'):
                        css_files.append(os.path.join(css_dir_path, file))
            
            user_queries.append(rewritten_query)
            html_urls.append(html_url)
            flashapp_folders.append({
                'folder_path': folder_path,
                'release_dist_path': release_dist_path,
                'html_file_path': html_file_path,
                'js_files': sorted(js_files),  # 排序以保证一致性
                'css_files': sorted(css_files),  # 排序以保证一致性
                'mode': 'html_js'
            })
            # 保存原始打分结果
            original_scores.append({
                'intension_score': row.get('intension_score', ''),
                'intension_reason': row.get('intension_reason', ''),
                'static_score': row.get('static_score', ''),
                'static_reason': row.get('static_reason', ''),
            })
            # 保存用户反馈信息
            user_feedbacks.append({
                'is_like': row.get('is_like', ''),
                'is_dislike': row.get('is_dislike', ''),
                'prev_origin_query': row.get('prev_origin_query', ''),
                'next_origin_query': row.get('next_origin_query', ''),
            })
    
    logging.info(f"Loaded {len(user_queries)} queries total: {react_ts_count} react_ts mode, {html_js_count} html_js mode")
    
    code_given_flag = True
    code_snippet = ""

    for user_query, html_url, flashapp_folder_info, original_score, user_feedback in zip(
        user_queries, html_urls, flashapp_folders, original_scores, user_feedbacks
    ):

        if code_given_flag:
            task_description = """对于每个任务，你会得到一个**用户需求(User Query)**、一个**网页地址(Target URL)**、**原始模型的打分结果**、**用户反馈信息**和闪应用制作时的源代码内容（Code Snippet）。
你需要根据用户反馈（点赞/点踩信号和前轮、后轮的query）来重新评估并修改原始模型对Intention和Static两个维度的打分。"""

            # 从flashapp_folder_info中获取模式
            mode = flashapp_folder_info.get('mode', 'html_js')
            
            if mode == "html_js":
                release_dist_path = flashapp_folder_info['release_dist_path']
                html_file_path = flashapp_folder_info['html_file_path']
                js_files = flashapp_folder_info['js_files']
                css_files = flashapp_folder_info.get('css_files', [])
                
                html_file_name = os.path.basename(html_file_path)
                
                # 按照顺序读取文件内容：CSS在前，然后是HTML，最后是JS
                file_contents = []
                
                # 添加CSS文件内容（如果存在）
                if css_files:
                    for css_file in css_files:
                        css_content = read_code_file(css_file)
                        css_file_name = os.path.basename(css_file)
                        file_contents.append(f"======= {css_file_name} =======\n{css_content}")
                
                # 添加HTML文件内容
                index_html_content = read_code_file(html_file_path)
                file_contents.append(f"======= {html_file_name} =======\n{index_html_content}")
                
                # 添加JS文件内容
                for js_file in js_files:
                    js_content = read_code_file(js_file)
                    js_file_name = os.path.basename(js_file)
                    file_contents.append(f"======= {js_file_name} =======\n{js_content}")
                
                all_code_content = "\n\n".join(file_contents)
                
                # 拼接代码片段
                code_snippet = f"""提供给你的代码文件如下：

{all_code_content}
"""
            
            elif mode == "react_ts":
                html_file_path = flashapp_folder_info['html_file_path']
                code_files = flashapp_folder_info['code_files']  # 这是(file_key, file_path)元组列表
                
                # 定义文件显示顺序
                file_order = ['index.html', 'src/App.tsx', 'src/index.css', 'src/App.css', 'src/base.js']
                
                # 按照指定顺序读取文件内容
                file_contents = []
                for file_key in file_order:
                    # 在code_files中查找对应的文件
                    for key, path in code_files:
                        if key == file_key:
                            file_content = read_code_file(path)
                            file_contents.append(f"======= {file_key} =======\n{file_content}")
                            break
                
                all_code_content = "\n\n".join(file_contents)
                
                # 拼接代码片段
                code_snippet = f"""# 提供的代码文件

{all_code_content}
"""
            else:
                code_snippet = ""

        # 构建原始打分结果展示
        original_score_text = f"""
**原始模型的打分结果:**

1. **Intention (意图达成度):**
   - 打分: {original_score.get('intension_score', 'N/A')}
   - 原因: {original_score.get('intension_reason', 'N/A')}

2. **Static (页面美观度与功能完整性):**
   - 打分: {original_score.get('static_score', 'N/A')}
   - 原因: {original_score.get('static_reason', 'N/A')}
"""

        # 构建用户反馈展示
        user_feedback_text = f"""
**用户反馈信息:**

1. 点赞信号: {user_feedback.get('is_like', 'N/A')}
2. 点踩信号: {user_feedback.get('is_dislike', 'N/A')}
3. 前轮query: {user_feedback.get('prev_origin_query', 'N/A')}
4. 后轮query: {user_feedback.get('next_origin_query', 'N/A')}
"""

        search_sys_prompt = f"""
# 角色说明
你是一名专业的QA自动化工程师和网页可用性评测专家。你需要根据用户反馈信息，对原始模型的打分结果进行修正，特别是针对Intention（意图达成度）和Static（页面美观度与功能完整性）两个维度。

# 任务说明
{task_description}

# 评估流程和方法说明
直接使用`mcp__ms-playwright__browser_navigate`工具打开`**目标网址:**`中的闪应用，然后根据以下信息进行重新评估（不要使用`mcp__ms-playwright__browser_install`工具）：

1. **分析用户反馈信号：**
   - 仔细分析用户的点赞/点踩信号，判断用户对这轮闪应用的满意度
   - 分析前轮query和后轮query，理解用户的意图变化和需求演进
   - 如果用户点踩，说明应用存在问题，需要降低相应维度的打分
   - 如果用户点赞，说明应用基本满足需求，但需要结合前后轮query判断是否还有改进空间
   - 如果后轮query是对前轮query的细化或修正，说明前轮生成的应用可能在某些方面未完全满足用户需求

2. **重新评估Intention（意图达成度）：**
   - 结合用户反馈和前后轮query，判断原始模型的打分是否合理
   - 如果用户点踩且后轮query显示用户需要修改功能，说明Intention打分可能过高
   - 如果用户点赞且后轮query是新增功能而非修正，说明Intention打分可能合理或偏低
   - 通过打开网页验证页面标题、主要Header等是否真正体现用户需求的核心意图

3. **重新评估Static（页面美观度与功能完整性）：**
   - 结合用户反馈和代码分析，判断原始模型的打分是否合理
   - 如果用户点踩，需要仔细检查页面美观度和功能完整性是否存在问题
   - 如果后轮query要求修改UI或添加功能，说明Static维度可能存在问题
   - 通过检查DOM结构、CSS样式等，验证页面美观度和功能完整性
   - 仅分析页面snapshot（HTML结构、DOM元素及其内容）和代码等文本信息，不用也不能参考截图或视觉渲染效果，禁止使用任何"截图"相关的工具（例如 mcp__ms-playwright__browser_take_screenshot）

# 评分维度与标准

请重新评估并输出以下两个维度的打分，打分严格，并输出原因：

- **Intention (意图达成度，按区间评分)：** 网页标题、主要Header等是否体现出用户需求的核心意图。
    - 0.8~1.0：标题、核心区域高度相关，功能流程在DOM中能明确找到，页面内容紧密匹配用户需求。
    - 0.5~0.7：标题、核心区域部分相关，功能流程在DOM中基本能找到，但存在一定不匹配或覆盖不完整的情况。
    - 0.0~0.4：页面与需求无关，或出现重大异常（如404、500），主要内容缺失或核心区域完全不相关。

- **Static (页面美观度与功能完整性综合评分)：** 静态评分时须严格衡量页面UI美观性和需求相关元素/模块的覆盖完整性，对页面的静态质量进行严苛评估。美观度是评分的重要组成部分，必须通过CSS样式检查（`getComputedStyle`）来客观评估配色、布局、字体、组件样式等。
    - 必须详细核查页面的布局合理性、视觉层级、颜色搭配和文本/组件排版规范性等美观指标，并逐项检查DOM结构中是否全面存在所有用户需求相关的核心元素（如表单、按钮、输入框、列表、标题等），不可遗漏。
    - 仅当页面美观性优秀（配色协调、布局合理、字体规范、组件样式统一现代化）、所有需求必需元素均完整无缺、功能设计合理且完整，才可得高分（0.8-1.0）；如仅部分必需元素缺失（如3个必须元素仅找到2个）、应用设计不合理或美观性一般（存在配色不协调、布局不够合理、字体不够规范、组件样式不够统一等问题），则应严控评分（0.5-0.8）；如有较多元素缺失、结构严重混乱或美观性不达标（配色混乱、布局混乱、字体不规范、组件样式不统一等），应给低分或直接0分。即使功能可用，美观度严重不达标也必须给予低分。
    - 必须明确指出所有功能设计不合理、功能模块缺失或美观度表现不佳的具体问题，评分不允许宽松或笼统评判，需有详实的snapshot或代码细节作为依据。

# 特别注意
- 不允许也不需要使用"browser_take_screenshot"等任何截图/视觉截图相关的工具或请求。仅依据DOM结构、属性和操作反馈进行分析，不参考截图或视觉效果。
- 必须结合用户反馈（点赞/点踩、前后轮query）来调整打分，不能完全依赖原始模型的打分结果
- 如果用户点踩，说明应用存在问题，需要相应降低打分
- 如果后轮query是对前轮query的修正或细化，说明前轮应用可能未完全满足需求，需要相应调整打分
- 由于上下文长度的限制，尽可能通过最少的操作步骤，完成评估。

# 输出格式
只输出被<answer>```json```</answer>包裹的单个JSON对象（按以下示例格式），不要额外输出任何playwright代码、分析日志或提示。你的原因(reason)应尽量细致分析，并说明你是如何根据用户反馈调整原始打分的。

示例：
<answer>```json
{example_json_str}
```</answer>

{original_score_text}

{user_feedback_text}

{code_snippet}
"""

        question = f"""
# 输入数据

**用户需求:**
{user_query}

**目标网址:**
{html_url}
"""

        agent_config = AgentConfig(
            llm_provider="openai",
            llm_model_name=os.getenv("LLM_MODEL_NAME", "gpt-4o"),
            llm_api_key=os.getenv("LLM_API_KEY", "your_openai_api_key"), 
            llm_base_url=os.getenv("LLM_BASE_URL", "your_openai_base_url"),
        )


        logging.info(f"Question Final: {question}")
        
        super = Agent(
            conf=agent_config,
            name="gaia_revise_agent",
            system_prompt=search_sys_prompt,
            mcp_servers=[
                "ms-playwright",
            ],
            history_messages=100,
        )

        try:
            result = Runners.sync_run_task(task=Task(input=question, agent=super, conf=TaskConfig(max_steps=100)))
            # import pdb;pdb.set_trace()
        except Exception as e:
            logging.error(f"LLM request failed for query: {user_query[:100]}... Error: {str(e)}")
            logging.info(f"Skipping to next query due to error")
            try:
                error_file = "/Users/yuchengyue/AWorld_local/runs/闪应用采样_revise_error.txt"
                os.makedirs(os.path.dirname(error_file), exist_ok=True)
                with open(error_file, "a", encoding="utf-8") as ef:
                    ef.write(f"User Query: {user_query}\nError: {str(e)}\n\n")
            except Exception as log_e:
                logging.error(f"Failed to write error info to 闪应用采样_revise_error.txt: {log_e}")
            continue
        
        match = re.search(r'<answer>(.*?)</answer>', result["task_0"]["answer"])
        if match:
            answer = match.group(1)
            logging.info(f"Agent answer: {answer}")
            
            logging.info(f"Question processed successfully!")

    # 脚本结束前，确保兜底HTTP服务被关闭
    if http_server_process is not None:
        logging.info("Shutting down fallback HTTP server")
        http_server_process.terminate()
