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

    file_handler = logging.FileHandler(os.getenv("LOG_FILE_PATH", "run_super_agent.log"), mode="a", encoding="utf-8")
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
        },
        "dynamic": {
            "score": 0.4,
            "reason": "部分核心操作按钮（如'提交'、'下一步'或'确认'）在实际交互时无法被正常点击，导致流程中断，也有部分交互式组件没有产生预期的响应。例如，点击'提交'按钮无反应，或表单中的输入后无法正常提交数据。有的弹窗或下拉选择卡死，影响了后续的互动步骤，但基础的点击操作依然部分可用。"
        },
        "fix_reason": "static分值由原先的0.8下调至0.6。用户反馈中指出页面部分功能按钮缺失，且推荐位图标没有显示。经代码与DOM结构检查，发现'个人中心'区不可见，辅助按钮等模块渲染不完整，布局虽基本合理但存在部分次要元素缺失，因此将static评分做出相应调整。"
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
    df_rows = []  # 保存对应的DataFrame行数据
    
    # 尝试读取CSV文件，如果不存在则读取Excel文件
    csv_path = '/Users/yuchengyue/AWorld_local/runs/flashapp/线上query-用户反馈-200条-mockapi_v2_带打分.csv'
    excel_path = '/Users/yuchengyue/AWorld_local/gaia_dataset/点踩query抽样-dt1207-rand1000.xlsx'
    
    df = None
    if os.path.exists(csv_path):
        logging.info("Reading CSV file")
        df = pd.read_csv(csv_path)
    elif os.path.exists(excel_path):
        logging.info("Reading Excel file")
        df = pd.read_excel(
            excel_path,
            sheet_name='工作表 1 - result_24',
            header=1
        )
    else:
        logging.error(f"Neither CSV nor Excel file found. CSV: {csv_path}, Excel: {excel_path}")
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
            # 这样可以避免误截断query内容中包含'rewritten_query:'字符串的情况
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
            html_url = row.get('mock_html_url', '') # 使用mock api的url
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
                df_rows.append(row)  # 保存对应的行数据
        
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
            df_rows.append(row)  # 保存对应的行数据
    
    logging.info(f"Loaded {len(user_queries)} queries total: {react_ts_count} react_ts mode, {html_js_count} html_js mode")
    
    code_given_flag = True
    user_flag = True
    code_snippet = ""

    for user_query, html_url, flashapp_folder_info, row in zip(user_queries, html_urls, flashapp_folders, df_rows):

        if not code_given_flag and not user_flag:
            task_description = """对于每个任务，你会得到一个**用户需求(User Query)**和一个**网页地址(Target URL)**。"""
            code_snippet = ""
        elif code_given_flag and not user_flag:
            task_description = """对于每个任务，你会得到一个**用户需求(User Query)**、一个**网页地址(Target URL)**和闪应用制作时的源代码内容（Code Snippet）。
你可以结合提供的代码去判断应用实现的合理性、美观度；或参考代码里的变量名称，通过js代码来操作页面、获取页面状态。"""
        elif code_given_flag and user_flag:
            task_description = """对于每个任务，你会得到一个**用户需求(User Query)**、一个**网页地址(Target URL)**、一段**用户反馈**和闪应用制作时的源代码内容（Code Snippet）。
首先，用户反馈的内容包含他的点赞/点踩行为信号，以及所有轮次的query，query之间以`->`分隔。你需要根据用户所有轮次的query，分析用户对这轮闪应用生成的情感变化；并结合用户点赞/点踩的行为信号，一起判断用户对这轮闪应用的满意度。
同时，你可以结合提供的代码去判断应用实现的合理性、美观度；或参考代码里的变量名称，通过js代码来操作页面、获取页面状态。"""

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

        # 注意：pandas的NaN值需要用pd.isna()检查，.get()方法不会将NaN转换为默认值
        def safe_get(row, key, default=""):
            """安全获取值，处理NaN情况"""
            value = row.get(key, default)
            return default if pd.isna(value) else value

        # 拼接用户反馈
        user_feedback = ""
        if user_flag:
            is_like = safe_get(row, 'is_like', '无')
            is_dislike = safe_get(row, 'is_dislike', '无')
            prev_origin_query = safe_get(row, 'prev_origin_query', '无')
            next_origin_query = safe_get(row, 'next_origin_query', '无')

            user_feedback = f"""1. 点赞信号：{is_like}
2. 点踩信号：{is_dislike}
2. 前轮query：{prev_origin_query}
3. 后轮query：{next_origin_query}
"""

        # 从csv中取出专家评分和原因
        
        expert_result = {
            "intension_score": safe_get(row, "intension_score", ""),
            "intension_reason": safe_get(row, "intension_reason", ""),
            "static_score": safe_get(row, "static_score", ""),
            "static_reason": safe_get(row, "static_reason", ""),
            "dynamic_score": safe_get(row, "dynamic_score", ""),
            "dynamic_reason": safe_get(row, "dynamic_reason", "")
        }


        search_sys_prompt = f"""
# 角色说明
你是一名专业的QA自动化工程师和网页可用性评测专家，你正在对闪应用的生成质量进行评估。
目前，已有另外一位专家已经完成了对闪应用的质量评估工作，从3个维度进行了打分，包括：页面和用户需求一致性(intension)、页面美观度与元素覆盖度(static)、交互功能可用性(dynamic)。
现在需要你根据用户反馈，判断原始模型的static打分及原因是否需要纠正，修正的维度只需要static，intension和dynamic不需要纠正。用户反馈的类型包括：点赞/点踩信号、上下轮用户query。用户反馈中，前后轮query的重要性更高，点赞/点踩信号可能有噪声。
你在修正static的时候，只要关注用户反馈中提到的优化点，比如功能完整性、功能设计合理性，如果用户没有提到，不要太严格的降分。
你能获取到的所有输入信息是：用户需求、闪应用目标网址、相关代码、用户反馈和另一位专家的打分结果。

# 评估流程和方法说明
1. 由于不需要对dynamic进行纠正，尽量只通过代码进行纠正。
2. 如果一定需要打开闪应用，直接使用`mcp__ms-playwright__browser_navigate`工具打开`**目标网址:**`中的闪应用，并可以使用`mcp__ms-playwright__browser_click`等playwright api来触发用户行为，或`mcp__ms-playwright__browser_evaluate`进行js代码编写和操作。

# 评估维度介绍
1. **页面美观度与功能完整性：**
要求：仅分析页面snapshot（HTML结构、DOM元素及其内容）和代码等文本信息，不用也不能参考截图或视觉渲染效果，禁止使用任何"截图"相关的工具（例如 mcp__ms-playwright__browser_take_screenshot）
- 页面美观度：美观度评估必须严格，不能因为功能可用就忽略美观度问题，需要严格检查配色方案、排版布局、字体规范、组件样式、视觉层级、统一性等美观指标，以及是否具备需求相关的UI元素（如文本、表单、按钮等）。
- 功能完整性：判断应用中的模块是否满足用户需求，包括功能设计的合理性和完整性(是否缺失必要功能)。例如：用户需求为“设计一个健身打卡应用”，那应用中需要能够支持设置打卡日期、添加健身项目、记录健身数据等功能。如果应用中不支持设置打卡日期，则认为功能缺失。

# 打分细则
- **Static (页面美观度与功能完整性综合评分)：** 静态评分时须严格衡量页面UI美观性和需求相关元素/模块的覆盖完整性，对页面的静态质量进行严苛评估。美观度是评分的重要组成部分，必须通过CSS样式检查（`getComputedStyle`）来客观评估配色、布局、字体、组件样式等。
    - 必须详细核查页面的布局合理性、视觉层级、颜色搭配和文本/组件排版规范性等美观指标，并逐项检查DOM结构中是否全面存在所有用户需求相关的核心元素（如表单、按钮、输入框、列表、标题等），不可遗漏。
    - 仅当页面美观性优秀（配色协调、布局合理、字体规范、组件样式统一现代化）、所有需求必需元素均完整无缺、功能设计合理且完整，才可得高分（0.8-1.0）；如仅部分必需元素缺失（如3个必须元素仅找到2个）、应用设计不合理或美观性一般（存在配色不协调、布局不够合理、字体不够规范、组件样式不够统一等问题），则应严控评分（0.5-0.8）；如有较多元素缺失、结构严重混乱或美观性不达标（配色混乱、布局混乱、字体不规范、组件样式不统一等），应给低分或直接0分。即使功能可用，美观度严重不达标也必须给予低分。
    - 必须明确指出所有功能设计不合理、功能模块缺失或美观度表现不佳的具体问题，评分不允许宽松或笼统评判，需有详实的snapshot或代码细节作为依据。

# 输出格式
1. 保留intension和dynamic的打分不动。
2. 新增fix_reason字段，里面写上得分修改的原因，如果没有修改，就置为空。
3. 只输出被<answer>```json```</answer>包裹的单个JSON对象（按以下示例格式），不要额外输出任何playwright代码、分析日志或提示，且你的原因(reason)应尽量细致分析。
4. 在intension、static和dynamic的reason中，一定不要出现“但鉴于用户给出了明确的'点赞'反馈”等类似的理由，不要显式出现用户反馈。用户反馈相关理由只能在fix_reason中体现。

示例：
<answer>```json
{example_json_str}
```</answer>

# 特别注意
- 不允许也不需要使用"browser_take_screenshot"等任何截图/视觉截图相关的工具或请求。仅依据DOM结构、属性和操作反馈进行分析，不参考截图或视觉效果。
- 由于上下文长度的限制，尽可能通过最少的操作步骤，完成评估。
"""

        question = f"""
# 输入数据

**用户需求:**
{user_query}

**目标网址:**
{html_url}

**用户反馈:**
{user_feedback}

**另一位专家的打分结果:**
{expert_result}

**相关代码:**
{code_snippet}
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
            name="gaia_super_agent",
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
                error_file = "/Users/yuchengyue/AWorld_local/runs/闪应用采样_error.txt"
                os.makedirs(os.path.dirname(error_file), exist_ok=True)
                with open(error_file, "a", encoding="utf-8") as ef:
                    ef.write(f"User Query: {user_query}\nError: {str(e)}\n\n")
            except Exception as log_e:
                logging.error(f"Failed to write error info to 闪应用采样_error.txt: {log_e}")
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

