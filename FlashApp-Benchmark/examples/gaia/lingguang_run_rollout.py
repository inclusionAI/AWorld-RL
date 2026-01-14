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
    
    # 尝试读取CSV文件，如果不存在则读取Excel文件
    # csv_path = '/Users/yuchengyue/AWorld_local/gaia_dataset/点踩query抽样-dt1214-rand1000-reactts.csv'
    csv_path = '/Users/yuchengyue/AWorld_local/runs/flashapp/线上query-用户反馈-200条-带打分.csv'
    excel_path = '/Users/yuchengyue/AWorld_local/gaia_dataset/点踩query抽样-dt1207-rand1000.xlsx'
    
    df = None
    if os.path.exists(csv_path):
        logging.info("Reading CSV file")
        df = pd.read_csv(csv_path)
        # df = df.iloc[[765]]
        df = df.iloc[:130]
    elif os.path.exists(excel_path):
        logging.info("Reading Excel file")
        df = pd.read_excel(
            excel_path,
            sheet_name='工作表 1 - result_24',
            header=1
        )
        df = df.iloc[679:]  # 第680个数据开始(索引679)
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
            html_url = row.get('mock_html_url', '')
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
            
            # 优先使用数据集中提供的线上地址 mock_html_url，如果没有则使用 file:// URL
            html_url = row.get('mock_html_url', '')
            # 处理 pandas 读取 CSV 时可能出现的 NaN 值
            if pd.isna(html_url):
                html_url = ''
            elif not isinstance(html_url, str):
                html_url = str(html_url) if not pd.isna(html_url) else ''
            if not html_url:
                # 如果没有提供线上 html_url，则使用 file:// URL
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
    
    logging.info(f"Loaded {len(user_queries)} queries total: {react_ts_count} react_ts mode, {html_js_count} html_js mode")
    
    code_given_flag = True
    code_snippet = ""

    for user_query, html_url, flashapp_folder_info in zip(user_queries, html_urls, flashapp_folders):

        if not code_given_flag:
            task_description = """对于每个任务，你会得到一个**用户需求(User Query)**和一个**网页地址(Target URL)**。"""
            code_snippet = ""
        else:
            task_description = """对于每个任务，你会得到一个**用户需求(User Query)**、一个**网页地址(Target URL)**和闪应用制作时的源代码内容（Code Snippet）。
你可以结合提供的代码去判断应用实现的合理性、美观度；或参考代码里的变量名称，通过js代码来操作页面、获取页面状态。"""

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

        search_sys_prompt = f"""
# 角色说明
你是一名专业的QA自动化工程师和网页可用性评测专家。你需要判断给定闪应用的质量高低，包括页面和用户需求一致性、页面美观度与元素覆盖度、交互功能可用性。

# 任务说明
{task_description}

# 评估流程和方法说明
直接使用`mcp__ms-playwright__browser_navigate`工具打开`**目标网址:**`中的闪应用，然后分三步进行评估（不要使用`mcp__ms-playwright__browser_install`工具）：

1. **页面和用户需求一致性：**
- 首先，判断目前生成的闪应用是否能够解决用户的核心诉求，且能够真切帮助用户解决问题。
- 其次，判断页面的标题（title）、主要Header内容是否展现出完成用户任务的关键信息和功能。重点对页面核心意图与用户诉求之间的一致性进行检查。例如，页面标题、主要内容区、重要功能菜单是否直接响应了用户的目标需求。

2. **页面美观度与功能完整性：**
要求：仅分析页面snapshot（HTML结构、DOM元素及其内容）和代码等文本信息，不用也不能参考截图或视觉渲染效果，禁止使用任何"截图"相关的工具（例如 mcp__ms-playwright__browser_take_screenshot）
- 页面美观度：美观度评估必须严格，不能因为功能可用就忽略美观度问题，需要严格检查配色方案、排版布局、字体规范、组件样式、视觉层级、统一性等美观指标，以及是否具备需求相关的UI元素（如文本、表单、按钮等）。
- 功能完整性：判断应用中的模块是否满足用户需求，包括功能设计的合理性和完整性(是否缺失必要功能)。例如：用户需求为“设计一个健身打卡应用”，那应用中需要能够支持设置打卡日期、添加健身项目、记录健身数据等功能。如果应用中不支持设置打卡日期，则认为功能缺失。

3. **交互功能可用性：**
- 仅判断操作返回的结果与DOM状态变化，严禁主观假设交互是否可用，必须以真实操作为准。
- 严格评分要求：如果发现代码实现存在严重问题（如渲染时机错误、状态更新不合理等），即使功能看起来能工作，也必须给予低分。代码质量问题直接影响用户体验，不能因为自动化测试能触发功能就忽略。
- 通用的交互测试流程应该：
  1. 首先通过`mcp__ms-playwright__browser_click`等真实交互操作来触发用户行为，观察页面响应和视觉反馈是否正常
  2. 同时使用`mcp__ms-playwright__browser_evaluate`来检测渲染和状态变化情况，例如：
     - 检查相关DOM元素是否存在且属性正确（如按钮的disabled状态、输入框的值等）
     - 检查页面状态是否正确更新（如数据变化、样式变化、类名变化等）
     - 检查关键变量和函数是否正常工作
     - 对于Canvas/游戏类应用，检查Canvas元素是否存在、context是否可用、是否有绘制内容（可通过`getImageData`等方法检测）、游戏循环是否正常运行
  3. 必须严格对比两种方式的验证结果：
     - 如果通过代码检测发现逻辑正常，但真实交互操作时页面不响应或视觉反馈异常，**必须给予低分（Dynamic评分应低于0.5）**，因为这说明存在严重的渲染时机或事件绑定问题
     - 如果通过代码检查发现实现存在明显问题（如Canvas绘制在setState回调中、嵌套setState调用、事件绑定不当等），即使真实交互看起来能工作，**也必须给予低分（Dynamic评分应低于0.7）**，因为这些问题会导致用户体验不稳定或在不同场景下失效
  4. 如果发现交互后页面不响应或渲染异常，通过`mcp__ms-playwright__browser_evaluate`检查代码实现，特别关注：
     - 事件处理函数是否正确绑定
     - 状态更新逻辑是否正确（如React应用中setState的使用是否合理，是否存在嵌套setState、在setState回调中执行渲染等反模式）
     - 对于Canvas应用，**必须检查绘制逻辑是否在setState回调中**（这是严重的代码质量问题，会导致渲染不及时）
  5. 结合使用`mcp__ms-playwright__browser_run_code`来执行更复杂的交互序列，并使用`mcp__ms-playwright__browser_evaluate`来检查交互后的DOM状态、属性变化或页面逻辑是否生效
- **特别注意：对于Canvas/游戏类应用，如果代码中Canvas绘制逻辑在React的setState回调中执行（例如在`setGameObjects`或`setRabbit`的回调中绘制Canvas），这是严重的代码质量问题。虽然通过`mcp__ms-playwright__browser_evaluate`直接执行JavaScript代码可能能触发渲染（因为绕过了React的状态更新机制），但实际用户通过点击等交互操作时可能无法看到画面更新。发现此类问题必须给予低分（Dynamic评分应低于0.6），即使功能看起来能工作也不能忽略。**
- 禁止使用`browser_take_screenshot`及任何截图、视觉识别类指令。

# 评分维度与标准

请从以下三方面打分，打分严格，并输出原因：

- **Intention (意图达成度，按区间评分)：** 网页标题、主要Header等是否体现出用户需求的核心意图。
    - 0.8~1.0：标题、核心区域高度相关，功能流程在DOM中能明确找到，页面内容紧密匹配用户需求。
    - 0.5~0.7：标题、核心区域部分相关，功能流程在DOM中基本能找到，但存在一定不匹配或覆盖不完整的情况。
    - 0.0~0.4：页面与需求无关，或出现重大异常（如404、500），主要内容缺失或核心区域完全不相关。

- **Static (页面美观度与功能完整性综合评分)：** 静态评分时须严格衡量页面UI美观性和需求相关元素/模块的覆盖完整性，对页面的静态质量进行严苛评估。美观度是评分的重要组成部分，必须通过CSS样式检查（`getComputedStyle`）来客观评估配色、布局、字体、组件样式等。
    - 必须详细核查页面的布局合理性、视觉层级、颜色搭配和文本/组件排版规范性等美观指标，并逐项检查DOM结构中是否全面存在所有用户需求相关的核心元素（如表单、按钮、输入框、列表、标题等），不可遗漏。
    - 仅当页面美观性优秀（配色协调、布局合理、字体规范、组件样式统一现代化）、所有需求必需元素均完整无缺、功能设计合理且完整，才可得高分（0.8-1.0）；如仅部分必需元素缺失（如3个必须元素仅找到2个）、应用设计不合理或美观性一般（存在配色不协调、布局不够合理、字体不够规范、组件样式不够统一等问题），则应严控评分（0.5-0.8）；如有较多元素缺失、结构严重混乱或美观性不达标（配色混乱、布局混乱、字体不规范、组件样式不统一等），应给低分或直接0分。即使功能可用，美观度严重不达标也必须给予低分。
    - 必须明确指出所有功能设计不合理、功能模块缺失或美观度表现不佳的具体问题，评分不允许宽松或笼统评判，需有详实的snapshot或代码细节作为依据。

- **Dynamic (动态交互能力)：**
    - 0.8~1.0：所有关键交互步骤都经过真实操作验证且完全可执行，操作后页面DOM、数据状态和功能流转准确实现预期业务逻辑，实现任务闭环。**代码实现必须合理，不存在明显的反模式或渲染时机问题。** 若操作流畅无异常且交互体验良好，可给高分（0.9-1.0）；若仅有极轻微的可忽略问题或偶发性小Bug，可酌情打0.8-0.9分。
    - 0.5~0.7：大部分主要交互操作可以执行，但有部分操作的反馈出现异常（如点击无明显反应、页面变化不符合预期、表单提交后无反馈、流程未能完全闭环等），或部分交互存在卡死、Bug、DOM未按预期刷新，仅能部分完成核心任务。**如果发现代码实现存在明显问题（如Canvas绘制在setState回调中、嵌套setState、事件绑定不当等），即使功能看起来能工作，评分也应低于0.7。** 根据问题严重程度、影响范围细分（如仅少量次要流程出问题可给0.7，出现较明显阻碍或代码质量问题则可低至0.5）。
    - 0.0~0.4：只要有任何一处必需交互操作无法完成（如按钮不可点击、表单不可输入/提交、操作后页面卡死无响应等），或交互能力严重缺失，致使主要任务流程无法推进，则此项分数应低于0.5，并根据可用性实际情况，区分为无交互可用（0.0），或仅极个别操作可用/严重损坏（0.1-0.4）。**如果通过代码检测发现逻辑正常，但真实交互操作时页面不响应或视觉反馈异常，说明存在严重的渲染时机或事件绑定问题，必须给予低分（低于0.5）。**
    
# 特别注意
- 不允许也不需要使用"browser_take_screenshot"等任何截图/视觉截图相关的工具或请求。仅依据DOM结构、属性和操作反馈进行分析，不参考截图或视觉效果。
- 鼓励通过操作步骤获取真实能力反馈（如click/fill返回是否报错、操作后DOM变化），不要仅仅"假设"交互能力。
- 若发现某应有的元素在DOM/Snapshot缺失，或交互操作无法完成，应明确指出原因。
- 由于上下文长度的限制，尽可能通过最少的操作步骤，完成评估。

# 输出格式
只输出被<answer>```json```</answer>包裹的单个JSON对象（按以下示例格式），不要额外输出任何playwright代码、分析日志或提示。你的原因(reason)应尽量细致分析。

示例：
<answer>```json
{example_json_str}
```</answer>

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

