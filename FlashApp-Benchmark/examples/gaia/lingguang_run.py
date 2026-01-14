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
            "reason": "页面中绝大多数需要的UI元素都能被正确渲染并可见，例如导航栏、主要内容区、部分功能按钮都出现在DOM结构中且布局合理。但仍有部分次要模块（如用户头像组件或一些辅助按钮）缺失，或没有渲染完整，比如“个人中心”区不可见，推荐位图标没有显示。整体排版和结构基本符合预期，但仍有部分次要元素未加载或因异常未被渲染。"
        },
        "dynamic": {
            "score": 0.4,
            "reason": "部分核心操作按钮（如“提交”、“下一步”或“确认”）在实际交互时无法被正常点击，导致流程中断，也有部分交互式组件没有产生预期的响应。例如，点击“提交”按钮无反应，或表单中的输入后无法正常提交数据。有的弹窗或下拉选择卡死，影响了后续的互动步骤，但基础的点击操作依然部分可用。"
        }
    }

    example_json_str = json.dumps(example_output, indent=2, ensure_ascii=False)


    # /Users/yuchengyue/AWorld_local/gaia_dataset/lingguang/策策90条benchmark-详细数据.xlsx
    import pandas as pd
    df = pd.read_excel('/Users/yuchengyue/AWorld_local/gaia_dataset/lingguang/策策90条benchmark-详细数据.xlsx', sheet_name='Sheet1')
    user_queries = df['query'].tolist()
    # url 是这样的，/Users/jcc/Documents/workspace_bak/asap-service/framework/agents/workspace/workspace/1760412257571863/index.html我要获取1760412257571863这个id
    origin_html_urls = df['url'].tolist()
    # 从原始url里正确提取flashapp_id，无论是否有file://前缀
    import re
    def extract_flashapp_id(url):
        # 匹配最后出现的 workspace/<id>/ 结构
        match = re.search(r'/workspace/(\d+)', url)
        if match:
            return match.group(1)
        return None
    # html_urls 要用extract_flashapp_id(url)后加上前缀和index.html，组成新的url
    html_urls = [f"file:///Users/yuchengyue/Downloads/framework/agents/workspace/workspace/{extract_flashapp_id(origin_html_url)}/index.html" for origin_html_url in origin_html_urls]

    # import pdb;pdb.set_trace()

    user_queries = user_queries[48:]
    html_urls = html_urls[48:]

    user_queries = [
        # "创建一个福彩3D应用",
        # "制作一个营业执照在线生成电子模拟器",
        # "请根据用户诉求: '可以画画吗'，生成对应的轻应用。",
        # "请根据用户诉求: '放烟花'，生成对应的轻应用。",
        # "请根据用户诉求: '给我一个计算器'，生成对应的轻应用。",
        # "请根据用户诉求: '生成一个马拉松破3小时的计划'，生成对应的轻应用。",
        # "请根据用户诉求: '周五了，现在还在加班，还有谁比我惨'，生成对应的轻应用。"
        # "请根据用户诉求: '如何不变老'，生成对应的轻应用。",
        # "请根据用户诉求: '帮我生成一个能显示时分秒的翻页时钟代码'，生成对应的轻应用。",
        "帮我做一个做饭的应用，输入菜名获得食材做法耗时等"
    ]
    html_urls = [
        # "https://render.lingguangcontent.com/p/lingguang/0b22282517646296792058518e90cf/lottery3d.html",
        # "https://render.lingguangcontent.com/p/lingguang/0b90097017646505474535156e6b83/index.html",
        # "file:///Users/yuchengyue/Downloads/framework/agents/workspace/workspace/1760412257571863/index.html",
        # "file:///Users/yuchengyue/Downloads/framework/agents/workspace/workspace/1760412259347301/index.html",
        # "file:///Users/yuchengyue/Downloads/framework/agents/workspace/workspace/1760412256028591/index.html",
        # "file:///Users/yuchengyue/Downloads/framework/agents/workspace/workspace/1760412256619558/index.html",
        # "file:///Users/yuchengyue/Downloads/framework/agents/workspace/workspace/1760412258193890/index.html",
        # "file:///Users/yuchengyue/Downloads/framework/agents/workspace/workspace/1760412306615268/index.html"
        # "file:///Users/yuchengyue/Downloads/framework/agents/workspace/workspace/1760412305128888/index.html",
        "https://render.lingguangcontent.com/p/lingguang/mini-app-wrapper-20251230115825/mini_app_wrapper.html?mini_app_url=https://render.lingguangcontent.com/p/lingguang/21962205176649100271410631050/index.html&api_base=https://agi.alipay.com"
    ]

    flashapp_ids = [extract_flashapp_id(url) for url in html_urls]

    code_given_flag = False
    code_snippet = ""


    for user_query, html_url, flashapp_id in zip(user_queries, html_urls, flashapp_ids):

        if not code_given_flag:
            task_description = """对于每个任务，你会得到一个**用户需求(User Query)**和一个**网页地址(Target URL)**。"""
        else:
            task_description = """对于每个任务，你会得到一个**用户需求(User Query)**、一个**网页地址(Target URL)**和页面的html及js代码（Code Snippet）。
当你获取了用户的代码，你可以参考js代码里的变量名称，写js代码来获取页面的信息；也可以结合提供的代码去判断应用实现的合理性、美观度等。"""
            
            workspace_path = f"/Users/yuchengyue/Downloads/framework/agents/workspace/workspace/{flashapp_id}"
            def read_code_file(file_path):
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        return f.read()
                except Exception as e:
                    return f"[读取失败：{file_path}, 错误：{e}]"

            index_html_path = os.path.join(workspace_path, "index.html")
            app_js_path = os.path.join(workspace_path, "app.js")
            if not os.path.exists(app_js_path):
                # 如果app.js不存在，查找workspace_path下的其他.js文件
                js_files = [f for f in os.listdir(workspace_path) if f.endswith('.js')]
                if js_files:
                    # 取第一个非app.js的js文件
                    for js_file in js_files:
                        if js_file != 'app.js':
                            app_js_path = os.path.join(workspace_path, js_file)
                            break

            index_html_content = read_code_file(index_html_path)
            app_js_content = read_code_file(app_js_path)

            code_snippet = f"""提供给你的html和js代码如下：

======= index.html =======
{index_html_content}
======= app.js =======
{app_js_content}
"""

        search_sys_prompt = f"""
# 角色说明
你是一名专业的QA自动化工程师和网页可用性评测专家。你需要判断给定“Target URL”页面中的美观度、元素和交互能力，是否满足“User Query”中的需求。

# 任务说明
{task_description}

你需要分三步进行评估：

1. **页面和用户需求一致性：**
- 首先，判断目前生成的闪应用是否能够解决用户的核心诉求，且能够真切帮助用户解决问题。
- 其次，判断页面的标题（title）、主要Header内容是否展现出完成用户任务的关键信息和功能。重点对页面核心意图与用户诉求之间的一致性进行检查。例如，页面标题、主要内容区、重要功能菜单是否直接响应了用户的目标需求。

2. **页面美观度与元素覆盖度：**
- 根据网页的snapshot（HTML结构、DOM元素及其内容），分析页面美观性（如配色、排版、视觉层级）以及是否具备需求相关的基础UI元素（如文本、表单、按钮等）。
- 可通过`mcp__ms-playwright__browser_evaluate`工具执行JavaScript代码来获取和分析页面CSS样式、颜色搭配等信息。统计页面是否具备需求中应有的关键元素，指出缺失部分。
- 仅分析DOM结构和文本信息，不用也不能参考截图或视觉渲染效果，禁止使用任何“截图”相关的工具（例如 mcp__ms-playwright__browser_take_screenshot）。 

3. **交互功能可用性：**
    - - 仅判断操作返回的结果与DOM状态变化，严禁主观假设交互是否可用，必须以真实操作为准。
    - 在烟花、射击等游戏类应用中，优先使用`mcp__ms-playwright__browser_evaluate`工具进行功能的快速检测；如果需要更可靠的等待和重试机制，可以结合使用`mcp__ms-playwright__browser_run_code`来执行交互操作，并使用`mcp__ms-playwright__browser_evaluate`来检查交互后的DOM状态、属性变化或页面逻辑是否生效。
    - 禁止使用`browser_take_screenshot`及任何截图、视觉识别类指令。

# 评分维度与标准

请从以下三方面打分，打分严格，并输出原因：

- **Intention (意图达成度，按区间评分)：** 网页标题、主要Header等是否体现出用户需求的核心意图。
    - 0.8~1.0：标题、核心区域高度相关，功能流程在DOM中能明确找到，页面内容紧密匹配用户需求。
    - 0.5~0.7：标题、核心区域部分相关，功能流程在DOM中基本能找到，但存在一定不匹配或覆盖不完整的情况。
    - 0.0~0.4：页面与需求无关，或出现重大异常（如404、500），主要内容缺失或核心区域完全不相关。

- **Static (静态元素覆盖度与美观度综合评分)：** 静态评分时须严格衡量页面UI美观性和需求相关元素/模块的覆盖完整性，对页面的静态质量进行严苛评估。
    - 必须详细核查页面的布局合理性、视觉层级、颜色搭配和文本/组件排版规范性等美观指标，并逐项检查DOM结构中是否全面存在所有用户需求相关的核心元素（如表单、按钮、输入框、列表、标题等），不可遗漏；
    - 仅当页面美观性优秀且所有需求必需元素均完整无缺，才可得高分（0.8-1.0）；如仅部分必需元素缺失（如3个必须元素仅找到2个，或布局出现明显混乱），或美观性一般，则应严控评分（0.5-0.8）；如有较多元素缺失、结构严重混乱或美观性不达标，应给低分或直接0分；
    - 必须明确指出所有缺失或表现不佳的具体元素或区域，并结合实际页面情况详细说明美观度和结构的不足。评分不允许宽松或笼统评判，需有详实依据。

- **Dynamic (动态交互能力)：**
    - 0.8~1.0：所有关键交互步骤都经过真实操作验证且完全可执行，操作后页面DOM、数据状态和功能流转准确实现预期业务逻辑，实现任务闭环。若操作流畅无异常且交互体验良好，可给高分（0.9-1.0）；若仅有极轻微的可忽略问题或偶发性小Bug，可酌情打0.8-0.9分。
    - 0.5~0.7：大部分主要交互操作可以执行，但有部分操作的反馈出现异常（如点击无明显反应、页面变化不符合预期、表单提交后无反馈、流程未能完全闭环等），或部分交互存在卡死、Bug、DOM未按预期刷新，仅能部分完成核心任务。根据问题严重程度、影响范围细分（如仅少量次要流程出问题可给0.7，出现较明显阻碍则可低至0.5）。
    - 0.0~0.4：只要有任何一处必需交互操作无法完成（如按钮不可点击、表单不可输入/提交、操作后页面卡死无响应等），或交互能力严重缺失，致使主要任务流程无法推进，则此项分数应低于0.5，并根据可用性实际情况，区分为无交互可用（0.0），或仅极个别操作可用/严重损坏（0.1-0.4）。
    
# 特别注意
- 不允许也不需要使用“browser_take_screenshot”等任何截图/视觉截图相关的工具或请求。仅依据DOM结构、属性和操作反馈进行分析，不参考截图或视觉效果。
- 鼓励通过操作步骤获取真实能力反馈（如click/fill返回是否报错、操作后DOM变化），不要仅仅“假设”交互能力。
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

        result = Runners.sync_run_task(task=Task(input=question, agent=super, conf=TaskConfig(max_steps=100)))
        
        match = re.search(r'<answer>(.*?)</answer>', result["task_0"]["answer"])
        # if match:
        #     answer = match.group(1)
        #     logging.info(f"Agent answer: {answer}")
        #     logging.info(f"Correct answer: {full_dataset[i]['Final answer']}")
            
        #     if answer == full_dataset[i]["Final answer"]:
        #         logging.info(f"Question {i} Correct!")
        #     else:
        #         logging.info("Incorrect!")