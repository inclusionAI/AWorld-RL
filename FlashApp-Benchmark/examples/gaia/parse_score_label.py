import json
import re
from typing import Dict, Any


def parse_and_label(raw_str: str, threshold: float = 0.8) -> Dict[str, Any]:
    """
    从原始字符串中提取JSON，计算总分并判断正负样本。
    
    Args:
        raw_str: 包含 ```json ... ``` 的原始字符串
        threshold: 阈值，默认0.8，任一维度低于此值则为负样本
        
    Returns:
        包含各维度分数、总分和标签的字典
    """
    # 1. 提取 ```json ... ``` 里的 JSON 内容
    match = re.search(r"```json\s*(\{.*?\})\s*```", raw_str, re.DOTALL)
    if not match:
        raise ValueError("未找到 ```json ... ``` 片段")
    
    json_str = match.group(1)
    
    # 2. 解析 JSON
    data = json.loads(json_str)
    
    # 3. 提取三个维度的分数
    intention_score = float(data["intention"]["score"])
    static_score = float(data["static"]["score"])
    dynamic_score = float(data["dynamic"]["score"])
    
    # 4. 判断正负样本：任一维度低于阈值则为负样本
    if intention_score <= threshold or static_score <= threshold or dynamic_score <= threshold:
        label = "neg"
    else:
        label = "pos"
    
    # 5. 构造最终输出 JSON
    result = {
        "label": label,
        "intention": {
            "score": intention_score,
            "reason": data["intention"]["reason"]
        },
        "static": {
            "score": static_score,
            "reason": data["static"]["reason"]
        },
        "dynamic": {
            "score": dynamic_score,
            "reason": data["dynamic"]["reason"]
        }
    }
    
    return result


if __name__ == "__main__":
    # 测试示例
    test_input = """<answer>```json
{
  "intention": {
    "score": 1.0,
    "reason": "当前页面完全符合用户需求。标题明确为'交叉点棋盘'，核心功能成功将棋子放置逻辑改为'交叉点'模式（代码中通过计算最近的行列交点坐标实现），且底层数据结构明确采用了9x10的矩阵（this.rows=10; this.cols=9），完美响应了重建需求。"
  },
  "static": {
    "score": 1.0,
    "reason": "页面UI美观度高，结构清晰。包含了所有必要的交互元素：棋盘区域（Canvas）、坐标和状态显示栏（深色背景突出显示）、控制面板（包含棋子大小滑块、颜色选择单选框及清空按钮）。使用了Tailwind CSS和自定义CSS进行样式修饰，具备圆角、阴影和合理的配色（米色背景、木纹色棋盘），视觉体验良好。"
  },
  "dynamic": {
    "score": 1.0,
    "reason": "动态交互能力完美。实际测试显示：1. 点击棋盘任意位置，程序能正确计算并吸附到最近的交叉点（9x10网格）放置棋子；2. 再次点击相同位置可移除棋子（Toggle逻辑有效）；3. 状态栏实时准确更新当前的坐标和棋子数量；4. '清空棋盘'按钮功能正常。整个交互流程流畅，无报错，完全满足'精确放置'的需求。"
  }
}
```</answer>"""
    
    result = parse_and_label(test_input)
    print(json.dumps(result, ensure_ascii=False, indent=2))

