#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LibreOffice Writer 任务准确率分析脚本
分析指定目录下所有任务的结果并统计正确率
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple


def read_result_file(result_path: str) -> float:
    """
    读取result.txt文件并返回结果值
    
    Args:
        result_path: result.txt文件的路径
        
    Returns:
        float: 结果值 (0.0 或 1.0)
        
    Raises:
        Exception: 如果文件不存在或格式错误
    """
    try:
        with open(result_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            
        # 处理可能的格式: "0", "1", "0.0", "1.0"
        if content in ['0', '0.0']:
            return 0.0
        elif content in ['1', '1.0']:
            return 1.0
        else:
            raise ValueError(f"未知的结果值: {content}")
            
    except FileNotFoundError:
        raise Exception(f"文件不存在: {result_path}")
    except Exception as e:
        raise Exception(f"读取文件失败: {e}")


def analyze_tasks(base_dir: str) -> Tuple[Dict[str, float], float, int, int]:
    """
    分析指定目录下所有任务的结果
    
    Args:
        base_dir: 基础目录路径
        
    Returns:
        tuple: (任务结果字典, 总体正确率, 正确任务数, 总任务数)
    """
    base_path = Path(base_dir)
    
    if not base_path.exists():
        raise Exception(f"目录不存在: {base_dir}")
    
    if not base_path.is_dir():
        raise Exception(f"路径不是目录: {base_dir}")
    
    task_results = {}
    successful_tasks = 0
    total_tasks = 0
    skipped_tasks = []
    
    # 遍历所有子目录
    for task_dir in sorted(base_path.iterdir()):
        if not task_dir.is_dir():
            continue
            
        task_id = task_dir.name
        result_file = task_dir / "result.txt"
        
        total_tasks += 1
        
        try:
            result = read_result_file(str(result_file))
            task_results[task_id] = result
            
            if result == 1.0:
                successful_tasks += 1
                
        except Exception as e:
            print(f"⚠️  跳过任务 {task_id}: {e}")
            skipped_tasks.append(task_id)
            task_results[task_id] = None
            total_tasks -= 1  # 不计入总数
    
    # 计算正确率
    accuracy = (successful_tasks / total_tasks * 100) if total_tasks > 0 else 0.0
    
    return task_results, accuracy, successful_tasks, total_tasks


def print_results(task_results: Dict[str, float], accuracy: float, 
                 successful_tasks: int, total_tasks: int):
    """
    打印分析结果
    
    Args:
        task_results: 任务结果字典
        accuracy: 总体正确率
        successful_tasks: 正确任务数
        total_tasks: 总任务数
    """
    print("=" * 60)
    print("📊 LibreOffice Writer 任务准确率分析报告")
    print("=" * 60)
    
    # 总体统计
    print(f"\n📈 总体统计:")
    print(f"   总任务数: {total_tasks}")
    print(f"   正确任务数: {successful_tasks}")
    print(f"   错误任务数: {total_tasks - successful_tasks}")
    print(f"   总体正确率: {accuracy:.2f}%")
    
    # 详细结果
    print(f"\n📋 详细结果:")
    correct_tasks = []
    incorrect_tasks = []
    skipped_tasks = []
    
    for task_id, result in task_results.items():
        if result is None:
            skipped_tasks.append(task_id)
        elif result == 1.0:
            correct_tasks.append(task_id)
        else:
            incorrect_tasks.append(task_id)
    
    if correct_tasks:
        print(f"\n✅ 正确的任务 ({len(correct_tasks)} 个):")
        for i, task_id in enumerate(correct_tasks, 1):
            print(f"   {i:2d}. {task_id}")
    
    if incorrect_tasks:
        print(f"\n❌ 错误的任务 ({len(incorrect_tasks)} 个):")
        for i, task_id in enumerate(incorrect_tasks, 1):
            print(f"   {i:2d}. {task_id}")
    
    if skipped_tasks:
        print(f"\n⚠️  跳过的任务 ({len(skipped_tasks)} 个):")
        for i, task_id in enumerate(skipped_tasks, 1):
            print(f"   {i:2d}. {task_id}")
    
    print("\n" + "=" * 60)


def save_results_to_file(task_results: Dict[str, float], accuracy: float,
                        successful_tasks: int, total_tasks: int, output_file: str):
    """
    将结果保存到文件
    
    Args:
        task_results: 任务结果字典
        accuracy: 总体正确率
        successful_tasks: 正确任务数
        total_tasks: 总任务数
        output_file: 输出文件路径
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("LibreOffice Writer 任务准确率分析报告\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"总体统计:\n")
        f.write(f"总任务数: {total_tasks}\n")
        f.write(f"正确任务数: {successful_tasks}\n")
        f.write(f"错误任务数: {total_tasks - successful_tasks}\n")
        f.write(f"总体正确率: {accuracy:.2f}%\n\n")
        
        f.write("详细结果:\n")
        for task_id, result in sorted(task_results.items()):
            if result is None:
                status = "跳过"
            elif result == 1.0:
                status = "正确"
            else:
                status = "错误"
            f.write(f"{task_id}: {status}\n")


def main():
    """主函数"""
    # 默认目录
    
    # 检查命令行参数
    if len(sys.argv) > 1:
        target_dir = sys.argv[1]
    else:
        print("No resultDir input")
        return
    
    print(f"🔍 分析目录: {target_dir}")
    
    try:
        # 分析任务
        task_results, accuracy, successful_tasks, total_tasks = analyze_tasks(target_dir)
        
        # 打印结果
        print_results(task_results, accuracy, successful_tasks, total_tasks)
        
        # 保存结果到文件
        output_file = "task_accuracy_report.txt"
        save_results_to_file(task_results, accuracy, successful_tasks, total_tasks, output_file)
        print(f"📄 详细报告已保存到: {output_file}")
        
    except Exception as e:
        print(f"❌ 错误: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
