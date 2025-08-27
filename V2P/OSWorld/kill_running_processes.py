#!/usr/bin/env python3
"""
脚本用于杀掉所有Docker容器和Python进程
"""

import subprocess
import sys
import time

def run_command(command):
    """执行命令并返回结果"""
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        return result.stdout.strip(), result.stderr.strip(), result.returncode
    except Exception as e:
        return "", str(e), -1

def kill_docker_containers():
    """杀掉所有Docker容器"""
    print("正在杀掉所有Docker容器...")
    
    # 获取所有运行中的容器ID
    stdout, stderr, returncode = run_command("docker ps -q")
    if returncode != 0:
        print(f"获取Docker容器列表失败: {stderr}")
        return False
    
    if not stdout:
        print("没有运行中的Docker容器")
        return True
    
    container_ids = stdout.split('\n')
    print(f"找到 {len(container_ids)} 个运行中的容器")
    
    # 杀掉每个容器
    for container_id in container_ids:
        if container_id.strip():
            print(f"正在杀掉容器: {container_id}")
            stdout, stderr, returncode = run_command(f"docker kill {container_id}")
            if returncode == 0:
                print(f"成功杀掉容器: {container_id}")
            else:
                print(f"杀掉容器失败 {container_id}: {stderr}")
    
    return True

def kill_python_processes():
    """杀掉所有Python进程"""
    print("正在杀掉所有Python进程...")
    
    # 获取所有Python进程的PID
    stdout, stderr, returncode = run_command("ps aux | grep python | grep -v grep | awk '{print $2}'")
    if returncode != 0:
        print(f"获取Python进程列表失败: {stderr}")
        return False
    
    if not stdout:
        print("没有运行中的Python进程")
        return True
    
    pids = stdout.split('\n')
    print(f"找到 {len(pids)} 个Python进程")
    
    # 杀掉每个Python进程
    for pid in pids:
        if pid.strip():
            print(f"正在杀掉Python进程: {pid}")
            stdout, stderr, returncode = run_command(f"kill -9 {pid}")
            if returncode == 0:
                print(f"成功杀掉Python进程: {pid}")
            else:
                print(f"杀掉Python进程失败 {pid}: {stderr}")
    
    return True

def main():
    """主函数"""
    print("开始清理进程...")
    
    # 杀掉Docker容器
    if not kill_docker_containers():
        print("清理Docker容器失败")
        sys.exit(1)
    
    # 等待一下确保容器完全停止
    time.sleep(2)
    
    # 杀掉Python进程
    if not kill_python_processes():
        print("清理Python进程失败")
        sys.exit(1)
    
    print("所有进程清理完成!")
    
    # 验证清理结果
    print("\n验证清理结果:")
    print("Docker容器状态:")
    run_command("docker ps")
    
    print("\nPython进程状态:")
    run_command("ps aux | grep python | grep -v grep")

if __name__ == "__main__":
    main()
