# AWorld-RL Environment Tuning

## 项目介绍

本项目基于Berkeley Function Calling Leaderboard (BFCL)进行多回合函数调用的强化学习训练，使用VERL (Versatile Efficient Reinforcement Learning)框架和GRPO (Generalized Reward Policy Optimization)算法来优化大语言模型在工具调用任务上的表现。

## 项目结构

```
EnvTuning/
├── verl/                           # VERL强化学习框架
│   ├── examples/                   # 示例脚本和配置
│   ├── recipe/                     # 训练配方
│   ├── verl/                       # 核心代码
│   ├── setup.py                    # 安装文件
│   └── requirements.txt            # 依赖列表
├── env_tuning/                     # 环境调优模块
│   ├── config/                     # 配置文件
│   │   ├── multi_turn_fc_grpo_stage1.yaml
│   │   ├── multi_turn_fc_grpo_stage2.yaml
│   │   ├── multi_turn_fc_grpo_stage3.yaml
│   │   └── multi_turn_fc_interaction_config.yaml
│   ├── interaction/                # 交互管理模块
│   ├── bfcl_reward.py             # BFCL奖励函数
│   └── format_reward.py           # 格式化奖励函数
├── run_multi_turn_fc_grpo_stage1.sh  # 第一阶段训练脚本
├── run_multi_turn_fc_grpo_stage2.sh  # 第二阶段训练脚本
├── run_multi_turn_fc_grpo_stage3.sh  # 第三阶段训练脚本
├── bfcl_env/                      # BFCL环境
└── assets/                        # 资源文件
```

## Quick Start

### 1. 环境准备

首先确保你有合适的Python环境（推荐Python 3.8+）和CUDA环境。

### 2. 安装VERL框架

```bash
# 进入verl目录
cd verl

# 安装VERL框架（开发模式）
pip install -e .

```

### 3. 配置模型和数据路径

在运行训练脚本之前，需要修改以下配置：

1. **修改模型路径**：编辑训练脚本中的`MODEL`变量
   ```bash
   # 在脚本中找到这行并修改为你的模型路径
   MODEL="/your/local/model/path"
   ```

2. **准备数据**：确保以下数据文件存在
   - `processed_data/bfcl_v3/train_format_75.parquet` (Stage 1)
   - `processed_data/bfcl_v3/bfcl_train_base.parquet` (Stage 2)
   - `processed_data/bfcl_v3_new_prompt/bfcl_train.parquet` (Stage 3)
   - `processed_data/bfcl_v3/bfcl_val.parquet` (验证集)

### 4. 运行训练

回到项目根目录执行训练脚本：

```bash
# 返回到EnvTuning目录
cd ..

# 第一阶段训练（基础格式训练）
bash run_multi_turn_fc_grpo_stage1.sh

# 第二阶段训练（基础能力训练）
bash run_multi_turn_fc_grpo_stage2.sh

# 第三阶段训练（完整数据训练）
bash run_multi_turn_fc_grpo_stage3.sh
```

## 训练阶段说明

### Stage 1: 格式化训练
- **目标**: 训练模型学习正确的函数调用格式
- **数据集**: `train_format_75.parquet` - 包含格式化的函数调用样本
- **批次大小**: 16
- **特点**: 专注于输出格式的规范化

### Stage 2: 基础能力训练  
- **目标**: 在基础数据集上进行强化学习训练
- **数据集**: `bfcl_train_base.parquet` - 基础函数调用数据集
- **批次大小**: 32
- **特点**: 建立基本的函数调用能力

### Stage 3: 完整训练
- **目标**: 使用完整数据集进行最终训练
- **数据集**: `bfcl_train.parquet` - 完整的BFCL训练数据
- **批次大小**: 32
- **特点**: 处理更复杂和多样化的场景

## 核心配置说明

### 训练参数
- **算法**: GRPO (Generalized Reward Policy Optimization)
- **学习率**: 1e-6
- **KL散度系数**: 0.1
- **熵系数**: 0.001
- **梯度裁剪**: 1.0
- **训练轮数**: 20 epochs

### 硬件要求
- **GPU**: 建议8卡GPU配置
- **显存**: 每个GPU建议24GB+显存
- **并行策略**: 支持张量并行和数据并行

## 监控和日志

### TensorBoard
训练过程会自动启动TensorBoard记录：
- 损失曲线
- 奖励变化
- KL散度
- 学习率变化

### 日志文件
训练日志保存在 `logs/` 目录下，文件名包含时间戳便于区分。

### 模型检查点
模型检查点默认保存在用户配置的路径下，每5个epoch保存一次。

## 自定义配置

### 修改训练参数
编辑 `env_tuning/config/` 目录下的YAML配置文件来调整训练参数。

### 自定义奖励函数
修改 `env_tuning/format_reward.py` 或 `env_tuning/bfcl_reward.py` 来实现自定义的奖励计算逻辑。

### 调整多回合交互
编辑 `env_tuning/config/multi_turn_fc_interaction_config.yaml` 来配置多回合交互的参数。

## 故障排除

### 常见问题
1. **CUDA内存不足**: 减小 `TRAIN_BATCH_SIZE` 或 `MINI_BATCH_SIZE`
2. **文件路径错误**: 确保所有数据文件路径正确且文件存在
3. **模型路径问题**: 检查 `MODEL` 变量是否指向正确的模型目录

### 依赖问题
如果遇到依赖问题，尝试：
```bash
cd verl
pip install -r requirements.txt
# 或者安装sglang相关依赖
pip install -r requirements_sglang.txt
```

## 许可证

请参考项目根目录下的LICENSE文件。

## 贡献指南

欢迎提交Issue和Pull Request来改进项目。
