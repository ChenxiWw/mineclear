# 扫雷AI训练项目

使用深度强化学习训练AI玩扫雷游戏。

## 项目结构

```
mineclear/
├── environment/      # 扫雷游戏环境模拟器
│   ├── minesweeper.py         # 基于Gymnasium的扫雷环境
│   └── minesweeper_solver.py  # 专业扫雷求解器用于数据生成
├── agent/            # 强化学习智能体
│   ├── dqn_agent.py  # DQN智能体实现
│   ├── ppo_agent.py  # PPO智能体实现
│   └── human_reasoning.py  # 人类推理模块
├── training/         # 训练逻辑和流程
│   ├── dqn_trainer.py # DQN训练器
│   └── ppo_trainer.py # PPO训练器
├── visualization/    # 可视化AI决策过程
├── real_game/        # 真实游戏对接
├── utils/            # 辅助函数
├── main.py           # 主程序
├── test_project.py   # 测试脚本
└── requirements.txt  # 项目依赖
```

## 功能特点

1. **增强的扫雷游戏环境**
   - 基于OpenAI Gymnasium的环境设计
   - 丰富的奖励系统，包括策略奖励和信息增益奖励
   - 自动避免首次点击踩雷

2. **高级DQN智能体**
   - 深度卷积神经网络带有注意力机制
   - 四通道状态表示，增强模式识别
   - 双重DQN实现，提升训练稳定性
   - 奖励整形和优先级经验回放

3. **人类推理能力**
   - 模拟人类玩扫雷的思维过程
   - 基于约束满足问题(CSP)的推理系统
   - 三层推理策略：单格推理、重叠约束推理和概率分析
   - 与神经网络无缝集成，智能选择决策方法

4. **专业扫雷求解器**
   - 提供高质量的预训练数据
   - 实现多种扫雷求解策略
   - 用于评估基准的标准解决方案

5. **先进的训练系统**
   - 课程学习：从简单到复杂的训练流程
   - 自监督预训练：使用专业求解器生成的样本
   - 记忆机制：跟踪确定安全的格子
   - 自动难度调整：根据胜率提高训练难度

6. **方法比较与评估**
   - 对比纯人类逻辑、纯DQN和二者结合的效果
   - 详细性能指标展示：平均奖励、胜率和推理使用率
   - 自动生成对比报告

7. **可视化与现实游戏对接**
   - 实时决策可视化
   - 真实游戏对接（通过OpenCV捕获屏幕，自动控制鼠标）

8. **完整测试套件**
   - 自动化测试环境、智能体和训练模块
   - 性能基准测试
   - 自动生成测试报告

## 安装

```bash
pip install -r requirements.txt
```

## 使用方法

### 训练模式

```bash
python main.py --mode train --num_episodes 10000 --initial_difficulty 3 --render --verbose
```

参数说明：
- `--algorithm`: 选择训练算法，支持 'dqn' 或 'ppo'
- `--num_episodes`: 训练回合数
- `--initial_difficulty`: 初始难度(地雷数)
- `--disable_reasoning`: 禁用人类推理能力

### 评估模式

```bash
python main.py --mode eval --model_path models/dqn_model_ep3000_diff10.pth --eval_episodes 50
```

参数说明：
- `--model_path`: 加载预训练模型的路径
- `--eval_episodes`: 评估的回合数
- `--eval_difficulty`: 评估的难度级别
- `--disable_reasoning`: 禁用人类推理能力

### 比较不同方法

```bash
python main.py --mode compare --model_path models/dqn_model_ep3000_diff10.pth --compare_episodes 50 --eval_difficulty 5
```

参数说明：
- `--model_path`: 加载预训练模型的路径
- `--compare_episodes`: 每种方法评估的回合数
- `--eval_difficulty`: 评估的难度级别

### 游戏模式

```bash
python main.py --mode play --width 9 --height 9 --num_mines 10 --model_path dqn_model_ep3000_diff10.pth
```

参数说明：
- `--width`: 游戏板宽度
- `--height`: 游戏板高度
- `--num_mines`: 地雷数量
- `--model_path`: 加载AI模型

### 测试模式

```bash
python test_project.py [--all] [--dependencies] [--environment] [--agent] [--training] [--cli] [--benchmark]
```

参数说明：
- `--all`: 运行所有测试（默认）
- `--dependencies`: 仅测试项目依赖
- `--environment`: 仅测试环境模块
- `--agent`: 仅测试智能体模块
- `--training`: 仅测试训练模块
- `--cli`: 仅测试命令行接口
- `--benchmark`: 仅运行性能基准测试
- `--seed`: 设置随机种子（默认42）

## 最新改进

### 1. 人类推理模块

项目现在集成了强大的人类推理模块，大幅提升AI性能：
- **单格推理**：识别确定安全或确定是地雷的格子
- **重叠约束推理**：分析多个数字格子的共同约束
- **概率分析**：计算每个未知格子是地雷的概率
- **整合机制**：在能确定安全格子时使用推理，不确定时使用神经网络

### 2. 增强的神经网络架构

- **四通道状态表示**：更丰富的状态信息
- **注意力机制**：帮助模型关注棋盘上的重要区域
- **更深的网络**：三层卷积网络配合两层全连接层
- **批量归一化和Dropout**：提高训练稳定性，防止过拟合

### 3. 方法对比功能

- 全新的对比评估系统，可比较三种方法：
  - **纯人类逻辑**：仅使用逻辑推理
  - **纯DQN**：仅使用神经网络
  - **混合方法**：结合逻辑推理和神经网络
- 自动生成详细对比报告，包括奖励、胜率和推理使用率

### 4. 优化的训练流程

- 动态难度调整阈值：从0.7降低到0.4，更合理的进阶策略
- 更细致的奖励整形，鼓励探索和策略决策
- 人类推理使用统计跟踪，帮助了解决策过程

### 5. 测试套件

- 全面的自动化测试系统，覆盖所有模块和功能
- 性能基准测试，实时监控系统效率
- 人类推理模块性能特别突出，较神经网络推理快约200倍以上

## 贡献

欢迎贡献代码、报告问题或提供改进建议！

## 许可证

此项目基于MIT许可证开源。 