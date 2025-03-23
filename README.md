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
│   └── minesweeper_game.py # 可玩的扫雷游戏实现
├── utils/            # 辅助函数
├── main.py           # 主程序
├── test_project.py   # 测试脚本
├── evaluate_models.py # 模型评估脚本
└── requirements.txt  # 项目依赖
```

## 功能特点

1. **增强的扫雷游戏环境**
   - 基于OpenAI Gymnasium的环境设计
   - 丰富的奖励系统，包括策略奖励和信息增益奖励
   - 自动避免首次点击踩雷

2. **先进强化学习算法**
   - **Dueling DQN**: 分离状态价值和动作优势函数
   - **优先经验回放(PER)**: 根据TD误差为经验分配优先级
   - **双重DQN(Double DQN)**: 减少Q值过估计问题
   - **深度卷积网络**: 带有注意力机制的深度卷积网络
   - **四通道状态表示**: 增强模式识别能力
   - **高级奖励塑形**: 根据游戏状态定制奖励信号

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

7. **可视化与游戏实现**
   - 交互式扫雷游戏实现，支持人类和AI玩家
   - 实时AI辅助功能
   - 详细的游戏统计和结果分析

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

高级训练参数：
```bash
python main.py --mode train --num_episodes 10000 --batch_size 128 --learning_rate 0.0001 --use_human_reasoning True --disable_reasoning False
```

参数说明：
- `--algorithm`: 选择训练算法，支持 'dqn' 或 'ppo'
- `--num_episodes`: 训练回合数
- `--initial_difficulty`: 初始难度(地雷数)
- `--disable_reasoning`: 禁用人类推理能力
- `--difficulty_threshold`: 提高难度的胜率阈值(默认0.7)

### 评估模式

```bash
python main.py --mode eval --model_path models/dqn_model_ep8100_diff4.pth --eval_episodes 50
```

参数说明：
- `--model_path`: 加载预训练模型的路径
- `--eval_episodes`: 评估的回合数
- `--eval_difficulty`: 评估的难度级别
- `--disable_reasoning`: 禁用人类推理能力
- `--stochastic`: 使用随机策略而非确定性策略

### 比较不同方法

```bash
python main.py --mode compare --model_path models/dqn_model_ep8100_diff4.pth --compare_episodes 50 --eval_difficulty 5
```

批量评估模型：
```bash
python evaluate_models.py
```

参数说明：
- `--model_path`: 加载预训练模型的路径
- `--compare_episodes`: 每种方法评估的回合数
- `--eval_difficulty`: 评估的难度级别

### 游戏模式

```bash
python main.py --mode play --width 9 --height 9 --num_mines 10 --model_path models/dqn_model_ep8100_diff4.pth
```

自定义控制键：
```bash
python main.py --mode play --width 9 --height 9 --num_mines 10 --ai-key=s --reset-key=n --quit-key=e
```

参数说明：
- `--width`: 游戏板宽度
- `--height`: 游戏板高度
- `--num_mines`: 地雷数量
- `--model_path`: 加载AI模型
- `--ai-key`: AI辅助键 (默认为'a')
- `--reset-key`: 重置游戏键 (默认为'r')
- `--quit-key`: 退出游戏键 (默认为'q')

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

## 改进

### 1. 先进强化学习算法实现

- **Dueling DQN架构**：将Q值分解为状态价值函数V(s)和优势函数A(s,a)，更高效地学习状态价值
- **优先经验回放(PER)**：基于TD误差为重要经验分配更高采样概率，提高学习效率
- **高级奖励塑形策略**：
  - 连锁反应奖励：根据一次操作打开的格子数量给予额外奖励
  - 边缘探索奖励：鼓励在已知区域边缘探索的行为
  - 逻辑推理奖励：对符合逻辑推理的决策给予额外奖励
  - 基于进度的失败惩罚：根据游戏进度调整失败惩罚的严重程度

### 2. 改进的人类推理模块

- **CSP求解器优化**：更高效的约束满足问题求解器
- **概率模型改进**：更准确的地雷概率计算方法
- **推理-网络协同机制**：智能地在确定性推理和概率性决策之间切换
- **边缘格子识别**：识别和优先探索信息丰富的边缘格子

### 3. 增强的游戏体验

- **交互式游戏界面**：使用pygame实现的完整扫雷游戏
- **AI辅助功能**：一键获取AI建议的最佳下一步
- **自定义控制键**：可通过命令行参数自定义游戏控制键
- **游戏内状态显示**：实时显示地雷数量和游戏状态

### 4. 训练与评估改进

- **自动难度调整**：基于胜率动态调整训练难度
- **批量模型评估**：使用`evaluate_models.py`脚本批量评估多个模型
- **不同决策策略的详细统计**：记录并分析推理vs神经网络决策的效果
- **训练过程可视化**：显示实时训练进度和性能指标

## 性能表现

依托答辩


## 贡献

欢迎贡献代码、报告问题或提供改进建议！

## 许可证

此项目基于MIT许可证开源。 