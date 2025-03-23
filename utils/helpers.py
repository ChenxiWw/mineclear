import argparse
import os
import torch
from environment import MinesweeperEnv
from agent import DQNAgent, PPOAgent
import random
import numpy as np

def parse_args():
    """
    解析命令行参数
    
    返回:
        解析后的参数
    """
    parser = argparse.ArgumentParser(description='扫雷AI训练和评估')
    
    # 通用参数
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'eval', 'play', 'compare'],
                        help='运行模式: train, eval, play, compare')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--render', action='store_true', help='是否渲染环境')
    parser.add_argument('--verbose', action='store_true', help='是否输出详细信息')
    parser.add_argument('--disable_reasoning', action='store_true', 
                        help='禁用人类推理能力（默认启用）')
    parser.add_argument('--algorithm', type=str, default='dqn', choices=['dqn', 'ppo'],
                        help='使用的强化学习算法：dqn, ppo')
    parser.add_argument('--compare_episodes', type=int, default=100,
                        help='比较不同方法时每种方法的评估回合数')
    
    # 训练相关参数
    parser.add_argument('--num_episodes', type=int, default=10000, help='训练回合数')
    parser.add_argument('--save_dir', type=str, default='models', help='模型保存目录')
    parser.add_argument('--save_freq', type=int, default=500, help='保存模型的频率')
    parser.add_argument('--eval_freq', type=int, default=100, help='评估的频率')
    parser.add_argument('--initial_difficulty', type=int, default=3, help='初始游戏难度（地雷数）')
    parser.add_argument('--max_difficulty', type=int, default=10, help='最大游戏难度')
    parser.add_argument('--difficulty_threshold', type=float, default=0.4, 
                        help='提高难度的胜率阈值')
    
    # 评估相关参数
    parser.add_argument('--model_path', type=str, help='要加载的模型路径')
    parser.add_argument('--eval_episodes', type=int, default=100, help='评估的回合数')
    parser.add_argument('--eval_difficulty', type=int, help='评估的游戏难度')
    parser.add_argument('--stochastic', action='store_true', 
                        help='评估时使用随机策略（默认使用确定性策略）')
    
    # 游戏相关参数
    parser.add_argument('--width', type=int, default=9, help='游戏板宽度')
    parser.add_argument('--height', type=int, default=9, help='游戏板高度')
    parser.add_argument('--num_mines', type=int, default=10, help='地雷数量')
    
    return parser.parse_args()

def create_env(args):
    """
    创建游戏环境
    
    参数:
        args: 命令行参数
        
    返回:
        MinesweeperEnv实例
    """
    render_mode = "human" if args.render else None
    env = MinesweeperEnv(
        width=args.width,
        height=args.height,
        num_mines=args.num_mines,
        render_mode=render_mode
    )
    
    return env

def create_agent(env, args):
    """
    创建强化学习智能体
    
    参数:
        env: 游戏环境
        args: 命令行参数
        
    返回:
        智能体实例
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 确定是否使用人类推理
    use_human_reasoning = not args.disable_reasoning if hasattr(args, 'disable_reasoning') else True
    
    if args.algorithm == 'dqn':
        agent = DQNAgent(
            height=env.height,
            width=env.width,
            device=device,
            learning_rate=0.0001,
            gamma=0.99,
            epsilon_start=1.0,
            epsilon_end=0.1,
            epsilon_decay=0.995,
            use_human_reasoning=use_human_reasoning  # 添加人类推理参数
        )
    elif args.algorithm == 'ppo':
        agent = PPOAgent(
            height=env.height,
            width=env.width,
            device=device,
            lr=0.0002,
            gamma=0.99,
            gae_lambda=0.95,
            policy_clip=0.2,
            batch_size=64,
            n_epochs=5,
            entropy_coef=0.01
        )
    else:
        raise ValueError(f"不支持的算法: {args.algorithm}")
    
    # 加载模型（如果指定）
    if args.model_path is not None:
        if os.path.exists(args.model_path):
            print(f"加载模型: {args.model_path}")
            agent.load_model(args.model_path)
        else:
            print(f"警告: 模型文件 {args.model_path} 不存在")
    
    return agent

def set_seed(seed):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False 