#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import time
import numpy as np
import torch
import argparse

# 项目模块
from environment import MinesweeperEnv
from agent import DQNAgent
from training import DQNTrainer, PPOTrainer
from visualization import PygameRenderer, MatplotlibRenderer, VisualizeGameplay
# real_game模块仅在需要时导入，避免GPU服务器环境缺少显示服务器引起的错误
# from real_game import ScreenCapture, MouseController, RealGameController
from utils.helpers import create_env, create_agent, parse_args, set_seed

def train_mode(args):
    """
    训练模式
    
    参数:
        args: 命令行参数
    """
    print("进入训练模式...")
    
    # 创建环境和智能体
    env = create_env(args)
    agent = create_agent(env, args)
    
    # 创建训练器
    if args.algorithm == 'dqn':
        trainer = DQNTrainer(env=env, agent=agent)
    elif args.algorithm == 'ppo':
        trainer = PPOTrainer(env=env, agent=agent)
    else:
        raise ValueError(f"不支持的算法: {args.algorithm}")
    
    # 开始训练
    rewards, steps, win_rates = trainer.train(
        num_episodes=args.episodes, 
        eval_interval=100,
        save_interval=500
    )
    
    print("训练完成!")
    return True

def visualize_mode(args):
    """
    可视化模式
    
    参数:
        args: 命令行参数
    """
    print("进入可视化模式...")
    
    # 创建环境和智能体
    env = create_env(args)
    agent = create_agent(env, args)
    
    # 加载预训练模型
    if args.model:
        agent.load_model(args.model)
        print(f"已加载模型: {args.model}")
    
    # 创建可视化器
    if args.renderer == 'pygame':
        renderer = PygameRenderer(env)
    else:
        renderer = MatplotlibRenderer(env)
    
    # 创建可视化游戏玩法
    visualizer = VisualizeGameplay(env, agent, renderer)
    
    # 开始可视化
    visualizer.visualize(num_episodes=args.episodes, speed=args.speed)
    
    print("可视化结束!")
    return True

def real_game_mode(args):
    """
    真实游戏对接模式
    
    参数:
        args: 命令行参数
    """
    print("进入真实游戏对接模式...")
    
    # 在这个模式下才导入real_game模块
    try:
        from real_game import ScreenCapture, MouseController, RealGameController
    except Exception as e:
        print(f"错误: 无法导入real_game模块，可能缺少GUI环境: {e}")
        print("请确保运行在有图形界面的环境中，并安装了所有依赖: PIL, pyautogui, opencv等")
        return False
    
    # 创建智能体
    agent = create_agent(None, args)
    
    # 加载预训练模型
    if args.model:
        agent.load_model(args.model)
        print(f"已加载模型: {args.model}")
    else:
        print("警告: 未指定预训练模型，将使用随机策略")
    
    # 创建游戏控制器
    controller = RealGameController(agent, args)
    
    # 开始控制真实游戏
    controller.play(num_games=args.num_games)
    
    print("游戏控制结束!")
    return True

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='扫雷游戏AI训练项目')
    # 通用参数
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'eval', 'play', 'compare'],
                        help='运行模式: train (训练), eval (评估), play (游戏), compare (方法比较)')
    parser.add_argument('--algorithm', type=str, default='dqn', choices=['dqn', 'ppo'],
                        help='强化学习算法: dqn (深度Q网络), ppo (近端策略优化)')
    
    # 环境参数
    parser.add_argument('--width', type=int, default=5, help='游戏板宽度')
    parser.add_argument('--height', type=int, default=5, help='游戏板高度')
    parser.add_argument('--num_mines', type=int, default=3, help='地雷数量')
    
    # 训练参数
    parser.add_argument('--num_episodes', type=int, default=10000, help='训练回合数')
    parser.add_argument('--batch_size', type=int, default=128, help='批量大小')
    parser.add_argument('--save_freq', type=int, default=100, help='模型保存频率')
    parser.add_argument('--eval_freq', type=int, default=100, help='评估频率')
    parser.add_argument('--render', action='store_true', help='是否渲染游戏')
    parser.add_argument('--verbose', action='store_true', help='是否打印详细信息')
    parser.add_argument('--initial_difficulty', type=int, default=3, help='初始难度')
    parser.add_argument('--max_difficulty', type=int, default=10, help='最大难度')
    parser.add_argument('--difficulty_threshold', type=float, default=0.7, 
                        help='提高难度的胜率阈值（当胜率超过此值时提高难度）')
    parser.add_argument('--disable_reasoning', action='store_true', help='禁用人类推理能力')
    
    # 评估参数
    parser.add_argument('--model_path', type=str, help='加载模型的路径')
    parser.add_argument('--eval_episodes', type=int, default=100, help='评估回合数')
    parser.add_argument('--eval_difficulty', type=int, help='评估难度')
    parser.add_argument('--stochastic', action='store_true', help='评估时使用随机策略而非确定性策略')
    
    # 比较参数
    parser.add_argument('--compare_episodes', type=int, default=50, help='比较评估的回合数')
    
    # 游戏控制键设置
    parser.add_argument('--ai-key', type=str, default='a', help='AI辅助键')
    parser.add_argument('--reset-key', type=str, default='r', help='重置游戏键')
    parser.add_argument('--quit-key', type=str, default='q', help='退出游戏键')
    
    return parser.parse_args()

def main():
    """主要训练和评估入口"""
    args = parse_args()
    
    if hasattr(args, 'seed') and args.seed is not None:
        set_seed(args.seed)
    
    # 创建环境和智能体
    env = create_env(args)
    
    if args.mode == 'train':
        # 创建智能体
        agent = create_agent(env, args)
        
        # 创建训练器
        trainer = DQNTrainer(
            env=env, 
            agent=agent, 
            models_dir=args.save_dir if hasattr(args, 'save_dir') else 'models'
        )
        
        # 开始训练
        print("开始训练...")
        trainer.train(
            num_episodes=args.num_episodes, 
            initial_difficulty=args.initial_difficulty,
            max_difficulty=args.max_difficulty,
            difficulty_increase_threshold=getattr(args, 'difficulty_threshold', 0.7),
            eval_freq=args.eval_freq,
            save_freq=args.save_freq,
            render=args.render,
            verbose=args.verbose,
            enable_human_reasoning=not args.disable_reasoning  # 默认启用人类推理
        )
        
    elif args.mode == 'eval':
        # 创建智能体
        agent = create_agent(env, args)
        
        # 加载训练好的模型进行评估
        trainer = DQNTrainer(env=env, agent=agent)
        if args.model_path:
            trainer.load_model(args.model_path)
            print(f"加载模型: {args.model_path}")
        
        # 进行评估
        print("开始评估...")
        avg_reward, win_rate = trainer.evaluate(
            num_episodes=args.eval_episodes,
            difficulty=args.eval_difficulty,
            render=args.render,
            deterministic=not getattr(args, 'stochastic', False),
            use_human_reasoning=not args.disable_reasoning  # 默认启用人类推理
        )
        
        print(f"评估结果: 平均奖励: {avg_reward:.2f}, 胜率: {win_rate:.2f}")
        
    elif args.mode == 'compare':
        # 创建智能体
        agent = create_agent(env, args)
        
        # 创建训练器
        trainer = DQNTrainer(env=env, agent=agent)
        
        # 如果指定了模型路径，加载模型
        if args.model_path:
            trainer.load_model(args.model_path)
            print(f"加载模型: {args.model_path}")
        
        # 比较不同方法的效果
        print("开始对比不同方法...")
        trainer.compare_methods(
            num_episodes=args.compare_episodes,
            difficulty=args.eval_difficulty,
            verbose=args.verbose
        )
        
    elif args.mode == 'play':
        from real_game.minesweeper_game import MinesweeperGame
        from agent.dqn_agent import DQNAgent
        
        # 如果指定了模型路径，加载AI智能体
        ai_agent = None
        if args.model_path and os.path.exists(args.model_path):
            print(f"加载AI模型: {args.model_path}")
            ai_agent = DQNAgent(
                height=args.height,
                width=args.width,
                use_human_reasoning=not args.disable_reasoning
            )
            ai_agent.load_model(args.model_path)
            print("AI模型加载成功!")
        
        # 创建并运行游戏
        game = MinesweeperGame(
            width=args.width,
            height=args.height,
            num_mines=args.num_mines,
            cell_size=40,
            ai_agent=ai_agent,
            use_human_reasoning=not args.disable_reasoning,
            control_keys={
                'ai': args.ai_key,
                'reset': args.reset_key,
                'quit': args.quit_key
            }
        )
        
        print(f"游戏开始! 按{args.ai_key.upper()}键让AI移动，{args.reset_key.upper()}键重置游戏，{args.quit_key.upper()}键退出。")
        game.run_game()
        
        # 显示结果
        result = game.get_game_result()
        if result:
            print(f"游戏结束！{'胜利！' if result['win'] else '失败！'}")
            print(f"用时: {result['time']:.2f}秒")
    else:
        print(f"未知模式: {args.mode}")

if __name__ == '__main__':
    sys.exit(main()) 