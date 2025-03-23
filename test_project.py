#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import subprocess
import argparse
import torch
import numpy as np
import random
from datetime import datetime

# 导入项目模块
try:
    from environment.minesweeper import MinesweeperEnv
    from agent.dqn_agent import DQNAgent, DQNNetwork
    from agent.human_reasoning import HumanReasoning
    from training.dqn_trainer import DQNTrainer
    from utils.helpers import set_seed
    print("✅ 项目模块导入成功")
except ImportError as e:
    print(f"❌ 模块导入失败: {str(e)}")
    exit(1)

def print_header(text):
    """打印带有格式的标题"""
    print("\n" + "=" * 80)
    print(f" {text} ".center(80, "="))
    print("=" * 80)

def print_section(text):
    """打印带有格式的节标题"""
    print("\n" + "-" * 80)
    print(f" {text} ".center(80, "-"))
    print("-" * 80)

def run_command(cmd, desc=None):
    """运行一个命令并返回结果"""
    if desc:
        print(f"➤ {desc}")
    
    print(f"执行: {cmd}")
    start_time = time.time()
    
    try:
        # 设置encoding和errors参数处理编码问题
        result = subprocess.run(
            cmd, 
            shell=True, 
            capture_output=True, 
            text=True,
            encoding='utf-8',  # 使用UTF-8编码
            errors='replace'   # 替换无法解码的字符
        )
        elapsed = time.time() - start_time
        
        if result.returncode == 0:
            print(f"✅ 命令成功完成 ({elapsed:.2f}秒)")
            return True, result.stdout
        else:
            print(f"❌ 命令失败 ({elapsed:.2f}秒)")
            print(f"错误信息: {result.stderr}")
            return False, result.stderr
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"❌ 执行错误 ({elapsed:.2f}秒): {str(e)}")
        return False, str(e)

def test_dependencies():
    """测试项目依赖是否已安装"""
    print_section("依赖测试")
    
    dependencies = [
        "torch", "numpy", "matplotlib", "gymnasium"
    ]
    
    all_ok = True
    for dep in dependencies:
        try:
            __import__(dep)
            print(f"✅ {dep} 已安装")
        except ImportError:
            print(f"❌ {dep} 未安装")
            all_ok = False
    
    return all_ok

def test_environment():
    """测试扫雷环境模块"""
    print_section("环境模块测试")
    
    try:
        # 创建环境
        env = MinesweeperEnv(width=5, height=5, num_mines=3)
        print("✅ 环境创建成功")
        
        # 测试环境重置
        state, _ = env.reset()
        print(f"✅ 环境重置成功，状态形状: {state['board'].shape}")
        
        # 测试执行动作
        action = env.action_space.sample()
        next_state, reward, done, _, info = env.step(action)
        print(f"✅ 动作执行成功，奖励: {reward}, 完成: {done}")
        
        # 测试难度设置
        env.set_difficulty(5)
        print(f"✅ 难度设置成功: {env.get_difficulty()}")
        
        # 测试渲染
        env.render(disable_text_output=True)
        print("✅ 渲染测试成功")
        
        return True
    except Exception as e:
        print(f"❌ 环境测试失败: {str(e)}")
        return False

def test_agent():
    """测试DQN智能体模块"""
    print_section("智能体模块测试")
    
    try:
        # 创建环境
        env = MinesweeperEnv(width=5, height=5, num_mines=3)
        state, _ = env.reset()
        board_shape = state['board'].shape
        
        # 创建智能体 - 修复参数
        agent = DQNAgent(
            height=board_shape[0],
            width=board_shape[1],
            use_human_reasoning=True
        )
        print("✅ 智能体创建成功")
        
        # 测试动作选择
        action_mask = env.get_valid_actions()
        action = agent.act(state, action_mask)
        print(f"✅ 动作选择成功: {action}")
        
        # 测试经验记忆
        next_state, reward, done, _, _ = env.step(action)
        agent.remember(state, action, reward, next_state, done, action_mask)
        print("✅ 经验记忆成功")
        
        # 测试预训练（跳过实际训练，仅测试函数调用）
        if hasattr(agent, 'pretrain'):
            # 添加一些假的预训练数据
            for _ in range(10):
                agent.pretrain_memory.append((np.random.rand(4, 5, 5), np.random.randint(0, 25)))
            
            # 只运行一个快速的epoch进行测试
            agent.pretrain(num_epochs=1, batch_size=2)
            print("✅ 预训练函数测试成功")
        
        # 测试模型保存/加载
        if not os.path.exists('test_models'):
            os.makedirs('test_models')
        
        test_model_path = 'test_models/test_agent.pth'
        agent.save_model(test_model_path)
        print(f"✅ 模型保存成功: {test_model_path}")
        
        try:
            agent.load_model(test_model_path)
            print("✅ 模型加载成功")
        except Exception as e:
            print(f"❌ 模型加载失败: {str(e)}")
        
        # 测试人类推理模块
        if hasattr(agent, 'human_reasoning'):
            action = agent.human_reasoning.find_safe_move(agent.preprocess_state(state['board']), action_mask)
            print(f"✅ 人类推理模块测试成功: {action}")
        
        return True
    except Exception as e:
        print(f"❌ 智能体测试失败: {str(e)}")
        return False

def test_training():
    """测试训练模块的基本功能"""
    print_section("训练模块测试")
    
    try:
        # 创建环境和智能体
        env = MinesweeperEnv(width=5, height=5, num_mines=2)
        agent = DQNAgent(
            height=5,
            width=5,
            use_human_reasoning=True
        )
        
        # 创建训练器
        trainer = DQNTrainer(env=env, agent=agent, models_dir='test_models')
        print("✅ 训练器创建成功")
        
        # 训练极少量回合以测试功能
        print("开始微型训练(3回合)...")
        results = trainer.train(
            num_episodes=3,  # 极少量回合，仅测试功能
            initial_difficulty=2,
            max_difficulty=3,
            difficulty_increase_threshold=0.9,  # 设高一点避免增加难度
            eval_freq=5,  # 这里不会触发评估
            save_freq=5,  # 这里不会触发保存
            render=False,
            verbose=True,
            enable_human_reasoning=True
        )
        print("✅ 微型训练完成")
        
        # 测试评估功能
        print("测试评估功能...")
        reward, win_rate = trainer.evaluate(num_episodes=3, use_human_reasoning=True)
        print(f"✅ 评估功能测试成功: 奖励={reward:.2f}, 胜率={win_rate:.2f}")
        
        # 测试比较功能（极少量回合，仅测试功能是否正常）
        print("测试方法比较功能...")
        comparison_results = trainer.compare_methods(num_episodes=2, verbose=True)
        print("✅ 比较功能测试成功")
        
        return True
    except Exception as e:
        print(f"❌ 训练模块测试失败: {str(e)}")
        return False

def test_cli():
    """测试命令行接口"""
    print_section("命令行接口测试")
    
    # 测试训练模式（极少量回合）
    success, output = run_command(
        "python main.py --mode train --num_episodes 3 --initial_difficulty 2 --verbose",
        "测试训练模式"
    )
    
    # 测试评估模式
    success, output = run_command(
        "python main.py --mode eval --eval_episodes 3",
        "测试评估模式"
    )
    
    # 测试比较模式
    success, output = run_command(
        "python main.py --mode compare --compare_episodes 2 --verbose",
        "测试比较模式"
    )
    
    return success

def run_performance_benchmark():
    """运行性能基准测试"""
    print_section("性能基准测试")
    
    # 创建环境和智能体
    try:
        env = MinesweeperEnv(width=5, height=5, num_mines=3)
        agent = DQNAgent(
            height=5,
            width=5,
            use_human_reasoning=True
        )
        
        # 测试推理速度
        print("测试推理速度...")
        state, _ = env.reset()
        action_mask = env.get_valid_actions()
        
        # 预热
        for _ in range(10):
            agent.act(state, action_mask)
        
        # 计时
        start_time = time.time()
        iterations = 100
        for _ in range(iterations):
            agent.act(state, action_mask)
        elapsed = time.time() - start_time
        
        print(f"✅ 推理性能: 平均每次推理耗时 {(elapsed / iterations) * 1000:.2f} 毫秒")
        
        # 测试人类推理模块速度
        if hasattr(agent, 'human_reasoning'):
            print("测试人类推理模块速度...")
            processed_state = agent.preprocess_state(state['board'])
            
            # 预热
            for _ in range(10):
                agent.human_reasoning.find_safe_move(processed_state, action_mask)
            
            # 计时
            start_time = time.time()
            iterations = 100
            for _ in range(iterations):
                agent.human_reasoning.find_safe_move(processed_state, action_mask)
            elapsed = time.time() - start_time
            
            print(f"✅ 人类推理性能: 平均每次推理耗时 {(elapsed / iterations) * 1000:.2f} 毫秒")
        
        return True
    except Exception as e:
        print(f"❌ 性能测试失败: {str(e)}")
        return False

def generate_report(results):
    """生成测试报告"""
    print_header("测试报告")
    
    total_tests = len(results)
    passed_tests = sum(1 for success, _ in results.values() if success)
    failed_tests = total_tests - passed_tests
    
    print(f"总测试数: {total_tests}")
    print(f"通过: {passed_tests} ({(passed_tests / total_tests) * 100:.1f}%)")
    print(f"失败: {failed_tests} ({(failed_tests / total_tests) * 100:.1f}%)")
    print("\n详细测试结果:")
    
    for name, (success, message) in results.items():
        status = "✅ 通过" if success else "❌ 失败"
        print(f"{status} - {name}")
        if not success and message:
            print(f"  错误信息: {message}")
    
    # 保存报告到文件
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = f"test_report_{timestamp}.txt"
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("扫雷AI项目测试报告\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write(f"总测试数: {total_tests}\n")
        f.write(f"通过: {passed_tests} ({(passed_tests / total_tests) * 100:.1f}%)\n")
        f.write(f"失败: {failed_tests} ({(failed_tests / total_tests) * 100:.1f}%)\n\n")
        
        f.write("详细测试结果:\n")
        for name, (success, message) in results.items():
            status = "通过" if success else "失败"
            f.write(f"{status} - {name}\n")
            if not success and message:
                f.write(f"  错误信息: {message}\n")
    
    print(f"\n测试报告已保存到: {report_path}")

def main():
    """运行测试套件"""
    parser = argparse.ArgumentParser(description="扫雷AI项目测试套件")
    parser.add_argument("--all", action="store_true", help="运行所有测试")
    parser.add_argument("--dependencies", action="store_true", help="测试项目依赖")
    parser.add_argument("--environment", action="store_true", help="测试环境模块")
    parser.add_argument("--agent", action="store_true", help="测试智能体模块")
    parser.add_argument("--training", action="store_true", help="测试训练模块")
    parser.add_argument("--cli", action="store_true", help="测试命令行接口")
    parser.add_argument("--benchmark", action="store_true", help="运行性能基准测试")
    parser.add_argument("--seed", type=int, default=42, help="设置随机种子")
    
    args = parser.parse_args()
    
    # 如果没有指定任何测试，则默认运行所有测试
    if not any([args.all, args.dependencies, args.environment, args.agent, 
                args.training, args.cli, args.benchmark]):
        args.all = True
    
    # 设置随机种子
    set_seed(args.seed)
    print(f"随机种子已设置为: {args.seed}")
    
    print_header("扫雷AI项目测试套件")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    results = {}
    
    # 运行指定的测试
    if args.all or args.dependencies:
        results["依赖测试"] = (test_dependencies(), None)
    
    if args.all or args.environment:
        results["环境模块"] = (test_environment(), None)
    
    if args.all or args.agent:
        results["智能体模块"] = (test_agent(), None)
    
    if args.all or args.training:
        results["训练模块"] = (test_training(), None)
    
    if args.all or args.cli:
        success, output = run_command("python --version", "检查Python版本")
        results["命令行接口"] = (test_cli(), None)
    
    if args.all or args.benchmark:
        results["性能基准"] = (run_performance_benchmark(), None)
    
    # 生成报告
    generate_report(results)
    
    print(f"\n结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    # 返回是否所有测试都通过
    return all(success for success, _ in results.values())

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 