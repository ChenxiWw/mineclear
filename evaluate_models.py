import os
import subprocess
import re

# 模型目录
models_dir = "models"
# 评估难度
difficulty = 5
# 每个模型评估的回合数
episodes = 20

# 获取所有模型文件
model_files = [f for f in os.listdir(models_dir) if f.endswith(".pth")]

# 结果存储
results = []

# 评估每个模型
for model in model_files:
    model_path = os.path.join(models_dir, model)
    print(f"评估模型: {model}")

    # 运行评估命令
    cmd = f"python main.py --mode eval --model_path {model_path} --eval_episodes {episodes} --eval_difficulty {difficulty}"
    output = subprocess.check_output(cmd, shell=True).decode("utf-8")

    # 从输出中提取胜率
    match = re.search(r"胜率: ([\d\.]+)", output)
    if match:
        win_rate = float(match.group(1))
        results.append((model, win_rate))
        print(f"  胜率: {win_rate:.2f}")
    else:
        print(f"  无法提取胜率")

# 按胜率排序
results.sort(key=lambda x: x[1], reverse=True)

# 打印结果
print("\n模型性能排名:")
print("=" * 50)
print(f"{'模型名称':<30} {'胜率':<10}")
print("-" * 50)
for model, win_rate in results:
    print(f"{model:<30} {win_rate:<10.2f}")