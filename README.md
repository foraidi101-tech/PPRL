# Multi-Agent Path Planning: A* + RL

一个可运行的示例项目，目标是把：

- 全局路径规划（A*）
- 局部动态避障（Q-learning）

结合到同一个多智能体网格环境中。

## 1. 项目结构

- `main.py`：程序入口（训练 + 评估 + 可视化）
- `planner/config.py`：超参数
- `planner/grid_map.py`：地图、障碍、采样
- `planner/astar.py`：A* 全局规划
- `planner/rl_local_avoider.py`：局部避障 Q-learning
- `planner/multi_agent_system.py`：多智能体融合调度
- `planner/simulation.py`：训练、评估、轨迹绘制

## 2. 运行方式

```bash
pip install -r requirements.txt
python main.py
```

运行后会看到：

1. 训练日志（epsilon 递减）
2. 评估结果（到达率、总回报）
3. Matplotlib 轨迹图（不同颜色代表不同智能体）

## 3. 核心思路

- A* 负责给每个智能体提供静态地图下的全局参考路径。
- RL 负责在每一步处理局部冲突（障碍/其他智能体冲突风险）。
- 如果 RL 动作无效（越界或撞障碍），系统回退到 A* 下一路标点，提升稳定性。

## 4. 可扩展建议

- 把 Q-learning 换成 DQN/PPO（可接入 PyTorch）。
- 加入时间维约束做真正的多智能体冲突消解（CBS、优先级规划等）。
- 增加动态障碍和部分可观测传感器。
- 将奖励函数拆分为更细粒度的多目标权重。

