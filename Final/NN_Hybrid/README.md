# 五子棋 AI - 神经网络混合版 (NN_Hybrid)

> **版本**: v1.0 (2026-01-08)  
> **作者**: Elysium-Seeker & Gemini 3 Pro (Preview)

---

## 📋 项目简介

这是一个功能完整的五子棋程序，使用标准 C 语言 (C99) 编写。本项目重点集成了一个基于**深度强化学习**的 AI 引擎，采用 **AlphaZero 风格**的神经网络 + MCTS 方案。

项目实现了标准的五子棋规则，包括黑棋的禁手规则（三三禁手、四四禁手、长连禁手）。

> 💡 虽然训练脚本是 Python 的，但最终的 `model_weights.h` 和 `ai_engine.c` 都是纯 C 实现，无需 Python 环境即可运行。

---

## 🚀 快速开始

### 编译
```bash
mingw32-make
```

### 运行
```bash
.\gomoku.exe
```

---

## 🎮 游戏模式

| 模式 | 说明 |
|------|------|
| **1. PvP** | 双人对战 |
| **2. PvE (执黑)** | 玩家先手，AI 后手 |
| **3. PvE (执白)** | AI 先手，玩家后手 |
| **4. EvE** | AI 自我对战 |

**落子**: 输入坐标如 `H8`、`A1`、`O15`

---

## 🧠 AI 核心技术

本项目的核心亮点是实现了一个轻量级的 **AlphaZero 风格** AI。采用神经网络 + MCTS 方案，而非传统 Minimax。

### 1. 神经网络架构 (Neural Network)

使用 **PyTorch** 构建的双头卷积神经网络 (`GomokuNet`)：

| 组件 | 说明 |
|------|------|
| **输入** | 15×15 棋盘状态矩阵 |
| **骨干网络** | 多层 Conv2d + ReLU，提取空间特征和棋型模式 |
| **策略头** | 输出 225 维向量，代表每个位置的落子概率 |
| **价值头** | 输出标量 [-1, 1]，评估当前局面胜率 |

### 2. 训练流程 (Training Pipeline)

训练代码位于 `ai_training/train.py`：

1. **自我对弈 (Self-Play)**: MCTS 结合神经网络预测进行大量自我对弈
2. **数据生成**: 保存每局棋谱（状态、MCTS概率、胜负结果）
3. **模型更新**: 梯度下降最小化策略损失和价值损失

### 3. 模型部署到 C 语言

实现了**跨语言模型移植**：

| 步骤 | 工具 | 输出 |
|------|------|------|
| 权重导出 | `export_weights.py` | `model_weights.h` |
| C推理引擎 | 手动实现 | `ai_engine.c` |
| 零依赖 | 纯 C 编译 | 无需 Python 运行时 |

---

## 📂 文件结构

```
NN_Hybrid/
├── src/
│   ├── main.c          # 主程序入口
│   ├── board.c/h       # 棋盘显示
│   ├── game.c/h        # 游戏逻辑
│   ├── rules.c/h       # 禁手规则
│   ├── ai_engine.c/h   # 神经网络推理引擎
│   └── model_weights.h # 导出的模型权重
├── ai_training/
│   ├── train.py        # 训练脚本
│   ├── export_weights.py # 权重导出
│   └── gomoku_model.pth  # 训练好的模型
├── Makefile
└── README.md
```

---

## ⚠️ 注意事项

- 神经网络版本文件较大（~15MB），主要是 `model_weights.h` 包含大量浮点权重
- 如需重新训练，需要 Python 环境和 PyTorch
- 推荐使用 Minimax 版本，更轻量且棋力相当

---

## 👤 作者

Elysium-Seeker & Gemini 3 Pro (Preview)
