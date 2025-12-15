# 五子棋 (Gomoku)

## 项目简介
这是一个功能完整的五子棋程序，使用标准 C 语言 (C99) 编写。除了基础的人人对战功能外，本项目重点集成了一个基于**深度强化学习**的 AI 引擎，支持人机对战和 AI 自我对战。

项目实现了标准的五子棋规则，包括黑棋的禁手规则（三三禁手、四四禁手、长连禁手）。

理论上来讲是可以交作业的，虽然训练的脚本是 Python 的但是最后的 model_weights 和 ai_engine 都是 C 的。

## 编译与运行

### 编译环境
- Windows (MinGW/GCC) 或 Linux (GCC)
- 建议使用 GCC 编译器

### 编译命令
在项目根目录下运行以下命令：
```bash
gcc -Wall -Wextra -std=c99 -I./src src/main.c src/board.c src/game.c src/rules.c src/ai_engine.c -o gomoku.exe
```
或者如果安装了 `make`，可以直接运行：
```bash
make
```

### 运行
Windows:
```bash
.\gomoku.exe
```
Linux/Mac:
```bash
./gomoku
```

## 游戏模式
程序启动后，您可以选择以下三种模式：
1.  **PvP (Player vs Player)**: 经典的双人对弈模式，轮流落子。
2.  **PvE (Player vs AI)**: 挑战 AI。您可以选择：
    -   **执黑 (Black)**: 玩家先手。
    -   **执白 (White)**: 玩家后手，AI 先手。
3.  **EvE (AI vs AI)**: 观赏模式，两个 AI 互搏，用于测试 AI 强度或观察策略。

## AI 训练与技术实现 (核心亮点)
本项目的一大特色是实现了一个轻量级的 **AlphaZero 风格** AI。我们没有使用传统的极大极小值搜索 (Minimax) 加 Alpha-Beta 剪枝，而是采用了更先进的神经网络加蒙特卡洛树搜索 (MCTS)。

### 1. 神经网络架构 (Neural Network)
我们在 `ai_training/` 目录下使用 **PyTorch** 构建了一个双头卷积神经网络 (`GomokuNet`)：
-   **输入**: 15x15 的棋盘状态矩阵。
-   **骨干网络 (Backbone)**: 包含多层卷积层 (`Conv2d`) 和 ReLU 激活函数，用于提取棋盘的空间特征和棋型模式。
-   **双头输出 (Dual Head)**:
    -   **策略头 (Policy Head)**: 输出一个 225 维的向量，代表在棋盘每一个位置落子的概率分布。
    -   **价值头 (Value Head)**: 输出一个标量 (范围 -1 到 1)，代表当前局面的胜率评估（1 为黑胜，-1 为白胜）。

### 2. 训练流程 (Training Pipeline)
训练过程完全自动化，代码位于 `ai_training/train.py`：
-   **自我对弈 (Self-Play)**: AI 使用 MCTS 算法进行大量的自我对弈。在每一步中，MCTS 结合神经网络的预测（先验概率）和模拟结果（实际价值）来决定落子。
-   **数据生成**: 每一局对弈的棋谱（状态、MCTS 搜索出的概率、最终胜负）被保存为训练数据。
-   **模型更新**: 神经网络通过梯度下降更新权重，目标是最小化：
    -   **策略损失**: 预测概率与 MCTS 搜索概率的差异。
    -   **价值损失**: 预测胜率与实际对局结果的均方误差。

### 3. 模型部署 (Deployment to C)
为了在纯 C 语言环境中高效运行深度学习模型，我们实现了**跨语言模型移植**：
1.  **权重导出**: 使用 `export_weights.py` 脚本，将 PyTorch 训练好的模型参数（权重和偏置）提取并转换为 C 语言头文件 (`src/model_weights.h`)。
2.  **C 语言推理引擎**: 在 `src/ai_engine.c` 中，我们手动实现了卷积层 (Convolution)、全连接层 (Linear) 和激活函数的前向传播算法。
3.  **零依赖**: 这种方法使得编译后的程序极其轻量，无需依赖庞大的 PyTorch C++ 库或 Python 环境即可运行 AI。

## 操作说明
1.  游戏开始后，根据提示选择模式。
2.  输入坐标进行落子，格式为 **字母+数字** (例如 `H8`, `A1`, `O15`)。
    -   字母 A-O 代表列。
    -   数字 1-15 代表行。
    -   输入不区分大小写。
3.  黑棋落子时，程序会自动检查是否为禁手。如果是禁手，黑棋直接判负。

## 功能列表
- [x] 15x15 标准棋盘显示
- [x] 三种游戏模式 (PvP, PvE, EvE)
- [x] **深度学习 AI 引擎**
- [x] 禁手规则实现 (三三、四四、长连)

## 文件结构
- `src/main.c`: 主程序入口，游戏循环。
- `src/board.c/h`: 棋盘数据结构及显示函数。
- `src/game.c/h`: 游戏逻辑（胜负判断、输入解析）。
- `src/rules.c/h`: 禁手规则判断逻辑。
- `src/ai_engine.c/h`: C 语言实现的神经网络推理引擎。
- `src/model_weights.h`: 导出的模型权重。
- `ai_training/`: Python 训练代码。
- `Makefile`: 编译脚本。
- `README.md`: 项目说明文档。

## 作者
Elysium-Seeker & Gemini 3 Pro(Preview)
