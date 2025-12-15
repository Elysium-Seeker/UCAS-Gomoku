# 五子棋 (Gomoku)

## 项目简介
这是一个基于命令行的人人对战版五子棋程序，使用标准 C 语言 (C99) 编写。
项目实现了标准的五子棋规则，包括黑棋的禁手规则（三三禁手、四四禁手、长连禁手）。

## 编译与运行

### 编译环境
- Windows (MinGW/GCC) 或 Linux (GCC)
- 建议使用 GCC 编译器

### 编译命令
在项目根目录下运行以下命令：
```bash
gcc -Wall -Wextra -std=c99 -I./src src/main.c src/board.c src/game.c src/rules.c -o gomoku.exe
```
或者如果安装了 `make` (Windows 上通常是 `mingw32-make`)，可以直接运行：
```bash
make
# 或者
mingw32-make
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

## 操作说明
1. 游戏开始后，黑棋先手。
2. 输入坐标进行落子，格式为 **字母+数字** (例如 `H8`, `A1`, `O15`)。
   - 字母 A-O 代表列。
   - 数字 1-15 代表行。
   - 输入不区分大小写。
3. 黑棋落子时，程序会自动检查是否为禁手。如果是禁手，黑棋直接判负。
4. 率先连成 5 子的一方获胜（黑棋若同时形成五连和禁手，算黑胜）。

## 功能列表
- [x] 15x15 标准棋盘显示
- [x] 命令行交互界面
- [x] 坐标输入解析
- [x] 黑白轮流落子
- [x] 胜负判断 (五连)
- [x] 平局判断 (棋盘满)
- [x] **禁手规则实现**:
    - 长连禁手 (黑棋 > 5 子)
    - 四四禁手 (双四)
    - 三三禁手 (双活三)//这部分的思路是Gemini提供的
- [x] 最后一步落子特殊标记 (▲/△)

## 文件结构
- `src/main.c`: 主程序入口，游戏循环。
- `src/board.c/h`: 棋盘数据结构及显示函数。
- `src/game.c/h`: 游戏逻辑（胜负判断、输入解析）。
- `src/rules.c/h`: 禁手规则判断逻辑。
- `Makefile`: 编译脚本。
- `README.md`: 项目说明文档。

## 作者
Elysium-Seeker & Gemini 3 Pro (Preview)

这个文档基本也是 Gemini 大人写的。
