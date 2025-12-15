#ifndef AI_ENGINE_H
#define AI_ENGINE_H

#include "board.h"

// AI 思考函数
// 输入当前棋盘，输出 AI 决定的落子位置 (row, col)
void get_ai_move(Board *board, int *row, int *col, int ai_color);

#endif
