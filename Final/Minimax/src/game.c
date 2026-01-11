#include <stdio.h>
#include <ctype.h>
#include <string.h>
#include "game.h"

// 计算连续棋子数量
int count_consecutive(Board *board, int x, int y, int dx, int dy, int piece) {
    int count = 0;
    int vx = x + dx;
    int vy = y + dy;
    while (is_valid_pos(vx, vy) && board->grid[vx][vy] == piece) {
        count++;
        vx += dx;
        vy += dy;
    }
    return count;
}

int dir[4][2] = {{0, 1}, {1, 0}, {1, 1}, {1, -1}};// 方向：水平 (0,1), 垂直 (1,0), 对角线 (1,1), 反对角线 (1,-1)

// 检查是否五连
int check_five_in_a_row(Board *board, int x, int y, int piece) {
    for (int i = 0; i < 4; i++) {
        int dx = dir[i][0];
        int dy = dir[i][1];
        int count = 1; // 当前棋子
        count += count_consecutive(board, x, y, dx, dy, piece);
        count += count_consecutive(board, x, y, -dx, -dy, piece);
        if (piece == BLACK) {
            if (count == 5) return 1;
        } else {
            if (count >= 5) return 1;
        }
    }
    return 0;
}

// 检查获胜
int check_win(Board *board, int x, int y, int piece) {
    return check_five_in_a_row(board, x, y, piece);
}

// 检查平局
int check_draw(Board *board) {
    for (int i = 0; i < BOARD_SIZE; i++) 
        for (int j = 0; j < BOARD_SIZE; j++)
            if (board->grid[i][j] == EMPTY) return 0;
    return 1;
}

// 解析用户输入
// 将 "H8" 格式的字符串转换为坐标 (x, y)
// 返回 1 表示解析成功，0 表示失败
int parse_input(char *input, int *x, int *y) {
    if (input == NULL || strlen(input) < 2) return 0;
    char letter = toupper(input[0]);//转大写
    if (letter < 'A' || letter > 'O') return 0;
    *y = letter - 'A';
    int num;
    if (sscanf(input + 1, "%d", &num) != 1) return 0;
    if (num < 1 || num > 15) return 0;
    *x = num - 1; // 0起始！
    return 1;
}
