#include <stdio.h>
#include <ctype.h>
#include <string.h>
#include "game.h"

// 计算连续棋子数量
// 从 (row, col) 开始，沿着 (d_row, d_col) 方向统计连续的 piece 数量
int count_consecutive(Board *board, int row, int col, int d_row, int d_col, int piece) {
    int count = 0;
    int r = row + d_row;
    int c = col + d_col;
    while (is_valid_pos(r, c) && board->grid[r][c] == piece) {
        count++;
        r += d_row;
        c += d_col;
    }
    return count;
}

int dir[4][2] = {{0, 1}, {1, 0}, {1, 1}, {1, -1}};// 方向：水平 (0,1), 垂直 (1,0), 对角线 (1,1), 反对角线 (1,-1)

// 检查是否五连
// 检查四个方向上是否有连续 5 个（或更多，对于白棋）棋子
int check_five_in_a_row(Board *board, int row, int col, int piece) {
    for (int i = 0; i < 4; i++) {
        int d_row = dir[i][0];
        int d_col = dir[i][1];
        int count = 1; // 当前棋子
        count += count_consecutive(board, row, col, d_row, d_col, piece);
        count += count_consecutive(board, row, col, -d_row, -d_col, piece);
        if (piece == BLACK) {
            if (count == 5) return 1;
        } else {
            if (count >= 5) return 1;
        }
    }
    return 0;
}

// 检查获胜
// 判断当前落子是否导致获胜
int check_win(Board *board, int row, int col, int piece) {
    return check_five_in_a_row(board, row, col, piece);
}

// 检查平局
// 如果棋盘已满且无人获胜，则为平局
int check_draw(Board *board) {
    for (int i = 0; i < BOARD_SIZE; i++) 
        for (int j = 0; j < BOARD_SIZE; j++)
            if (board->grid[i][j] == EMPTY) return 0;
    return 1;
}

// 解析用户输入
// 将 "H8" 格式的字符串转换为坐标 (row, col)
// 返回 1 表示解析成功，0 表示失败
int parse_input(char *input, int *row, int *col) {
    // 预期格式：字母 + 数字 (例如 H7, A15)
    // 不区分大小写
    if (input == NULL || strlen(input) < 2) return 0;
    char letter = toupper(input[0]);
    if (letter < 'A' || letter > 'O') return 0;
    *col = letter - 'A';
    int num;
    if (sscanf(input + 1, "%d", &num) != 1) return 0;
    if (num < 1 || num > 15) return 0;
    *row = num - 1; // 0起始！
    return 1;
}

// 保存游戏
// 将当前棋局历史记录保存到文件
void save_game(Board *board, const char *filename) {
    FILE *f = fopen(filename, "w");
    if (!f) {
        printf("Error: Could not open file %s for writing.\n", filename);
        return;
    }
    
    fprintf(f, "Gomoku Game Record\n");
    fprintf(f, "------------------\n");
    
    for (int i = 0; i < board->move_count; i++) {
        int r = board->history[i].row;
        int c = board->history[i].col;
        int p = board->history[i].player;
        const char *player_name = (p == BLACK) ? "Black" : "White";
        fprintf(f, "%d. %s: %c%d\n", i + 1, player_name, 'A' + c, r + 1);
    }
    
    fclose(f);
    printf("Game saved to %s\n", filename);
}
