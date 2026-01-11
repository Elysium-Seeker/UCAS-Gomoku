#include <stdio.h>
#include <stdlib.h>
#include "board.h"

// 初始化棋盘
// 将所有位置设为 EMPTY，重置历史记录
void init_board(Board *board) {
    for (int i = 0; i < BOARD_SIZE; i++) {
        for (int j = 0; j < BOARD_SIZE; j++) {
            board->grid[i][j] = EMPTY;
        }
    }
    board->last_move_row = -1;
    board->last_move_col = -1;
    board->move_count = 0;
}

// 打印棋盘
// 在控制台显示当前棋盘状态，包括坐标轴和棋子
void print_board(Board *board) {
    // 清屏（简单实现）
    #ifdef _WIN32
        system("cls");
    #else
        system("clear");
    #endif
    printf("   ");
    for (int i = 0; i < BOARD_SIZE; i++) {
        printf("%c ", 'A' + i);
    }
    printf("\n");
    for (int i = BOARD_SIZE - 1; i >= 0; i--) {
        printf("%2d ", i + 1);
        for (int j = 0; j < BOARD_SIZE; j++) {
            int piece = board->grid[i][j];
            int is_last = (i == board->last_move_row && j == board->last_move_col);
            if (piece != EMPTY) {
                // 显示棋子
                if (piece == BLACK) {
                    if (is_last) printf("▲"); 
                        else printf("●");
                } else {
                    if (is_last) printf("△");
                        else printf("○");
                }
                // 补上右侧连接线（除非是最后一列），保持与网格对齐
                if (j < BOARD_SIZE - 1) {
                    printf("─");
                }
            } else {
                // 显示棋盘网格 (占2格: 符号+横线)
                // 角落和边缘处理
                if (i == BOARD_SIZE - 1) {
                    if (j == 0) printf("┌─");
                        else if (j == BOARD_SIZE - 1) printf("┐");
                            else printf("┬─");
                } else if (i == 0) {
                    if (j == 0) printf("└─");
                        else if (j == BOARD_SIZE - 1) printf("┘");
                            else printf("┴─");
                } else {
                    if (j == 0) printf("├─");
                        else if (j == BOARD_SIZE - 1) printf("┤");
                            else printf("┼─");
                }
            }
        }
        printf("%d", i + 1);
        printf("\n");
    }
    printf("   ");
    for (int i = 0; i < BOARD_SIZE; i++) {
        printf("%c ", 'A' + i);
    }
    printf("\n");
}

// 检查坐标是否有效
// 返回 1 表示有效，0 表示越界
int is_valid_pos(int row, int col) {
    return row >= 0 && row < BOARD_SIZE && col >= 0 && col < BOARD_SIZE;
}

// 检查位置是否为空
// 返回 1 表示为空且有效，0 表示非空或越界
int is_empty(Board *board, int row, int col) {
    return is_valid_pos(row, col) && board->grid[row][col] == EMPTY;
}

// 落子
// 在指定位置放置棋子，并记录到历史记录中
void place_piece(Board *board, int row, int col, int piece) {
    if (is_valid_pos(row, col)) {
        board->grid[row][col] = piece;
        board->last_move_row = row;
        board->last_move_col = col;
        
        // 记录历史
        if (board->move_count < BOARD_SIZE * BOARD_SIZE) {
            board->history[board->move_count].row = row;
            board->history[board->move_count].col = col;
            board->history[board->move_count].player = piece;
            board->move_count++;
        }
    }
}

// 悔棋
// 撤销最后一步落子，恢复棋盘状态
// 返回 1 表示成功，0 表示无棋可悔
int undo_move(Board *board) {
    if (board->move_count <= 0) return 0;
    
    // 获取最后一步
    int idx = board->move_count - 1;
    int r = board->history[idx].row;
    int c = board->history[idx].col;
    
    // 清空棋盘位置
    board->grid[r][c] = EMPTY;
    board->move_count--;
    
    // 更新最后一步指示器
    if (board->move_count > 0) {
        board->last_move_row = board->history[board->move_count - 1].row;
        board->last_move_col = board->history[board->move_count - 1].col;
    } else {
        board->last_move_row = -1;
        board->last_move_col = -1;
    }
    
    return 1;
}
