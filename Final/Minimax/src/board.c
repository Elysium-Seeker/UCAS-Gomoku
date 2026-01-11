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
    board->last_move_x = -1;
    board->last_move_y = -1;
    board->move_count = 0;
}

// 打印棋盘
// 在控制台显示当前棋盘状态，包括坐标轴和棋子
void print_board(Board *board) {
    // Clear screen disabled for log visibility
    // #ifdef _WIN32
    //     system("cls");
    // #else
    //     system("clear");
    // #endif
    printf("   ");
    for (int i = 0; i < BOARD_SIZE; i++) {
        printf("%c ", 'A' + i);
    }
    printf("\n");
    for (int i = BOARD_SIZE - 1; i >= 0; i--) {
        printf("%2d ", i + 1);
        for (int j = 0; j < BOARD_SIZE; j++) {
            int piece = board->grid[i][j];//表示格点状态
            int is_last = (i == board->last_move_x && j == board->last_move_y);
            if (piece != EMPTY) {// 有棋
                if (piece == BLACK) {
                    if (is_last) printf("▲"); 
                        else printf("●");
                } else {
                    if (is_last) printf("△");
                        else printf("○");
                }
                // 补上右侧连接线（除非是最后一列）
                if (j < BOARD_SIZE - 1) {
                    printf("─");
                }
            } else {//无棋
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
int is_valid_pos(int x, int y) {
    return x >= 0 && x < BOARD_SIZE && y >= 0 && y < BOARD_SIZE;
}

// 检查位置是否为空
int is_empty(Board *board, int x, int y) {
    return is_valid_pos(x, y) && board->grid[x][y] == EMPTY;
}

// 落子
// 在指定位置放置棋子，并记录到历史记录中
void place_piece(Board *board, int x, int y, int piece) {
    if (is_valid_pos(x, y)) {
        board->grid[x][y] = piece;
        board->last_move_x = x;
        board->last_move_y = y;
        board->move_count++;
    }
}
