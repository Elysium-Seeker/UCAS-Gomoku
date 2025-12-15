#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef _WIN32
#include <windows.h>
#endif
#include "board.h"
#include "game.h"
#include "rules.h"
int main() {
    #ifdef _WIN32
    SetConsoleOutputCP(65001); // 设置控制台为 UTF-8 编码，防止乱码，上网查的
    #endif
    Board board;
    init_board(&board);
    int current_player = BLACK; // 黑先
    int game_over = 0;
    char input[10];
    while (!game_over) {
        print_board(&board); // 打印棋盘
        if (current_player == BLACK) printf("黑方落子在： ");
            else printf("白方落子在： ");
        if (fgets(input, sizeof(input), stdin) == NULL) break;
        // 移除换行符（C语言输入真麻烦，不如C++）
        input[strcspn(input, "\n")] = 0;
        int x, y;
        // 解析玩家输入的坐标是否合法
        if (!parse_input(input, &x, &y)) {
            printf("输入格式无效\n");
            printf("按任意键继续...");
            getchar();
            continue;
        }
        // 检查位置是否已被占用
        if (!is_empty(&board, x, y)) {
            printf("该位置已被占用。\n");
            printf("按任意键继续...");
            getchar();
            continue;
        }
        // 检查黑棋禁手
        if (current_player == BLACK) {
            if (is_forbidden(&board, x, y)) {
                printf("禁手！黑方落子无效，黑方输。\n");
                game_over = 1;
                // 放置棋子以显示禁手位置
                place_piece(&board, x, y, BLACK);
                print_board(&board);
                printf("游戏结束。白方因禁手获胜！\n");
                break;
            }
        }
        // 落子
        place_piece(&board, x, y, current_player);
        // 检查获胜
        if (check_win(&board, x, y, current_player)) {
            print_board(&board);
            if (current_player == BLACK) printf("黑方获胜！\n");
                else printf("白方获胜！\n");
            game_over = 1;
            break;
        }
        // 检查平局
        if (check_draw(&board)) {
            print_board(&board);
            printf("平局！\n");
            game_over = 1;
            break;
        }
        // 切换回合
        current_player = (current_player == BLACK) ? WHITE : BLACK;
    }
    
    printf("按回车键退出...");//避免直接return 0导致程序闪退
    getchar();

    return 0;
}
