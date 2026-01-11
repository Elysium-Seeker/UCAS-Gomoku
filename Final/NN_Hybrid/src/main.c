#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifdef _WIN32
#include <windows.h>
#endif
#include "board.h"
#include "game.h"
#include "rules.h"
#include "ai_engine.h"

// 主函数
// 游戏程序的入口点，负责初始化游戏、处理用户输入、管理游戏循环和模式选择
int main() {
    srand(time(NULL)); // 初始化随机数种子
    #ifdef _WIN32
    SetConsoleOutputCP(65001); // 设置控制台为 UTF-8 编码，防止乱码
    #endif

    int play_again = 1;
    // 游戏主循环，支持“再来一局”功能
    while (play_again) {
        Board board;
        init_board(&board); // 初始化棋盘        
        // AI 思考时间统计
        double total_ai_time = 0.0;
        int ai_move_count = 0;
        int mode = 0;
        int ai_player = 0; // 0: 无AI, 1: AI执黑, 2: AI执白, 3: 双方AI

        // 模式选择菜单
        printf("Select Mode:\n1. PvP (Player vs Player)\n2. PvE (Player vs AI)\n3. EvE (AI vs AI)\n> ");
        char mode_input[10];
        fgets(mode_input, sizeof(mode_input), stdin);
        if (mode_input[0] == '2') {
            mode = 2;
            printf("Select your color:\n1. Black (First)\n2. White (Second)\n> ");
            char color_input[10];
            fgets(color_input, sizeof(color_input), stdin);
            if (color_input[0] == '2') {
                // 玩家选白棋，AI 执黑
                ai_player = BLACK;
            } else {
                // 玩家选黑棋（默认），AI 执白
                ai_player = WHITE;
            }
        } else if (mode_input[0] == '3') {
            mode = 3;
            ai_player = 3; // 双方都是 AI
        } else {
            mode = 1; // 默认 PvP
        }

        int current_player = BLACK; // 黑棋先行
        int game_over = 0;
        char input[10];

        // 单局游戏循环
        while (!game_over) {
            print_board(&board); // 打印棋盘

            // AI 回合处理逻辑
            if ((mode == 2 && current_player == ai_player) || (mode == 3)) {
                const char* color_name = (current_player == BLACK) ? "Black" : "White";
                printf("%s's turn (AI) is thinking...\n", color_name);
                
                // EvE 模式下增加延迟，方便观察
                if (mode == 3) {
                    #ifdef _WIN32
                    Sleep(500); // 延迟 0.5 秒
                    #else
                    usleep(500000);
                    #endif
                }

                int r, c;
                // 记录 AI 开始思考时间
                clock_t ai_start = clock();
                
                // 获取 AI 的落子位置
                get_ai_move(&board, &r, &c, current_player);
                
                // 计算 AI 思考时长
                clock_t ai_end = clock();
                double ai_duration = (double)(ai_end - ai_start) / CLOCKS_PER_SEC;
                total_ai_time += ai_duration;
                ai_move_count++;
                
                printf("AI plays at %c%d\n", 'A' + c, r + 1);
                printf("Time: %.2fs (Avg: %.2fs)\n", ai_duration, total_ai_time / ai_move_count);

                // AI 执黑时，理论上也需要检查禁手（虽然 AI 应该避免禁手）
                if (current_player == BLACK && is_forbidden(&board, r, c)) {
                    printf("AI made a forbidden move at %c%d! AI loses.\n", 'A' + c, r + 1);
                    game_over = 1;
                    place_piece(&board, r, c, BLACK);
                    print_board(&board);
                    printf("Game Over. White Wins!\n");
                    break;
                }

                place_piece(&board, r, c, current_player);
                
                // 检查 AI 是否获胜
                if (check_win(&board, r, c, current_player)) {
                    print_board(&board);
                    printf("%s (AI) Wins!\n", color_name);
                    game_over = 1;
                } else if (check_draw(&board)) {
                    print_board(&board);
                    printf("Draw!\n");
                    game_over = 1;
                }
                current_player = (current_player == BLACK) ? WHITE : BLACK;
                continue;
            }

            // 玩家回合提示
            if (current_player == BLACK) printf("Black's turn (e.g., H8, 'u' to undo, 's' to save): ");
            else printf("White's turn (e.g., H8, 'u' to undo, 's' to save): ");
            
            if (fgets(input, sizeof(input), stdin) == NULL) break;
            // 移除换行符
            input[strcspn(input, "\n")] = 0;
            
            // 处理悔棋功能
            if (strcmp(input, "u") == 0 || strcmp(input, "undo") == 0) {
                if (mode == 2) {
                    // PvE 模式：悔棋需要回退两步（玩家一步 + AI 一步）
                    if (undo_move(&board)) {
                        undo_move(&board); // 尝试回退第二步
                        printf("Undid 2 moves.\n");
                    } else {
                        printf("Cannot undo.\n");
                    }
                } else {
                    // PvP 模式：悔棋回退一步
                    if (undo_move(&board)) {
                        printf("Undid 1 move.\n");
                        current_player = (current_player == BLACK) ? WHITE : BLACK;
                    } else {
                        printf("Cannot undo.\n");
                    }
                }
                continue;
            }
            
            // 处理保存功能
            if (strcmp(input, "s") == 0 || strcmp(input, "save") == 0) {
                save_game(&board, "game_record.txt");
                printf("Press Enter to continue...");
                getchar();
                continue;
            }

            int row, col;
            // 解析玩家输入的坐标
            if (!parse_input(input, &row, &col)) {
                printf("Invalid input format. Use Letter+Number (e.g., H8).\n");
                printf("Press Enter to continue...");
                getchar();
                continue;
            }
            // 检查位置是否已被占用
            if (!is_empty(&board, row, col)) {
                printf("Position already occupied.\n");
                printf("Press Enter to continue...");
                getchar();
                continue;
            }
            // 检查黑棋禁手
            if (current_player == BLACK) {
                if (is_forbidden(&board, row, col)) {
                    printf("Forbidden move at %s! Black loses.\n", input);
                    game_over = 1;
                    // 放置棋子以显示禁手位置
                    place_piece(&board, row, col, BLACK);
                    print_board(&board);
                    printf("Game Over. White Wins by Forbidden Move!\n");
                    break;
                }
            }
            // 落子
            place_piece(&board, row, col, current_player);
            // 检查获胜
            if (check_win(&board, row, col, current_player)) {
                print_board(&board);
                if (current_player == BLACK) printf("Black Wins!\n");
                else printf("White Wins!\n");
                game_over = 1;
                break;
            }
            // 检查平局
            if (check_draw(&board)) {
                print_board(&board);
                printf("Draw!\n");
                game_over = 1;
                break;
            }
            // 切换回合
            current_player = (current_player == BLACK) ? WHITE : BLACK;
        }
        
        // 询问是否再来一局
        printf("Play again? (y/n): ");
        char resp[10];
        if (fgets(resp, sizeof(resp), stdin) != NULL) {
            if (resp[0] != 'y' && resp[0] != 'Y') {
                play_again = 0;
            }
        } else {
            play_again = 0;
        }
    }

    return 0;
}
