#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef _WIN32
#include <windows.h>
#endif
#include "board.h"
#include "game.h"
#include "rules.h"
#include "ai.h"

// Game Modes
#define MODE_PVP 1
#define MODE_PVE_BLACK 2 // Human is Black (First)
#define MODE_PVE_WHITE 3 // Human is White (Second)
#define MODE_EVE 4       // AI vs AI

int main() {
    #ifdef _WIN32
    SetConsoleOutputCP(65001); 
    #endif

    int mode = MODE_PVP;
    printf("=== Gomoku Game (Minimax Version) ===\n");
    printf("1. PvP (Player vs Player)\n");
    printf("2. PvE (Player vs AI) - You are Black (First)\n");
    printf("3. PvE (Player vs AI) - You are White (Second)\n");
    printf("4. EvE (AI vs AI) - Watch Mode\n");
    printf("Select Mode: ");
    char mode_buf[10];
    if (fgets(mode_buf, sizeof(mode_buf), stdin)) {
        int m = atoi(mode_buf);
        if (m >= 1 && m <= 4) mode = m;
    }

    Board board;
    init_board(&board);
    int current_player = BLACK; 
    int game_over = 0;
    char input[100];
    
    // AI Stats
    AIStats ai_stats_black;
    AIStats ai_stats_white;
    init_ai_stats(&ai_stats_black);
    init_ai_stats(&ai_stats_white);

    while (!game_over) {
        print_board(&board); 

        int is_ai_turn = 0;
        if (mode == MODE_PVP) is_ai_turn = 0;
        else if (mode == MODE_PVE_BLACK && current_player == WHITE) is_ai_turn = 1;
        else if (mode == MODE_PVE_WHITE && current_player == BLACK) is_ai_turn = 1;
        else if (mode == MODE_EVE) is_ai_turn = 1;

        if (is_ai_turn) {
            printf(current_player == BLACK ? "Black(AI) is thinking...\n" : "White(AI) is thinking...\n");
            int ax, ay;
            AIStats *current_stats = (current_player == BLACK) ? &ai_stats_black : &ai_stats_white;
            ai_get_move(&board, current_player, &ax, &ay, current_stats);
            
             if (current_player == BLACK) { 
                if (is_forbidden(&board, ax, ay)) {
                    printf("AI attempted forbidden move! AI Loss.\n");
                    place_piece(&board, ax, ay, BLACK);
                    print_board(&board);
                    game_over = 1;
                    break;
                }
            }
            place_piece(&board, ax, ay, current_player);
            
            if (check_win(&board, ax, ay, current_player)) {
                print_board(&board);
                printf(current_player == BLACK ? "Black Wins!\n" : "White Wins!\n");
                game_over = 1; break;
            }
        } else {
            // Human Turn
            if (current_player == BLACK) printf("Black's Turn (e.g. H8): ");
            else printf("White's Turn (e.g. H8): ");
            
            if (fgets(input, sizeof(input), stdin) == NULL) break;
            input[strcspn(input, "\n")] = 0;
            
            int x, y;
            if (!parse_input(input, &x, &y)) {
                printf("Invalid Format.\n");
                continue;
            }
            if (!is_empty(&board, x, y)) {
                printf("Position Occupied.\n");
                continue;
            }
            if (current_player == BLACK) {
                if (is_forbidden(&board, x, y)) {
                    printf("Forbidden Move! Black Loses.\n");
                    game_over = 1;
                    place_piece(&board, x, y, BLACK);
                    print_board(&board);
                    break;
                }
            }
            place_piece(&board, x, y, current_player);
            if (check_win(&board, x, y, current_player)) {
                print_board(&board);
                printf(current_player == BLACK ? "Black Wins!\n" : "White Wins!\n");
                game_over = 1; break;
            }
        }

        if (check_draw(&board)) {
            print_board(&board);
            printf("Draw!\n");
            game_over = 1;
            break;
        }

        current_player = (current_player == BLACK) ? WHITE : BLACK;
    }
    
    printf("Game Over. Press Enter to exit...");
    getchar();

    return 0;
}
