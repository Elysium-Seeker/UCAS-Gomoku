#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "ai_engine.h"
#include "board.h"
#include "game.h" // 为了使用 check_win 模拟简单的搜索
#include "rules.h"

// 引入我们"训练"出来的权重
#include "model_weights.h"

// 时间限制相关
#define TIME_LIMIT_MS 14500  // 14.5秒限制（留0.5秒余量）
static clock_t search_start_time;
static int time_up = 0;

static int check_time(void) {
    clock_t now = clock();
    double elapsed = (double)(now - search_start_time) * 1000.0 / CLOCKS_PER_SEC;
    return elapsed >= TIME_LIMIT_MS;
}

// 简单的激活函数 (ReLU)
// f(x) = max(0, x)
float relu(float x) {
    return x > 0 ? x : 0;
}

// Layer 1: Conv 5x5, 1 -> 16
// Input: [1][15][15], Output: [16][15][15]
// 卷积层 1：提取基础特征
void conv_layer1(float input[1][15][15], float output[C1_OUT][15][15]) {
    int pad = 2;
    for (int o = 0; o < C1_OUT; o++) {
        for (int i = 0; i < BOARD_SIZE; i++) {
            for (int j = 0; j < BOARD_SIZE; j++) {
                float sum = c1_bias[o];
                // 输入通道数为 1
                for (int ki = 0; ki < C1_K; ki++) {
                    for (int kj = 0; kj < C1_K; kj++) {
                        int r = i - pad + ki;
                        int c = j - pad + kj;
                        if (r >= 0 && r < BOARD_SIZE && c >= 0 && c < BOARD_SIZE) {
                            sum += input[0][r][c] * c1_weights[o][0][ki][kj];
                        }
                    }
                }
                output[o][i][j] = relu(sum);
            }
        }
    }
}

// Layer 2: Conv 3x3, 16 -> 32
// Input: [16][15][15], Output: [32][15][15]
// 卷积层 2：提取高级特征
void conv_layer2(float input[C1_OUT][15][15], float output[C2_OUT][15][15]) {
    int pad = 1;
    for (int o = 0; o < C2_OUT; o++) {
        for (int i = 0; i < BOARD_SIZE; i++) {
            for (int j = 0; j < BOARD_SIZE; j++) {
                float sum = c2_bias[o];
                // 遍历输入通道
                for (int ic = 0; ic < C1_OUT; ic++) {
                    for (int ki = 0; ki < C2_K; ki++) {
                        for (int kj = 0; kj < C2_K; kj++) {
                            int r = i - pad + ki;
                            int c = j - pad + kj;
                            if (r >= 0 && r < BOARD_SIZE && c >= 0 && c < BOARD_SIZE) {
                                sum += input[ic][r][c] * c2_weights[o][ic][ki][kj];
                            }
                        }
                    }
                }
                output[o][i][j] = relu(sum);
            }
        }
    }
}

// CNN 前向传播 (Enhanced)
// 使用神经网络评估当前局面的胜率
// 返回值范围 [-1, 1]，1 表示黑棋优势，-1 表示白棋优势
float evaluate_position_with_model(Board *board, int row, int col, int current_player) {
    // 1. 准备输入数据 [1][15][15]
    float input_layer[1][15][15];
    for(int i=0; i<BOARD_SIZE; i++) {
        for(int j=0; j<BOARD_SIZE; j++) {
            if (board->grid[i][j] == BLACK) input_layer[0][i][j] = 1.0f;
            else if (board->grid[i][j] == WHITE) input_layer[0][i][j] = -1.0f;
            else input_layer[0][i][j] = 0.0f;
        }
    }
    // 模拟在该位置落子
    input_layer[0][row][col] = (current_player == BLACK) ? 1.0f : -1.0f;

    // 2. Conv1
    float c1_out[C1_OUT][15][15];
    conv_layer1(input_layer, c1_out);

    // 3. Conv2
    float c2_out[C2_OUT][15][15];
    conv_layer2(c1_out, c2_out);

    // 4. Flatten & FC1
    // Flatten size: 32 * 15 * 15 = 7200
    // FC1 size: 128
    float fc1_out[FC1_SIZE];
    for (int h = 0; h < FC1_SIZE; h++) {
        float sum = fc1_bias[h];
        int idx = 0;
        // 展平顺序必须匹配 PyTorch 的 view(-1, 32*15*15)
        // PyTorch 默认为 NCHW -> 展平 -> C, H, W 顺序
        for (int c = 0; c < C2_OUT; c++) {
            for (int i = 0; i < BOARD_SIZE; i++) {
                for (int j = 0; j < BOARD_SIZE; j++) {
                    sum += c2_out[c][i][j] * fc1_weights[idx][h];
                    idx++;
                }
            }
        }
        fc1_out[h] = relu(sum);
    }

    // 5. FC2 (Output)
    float score = fc2_bias[0];
    for (int h = 0; h < FC1_SIZE; h++) {
        score += fc1_out[h] * fc2_weights[h][0];
    }
    
    // 加上 Tanh 激活函数，匹配 Python 训练代码
    return tanhf(score);
}

// ==========================================
// 增强版启发式评分 (Robust Heuristic)
// ==========================================

#define SCORE_FIVE          100000
#define SCORE_OPEN_FOUR     10000
#define SCORE_BROKEN_FOUR   1500   // 提高 Broken 4 的优先级
#define SCORE_OPEN_THREE    1200   // 提高 Open 3 的优先级
#define SCORE_BROKEN_THREE  100
#define SCORE_OPEN_TWO      10

// Helper: 1=Self, 2=Opponent, 0=Empty
// 获取相对棋子颜色：1=己方，2=对方，0=空
int get_rel_piece(int piece, int player) {
    if (piece == EMPTY) return 0;
    if (piece == player) return 1;
    return 2;
}

// 评估一个包含 6 个棋子的窗口
// 评估一个 6 格窗口内的棋型分数
int match_pattern_score(int* window, int size) {
    int count = 0;
    int empty = 0;
    int opponent = 0;
    
    for(int i=0; i<size; i++) {
        if(window[i] == 1) count++;
        else if(window[i] == 0) empty++;
        else opponent++;
    }

    if (opponent > 0) return 0; 

    if (count == 5) return SCORE_FIVE;

    if (count == 4) {
        // _XXXX_ -> 活四
        if (window[0] == 0 && window[size-1] == 0) return SCORE_OPEN_FOUR;
        // X_XXX, XX_XX 等 -> 冲四
        return SCORE_BROKEN_FOUR;
    }

    if (count == 3) {
        // _XXX_ -> 活三
        if (window[0] == 0 && window[size-1] == 0) {
            // 检查是否真的是活三（中间没有间隔）
            // 如果有间隔 (1011)，则是“跳三”，虽然危险但略逊于纯活三
            // 但为了防守，我们将其视为同等危险
            return SCORE_OPEN_THREE;
        }
        // _XXXO -> 眠三
        return SCORE_BROKEN_THREE;
    }
    
    if (count == 2 && window[0] == 0 && window[size-1] == 0) return SCORE_OPEN_TWO;

    return 0;
}

// 评估单条线上的分数
int evaluate_line(int *line, int length, int player) {
    int score = 0;
    int rel_line[BOARD_SIZE];
    for (int i = 0; i < length; i++) rel_line[i] = get_rel_piece(line[i], player);

    // 大小为 6 的滑动窗口
    for (int i = 0; i <= length - 6; i++) {
        score += match_pattern_score(rel_line + i, 6);
    }
    // 检查大小为 5 的窗口以寻找纯五连
    for (int i = 0; i <= length - 5; i++) {
        int window[5];
        for(int k=0; k<5; k++) window[k] = rel_line[i+k];
        int c=0;
        for(int k=0; k<5; k++) if(window[k]==1) c++;
        if(c==5) score += SCORE_FIVE;
    }
    return score;
}

// 传统的启发式评分（作为模型的补充或兜底）
// 结合进攻和防守，计算某一点的综合评分
int heuristic_score(Board *board, int row, int col, int piece) {
    int score = 0;
    
    // 1. 进攻评分 (如果我下这里能成什么)
    board->grid[row][col] = piece; // 临时落子
    
    int directions[4][2] = {{0,1}, {1,0}, {1,1}, {1,-1}};
    for (int k = 0; k < 4; k++) {
        int line[BOARD_SIZE];
        int len = 0;
        int dr = directions[k][0];
        int dc = directions[k][1];
        
        // 寻找起始点
        int r = row, c = col;
        while(r-dr >= 0 && r-dr < BOARD_SIZE && c-dc >= 0 && c-dc < BOARD_SIZE) {
            r -= dr; c -= dc;
        }
        
        // 收集线路上的棋子
        while(r >= 0 && r < BOARD_SIZE && c >= 0 && c < BOARD_SIZE) {
            line[len++] = board->grid[r][c];
            r += dr; c += dc;
        }
        
        score += evaluate_line(line, len, piece);
    }
    board->grid[row][col] = EMPTY; // 还原棋盘

    // 2. 防守评分 (如果我不下这里，对手下这里能成什么)
    int opponent = (piece == BLACK) ? WHITE : BLACK;
    board->grid[row][col] = opponent; 
    
    int defense_score = 0;
    for (int k = 0; k < 4; k++) {
        int line[BOARD_SIZE];
        int len = 0;
        int dr = directions[k][0];
        int dc = directions[k][1];
        
        int r = row, c = col;
        while(r-dr >= 0 && r-dr < BOARD_SIZE && c-dc >= 0 && c-dc < BOARD_SIZE) {
            r -= dr; c -= dc;
        }
        while(r >= 0 && r < BOARD_SIZE && c >= 0 && c < BOARD_SIZE) {
            line[len++] = board->grid[r][c];
            r += dr; c += dc;
        }
        
        int threat = evaluate_line(line, len, opponent);
        // 防守系数
        if (threat >= SCORE_FIVE) defense_score += SCORE_FIVE * 10; // 必输，必须堵
        else if (threat >= SCORE_OPEN_FOUR) defense_score += SCORE_OPEN_FOUR * 5; 
        else if (threat >= SCORE_BROKEN_FOUR) defense_score += SCORE_BROKEN_FOUR * 5;
        else if (threat >= SCORE_OPEN_THREE) defense_score += SCORE_OPEN_THREE * 5;
    }
    board->grid[row][col] = EMPTY; // 还原

    score += defense_score;

    // 3. 位置分
    int center = BOARD_SIZE / 2;
    int dist = abs(row - center) + abs(col - center);
    score += (30 - dist);

    return score;
}

// 检查某个位置是否形成连线威胁 (Deprecated, kept for compatibility if needed, but unused)
int check_line(Board *board, int r, int c, int dr, int dc, int player) {
    return 0; 
}

// ==========================================
// Minimax Search Implementation
// ==========================================

#define MAX_DEPTH 4
#define INF 1000000000

typedef struct {
    int r, c;
    int score;
} Move;

// 比较函数，用于 qsort 排序
// 按分数降序排列
int compare_moves(const void *a, const void *b) {
    return ((Move*)b)->score - ((Move*)a)->score;
}

// 获取候选点
// 筛选出所有邻近有棋子的空位，减少搜索空间
// 返回候选点数量
int get_candidates(Board *board, Move *moves) {
    int count = 0;
    int has_neighbor[BOARD_SIZE][BOARD_SIZE] = {0};
    int stones_found = 0;

    // 标记邻居节点
    for(int r=0; r<BOARD_SIZE; r++) {
        for(int c=0; c<BOARD_SIZE; c++) {
            if(board->grid[r][c] != EMPTY) {
                stones_found++;
                for(int dr=-2; dr<=2; dr++) {
                    for(int dc=-2; dc<=2; dc++) {
                        int nr = r+dr, nc = c+dc;
                        if(nr>=0 && nr<BOARD_SIZE && nc>=0 && nc<BOARD_SIZE && board->grid[nr][nc] == EMPTY) {
                            has_neighbor[nr][nc] = 1;
                        }
                    }
                }
            }
        }
    }
    
    // 如果棋盘为空（第一步），返回中心点
    if (stones_found == 0) {
        moves[0].r = BOARD_SIZE/2;
        moves[0].c = BOARD_SIZE/2;
        moves[0].score = 0;
        return 1;
    }

    // 收集候选点
    for(int r=0; r<BOARD_SIZE; r++) {
        for(int c=0; c<BOARD_SIZE; c++) {
            if(has_neighbor[r][c]) {
                moves[count].r = r;
                moves[count].c = c;
                moves[count].score = 0; // 待填充
                count++;
            }
        }
    }
    
    return count;
}

// 全局评估函数
// 评估当前棋盘对 ai_color 的优势程度
int evaluate_full_board(Board *board, int ai_color) {
    int score = 0;
    int my_color = ai_color;
    int opp_color = (ai_color == BLACK) ? WHITE : BLACK;
    
    // 水平方向
    for(int r=0; r<BOARD_SIZE; r++) {
        int line[BOARD_SIZE];
        for(int c=0; c<BOARD_SIZE; c++) line[c] = board->grid[r][c];
        score += evaluate_line(line, BOARD_SIZE, my_color);
        score -= evaluate_line(line, BOARD_SIZE, opp_color) * 1.5; // 防守偏置
    }
    // 垂直方向
    for(int c=0; c<BOARD_SIZE; c++) {
        int line[BOARD_SIZE];
        for(int r=0; r<BOARD_SIZE; r++) line[r] = board->grid[r][c];
        score += evaluate_line(line, BOARD_SIZE, my_color);
        score -= evaluate_line(line, BOARD_SIZE, opp_color) * 1.5;
    }
    // 对角线方向
    // (简化版：目前只扫描主对角线以节省代码量，或者完整实现)
    // 这里我们完整实现。
    // 左上到右下
    for(int k=0; k<BOARD_SIZE*2; k++) {
        int line[BOARD_SIZE];
        int len = 0;
        for(int j=0; j<=k; j++) {
            int i = k - j;
            if(i<BOARD_SIZE && j<BOARD_SIZE) {
                line[len++] = board->grid[i][j];
            }
        }
        if(len >= 5) {
            score += evaluate_line(line, len, my_color);
            score -= evaluate_line(line, len, opp_color) * 1.5;
        }
    }
    // 右上到左下
    for(int k=0; k<BOARD_SIZE*2; k++) {
        int line[BOARD_SIZE];
        int len = 0;
        for(int j=0; j<=k; j++) {
            int i = k - j;
            int c = BOARD_SIZE - 1 - j;
            if(i<BOARD_SIZE && c>=0) {
                line[len++] = board->grid[i][c];
            }
        }
        if(len >= 5) {
            score += evaluate_line(line, len, my_color);
            score -= evaluate_line(line, len, opp_color) * 1.5;
        }
    }

    return score;
}

// Minimax 搜索算法 (Alpha-Beta 剪枝)
// 递归搜索最佳走法
// depth: 剩余搜索深度
// alpha, beta: 剪枝边界
// is_maximizing: 当前是否为最大化层（AI 回合）
int minimax(Board *board, int depth, int alpha, int beta, int is_maximizing, int ai_color) {
    // 时间检查
    if (time_up || check_time()) {
        time_up = 1;
        return evaluate_full_board(board, ai_color);
    }
    
    if (depth == 0) {
        return evaluate_full_board(board, ai_color);
    }
    
    Move moves[400]; // 最大候选点数量
    int count = get_candidates(board, moves);
    
    int current_player = is_maximizing ? ai_color : ((ai_color == BLACK) ? WHITE : BLACK);
    
    // 启发式排序
    for(int i=0; i<count; i++) {
        // 使用现有的启发式评分进行走法排序
        moves[i].score = heuristic_score(board, moves[i].r, moves[i].c, current_player);
        
        // 仅在根节点或非常浅的深度使用模型以节省时间
        // 在递归 Minimax 内部，跳过繁重的 CNN 模型
        // float model_val = evaluate_position_with_model(board, moves[i].r, moves[i].c, current_player);
        // moves[i].score += (int)(model_val * 1000); 
    }
    qsort(moves, count, sizeof(Move), compare_moves);
    
    // 剪枝：只考虑前 N 个走法
    // 在更深层进行更窄的搜索以保持性能
    int limit = 8;
    if (depth >= 3) limit = 12;
    
    if (count > limit) count = limit;
    
    if (is_maximizing) {
        int max_eval = -INF;
        for (int i=0; i<count; i++) {
            int r = moves[i].r;
            int c = moves[i].c;
            
            if (current_player == BLACK && is_forbidden(board, r, c)) continue;
            
            board->grid[r][c] = current_player;
            int eval = minimax(board, depth-1, alpha, beta, 0, ai_color);
            board->grid[r][c] = EMPTY;
            
            if (eval > max_eval) max_eval = eval;
            if (eval > alpha) alpha = eval;
            if (beta <= alpha) break;
        }
        // 如果所有走法都被禁手规则禁止
        if (max_eval == -INF) return -INF + 1; 
        return max_eval;
    } else {
        int min_eval = INF;
        for (int i=0; i<count; i++) {
            int r = moves[i].r;
            int c = moves[i].c;
            
            if (current_player == BLACK && is_forbidden(board, r, c)) continue;

            board->grid[r][c] = current_player;
            int eval = minimax(board, depth-1, alpha, beta, 1, ai_color);
            board->grid[r][c] = EMPTY;
            
            if (eval < min_eval) min_eval = eval;
            if (eval < beta) beta = eval;
            if (beta <= alpha) break;
        }
        if (min_eval == INF) return INF - 1;
        return min_eval;
    }
}

// AI 决策入口函数
// 结合神经网络（直觉）和 Minimax 搜索（计算）来决定最佳落子
void get_ai_move(Board *board, int *row, int *col, int ai_color) {
    // 初始化时间限制
    search_start_time = clock();
    time_up = 0;
    
    // 使用 Minimax 搜索
    Move moves[400];
    int count = get_candidates(board, moves);
    
    // 初始排序
    for(int i=0; i<count; i++) {
        moves[i].score = heuristic_score(board, moves[i].r, moves[i].c, ai_color);
        float model_val = evaluate_position_with_model(board, moves[i].r, moves[i].c, ai_color);
        moves[i].score += (int)(model_val * 1000);
        
        // 增加一点随机性，打破平局，让 AI 每次表现略有不同
        moves[i].score += rand() % 10;

        // 禁手检查（根节点硬过滤）
        if (ai_color == BLACK && is_forbidden(board, moves[i].r, moves[i].c)) {
            moves[i].score = -INF;
        }
    }
    qsort(moves, count, sizeof(Move), compare_moves);
    
    // 挑选用于根搜索的顶级候选点
    int limit = 15;
    if (count > limit) count = limit;
    
    int best_val = -INF;
    
    // 用于存储最佳走法的数组（处理平局）
    typedef struct { int r; int c; } Point;
    Point best_moves[20];
    int best_count = 0;
    
    // 如果只有一步（中心），直接走
    if (count == 1) {
        *row = moves[0].r;
        *col = moves[0].c;
        return;
    }

    // 对顶级候选点运行 Minimax
    for(int i=0; i<count; i++) {
        if (moves[i].score == -INF) continue; // 跳过禁手
        
        int r = moves[i].r;
        int c = moves[i].c;
        
        board->grid[r][c] = ai_color;
        int val = minimax(board, MAX_DEPTH, -INF, INF, 0, ai_color); // 下一层是极小化层
        board->grid[r][c] = EMPTY;
        
        // printf("Move %c%d: Score %d\n", 'A'+c, r+1, val); // 调试信息
        
        if (val > best_val) {
            best_val = val;
            best_count = 0;
            best_moves[best_count].r = r;
            best_moves[best_count].c = c;
            best_count++;
        } else if (val == best_val) {
            if (best_count < 20) {
                best_moves[best_count].r = r;
                best_moves[best_count].c = c;
                best_count++;
            }
        }
    }
    
    if (best_count > 0) {
        // 随机选择一个最佳走法
        int idx = rand() % best_count;
        *row = best_moves[idx].r;
        *col = best_moves[idx].c;
    } else {
        // 兜底（除非所有走法都被禁，否则不应发生）
        *row = moves[0].r;
        *col = moves[0].c;
    }
}
