"""
可视化对战：Minimax vs 混合AI
实时打印棋盘
"""
import torch
import numpy as np
import time
import os
import sys

BOARD_SIZE = 15
EMPTY, BLACK, WHITE = 0, 1, 2

# ============ 棋盘显示 ============
def print_board(board, last_move=None):
    """打印棋盘，用 ● ○ 表示黑白棋"""
    os.system('cls' if os.name == 'nt' else 'clear')
    
    # 列标题
    print("   ", end="")
    for c in range(BOARD_SIZE):
        print(f" {chr(ord('A') + c)}", end="")
    print()
    
    for r in range(BOARD_SIZE):
        print(f"{r+1:2d} ", end="")
        for c in range(BOARD_SIZE):
            if board[r, c] == BLACK:
                if last_move and last_move == (r, c):
                    print(" ◆", end="")  # 最后一手黑棋
                else:
                    print(" ●", end="")
            elif board[r, c] == WHITE:
                if last_move and last_move == (r, c):
                    print(" ◇", end="")  # 最后一手白棋
                else:
                    print(" ○", end="")
            else:
                print(" ·", end="")
        print(f" {r+1}")
    
    print("   ", end="")
    for c in range(BOARD_SIZE):
        print(f" {chr(ord('A') + c)}", end="")
    print("\n")

# ============ 模型定义 ============
class GomokuNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = torch.nn.Linear(64 * 15 * 15, 256)
        self.fc2 = torch.nn.Linear(256, 1)
        self.fc_policy = torch.nn.Linear(256, 225)

    def forward(self, x):
        x = x.view(-1, 1, 15, 15)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(-1, 64 * 15 * 15)
        x = torch.relu(self.fc1(x))
        return self.fc_policy(x), torch.tanh(self.fc2(x))

# ============ 评估函数 ============
SCORE_FIVE = 100000
SCORE_OPEN_FOUR = 10000
SCORE_BROKEN_FOUR = 5000
SCORE_OPEN_THREE = 1000
SCORE_BROKEN_THREE = 100
SCORE_OPEN_TWO = 50

def get_rel_piece(piece, player):
    if piece == EMPTY: return 0
    return 1 if piece == player else -1

def match_pattern_score(window):
    count = sum(1 for x in window if x == 1)
    opponent = sum(1 for x in window if x == -1)
    if opponent > 0: return 0
    if count == 5: return SCORE_FIVE
    if count == 4:
        return SCORE_OPEN_FOUR if window[0] == 0 and window[-1] == 0 else SCORE_BROKEN_FOUR
    if count == 3:
        return SCORE_OPEN_THREE if window[0] == 0 and window[-1] == 0 else SCORE_BROKEN_THREE
    if count == 2 and window[0] == 0 and window[-1] == 0:
        return SCORE_OPEN_TWO
    return 0

def evaluate_line(line, player):
    rel_line = [get_rel_piece(p, player) for p in line]
    score = 0
    for i in range(len(rel_line) - 5):
        score += match_pattern_score(rel_line[i:i+6])
    for i in range(len(rel_line) - 4):
        if sum(1 for x in rel_line[i:i+5] if x == 1) == 5:
            score += SCORE_FIVE
    return score

def heuristic_score(board, row, col, piece):
    score = 0
    opponent = WHITE if piece == BLACK else BLACK
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
    
    original = board[row, col]
    board[row, col] = piece
    for dr, dc in directions:
        line = []
        r, c = row, col
        while 0 <= r - dr < BOARD_SIZE and 0 <= c - dc < BOARD_SIZE:
            r -= dr; c -= dc
        while 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE:
            line.append(board[r, c])
            r += dr; c += dc
        score += evaluate_line(line, piece)
    board[row, col] = original
    
    board[row, col] = opponent
    for dr, dc in directions:
        line = []
        r, c = row, col
        while 0 <= r - dr < BOARD_SIZE and 0 <= c - dc < BOARD_SIZE:
            r -= dr; c -= dc
        while 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE:
            line.append(board[r, c])
            r += dr; c += dc
        threat = evaluate_line(line, opponent)
        if threat >= SCORE_FIVE: score += SCORE_FIVE * 10
        elif threat >= SCORE_OPEN_FOUR: score += SCORE_OPEN_FOUR * 5
        elif threat >= SCORE_BROKEN_FOUR: score += SCORE_BROKEN_FOUR * 5
        elif threat >= SCORE_OPEN_THREE: score += SCORE_OPEN_THREE * 5
    board[row, col] = original
    
    center = BOARD_SIZE // 2
    score += (30 - abs(row - center) - abs(col - center))
    return score

def get_candidates(board):
    moves = []
    has_neighbor = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=bool)
    stones = 0
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if board[r, c] != EMPTY:
                stones += 1
                for dr in range(-2, 3):
                    for dc in range(-2, 3):
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE and board[nr, nc] == EMPTY:
                            has_neighbor[nr, nc] = True
    if stones == 0:
        return [(BOARD_SIZE // 2, BOARD_SIZE // 2, 0)]
    return [(r, c, 0) for r in range(BOARD_SIZE) for c in range(BOARD_SIZE) if has_neighbor[r, c]]

def evaluate_full_board(board, ai_color):
    score = 0
    opp = WHITE if ai_color == BLACK else BLACK
    for r in range(BOARD_SIZE):
        line = list(board[r, :])
        score += evaluate_line(line, ai_color) - int(evaluate_line(line, opp) * 1.5)
    for c in range(BOARD_SIZE):
        line = list(board[:, c])
        score += evaluate_line(line, ai_color) - int(evaluate_line(line, opp) * 1.5)
    for k in range(-BOARD_SIZE + 1, BOARD_SIZE):
        line = list(np.diag(board, k))
        if len(line) >= 5:
            score += evaluate_line(line, ai_color) - int(evaluate_line(line, opp) * 1.5)
        line = list(np.diag(np.fliplr(board), k))
        if len(line) >= 5:
            score += evaluate_line(line, ai_color) - int(evaluate_line(line, opp) * 1.5)
    return score

def minimax(board, depth, alpha, beta, is_max, ai_color):
    if depth == 0:
        return evaluate_full_board(board, ai_color)
    moves = get_candidates(board)
    curr = ai_color if is_max else (WHITE if ai_color == BLACK else BLACK)
    scored = [(m[0], m[1], heuristic_score(board, m[0], m[1], curr)) for m in moves]
    scored.sort(key=lambda x: -x[2])
    scored = scored[:10 if depth >= 2 else 6]
    
    if is_max:
        max_eval = float('-inf')
        for r, c, _ in scored:
            board[r, c] = curr
            val = minimax(board, depth - 1, alpha, beta, False, ai_color)
            board[r, c] = EMPTY
            max_eval = max(max_eval, val)
            alpha = max(alpha, val)
            if beta <= alpha: break
        return max_eval
    else:
        min_eval = float('inf')
        for r, c, _ in scored:
            board[r, c] = curr
            val = minimax(board, depth - 1, alpha, beta, True, ai_color)
            board[r, c] = EMPTY
            min_eval = min(min_eval, val)
            beta = min(beta, val)
            if beta <= alpha: break
        return min_eval

# ============ AI Agents ============
class MinimaxAgent:
    def __init__(self, max_depth=3):
        self.name = f"Minimax(深度{max_depth})"
        self.max_depth = max_depth
    
    def get_move(self, board, player):
        board = board.copy()
        moves = get_candidates(board)
        scored = [(m[0], m[1], heuristic_score(board, m[0], m[1], player)) for m in moves]
        scored.sort(key=lambda x: -x[2])
        candidates = scored[:12]
        
        best_val = float('-inf')
        best_move = candidates[0][:2]
        for r, c, _ in candidates:
            board[r, c] = player
            val = minimax(board, self.max_depth, float('-inf'), float('inf'), False, player)
            board[r, c] = EMPTY
            if val > best_val:
                best_val = val
                best_move = (r, c)
        return best_move

class HybridAgent:
    def __init__(self, model_path, max_depth=2):
        self.name = f"混合AI(NN+Minimax深度{max_depth})"
        self.max_depth = max_depth
        self.model = GomokuNet()
        self.model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
        self.model.eval()
    
    def nn_score(self, board, row, col, player):
        original = board[row, col]
        board[row, col] = player
        rel = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                rel[r, c] = get_rel_piece(board[r, c], player)
        with torch.no_grad():
            _, val = self.model(torch.from_numpy(rel.flatten()).unsqueeze(0))
        board[row, col] = original
        return val.item()
    
    def get_move(self, board, player):
        board = board.copy()
        moves = get_candidates(board)
        scored = []
        for r, c, _ in moves:
            h = heuristic_score(board, r, c, player)
            n = self.nn_score(board, r, c, player) * 1000
            scored.append((r, c, h + int(n)))
        scored.sort(key=lambda x: -x[2])
        candidates = scored[:12]
        
        best_val = float('-inf')
        best_move = candidates[0][:2]
        for r, c, _ in candidates:
            board[r, c] = player
            val = minimax(board, self.max_depth, float('-inf'), float('inf'), False, player)
            board[r, c] = EMPTY
            if val > best_val:
                best_val = val
                best_move = (r, c)
        return best_move

# ============ 胜负判断 ============
def check_winner(board):
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if board[r, c] == EMPTY: continue
            p = board[r, c]
            for dr, dc in directions:
                count = 1
                for i in range(1, 5):
                    nr, nc = r + dr * i, c + dc * i
                    if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE and board[nr, nc] == p:
                        count += 1
                    else: break
                if count >= 5: return p
    return None

# ============ 对战 ============
def play_visual(agent1, agent2, delay=1.0):
    """可视化对战"""
    board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int32)
    current = BLACK
    moves = 0
    last_move = None
    
    print_board(board)
    print(f"对战: {agent1.name} (●黑) vs {agent2.name} (○白)")
    print("=" * 50)
    time.sleep(1)
    
    while moves < 225:
        # 获取走法
        start = time.time()
        if current == BLACK:
            move = agent1.get_move(board, BLACK)
        else:
            move = agent2.get_move(board, WHITE)
        elapsed = time.time() - start
        
        r, c = move
        board[r, c] = current
        moves += 1
        last_move = (r, c)
        
        # 显示
        print_board(board, last_move)
        color_name = "●黑" if current == BLACK else "○白"
        agent_name = agent1.name if current == BLACK else agent2.name
        print(f"第{moves}手: {color_name} {agent_name} 下 {chr(ord('A')+c)}{r+1} (用时{elapsed:.1f}s)")
        print("=" * 50)
        
        # 检查胜负
        winner = check_winner(board)
        if winner:
            if winner == BLACK:
                print(f"\n🎉 {agent1.name} (黑) 获胜!")
            else:
                print(f"\n🎉 {agent2.name} (白) 获胜!")
            return winner
        
        current = WHITE if current == BLACK else BLACK
        time.sleep(delay)
    
    print("\n平局!")
    return 0

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "gomoku_model_original.pth")
    
    print("加载 AI...")
    minimax_agent = MinimaxAgent(max_depth=4)
    hybrid_agent = HybridAgent(model_path, max_depth=3)
    
    print("\n选择对战:")
    print("1. 混合AI(黑) vs Minimax(白)")
    print("2. Minimax(黑) vs 混合AI(白)")
    print("3. Minimax vs Minimax")
    
    choice = input("> ").strip()
    
    if choice == "2":
        play_visual(minimax_agent, hybrid_agent, delay=0.5)
    elif choice == "3":
        play_visual(minimax_agent, MinimaxAgent(max_depth=3), delay=0.5)
    else:
        play_visual(hybrid_agent, minimax_agent, delay=0.5)

if __name__ == "__main__":
    main()
