"""
对比原始权重 vs 当前权重
"""
import torch
import torch.nn as nn
import numpy as np
import os
import sys

BOARD_SIZE = 15
EMPTY, BLACK, WHITE = 0, 1, -1

class GomokuNet(nn.Module):
    def __init__(self):
        super(GomokuNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 15 * 15, 256)
        self.fc2 = nn.Linear(256, 1)
        self.fc_policy = nn.Linear(256, 225)

    def forward(self, x):
        x = x.view(-1, 1, 15, 15)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = x.view(-1, 64 * 15 * 15)
        x = self.relu(self.fc1(x))
        val = torch.tanh(self.fc2(x))
        pol = self.fc_policy(x)
        return pol, val


def check_winner(board):
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if board[r, c] == EMPTY:
                continue
            player = board[r, c]
            for dr, dc in directions:
                count = 1
                for i in range(1, 5):
                    nr, nc = r + dr * i, c + dc * i
                    if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE:
                        if board[nr, nc] == player:
                            count += 1
                        else:
                            break
                    else:
                        break
                if count >= 5:
                    return player
    return None


class NNAgent:
    def __init__(self, model_path, name="Agent"):
        self.name = name
        self.device = torch.device('cpu')
        self.model = GomokuNet()
        state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.eval()
    
    def get_move(self, board, player):
        canonical = board.flatten() * player
        input_tensor = torch.FloatTensor(canonical).unsqueeze(0)
        
        with torch.no_grad():
            policy, _ = self.model(input_tensor)
            policy = torch.softmax(policy, dim=1).squeeze().numpy()
        
        # 屏蔽非法位置
        mask = (board.flatten() == EMPTY)
        policy = policy * mask
        
        if policy.sum() == 0:
            return None
        
        policy /= policy.sum()
        move = np.argmax(policy)
        return move // BOARD_SIZE, move % BOARD_SIZE


def play_game(agent1, agent2, verbose=False):
    board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int32)
    current_player = BLACK
    move_count = 0
    
    while move_count < BOARD_SIZE * BOARD_SIZE:
        if current_player == BLACK:
            move = agent1.get_move(board, BLACK)
        else:
            move = agent2.get_move(board, WHITE)
        
        if move is None:
            break
        
        r, c = move
        board[r, c] = current_player
        move_count += 1
        
        winner = check_winner(board)
        if winner:
            if verbose:
                print(f"  {move_count}手结束")
            return winner, move_count
        
        current_player = WHITE if current_player == BLACK else BLACK
    
    return 0, move_count


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    original_path = os.path.join(script_dir, "gomoku_model_original.pth")
    current_path = os.path.join(script_dir, "gomoku_model.pth")
    
    print("加载模型...")
    original_agent = NNAgent(original_path, "原始权重")
    current_agent = NNAgent(current_path, "当前权重")
    
    print("\n" + "="*50)
    print("对比测试: 原始权重 vs 当前权重")
    print("="*50)
    
    original_wins = 0
    current_wins = 0
    draws = 0
    
    num_games = 10
    
    for i in range(num_games):
        if i % 2 == 0:
            # 原始执黑
            result, length = play_game(original_agent, current_agent)
            if result == BLACK:
                original_wins += 1
                print(f"第{i+1}局: 原始(黑) 胜 ({length}手)")
            elif result == WHITE:
                current_wins += 1
                print(f"第{i+1}局: 当前(白) 胜 ({length}手)")
            else:
                draws += 1
                print(f"第{i+1}局: 平局 ({length}手)")
        else:
            # 当前执黑
            result, length = play_game(current_agent, original_agent)
            if result == BLACK:
                current_wins += 1
                print(f"第{i+1}局: 当前(黑) 胜 ({length}手)")
            elif result == WHITE:
                original_wins += 1
                print(f"第{i+1}局: 原始(白) 胜 ({length}手)")
            else:
                draws += 1
                print(f"第{i+1}局: 平局 ({length}手)")
    
    print("\n" + "="*50)
    print(f"结果:")
    print(f"  原始权重: {original_wins} 胜")
    print(f"  当前权重: {current_wins} 胜")
    print(f"  平局:     {draws}")
    print("="*50)
    
    if original_wins > current_wins:
        print("\n结论: 原始权重更强，建议用原始权重重新训练")
    elif current_wins > original_wins:
        print("\n结论: 当前权重更强，继续训练有效果")
    else:
        print("\n结论: 两者水平相当")


if __name__ == "__main__":
    main()
