# IMPORTS
import os
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm # type: ignore
import keras # type: ignore
from keras import layers # type: ignore
import numpy as np # type: ignore
import tensorflow as tf # type: ignore

# BOARD DIMENSIONS AND PIECE INITIALIZATION
ROW_COUNT = 6
COLUMN_COUNT = 7
EMPTY = 2
X_PIECE = 1
O_PIECE = 0
MAX_DEPTH = 6

label = {X_PIECE: 'X', O_PIECE: 'O'}


# DQN Hyperparameters
DQN_EPISODES      = 300000   
DQN_EPSILON       = 1.0
DQN_EPSILON_DECAY = 0.99995
DQN_DISCOUNT      = 0.9
DQN_LR            = 0.001
DQN_BATCH_SIZE    = 256
DQN_REPLAY_SIZE   = 10000
DQN_TARGET_UPDATE = 200
DQN_TRAIN_FREQ    = 4       # train every N environment steps

# SIZE OF INPUT FOR DQN NETWORKS
STATE_SIZE = ROW_COUNT * COLUMN_COUNT * 3   # 126

# Precomputed index arrays for diagonal check_win
_IDX2 = np.arange(1, 3)
_IDX3 = np.arange(1, 4)

#-----------------------------------------------------------------



# Precomputed row/col index arrays for all heuristic windows (shape: N_WINDOWS x 4)
def build_window_indices():
    rows, cols = [], []
    # horizontal
    for r in range(ROW_COUNT):
        for c in range(COLUMN_COUNT - 3):
            rows.append([r] * 4)
            cols.append([c + i for i in range(4)])
    # vertical
    for c in range(COLUMN_COUNT):
        for r in range(ROW_COUNT - 3):
            rows.append([r + i for i in range(4)])
            cols.append([c] * 4)
    # diagonal /
    for r in range(ROW_COUNT - 3):
        for c in range(COLUMN_COUNT - 3):
            rows.append([r + i for i in range(4)])
            cols.append([c + i for i in range(4)])
    # diagonal backslash
    for r in range(3, ROW_COUNT):
        for c in range(COLUMN_COUNT - 3):
            rows.append([r - i for i in range(4)])
            cols.append([c + i for i in range(4)])
    return np.array(rows, dtype=np.int32), np.array(cols, dtype=np.int32)

WINDOW_ROWS, WINDOW_COLS = build_window_indices()

# Class ReplayBuffer to Simulate the replay buffer in DQN
class ReplayBuffer:
    """Fixed-size numpy buffer"""
    def __init__(self, capacity):
        self.capacity    = capacity
        self.ptr         = 0
        self.size        = 0
        self.states      = np.zeros((capacity, STATE_SIZE), dtype=np.float32)
        self.actions     = np.zeros(capacity, dtype=np.int32)
        self.rewards     = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, STATE_SIZE), dtype=np.float32)
        self.terminated  = np.zeros(capacity, dtype=bool)

    def add(self, state, action, reward, next_state, done):
        self.states[self.ptr]      = state
        self.actions[self.ptr]     = action
        self.rewards[self.ptr]     = reward
        self.next_states[self.ptr] = next_state
        self.terminated[self.ptr]  = done
        self.ptr  = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            self.states[idx], self.actions[idx], self.rewards[idx],
            self.next_states[idx], self.terminated[idx],
        )

    def __len__(self):
        return self.size

# Return state of the board as a byte string for faster processing
def get_state(board):
    return board.tobytes()

# initialize state if not in q_table. return state and valid moves
def init_state(board):
    state = get_state(board)
    valid_moves = get_valid_moves(board)
    if state not in q_table:
        q_table[state] = np.where(valid_moves != -1, 0.0, -np.inf)
    return state, valid_moves

# create an empty 6x7 board for connect4
def create_board():
    return np.full((ROW_COUNT, COLUMN_COUNT), EMPTY, dtype=np.int8)

# place a piece on the board
def make_move(board, row, col, piece):
    board[row][col] = piece

# get the next open row with 0 pieces on it
def get_next_open_row(board, col):
    empty_rows = np.where(board[:, col] == EMPTY)[0]
    return int(empty_rows[0]) if len(empty_rows) > 0 else None

# get the valid moves on the board
def get_valid_moves(board):
    empty_mask = (board == EMPTY)                      # (ROW_COUNT, COLUMN_COUNT)
    has_empty  = empty_mask.any(axis=0)                # (COLUMN_COUNT,) bool
    first_empty = np.argmax(empty_mask, axis=0)        # (COLUMN_COUNT,) — 0 when no empty
    valid_moves = np.full(COLUMN_COUNT, -1, dtype=np.int32)
    valid_moves[has_empty] = first_empty[has_empty]
    return valid_moves

# print the board
def print_board(board):
    symbol = {EMPTY: ' ', X_PIECE: 'X', O_PIECE: 'O'}
    for row in reversed(board):
        print('|' + '|'.join(f' {symbol[int(cell)]} ' for cell in row) + '|')
    print('  ' + '   '.join(str(c) for c in range(COLUMN_COUNT)))

# check for tie if all the places on the board are filled.
def check_tie(board):
    return bool(np.all(board != EMPTY))

def get_board_winner(board):
    for r in range(ROW_COUNT):
        for c in range(COLUMN_COUNT):
            if board[r][c] != EMPTY:
                is_win, winner = check_win(board, r, c, board[r][c])
                if is_win:
                    return winner
    return None

# check for win
def check_win(board, row, col, piece):
    dist_left   = col
    dist_right  = COLUMN_COUNT - 1 - col
    dist_bottom = row
    dist_top    = ROW_COUNT - 1 - row

    # Horizontal
    # check for case 'P', P, P, P  (piece is leftmost)
    if dist_left >= 3 and np.all(board[row, col-3:col] == piece):
        return True, piece
    # check for case P, P, P, 'P'  (piece is rightmost)
    if dist_right >= 3 and np.all(board[row, col+1:col+4] == piece):
        return True, piece
    if dist_left >= 1 and dist_right >= 1:
        # check for case P, P, 'P', P
        if dist_left >= 2 and np.all(board[row, col-2:col] == piece) and board[row, col+1] == piece:
            return True, piece
        # check for case P, 'P', P, P
        if dist_right >= 2 and board[row, col-1] == piece and np.all(board[row, col+1:col+3] == piece):
            return True, piece

    # Vertical — only downward (piece is always the highest in its column)
    if dist_bottom >= 3 and np.all(board[row-3:row, col] == piece):
        return True, piece

    # Diagonal '/' (bottom-left to top-right)
    # piece at pos 4 (top-right end): P P P 'P'
    if dist_left >= 3 and dist_bottom >= 3 and np.all(board[row - _IDX3, col - _IDX3] == piece):
        return True, piece
    # piece at pos 3: P P 'P' P
    if dist_left >= 2 and dist_bottom >= 2 and dist_right >= 1 and dist_top >= 1:
        if np.all(board[row - _IDX2, col - _IDX2] == piece) and board[row+1, col+1] == piece:
            return True, piece
    # piece at pos 2: P 'P' P P
    if dist_left >= 1 and dist_bottom >= 1 and dist_right >= 2 and dist_top >= 2:
        if board[row-1, col-1] == piece and np.all(board[row + _IDX2, col + _IDX2] == piece):
            return True, piece
    # piece at pos 1 (bottom-left end): 'P' P P P
    if dist_right >= 3 and dist_top >= 3 and np.all(board[row + _IDX3, col + _IDX3] == piece):
        return True, piece

    # Diagonal '\' (bottom-right to top-left)
    # piece at pos 4 (top-left end): P P P 'P'
    if dist_right >= 3 and dist_bottom >= 3 and np.all(board[row - _IDX3, col + _IDX3] == piece):
        return True, piece
    # piece at pos 3: P P 'P' P
    if dist_right >= 2 and dist_bottom >= 2 and dist_left >= 1 and dist_top >= 1:
        if np.all(board[row - _IDX2, col + _IDX2] == piece) and board[row+1, col-1] == piece:
            return True, piece
    # piece at pos 2: P 'P' P P
    if dist_right >= 1 and dist_bottom >= 1 and dist_left >= 2 and dist_top >= 2:
        if board[row-1, col+1] == piece and np.all(board[row + _IDX2, col - _IDX2] == piece):
            return True, piece
    # piece at pos 1 (bottom-right end): 'P' P P P
    if dist_left >= 3 and dist_top >= 3 and np.all(board[row + _IDX3, col - _IDX3] == piece):
        return True, piece

    return False, None

# return a reward
def get_reward(winner, num_moves, player=X_PIECE):
    # if player and winner are the same, return a positive reward, else a negative reward.
    # if the games ends in less than 15 moves, give a reward of 100-num_moves to encourage faster wins, else give a reward lower than 15.
    if player == X_PIECE:
        if winner == X_PIECE:
            if num_moves > 15:
                diff = num_moves - 15
                return max(1, diff)

            else:
                return 100 - num_moves    # win sooner = higher reward
        else:

            return -100 + num_moves   # lose sooner = more negative

    # same logic as above
    if player == O_PIECE:
        if winner == O_PIECE:
            if num_moves > 15:
                diff = num_moves - 15
                return max(1, diff)

            else:
                return 100 - num_moves

        else:
            return -100 + num_moves



def default_opponent(board, player=O_PIECE):
    opponent = 1 - player
    valid_moves = get_valid_moves(board)
    centre = COLUMN_COUNT // 2  # col 3

    # check if player can win immediately
    for col, row in enumerate(valid_moves):
        if row == -1:
            continue
        board[row][col] = player
        is_win, _ = check_win(board, row, col, player)
        board[row][col] = EMPTY
        if is_win:
            return col, int(row)

    # block opponent from winning
    for col, row in enumerate(valid_moves):
        if row == -1:
            continue
        board[row][col] = opponent
        is_win, _ = check_win(board, row, col, opponent)
        board[row][col] = EMPTY
        if is_win:
            return col, int(row)

    # pick lowest row; if tie pick closest to centre, prefer left on equidistance
    best_col = None
    best_row = None
    best_dist = float('inf')

    for col, row in enumerate(valid_moves):
        if row == -1:
            continue
        dist = abs(col - centre)
        if best_col is None:
            best_col, best_row, best_dist = col, int(row), dist
        elif row < best_row:
            best_col, best_row, best_dist = col, int(row), dist
        elif row == best_row:
            if dist < best_dist:
                best_col, best_row, best_dist = col, int(row), dist
            elif dist == best_dist and col < best_col:
                best_col, best_row, best_dist = col, int(row), dist

    return best_col, best_row


def heuristic_score(curr_board, curr_player, ai_player):
    opponent = 1 - curr_player
    all_windows = curr_board[WINDOW_ROWS, WINDOW_COLS]          # (N_WINDOWS, 4)
    player_counts   = np.sum(all_windows == curr_player, axis=1)
    opponent_counts = np.sum(all_windows == opponent,    axis=1)

    pure = ~((player_counts > 0) & (opponent_counts > 0))
    pc = player_counts[pure]
    oc = opponent_counts[pure]

    score_p = np.where(pc == 3, 9.0,  np.where(pc == 2, 3.0,  np.where(pc == 1, 0.5, 0.0)))
    score_o = np.where(oc == 3, 40.0, np.where(oc == 2, 2.0,  np.where(oc == 1, 0.5, 0.0)))
    final   = np.where(oc == 0, score_p, 0.0) + np.where(pc == 0, score_o, 0.0)

    num_pieces = max(int(np.sum(curr_board != EMPTY)), 1)
    total = float(np.sum(final)) / num_pieces
    sign  = 1 if curr_player == ai_player else -1
    return sign * total

def opening_board():
            board = create_board()
            c = int(np.random.randint(COLUMN_COUNT))
            r = int(get_valid_moves(board)[c])
            make_move(board, r, c, X_PIECE)
            print_board(board)
            return board

def minimax(curr_board, curr_player, depth=0, ai_player=X_PIECE):


    if depth >= MAX_DEPTH:
        return heuristic_score(curr_board, curr_player, ai_player), None, None

    valid_moves = get_valid_moves(curr_board)

    if check_tie(curr_board):
        return 0, None, None

    # check if last placed piece caused a win — scan all pieces
    for r in range(ROW_COUNT):
        for c in range(COLUMN_COUNT):
            if curr_board[r][c] != EMPTY:
                is_win, winner = check_win(curr_board, r, c, curr_board[r][c])
                if is_win:
                    return get_reward(winner, depth, player=ai_player), None, None

    opponent = 1 - curr_player

    # maximising player
    if curr_player == ai_player:
        best_score = -float('inf')
        best_col = None
        best_row = None
        for col, row in enumerate(valid_moves):
            if row == -1: # column full case
                continue
            new_board = curr_board.copy()
            new_board[row][col] = curr_player
            score, _, _ = minimax(new_board, opponent, depth + 1, ai_player)
            if score > best_score:
                best_score = score
                best_col = col
                best_row = int(row)
        return best_score, best_col, best_row

    else:
        best_score = float('inf')
        best_col = None
        best_row = None
        for col, row in enumerate(valid_moves):
            if row == -1:
                continue
            new_board = curr_board.copy()
            new_board[row][col] = curr_player
            score, _, _ = minimax(new_board, opponent, depth + 1, ai_player)
            if score < best_score:
                best_score = score
                best_col = col
                best_row = int(row)
        return best_score, best_col, best_row


def minimax_ab(curr_board, curr_player, depth=0, alpha=-float('inf'), beta=float('inf'), ai_player=X_PIECE):
   

    if depth >= MAX_DEPTH:
        return heuristic_score(curr_board, curr_player, ai_player), None, None

    valid_moves = get_valid_moves(curr_board)

    if check_tie(curr_board):
        return 0, None, None

    for r in range(ROW_COUNT):
        for c in range(COLUMN_COUNT):
            if curr_board[r][c] != EMPTY:
                is_win, winner = check_win(curr_board, r, c, curr_board[r][c])
                if is_win:
                    return get_reward(winner, depth, player=ai_player), None, None

    opponent = 1 - curr_player

    if curr_player == ai_player:
        best_score = -float('inf')
        best_col = None
        best_row = None
        for col, row in enumerate(valid_moves):
            if row == -1:
                continue
            new_board = curr_board.copy()
            new_board[row][col] = curr_player
            score, _, _ = minimax_ab(new_board, opponent, depth + 1, alpha, beta, ai_player)
            if score > best_score:
                best_score = score
                best_col = col
                best_row = int(row)
            alpha = max(alpha, best_score)
            if beta <= alpha:
                break
        return best_score, best_col, best_row

    else:
        best_score = float('inf')
        best_col = None
        best_row = None
        for col, row in enumerate(valid_moves):
            if row == -1:
                continue
            new_board = curr_board.copy()
            new_board[row][col] = curr_player
            score, _, _ = minimax_ab(new_board, opponent, depth + 1, alpha, beta, ai_player)
            if score < best_score:
                best_score = score
                best_col = col
                best_row = int(row)
            beta = min(beta, best_score)
            if beta <= alpha:
                break
        return best_score, best_col, best_row

#---------------------Q-Learning---------------------#
# Q-Learning Initialization and hyperparameters
q_table = {}
LR = 0.1
DISCOUNT = 0.95
EPSILON = 1.0
EPSILON_DECAY = 0.99995
EPSILON_MIN = 0

def random_opponent(board, player):
    valid_moves = get_valid_moves(board)
    valid_cols = np.where(np.array(valid_moves) != -1)[0]
    col = int(np.random.choice(valid_cols))
    return col, int(valid_moves[col])

def train_q_random(player, episodes=20000):
    global EPSILON
    opponent = 1 - player
    plot_data_random = {'episodes': [], 'win': [], 'loss': [], 'draw': []}
    wins = losses = draws = 0
    for ep_idx in tqdm(range(episodes), desc="Q-Learning vs Random"):
        board = create_board()
        terminated = False

        c = int(np.random.randint(COLUMN_COUNT))
        r = int(get_valid_moves(board)[c])
        make_move(board, r, c, X_PIECE)

        x_count = 0
        o_count = 0
        x_last_state = None
        x_last_col = None
        while not terminated:
            new_board = board.copy()

            if player == X_PIECE:
                # random opponent plays O first, then AI plays X
                o_col, o_row = random_opponent(new_board, opponent)
                new_board[o_row][o_col] = opponent
                o_count += 1
                is_win_o, _ = check_win(new_board, o_row, o_col, opponent)
                if is_win_o:
                    if x_last_state is not None:
                        reward = get_reward(opponent, o_count, player=player)
                        q_table[x_last_state][x_last_col] += LR * (reward - q_table[x_last_state][x_last_col])
                    terminated = True
                elif check_tie(new_board):
                    if x_last_state is not None:
                        q_table[x_last_state][x_last_col] += LR * (-50 - q_table[x_last_state][x_last_col])
                    terminated = True
                else:
                    # delayed Q-update for previous X action: next_state is now after O responded
                    if x_last_state is not None:
                        next_state, next_valid = init_state(new_board)
                        next_valid_cols = np.where(next_valid != -1)[0]
                        max_next_q = float(np.max(q_table[next_state][next_valid_cols]))
                        q_table[x_last_state][x_last_col] += LR * (DISCOUNT * max_next_q - q_table[x_last_state][x_last_col])

                if not terminated:
                    state, valid_moves = init_state(new_board)
                    valid_cols = np.where(valid_moves != -1)[0]
                    if np.random.random() < EPSILON:
                        col = int(np.random.choice(valid_cols))
                    else:
                        q_vals = q_table[state][valid_cols]
                        col = int(np.random.choice(valid_cols[q_vals == np.max(q_vals)]))
                    row = int(valid_moves[col])
                    new_board[row][col] = player
                    x_count += 1
                    x_last_state = state
                    x_last_col = col
                    is_win, winner = check_win(new_board, row, col, player)
                    if is_win:
                        reward = get_reward(winner, x_count, player=player)
                        q_table[state][col] += LR * (reward - q_table[state][col])
                        terminated = True
                    elif check_tie(new_board):
                        q_table[state][col] += LR * (-50 - q_table[state][col])
                        terminated = True
                    # non-terminal: Q-update deferred to next iteration after O responds

            elif player == O_PIECE:
                # AI plays O first, then random plays X
                state, valid_moves = init_state(new_board)
                valid_cols = np.where(valid_moves != -1)[0]
                if np.random.random() < EPSILON:
                    col = int(np.random.choice(valid_cols))
                else:
                    q_vals = q_table[state][valid_cols]
                    col = int(np.random.choice(valid_cols[q_vals == np.max(q_vals)]))
                row = int(valid_moves[col])
                new_board[row][col] = player
                o_count += 1
                is_win, winner = check_win(new_board, row, col, player)
                if is_win:
                    reward = get_reward(winner, o_count, player=player)
                    q_table[state][col] += LR * (reward - q_table[state][col])
                    terminated = True
                elif check_tie(new_board):
                    q_table[state][col] += LR * (-50 - q_table[state][col])
                    terminated = True

                if not terminated:
                    o_col, o_row = random_opponent(new_board, opponent)
                    new_board[o_row][o_col] = opponent
                    x_count += 1
                    is_win_x, _ = check_win(new_board, o_row, o_col, opponent)
                    if is_win_x:
                        reward = get_reward(opponent, x_count, player=player)
                        q_table[state][col] += LR * (reward - q_table[state][col])
                        terminated = True
                    elif check_tie(new_board):
                        q_table[state][col] += LR * (-50 - q_table[state][col])
                        terminated = True
                    else:
                        next_state, next_valid = init_state(new_board)
                        next_valid_cols = np.where(next_valid != -1)[0]
                        max_next_q = float(np.max(q_table[next_state][next_valid_cols]))
                        q_table[state][col] += LR * (DISCOUNT * max_next_q - q_table[state][col])

            board = new_board

        winner = get_board_winner(board)
        if winner == player:
            wins += 1
        elif winner is not None:
            losses += 1
        else:
            draws += 1

        if (ep_idx + 1) % 100 == 0:
            total = wins + losses + draws
            if total > 0:
                plot_data_random['episodes'].append(ep_idx + 1)
                plot_data_random['win'].append(wins / total)
                plot_data_random['loss'].append(losses / total)
                plot_data_random['draw'].append(draws / total)
                wins = losses = draws = 0

        EPSILON = EPSILON * EPSILON_DECAY

    print("Q-table pre-trained vs random. Continuing vs default opponent...")
    return plot_data_random

def train_qlearning(player, episodes=100000):
    global EPSILON
    plot_data_random = train_q_random(player)
    EPSILON = 1.0
    opponent = 1 - player
    plot_data_default = {'episodes': [], 'win': [], 'loss': [], 'draw': []}
    wins = losses = draws = 0
    for ep_idx in tqdm(range(episodes), desc="Q-Learning Training"):
        board = create_board()
        terminated = False

        
        c = int(np.random.randint(COLUMN_COUNT))
        r = int(get_valid_moves(board)[c])
        make_move(board, r, c, X_PIECE)

        x_count = 0
        o_count = 0
        x_last_state = None
        x_last_col = None
        while not terminated:
            new_board = board.copy()

            if player == X_PIECE:
                # opponent plays O first (response to opening X), then AI plays X
                o_col, o_row = default_opponent(new_board, opponent)
                new_board[o_row][o_col] = opponent
                o_count += 1
                is_win_o, _ = check_win(new_board, o_row, o_col, opponent)
                if is_win_o:
                    if x_last_state is not None:
                        reward = get_reward(opponent, o_count, player=player)
                        q_table[x_last_state][x_last_col] += LR * (reward - q_table[x_last_state][x_last_col])
                    terminated = True
                elif check_tie(new_board):
                    if x_last_state is not None:
                        q_table[x_last_state][x_last_col] += LR * (-50 - q_table[x_last_state][x_last_col])
                    terminated = True
                else:
                    # delayed Q-update for previous X action: next_state is now after O responded
                    if x_last_state is not None:
                        next_state, next_valid = init_state(new_board)
                        next_valid_cols = np.where(next_valid != -1)[0]
                        max_next_q = float(np.max(q_table[next_state][next_valid_cols]))
                        q_table[x_last_state][x_last_col] += LR * (DISCOUNT * max_next_q - q_table[x_last_state][x_last_col])

                if not terminated:
                    # AI plays X
                    state, valid_moves = init_state(new_board)
                    valid_cols = np.where(valid_moves != -1)[0]
                    if np.random.random() < EPSILON:
                        col = int(np.random.choice(valid_cols))
                    else:
                        q_vals = q_table[state][valid_cols]
                        col = int(np.random.choice(valid_cols[q_vals == np.max(q_vals)]))
                    row = int(valid_moves[col])
                    new_board[row][col] = player
                    x_count += 1
                    x_last_state = state
                    x_last_col = col
                    is_win, winner = check_win(new_board, row, col, player)
                    if is_win:
                        reward = get_reward(winner, x_count, player=player)
                        q_table[state][col] += LR * (reward - q_table[state][col])
                        terminated = True
                    elif check_tie(new_board):
                        q_table[state][col] += LR * (-50 - q_table[state][col])
                        terminated = True
                    # non-terminal: Q-update deferred to next iteration after O responds

            elif player == O_PIECE:
                # AI plays O first, then default plays X
                state, valid_moves = init_state(new_board)
                valid_cols = np.where(valid_moves != -1)[0]
                if np.random.random() < EPSILON:
                    col = int(np.random.choice(valid_cols))
                else:
                    q_vals = q_table[state][valid_cols]
                    col = int(np.random.choice(valid_cols[q_vals == np.max(q_vals)]))
                row = int(valid_moves[col])
                new_board[row][col] = player
                o_count += 1
                is_win, winner = check_win(new_board, row, col, player)
                if is_win:
                    reward = get_reward(winner, o_count, player=player)
                    q_table[state][col] += LR * (reward - q_table[state][col])
                    terminated = True
                elif check_tie(new_board):
                    q_table[state][col] += LR * (-50 - q_table[state][col])
                    terminated = True

                if not terminated:
                    # default plays X
                    o_col, o_row = default_opponent(new_board, opponent)
                    new_board[o_row][o_col] = opponent
                    x_count += 1
                    is_win_x, _ = check_win(new_board, o_row, o_col, opponent)
                    if is_win_x:
                        reward = get_reward(opponent, x_count, player=player)
                        q_table[state][col] += LR * (reward - q_table[state][col])
                        terminated = True
                    elif check_tie(new_board):
                        q_table[state][col] += LR * (-50 - q_table[state][col])
                        terminated = True
                    else:
                        next_state, next_valid = init_state(new_board)
                        next_valid_cols = np.where(next_valid != -1)[0]
                        max_next_q = float(np.max(q_table[next_state][next_valid_cols]))
                        q_table[state][col] += LR * (DISCOUNT * max_next_q - q_table[state][col])

            board = new_board

        winner = get_board_winner(board)
        if winner == player:
            wins += 1
        elif winner is not None:
            losses += 1
        else:
            draws += 1

        if (ep_idx + 1) % 100 == 0:
            total = wins + losses + draws
            if total > 0:
                plot_data_default['episodes'].append(ep_idx + 1)
                plot_data_default['win'].append(wins / total)
                plot_data_default['loss'].append(losses / total)
                plot_data_default['draw'].append(draws / total)
                wins = losses = draws = 0

        EPSILON = EPSILON * EPSILON_DECAY

    label = {X_PIECE: 'X', O_PIECE: 'O'}
    fname = f'connect4_q_table_{label[player]}.pkl'
    with open(fname, 'wb') as f:
        pickle.dump(q_table, f)
    print(f"Q-table saved to {fname}")
    _plot_q_training(plot_data_random, plot_data_default, player)

def _plot_q_training(plot_data_random, plot_data_default, player):
    label_p = {X_PIECE: 'X', O_PIECE: 'O'}[player]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    phase_info = [
        (plot_data_random,  f'Phase 1 – Random Opponent (Player {label_p})'),
        (plot_data_default, f'Phase 2 – Default Opponent (Player {label_p})'),
    ]
    for ax, (d, title) in zip(axes, phase_info):
        if not d['episodes']:
            ax.set_title(title + '\n(no data)')
            continue
        ax.plot(d['episodes'], d['win'],  label='Win Rate',  color='green')
        ax.plot(d['episodes'], d['loss'], label='Loss Rate', color='red')
        ax.plot(d['episodes'], d['draw'], label='Draw Rate', color='blue')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Rate')
        ax.set_title(title)
        ax.set_ylim(0, 1)
        ax.legend()
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'q_training_{label_p}.png', dpi=150)
    plt.show()

def qlearning(currboard):
    state, valid_moves = init_state(currboard)
    valid_cols = np.where(valid_moves != -1)[0]
    q_vals = q_table[state][valid_cols]
    col = int(np.random.choice(valid_cols[q_vals == np.max(q_vals)]))
    return col, int(valid_moves[col])

#---------------------Deep Q-Network---------------------#

def create_model():
    model = keras.Sequential([
        layers.Input(shape=(126,)),
        layers.Dense(150, activation="relu"),
        layers.Dense(150, activation="relu"),
        layers.Dense(150, activation="relu"),
        layers.Dense(150, activation="relu"),
        layers.Dropout(rate=0.3),
        layers.Dense(7)
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=DQN_LR), loss="mse")
    return model

def encode_state(currboard, player=X_PIECE):
    opponent = 1 - player
    flat = currboard.ravel()                                   # (ROW_COUNT*COLUMN_COUNT,)
    result = np.zeros(ROW_COUNT * COLUMN_COUNT * 3, dtype=np.float32)
    result[0::3] = (flat == player).astype(np.float32)        # AI's own piece
    result[1::3] = (flat == EMPTY).astype(np.float32)         # empty cell
    result[2::3] = (flat == opponent).astype(np.float32)      # opponent's piece
    return result

def dqn_choose_action(board, eps, player=X_PIECE):
    valid_moves = get_valid_moves(board)
    valid_cols = np.where(valid_moves != -1)[0]
    if np.random.random() < eps:
        return int(np.random.choice(valid_cols))
    state_tensor = tf.constant(encode_state(board, player)[np.newaxis])  # (1, STATE_SIZE)
    q_values = infer(state_tensor).numpy()[0]
    valid_q = q_values[valid_cols]
    return int(np.random.choice(valid_cols[valid_q == np.max(valid_q)]))

def dqn_replay_train():
    if len(replay_buffer) < DQN_BATCH_SIZE:
        return
    states, actions, rewards, next_states, terminated = replay_buffer.sample(DQN_BATCH_SIZE)
    train_step(states, actions, rewards, next_states, terminated)

model_policy = create_model()
model_target = create_model()
model_target.set_weights(model_policy.get_weights())
replay_buffer = ReplayBuffer(DQN_REPLAY_SIZE)
dqn_optimizer = keras.optimizers.Adam(learning_rate=DQN_LR)

@tf.function(input_signature=[tf.TensorSpec(shape=(1, STATE_SIZE), dtype=tf.float32)])
def infer(x):
    return model_policy(x, training=False)

@tf.function(input_signature=[
    tf.TensorSpec(shape=(None, STATE_SIZE), dtype=tf.float32),
    tf.TensorSpec(shape=(None,),            dtype=tf.int32),
    tf.TensorSpec(shape=(None,),            dtype=tf.float32),
    tf.TensorSpec(shape=(None, STATE_SIZE), dtype=tf.float32),
    tf.TensorSpec(shape=(None,),            dtype=tf.bool),
])
def train_step(states, actions, rewards, next_states, terminated):
    next_q   = model_target(next_states, training=False)
    max_next = tf.reduce_max(next_q, axis=1)
    targets  = tf.where(terminated, rewards, rewards + DQN_DISCOUNT * max_next)
    with tf.GradientTape() as tape:
        q_vals   = model_policy(states, training=True)
        idx      = tf.stack([tf.range(tf.shape(states)[0]), actions], axis=1)
        action_q = tf.gather_nd(q_vals, idx)
        loss     = tf.reduce_mean(tf.square(targets - action_q))
    grads = tape.gradient(loss, model_policy.trainable_variables)
    dqn_optimizer.apply_gradients(zip(grads, model_policy.trainable_variables))

def train_dqn(player=X_PIECE):
    global DQN_EPSILON
    opponent = 1 - player
    label = {X_PIECE: 'X', O_PIECE: 'O'}
    step = 0
    for ep in tqdm(range(DQN_EPISODES), desc="DQN Training"):
        board = create_board()
        terminated = False
        ai_moves = 0

        # opening: one random X move then one random O move
        
        c = int(np.random.randint(COLUMN_COUNT))
        r = int(get_valid_moves(board)[c])
        make_move(board, r, c, X_PIECE)

        while not terminated:

            if player == X_PIECE:
                # opponent plays O
                opp_col, opp_row = default_opponent(board, opponent)
                board[opp_row][opp_col] = opponent
                is_win_opp, _ = check_win(board, opp_row, opp_col, opponent)
                if is_win_opp or check_tie(board):
                    terminated = True
                    break

                # AI plays X
                state = encode_state(board, player)
                col = dqn_choose_action(board, DQN_EPSILON, player)
                row = int(get_valid_moves(board)[col])
                board[row][col] = player
                ai_moves += 1

                reward = 0
                is_win, _ = check_win(board, row, col, player)
                if is_win:
                    reward = get_reward(player, ai_moves, player=player)
                    terminated = True
                elif check_tie(board):
                    reward = -50
                    terminated = True

                next_state = encode_state(board, player)
                replay_buffer.add(state, col, reward, next_state, terminated)
                step += 1
                if step % DQN_TRAIN_FREQ == 0:
                    dqn_replay_train()

            elif player == O_PIECE:
                # AI plays O
                state = encode_state(board, player)
                col = dqn_choose_action(board, DQN_EPSILON, player)
                row = int(get_valid_moves(board)[col])
                board[row][col] = player
                ai_moves += 1

                reward = 0
                is_win, _ = check_win(board, row, col, player)
                if is_win:
                    reward = get_reward(player, ai_moves, player=player)
                    terminated = True
                elif check_tie(board):
                    reward = -50
                    terminated = True

                if not terminated:
                    # opponent plays X
                    opp_col, opp_row = default_opponent(board, opponent)
                    board[opp_row][opp_col] = opponent
                    is_win_opp, _ = check_win(board, opp_row, opp_col, opponent)
                    if is_win_opp or check_tie(board):
                        reward = get_reward(opponent, ai_moves, player=player) if is_win_opp else -50
                        terminated = True

                next_state = encode_state(board, player)
                replay_buffer.add(state, col, reward, next_state, terminated)
                step += 1
                if step % DQN_TRAIN_FREQ == 0:
                    dqn_replay_train()

        DQN_EPSILON = DQN_EPSILON * DQN_EPSILON_DECAY

        if (ep + 1) % DQN_TARGET_UPDATE == 0:
            model_target.set_weights(model_policy.get_weights())

        if (ep + 1) % 1000 == 0:
            model_policy.save(f'connect4_dqn_{label[player]}_{ep + 1}.keras')

    model_policy.save(f'connect4_dqn_{label[player]}.keras')

def dqn(board, player=X_PIECE):
    valid_moves = get_valid_moves(board)
    valid_cols = np.where(valid_moves != -1)[0]
    state_tensor = tf.expand_dims(tf.convert_to_tensor(encode_state(board, player)), axis=0)
    q_values = model_policy(state_tensor).numpy()[0]
    valid_q = q_values[valid_cols]
    col = int(np.random.choice(valid_cols[valid_q == np.max(valid_q)]))
    return col, int(valid_moves[col])

def AI_player(board, strategy, player=X_PIECE):
    if strategy == 1:
        _, col, row = minimax(board, player, ai_player=player)
        return col, row
    elif strategy == 2:
        _, col, row = minimax_ab(board, player, ai_player=player)
        return col, row
    elif strategy == 3:
        return qlearning(board)
    elif strategy == 4:
        return dqn(board, player)

def pick_algorithm(prompt):
    print(f"\n-------------------{prompt}-------------------\n")
    print("1. Minimax")
    print("2. Minimax with Alpha Beta Pruning")
    print("3. Q-Learning")
    print("4. Deep Q-Learning Network (DQN)")
    return int(input("\nEnter Choice: "))

first_time = True
def setup_algorithm(strategy, player):
    """Train or load a model for the given strategy/player if required."""
    global model_policy, first_time
    label = {X_PIECE: 'X', O_PIECE: 'O'}
    if strategy == 3 and first_time:
        first_time = False
        fname = f'connect4_q_table_{label[player]}.pkl'
        if os.path.exists(fname):
            print(f"Loading existing Q-table from {fname}...")
            with open(fname, 'rb') as f:
                q_table.update(pickle.load(f))
        else:
            train_qlearning(player=player)
    elif strategy == 4:
        fname = f'connect4_dqn_{label[player]}.keras'
        if os.path.exists(fname):
            print(f"Loading existing DQN model from {fname}...")
            model_policy = keras.models.load_model(fname)
        else:
            train_dqn(player=player)

def run_game(board, move_fn_x, move_fn_o):
    """Run a game loop given two move functions. Returns the winner label or 'Tie'."""

    current = X_PIECE
    while True:
        col, row = move_fn_x(board) if current == X_PIECE else move_fn_o(board)
        make_move(board, row, col, current)
        print_board(board)
        is_win, _ = check_win(board, row, col, current)
        if is_win:
            print(f"{label[current]} wins!")
            return label[current]
        if check_tie(board):
            print("It's a tie!")
            return "Tie"
        current = 1 - current


def play():
    global states_list, time_taken_list, win, draw, loss, num_states
    # MAPPING USER INPUT TO BOARD PIECES
    symbol = {'X': X_PIECE, 'O': O_PIECE, 'x': X_PIECE, 'o': O_PIECE}

    # AI vs Default / AI vs AI 
    print("\n===================== SELECT MODE =====================\n")
    print("1. AI vs Default Opponent")
    print("2. AI vs AI")
    mode = int(input("\nEnter Choice: "))

    # create an empty board
    board = create_board()
    
    # AI vs Default Mode
    if mode == 1:
        # Picking AI algorithm
        strategy = pick_algorithm("PICK AI ALGORITHM")
        
        # Picking 'X' or 'O' for AI player to play as
        player_input = input("\nEnter the player this algorithm will play as (X/O): ")
        if player_input not in symbol:
            print("Wrong Input !!! Terminating !!!")
            exit()
        player = symbol[player_input]

        # Train algorithms like qlearning or dqn or load their respective models
        setup_algorithm(strategy, player)

        # make the first 'X' random move
        c = int(np.random.randint(COLUMN_COUNT))
        r = int(get_valid_moves(board)[c])
        make_move(board, r, c, X_PIECE)
        print_board(board)
        
        # If player is X, O goes first as random move for X was made
        if player == X_PIECE:
            # Opponent responds with O first, then run_game: AI(X) → opponent(O) → ...
            col, row = default_opponent(board, O_PIECE)
            make_move(board, row, col, O_PIECE)
            print_board(board)

            # lambda functions to get board coordinates to place a piece upon
            move_x = lambda b: AI_player(b, strategy, X_PIECE)
            move_o = lambda b: default_opponent(b, O_PIECE)
        # If player is O, player goes first as random move for X was made
        else:
            # AI responds with O first, then run_game: default(X) → AI(O) → ...
            col, row = AI_player(board, strategy, O_PIECE)
            make_move(board, row, col, O_PIECE)
            print_board(board)
            move_x = lambda b: default_opponent(b, X_PIECE)
            move_o = lambda b: AI_player(b, strategy, O_PIECE)
        
        # run the game
        run_game(board, move_x, move_o)

    # AI vs AI mode
    elif mode == 2:

        # smaking a random 'X' first move
        c = int(np.random.randint(COLUMN_COUNT))
        r = int(get_valid_moves(board)[c])
        make_move(board, r, c, X_PIECE)
        print_board(board)

        # getting algorithms for both AI agents to play as
        strategy_x = pick_algorithm("PICK ALGORITHM FOR X (Player 1)")
        strategy_o = pick_algorithm("PICK ALGORITHM FOR O (Player 2)")

        # training or loading of models if required
        setup_algorithm(strategy_x, X_PIECE)
        setup_algorithm(strategy_o, O_PIECE)

        # AI(O) responds first, then run_game: AI(X) → AI(O) → ...
        col, row = AI_player(board, strategy_o, O_PIECE)
        make_move(board, row, col, O_PIECE)
        print_board(board)
        move_x = lambda b: AI_player(b, strategy_x, X_PIECE)
        move_o = lambda b: AI_player(b, strategy_o, O_PIECE)
        
        # run the game
        run_game(board, move_x, move_o)

    # handling invalid input
    else:
        print("Invalid choice. Terminating.")
        exit()

play()
