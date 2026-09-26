import os
import copy
import random
import numpy as np #type: ignore
import pickle
from tqdm import tqdm #type: ignore
import keras #type: ignore
from keras import layers #type: ignore
import tensorflow as tf #type: ignore
from collections import deque
import sys


# Initializing empty board
board = [' '] * 9

# winning combos
combos = [
        [0,1,2],[3,4,5],[6,7,8],  # rows
        [0,3,6],[1,4,7],[2,5,8],  # cols
        [0,4,8],[2,4,6]           # diagonals
    ]

# flags to monitor first moves for both players
first_move_X = True
first_move_O = True

# Q-Learning Initializations and Hyperparameters
Q = {}
learning_rate = 0.1
discount_factor = 0.9
exploration_rate = 1
num_episodes = 300000
decay_rate = 0.99995
is_training = False


#----------------------UTILITY FUNCTIONS----------------------#
def print_board():
    for i in range(0, 9, 3):
        print(f" {board[i]} | {board[i+1]} | {board[i+2]} ")
        if i < 6:
            print("---|---|---")
    print("\n\n")

def make_move(position, player):
    if board[position] == ' ':
        board[position] = player
        if not is_training:
            print_board()
        return True
    if not is_training:
        print("Invalid Move !!!")
    return False

def make_dummy_move(position, player, current_board):
    if current_board[position] == ' ':
        current_board[position] = player
        return True
    return False

def default_opponent_pos(player, current_board):
    opponent = "O" if player == "X" else "X"

    for a,b,c in combos:
        values = [current_board[a], current_board[b], current_board[c]]
        if values.count(player) == 2 and values.count(" ") == 1:
            return [a,b,c][values.index(' ')]

    for a,b,c in combos:
        values = [current_board[a], current_board[b], current_board[c]]
        if values.count(opponent) == 2 and values.count(" ") == 1:
            return [a,b,c][values.index(' ')]

    for pos in [0, 2, 6, 8, 4, 1, 3, 5, 7]:
        if current_board[pos] == " ":
            return pos

def default_opponent(player):
    pos = default_opponent_pos(player, board)
    if pos == None:
        return 
    make_move(pos, player)

def AI_opponent(strategy, player):
    successors = get_succesors(board)

    if strategy == 3:
        pos = best_action(board)
        make_move(pos, player)
    elif strategy == 4:
        pos = dqn_best_action(board,player)
        make_move(pos, player)
    else:
        best_score = -float('inf')
        best_pos = None
        opponent = "O" if player == "X" else "X"

        for pos in successors:
            new_board = copy.deepcopy(board)
            make_dummy_move(pos, player, new_board)
            if strategy == 2:
                score = minimax_AB(new_board, False, player=player, opponent=opponent, alpha=-float('inf'), beta=float('inf'))
            else:
                score = minimax(new_board, False, player=player, opponent=opponent, depth=0)
            if score > best_score:
                best_score = score
                best_pos = pos

        make_move(best_pos, player)

def get_succesors(current_board):
    index_list = []
    for i in range(9):
        if current_board[i] == ' ':
            index_list.append(i)
    return index_list

def get_reward(winner, depth, player="X"):
    if player == "X":
        if winner == "X":
            return 100 - depth
        elif winner == "O":
            return -100 + depth
    elif player == "O":
        if winner == "O":
            return 100 - depth
        elif winner == "X":
            return -100 + depth

def reset():
    global board, first_move_X, first_move_O
    board = [' '] * 9
    first_move_X = True
    first_move_O = True

def has_winner(current_board=None):
    current_board = current_board if current_board is not None else board
    for a, b, c in combos:
        if current_board[a] == current_board[b] == current_board[c] and current_board[a] != ' ':
            return True, current_board[a]
    return False, None

def has_tied(current_board=None):
    current_board = current_board if current_board is not None else board
    is_win, _ = has_winner(current_board)
    return ' ' not in current_board and not is_win


#----------------------MINIMAX ALGORITHMS----------------------#

def minimax(current_board, maximisingPlayer,
             player, opponent, depth=0):
    
    is_win, winner = has_winner(current_board)
    if is_win:
        return get_reward(winner, depth, player=player)
    if has_tied(current_board):
        return 0

    if maximisingPlayer:
        maxEval = -float("inf")
        for child in get_succesors(current_board):
            new_board = copy.deepcopy(current_board)
            make_dummy_move(child, player, new_board)
            eval = minimax(new_board, False, player=player, 
                           opponent=opponent, depth=depth + 1)
            maxEval = max(maxEval, eval)
        return maxEval
    else:
        minEval = float('inf')
        for child in get_succesors(current_board):
            new_board = copy.deepcopy(current_board)
            make_dummy_move(child, opponent, new_board)
            eval = minimax(new_board, True, player=player, 
                           opponent=opponent, depth=depth + 1)
            minEval = min(minEval, eval)
        return minEval

def minimax_AB(current_board, maximisingPlayer, player, 
               opponent, alpha, beta, depth=0):
    is_win, winner = has_winner(current_board)
    if is_win:
        return get_reward(winner, depth, player=player)
    if has_tied(current_board):
        return 0

    if maximisingPlayer:
        maxEval = -float("inf")
        for child in get_succesors(current_board):
            new_board = copy.deepcopy(current_board)
            make_dummy_move(child, player, new_board)
            eval = minimax_AB(new_board, False, player=player, 
                              opponent=opponent, alpha=alpha, beta=beta, depth=depth + 1)
            maxEval = max(maxEval, eval)
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        return maxEval
    else:
        minEval = float('inf')
        for child in get_succesors(current_board):
            new_board = copy.deepcopy(current_board)
            make_dummy_move(child, opponent, new_board)
            eval = minimax_AB(new_board, True, player=player, 
                    opponent=opponent, alpha=alpha, beta=beta, depth=depth + 1)
            minEval = min(minEval, eval)
            beta = min(beta, eval)
            if beta <= alpha:
                break
        return minEval


#----------------------Q LEARNING----------------------#
def board_to_string(current_board):
    return ''.join(current_board)

def choose_action(current_board, exp_rate, state):
    
    empty_cells = get_succesors(current_board)

    if random.uniform(0, 1) < exp_rate or state not in Q:
        action = random.choice(empty_cells)
    else:
        q_values = Q[state]
        empty_q_values = [q_values[cell] for cell in empty_cells]
        max_q_value = max(empty_q_values)
        max_q_indices = [i for i in range(len(empty_cells)) if empty_q_values[i] == max_q_value]
        max_q_index = random.choice(max_q_indices)
        action = empty_cells[max_q_index]

    return action

def update_q_table(state, action, next_state, reward):
    q_values = Q.get(state, np.zeros(9))
    next_q_values = Q.get(board_to_string(next_state), np.zeros(9))
    max_next_q_value = np.max(next_q_values)

    q_values[action] += learning_rate * (reward + discount_factor * max_next_q_value - q_values[action])
    Q[state] = q_values

def best_action(current_board):
    state = board_to_string(current_board)
    empty_cells = get_succesors(current_board)

    if state not in Q:
        
        return random.choice(empty_cells)

    q_values = Q[state]
    empty_q_values = [q_values[cell] for cell in empty_cells]
    max_q_value = max(empty_q_values)
    return random.choice([empty_cells[i] for i, v in enumerate(empty_q_values) if v == max_q_value])

def train(player):
    global  is_training, exploration_rate, Q
    string1 = "q_table_X.pkl" if player == "X" else "q_table_O.pkl"

    if os.path.exists(string1):
        with open(string1, 'rb') as f:
            Q = pickle.load(f)
        print("Loaded q_table from " + string1)

        return

    Q = {}
    is_training = True
    opponent = "O" if player == "X" else "X"

    random_episodes = 50000 
    phases = [ 
        (random_episodes, True,  "Q-Learning Phase 1 (Random Opp)"), 
        (num_episodes,    False, "Q-Learning Phase 2 (Default Opp)"), 
    ] 



    for n_eps, use_random, desc in phases: 
        exploration_rate = 1.0 

        for _ in tqdm(range(n_eps), desc=desc):  
            reset()

            # X plays a random opening move
            make_move(random.choice([0,1,2,3,4,5,6,7,8]), "X")

            game_over = False
            last_state = None
            last_action = None

            while not game_over:
                if player == "X":
                    terminated = False

                    if use_random: 
                        successors = get_succesors(board) 
                        if successors: 
                            make_move(random.choice(successors), opponent) 
                    else:  
                        default_opponent(opponent)

                    is_win, winner = has_winner()
                    if is_win:
                        terminated = True
                    if has_tied():
                        terminated = True

                    if last_state is not None:
                        next_state = board_to_string(board)
                        next_q_values = Q.get(next_state, np.zeros(9))
                        empty_cells = get_succesors(board)
                        if empty_cells:
                            max_next_q = max([next_q_values[c] for c in empty_cells])
                        else:
                            max_next_q = 0
                        q_values = Q.get(last_state, np.zeros(9))
                        q_values[last_action] += learning_rate * (discount_factor * max_next_q - q_values[last_action])
                        Q[last_state] = q_values


                    if not terminated:
                        state = board_to_string(board)
                        action = choose_action(board, exploration_rate,state)
                        make_move(action, player)
                        depth = 9 - len(get_succesors(board))

                        is_win, winner = has_winner()
                        if is_win:
                            reward = get_reward(winner, depth, player)
                            update_q_table(state, action, board, reward)
                            game_over = True
                            
                        if has_tied():
                            update_q_table(state, action, board, -50)
                            game_over = True
                            

                        last_state = state
                        last_action = action

                    if terminated: break



                elif player == "O":
                    state = board_to_string(board)
                    action = choose_action(board, exploration_rate, state)
                    make_move(action, player)
                    depth = 9 - len(get_succesors(board))
                    is_win, winner = has_winner()
                    if is_win:
                        reward = get_reward(winner, depth, player)
                        update_q_table(state, action, board, reward)
                        if last_state is not None:
                            update_q_table(last_state, last_action, board, reward)
                        game_over = True
                        break
                    if has_tied():
                        update_q_table(state, action, board, 0.001)
                        if last_state is not None:
                            update_q_table(last_state, last_action, board, 0.001)
                        game_over = True
                        break

                    if use_random:  
                        successors = get_succesors(board) 
                        if successors:  
                            make_move(random.choice(successors), opponent) 
                    else: 
                        default_opponent(opponent)
                    depth = 9 - len(get_succesors(board))

                    is_win, winner = has_winner()
                    if is_win:
                        if last_state is not None:
                            reward = get_reward(winner, depth, player)
                            update_q_table(last_state, last_action, board, reward)
                        game_over = True
            
                        break
                    if has_tied():
                        if last_state is not None:
                            update_q_table(last_state, last_action, board, 0.001)
                        game_over = True
                        break


                    if last_state is not None:
                        next_q_values = Q.get(state, np.zeros(9))
                        empty_cells = get_succesors(board)
                        if empty_cells:
                            max_next_q = max([next_q_values[c] for c in empty_cells])
                        else:
                            max_next_q = 0
                        q_values = Q.get(last_state, np.zeros(9))
                        q_values[last_action] += learning_rate * (discount_factor * max_next_q - q_values[last_action])
                        Q[last_state] = q_values

                    last_state = state
                    last_action = action

            exploration_rate = exploration_rate * decay_rate

    is_training = False
    with open(string1, 'wb') as f:
        pickle.dump(Q, f)

def print_qtable():
    for state, values in Q.items():
        print(f"State: {state}")
        print(f"Q-values: {[round(v, 4) for v in values]}")
        print()


#----------------------Deep Q NETWORK----------------------#
# DQN hyperparameters
dqn_episodes      = 100000
dqn_epsilon        = 1.0
dqn_epsilon_min    = 0.05
dqn_epsilon_decay  = 0.99995
dqn_discount       = 0.95
dqn_lr             = 0.0005
dqn_batch_size     = 128
dqn_replay_size    = 50000
dqn_target_model_update  = 500
dqn_train_freq     = 10       

STATE_SIZE = 27  # 9 cells * 3 one-hot channels

def dqn_create_model():
    model = keras.Sequential([
        layers.Input(shape=(STATE_SIZE,)),
        layers.Dense(36, activation="relu"),
        layers.Dense(36, activation="relu"),
        layers.Dense(9),
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=dqn_lr), loss="mse")
    return model

def dqn_encode_state(brd, player="X"):
    opponent = "O" if player == "X" else "X"
    vec = []
    for cell in brd:
        if cell == " ":      vec.extend((0, 1, 0))
        elif cell == player:  vec.extend((1, 0, 0))
        elif cell == opponent: vec.extend((0, 0, 1))
    return np.array(vec, dtype=np.float32)

def dqn_choose_action(brd, eps, player="X"):
    successors = get_succesors(brd)
    if random.random() < eps:
        return random.choice(successors)
    state_tensor = tf.expand_dims(
        tf.convert_to_tensor(dqn_encode_state(brd, player)), axis=0
    )
    q_values = dqn_policy_model(state_tensor, training=False).numpy()[0]
    mask = np.full(9, -np.inf)
    for i in successors:
        mask[i] = q_values[i]
    return int(np.argmax(mask))

def dqn_replay_train():
    if len(dqn_replay_buffer) < dqn_batch_size:
        return

    batch       = random.sample(dqn_replay_buffer, dqn_batch_size)
    states      = np.array([e[0] for e in batch], dtype=np.float32)
    actions     = np.array([e[1] for e in batch], dtype=np.int32)
    rewards     = np.array([e[2] for e in batch], dtype=np.float32)
    next_states = np.array([e[3] for e in batch], dtype=np.float32)
    terminated_ = np.array([e[4] for e in batch], dtype=bool)
    valid_masks = np.array([e[5] for e in batch], dtype=np.float32)

    next_q_policy = dqn_policy_model(next_states, training=False).numpy()
    next_q_target = dqn_target_model(next_states, training=False).numpy()

    next_q_policy_masked = np.where(valid_masks > 0, next_q_policy, -np.inf)
    best_next_actions = np.argmax(next_q_policy_masked, axis=1)

    target_q = dqn_policy_model(states, training=False).numpy()

    for i in range(dqn_batch_size):
        if terminated_[i]:
            target_q[i][actions[i]] = rewards[i]
        else:
            target_q[i][actions[i]] = rewards[i] 
            + dqn_discount * next_q_target[i][best_next_actions[i]]

    dqn_policy_model.fit(states, target_q, 
            verbose=0, batch_size=dqn_batch_size)

def dqn_best_action(brd, player="X"):
    successors = get_succesors(brd)
    if not successors:
        return None
    state_tensor = tf.expand_dims(
        tf.convert_to_tensor(dqn_encode_state(brd, player)), axis=0
    )
    q_values = dqn_policy_model(state_tensor, training=False).numpy()[0]
    mask = np.full(9, -np.inf)
    for i in successors:
        mask[i] = q_values[i]
    return int(np.argmax(mask))

def get_valid_mask(brd):
    return np.array([1.0 if brd[i] == ' ' else 0.0 for i in range(9)], dtype=np.float32)

def dqn_train(player):
    global dqn_epsilon, is_training, dqn_policy_model, dqn_target_model, dqn_replay_buffer
    model_path = f"model_ttt_{player}.keras"
    if os.path.exists(model_path):
        print(f"Loading existing DQN model from {model_path}...")
        dqn_policy_model = keras.models.load_model(model_path)
        dqn_target_model = dqn_create_model()
        dqn_target_model.set_weights(dqn_policy_model.get_weights())
        return
    dqn_policy_model = dqn_create_model()
    dqn_target_model = dqn_create_model()
    dqn_target_model.set_weights(dqn_policy_model.get_weights())
    dqn_replay_buffer = deque(maxlen=dqn_replay_size)
    dqn_epsilon_local = dqn_epsilon
    is_training = True
    opponent = "O" if player == "X" else "X"
    step = 0
    win_count = 0
    loss_count = 0
    tie_count = 0
    for ep in tqdm(range(dqn_episodes), desc=f"DQN Training ({player})"):
        reset()
        # X plays a random opening move
        make_move(random.choice(get_succesors(board)), "X")
        move_count = 0
        last_state = None
        last_action = None
        while True:
            if player == "X":
                terminated = False
                default_opponent(opponent)
                is_win, winner = has_winner()
                if is_win:
                    reward = get_reward(winner, move_count, player)
                    terminated = True
                elif has_tied():
                    reward = -50.0
                    terminated = True

                if last_state is not None:
                    next_state = dqn_encode_state(board, player)
                    valid_mask = get_valid_mask(board)
                    dqn_replay_buffer.append((last_state, 
                        last_action, reward, next_state, terminated, valid_mask))
                    step += 1

                    if step % dqn_train_freq == 0:
                        dqn_replay_train()

                if not terminated:
                    state = dqn_encode_state(board, player)
                    action = dqn_choose_action(board, dqn_epsilon_local, player)
                    make_move(action, player)
                    move_count += 1

                    reward = 0.0
                    terminated = False

                    is_win, winner = has_winner()
                    if is_win:
                        reward = get_reward(winner, move_count, player)
                        terminated = True
                        next_state = dqn_encode_state(board, player)
                        valid_mask = get_valid_mask(board)
                        dqn_replay_buffer.append((state, action, 
                            reward, next_state, terminated, valid_mask))
                        step += 1

                        if step % dqn_train_freq == 0:
                            dqn_replay_train()
                    
                    
                    elif has_tied():
                        reward = -50.0
                        terminated = True
                        next_state = dqn_encode_state(board, player)
                        valid_mask = get_valid_mask(board)
                        dqn_replay_buffer.append((state, action, 
                            reward, next_state, terminated, valid_mask))
                        step += 1

                        if step % dqn_train_freq == 0:
                            dqn_replay_train()
                if terminated:
                    if winner == player:
                        win_count += 1
                    elif winner == opponent:
                        loss_count += 1
                    else:
                        tie_count += 1
                    break
                last_state = state
                last_action = action
                
            elif player == "O":
                # O's Turn
                state = dqn_encode_state(board, player)
                action = dqn_choose_action(board, dqn_epsilon_local, player)
                make_move(action, player)
                move_count += 1
                reward = 0.0
                terminated = False
                is_win, winner = has_winner()
                if is_win:
                    reward = get_reward(winner, move_count, player)
                    terminated = True
                elif has_tied():
                    reward = -50.0
                    terminated = True
                # X's Turn
                if not terminated:
                    default_opponent(opponent)
                    is_win, winner = has_winner()
                    if is_win or has_tied():
                        if winner == player:
                            reward = get_reward(winner, move_count, player)
                            terminated = True
                            win_count += 1
                        elif winner == opponent:
                            reward = get_reward(winner, move_count, player)
                            terminated = True
                            loss_count += 1
                        else:
                            reward = -50
                            terminated = True
                            tie_count += 1
                next_state = dqn_encode_state(board, player)
                valid_mask = get_valid_mask(board)
                dqn_replay_buffer.append((state, action,
                     reward, next_state, terminated, valid_mask))
                step += 1

                if step % dqn_train_freq == 0:
                    dqn_replay_train()
                if terminated:
                    if winner == player:
                        win_count += 1
                    elif winner == opponent:
                        loss_count += 1
                    else:
                        tie_count += 1
                    break
        dqn_epsilon_local =  dqn_epsilon_local * dqn_epsilon_decay
        if (ep + 1) % dqn_target_model_update == 0:
            dqn_target_model.set_weights(dqn_policy_model.get_weights())
            win_count = 0
            loss_count = 0
            tie_count = 0
    is_training = False
    dqn_policy_model.save(model_path)
    print(f"DQN training complete. Model saved to {model_path}")

def dqn_train_vs_minimax(player):
    global is_training, dqn_policy_model, dqn_target_model, dqn_replay_buffer

    model_path = f"model_ttt_minimax_{player}.keras"
    if os.path.exists(model_path):
        print(f"Loading existing DQN minimax model from {model_path}...")
        dqn_policy_model = keras.models.load_model(model_path)
        dqn_target_model = dqn_create_model()
        dqn_target_model.set_weights(dqn_policy_model.get_weights())
        return

    dqn_policy_model = dqn_create_model()
    dqn_target_model = dqn_create_model()
    dqn_target_model.set_weights(dqn_policy_model.get_weights())
    dqn_replay_buffer = deque(maxlen=dqn_replay_size)
    dqn_epsilon_local = dqn_epsilon

    is_training = True
    opponent = "O" if player == "X" else "X"
    step = 0
    minimax_episodes = 100000

    win_count = 0
    loss_count = 0
    tie_count = 0

    plot_episodes = []
    plot_win_rates = []
    plot_draw_rates = []
    plot_loss_rates = []

    def minimax_opponent_move(opp):
        successors = get_succesors(board)
        if not successors:
            return
        best_score = -float('inf')
        best_pos = None
        p = "X" if opp == "X" else "O"
        o = "O" if opp == "X" else "X"
        for pos in successors:
            nb = copy.deepcopy(board)
            make_dummy_move(pos, opp, nb)
            score = minimax_AB(nb, False, player=p, 
                opponent=o, alpha=-float('inf'), beta=float('inf'))
            if score > best_score:
                best_score = score
                best_pos = pos
        make_move(best_pos, opp)

    for ep in tqdm(range(minimax_episodes), desc=f"DQN vs Minimax Training ({player})"):
        reset()

        make_move(random.choice(get_succesors(board)), "X")

        move_count = 0
        last_state = None
        last_action = None

        while True:
            if player == "X":
                terminated = False
                minimax_opponent_move(opponent)
                is_win, winner = has_winner()
                if is_win:
                    reward = get_reward(winner, move_count, player)
                    terminated = True
                elif has_tied():
                    reward = -50.0
                    terminated = True

                if last_state is not None:
                    next_state = dqn_encode_state(board, player)
                    valid_mask = get_valid_mask(board)
                    dqn_replay_buffer.append((last_state, last_action, 
                        reward, next_state, terminated, valid_mask))
                    step += 1
                    if step % dqn_train_freq == 0:
                        dqn_replay_train()

                if not terminated:
                    state = dqn_encode_state(board, player)
                    action = dqn_choose_action(board, dqn_epsilon_local, player)
                    make_move(action, player)
                    move_count += 1
                    reward = 0.0
                    terminated = False

                    is_win, winner = has_winner()
                    if is_win:
                        reward = get_reward(winner, move_count, player)
                        terminated = True
                        next_state = dqn_encode_state(board, player)
                        valid_mask = get_valid_mask(board)
                        dqn_replay_buffer.append((state, action, 
                            reward, next_state, terminated, valid_mask))
                        step += 1
                        if step % dqn_train_freq == 0:
                            dqn_replay_train()
                    elif has_tied():
                        reward = -50.0
                        terminated = True
                        next_state = dqn_encode_state(board, player)
                        valid_mask = get_valid_mask(board)
                        dqn_replay_buffer.append((state, action, 
                            reward, next_state, terminated, valid_mask))
                        step += 1
                        if step % dqn_train_freq == 0:
                            dqn_replay_train()

                if terminated:
                    if winner == player:
                        win_count += 1
                    elif winner == opponent:
                        loss_count += 1
                    else:
                        tie_count += 1
                    break
                last_state = state
                last_action = action

            elif player == "O":
                state = dqn_encode_state(board, player)
                action = dqn_choose_action(board, dqn_epsilon_local, player)
                make_move(action, player)
                move_count += 1
                reward = 0.0
                terminated = False

                is_win, winner = has_winner()
                if is_win:
                    reward = get_reward(winner, move_count, player)
                    terminated = True
                elif has_tied():
                    reward = -50.0
                    terminated = True

                if not terminated:
                    minimax_opponent_move(opponent)
                    is_win, winner = has_winner()
                    if is_win or has_tied():
                        if winner == player:
                            reward = get_reward(winner, move_count, player)
                            terminated = True
                            win_count += 1
                        elif winner == opponent:
                            reward = get_reward(winner, move_count, player)
                            terminated = True
                            loss_count += 1
                        else:
                            reward = -50
                            terminated = True
                            tie_count += 1

                next_state = dqn_encode_state(board, player)
                valid_mask = get_valid_mask(board)
                dqn_replay_buffer.append((state, action, 
                        reward, next_state, terminated, valid_mask))
                step += 1
                if step % dqn_train_freq == 0:
                    dqn_replay_train()

                if terminated:
                    if winner == player:
                        win_count += 1
                    elif winner == opponent:
                        loss_count += 1
                    else:
                        tie_count += 1
                    break

        dqn_epsilon_local = dqn_epsilon_local * dqn_epsilon_decay

        if (ep + 1) % dqn_target_model_update == 0:
            dqn_target_model.set_weights(dqn_policy_model.get_weights())

        if (ep + 1) % 100 == 0:
            total = win_count + loss_count + tie_count
            if total > 0:
                plot_episodes.append(ep + 1)
                plot_win_rates.append(win_count / total)
                plot_draw_rates.append(tie_count / total)
                plot_loss_rates.append(loss_count / total)
            win_count = 0
            loss_count = 0
            tie_count = 0

    is_training = False
    dqn_policy_model.save(model_path)
    print(f"DQN vs Minimax training complete. Model saved to {model_path}")

# initialise models
dqn_policy_model = dqn_create_model()
dqn_target_model = dqn_create_model()
dqn_target_model.set_weights(dqn_policy_model.get_weights())
dqn_replay_buffer = deque(maxlen=dqn_replay_size)

def play():

    print("\n===================== SELECT MODE =====================\n")
    print("1) AI v/s Default Opponent\n")
    print("2) AI v/s AI\n")
    mode = int(input("Enter mode: "))


    if mode == 1:
        print("\n-------------------PICK ALGORITHM FOR AI Player-------------------\n")
        print("\n1. Minimax")
        print("\n2. Minimax with Alpha Beta Pruning")
        print("\n3. Q-Learning")
        print("\n4. Deep Q-Learning Network (DQN)")
        strategy = int(input("\n\nEnter Choice: "))
        player = input("\nEnter the player this algorithm will play as (X/O): ")
        opponent = 'O' if player == 'X' else 'X'
        if strategy == 3:
            train(player)
            #print_qtable()
        elif strategy == 4:
            dqn_train(player)

        reset()
        # X plays a random opening move, O responds with default heuristic
        make_move(random.choice([0,1,2,3,4,5,6,7,8]), "X")
        

        if player == "X":
            while True:
                default_opponent(opponent)
                is_win, _ = has_winner()
                if is_win:
                    print("O wins!")
                 
                    break
                if has_tied():
                    print("It's a tie!")
                    
                    break

                AI_opponent(strategy, player)
                is_win, _ = has_winner()
                if is_win:
                    print("X wins!")
       
                    break
                if has_tied():
                    print("It's a tie!")
            
                    break

                

        elif player == "O":
            while True:
                AI_opponent(strategy, player)
                is_win, _ = has_winner()
                if is_win:
                    print("O wins!")
                   
                    break
                if has_tied():
                    print("It's a tie!")
                  
                    break
                default_opponent(opponent)
                is_win, _ = has_winner()
                if is_win:
                    print("X wins!")
        
                    break
                if has_tied():
            
                    print("It's a tie!")
                    break


    elif mode == 2:
        print("\n-------------------PICK ALGORITHM FOR X (Player 1)-------------------\n")
        print("\n1. Minimax")
        print("\n2. Minimax with Alpha Beta Pruning")
        print("\n3. Q-Learning")
        print("\n4. Deep Q-Learning Network (DQN)")
        strategy_x = int(input("\n\nEnter Choice for X: "))

        print("\n-------------------PICK ALGORITHM FOR O (Player 2)-------------------\n")
        print("\n1. Minimax")
        print("\n2. Minimax with Alpha Beta Pruning")
        print("\n3. Q-Learning")
        print("\n4. Deep Q-Learning Network (DQN)")
        strategy_o = int(input("\n\nEnter Choice for O: "))

        if strategy_x == 3: train("X")
        elif strategy_x == 4: dqn_train_vs_minimax("X")
        if strategy_o == 3: train("O")
        elif strategy_o == 4: dqn_train_vs_minimax("O")

        reset()
        # X plays a random opening move, O responds with default heuristic
        make_move(random.choice([0,1,2,3,4,5,6,7,8]), "X")

        while True:
            AI_opponent(strategy_o, "O")
            is_win, _ = has_winner()
            if is_win:
                print("O wins!")
            
                break
            if has_tied():
  
                print("It's a tie!")
                break

            AI_opponent(strategy_x, "X")
            is_win, _ = has_winner()
            if is_win:
                print("X wins!")

                break
            if has_tied():

                print("It's a tie!")
                break

    else:
        print("Invalid Input !!! Terminated !!!")
        exit()


play()