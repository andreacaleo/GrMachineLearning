import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras import models
from tensorflow.keras.optimizers import Adam
import numpy as np
desired_width = 320
np.set_printoptions(linewidth=desired_width)
import math
from collections import deque
import os
import time
from datetime import datetime
import h5py
import copy

import logging
FORMAT = '%(asctime)-15s %(message)s'
logging.basicConfig(format=FORMAT, level=logging.INFO)
logger = logging.getLogger("Logger for trainer_regular")

# hyperparameters
train_episodes = 50
mcts_search = 10
n_pit_network = 100
playgames_before_training = 200
cpuct = 4
training_epochs = 10
learning_rate = 0.0001

Q = {}  # state-action values
Nsa = {}  # number of times certain state-action pair has been visited
Ns = {}  # number of times state has been visited
W = {}  # number of total points collected after taking state action pair
P = {}  # initial predicted probabilities of taking certain actions in state


def flatten(l):
    return [item for sublist in l for item in sublist]


def get_empty_board():
    board = [[" ", " ", " "],
             [" ", " ", " "],
             [" ", " ", " "]]
    return board


def print_board(board):
    firstRow = ""
    secondRow = ""
    thirdRow = ""

    # Takes each board, saves the rows in a variable, then prints the variables
    firstRow = firstRow + "|" + " ".join(board[0]) + "|"
    secondRow = secondRow + "|" + " ".join(board[1]) + "|"
    thirdRow = thirdRow + "|" + " ".join(board[2]) + "|"

    print(" -----")
    print(firstRow)
    print(secondRow)
    print(thirdRow)
    print(" -----")


def possible_positions(board):
    possible = []

    for r in range(3):
        if board[r][0] != " " and board[r][0] == board[r][1] and board[r][0] == board[r][2]:
            return []

    for c in range(3):
        if board[0][c] != " " and board[0][c] == board[1][c] and board[0][c] == board[2][c]:
            return []

    if board[0][0] != " " and board[0][0] == board[1][1] and board[0][0] == board[2][2]:
        return []

    if board[0][2] != " " and board[0][2] == board[1][1] and board[0][2] == board[2][0]:
        return []

    for row in range(3):
        for column in range(3):
            if board[row][column] == " ":
                possible.append(row * 3 + column)

    return possible


def check_winner(board, turn):
    for r in range(3):
        if turn == board[r][0] and board[r][0] == board[r][1] and board[r][0] == board[r][2]:
            return True

    for c in range(3):
        if turn == board[0][c] and board[0][c] == board[1][c] and board[0][c] == board[2][c]:
            return True

    if turn == board[0][0] and board[0][0] == board[1][1] and board[0][0] == board[2][2]:
        return True

    if turn == board[0][2] and board[0][2] == board[1][1] and board[0][2] == board[2][0]:
        return True

    return False


def move(board, action, player, print_on_win=False):
    if player == 1:
        turn = 'X'
    elif player == -1:
        turn = 'O'
    else:
        raise Exception("Error in move(): player must be either 1 or -1.")

    position = []
    row = action // 3
    column = action % 3

    position.append(row)
    position.append(column)

    # place piece at position on board
    board[position[0]][position[1]] = turn

    win = False
    win = check_winner(board, turn)

    if win and print_on_win:
        print(f"The game is over! Winner is {turn}!")
        print_board(board)

    return board, win


def char_to_int(char, player):
    # based on the char in a box in the board, replaces 'X' with 1 and 'O' with -1
    if char == " ":
        return 0
    elif char == "X":
        return 1 * player
    elif char == "O":
        return -1 * player
    else:
        raise Exception(f"Cannot identify player from char {char}.")


def board_to_array(board, player):
    # takes a board in its normal state, and returns a 1x9 numpy array, changing 'X' = 1 and 'O' = -1
    array = []
    firstline = []
    secondline = []
    thirdline = []

    for item in board[0]:
        firstline.append(char_to_int(item, player))

    for item in board[1]:
        secondline.append(char_to_int(item, player))

    for item in board[2]:
        thirdline.append(char_to_int(item, player))

    array.append(firstline)
    array.append(secondline)
    array.append(thirdline)
    nparray = np.array(array)

    return nparray


test_board = [["X", "X", "O"], [" ", " ", "O"], [" ", " ", " "]]

print("Starting board:")
print_board(test_board)
print("Making a move for player X:")
move(test_board, 3, 1)
print_board(test_board)
print("Making a move for player O:")
move(test_board, 8, -1)
print_board(test_board)

print("Now transforming to array:")
print(board_to_array(test_board, 1))


def mcts(s, current_player):
    possibleA = possible_positions(s)

    sArray = board_to_array(s, current_player)
    sTuple = tuple(map(tuple, sArray))

    if len(possibleA) > 0:
        if sTuple not in P.keys():

            policy, v = nn.predict(sArray.reshape(1, 9))
            v = v[0][0]
            valids = np.zeros(9)
            np.put(valids, possibleA, 1)
            policy = policy.reshape(9) * valids
            policy = policy / np.sum(policy)
            P[sTuple] = policy

            Ns[sTuple] = 1

            for a in possibleA:
                Q[(sTuple, a)] = 0
                Nsa[(sTuple, a)] = 0
                W[(sTuple, a)] = 0
            return -v

        best_uct = -100
        for a in possibleA:

            uct_a = Q[(sTuple, a)] + cpuct * P[sTuple][a] * (math.sqrt(Ns[sTuple]) / (1 + Nsa[(sTuple, a)]))

            if uct_a > best_uct:
                best_uct = uct_a
                best_a = a

        next_state, wonBoard = move(s, best_a, current_player, print_on_win=False)

        if wonBoard:
            v = 1
        else:
            current_player *= -1
            v = mcts(next_state, current_player)
    else:
        return 0

    W[(sTuple, best_a)] += v
    Ns[sTuple] += 1
    Nsa[(sTuple, best_a)] += 1
    Q[(sTuple, best_a)] = W[(sTuple, best_a)] / Nsa[(sTuple, best_a)]
    return -v


def get_action_probs(init_board, current_player):
    for _ in range(mcts_search):
        s = copy.deepcopy(init_board)
        value = mcts(s, current_player)

    logger.info("Completed an iteration of MCTS.")

    actions_dict = {}

    sArray = board_to_array(init_board, current_player)
    sTuple = tuple(map(tuple, sArray))

    for a in possible_positions(init_board):
        actions_dict[a] = Nsa[(sTuple, a)] / Ns[sTuple]

    action_probs = np.zeros(9)

    for a in actions_dict:
        np.put(action_probs, a, actions_dict[a], mode='raise')

    return action_probs


def playgame():
    current_player = 1
    game_mem = []

    real_board = get_empty_board()

    while True:
        s = copy.deepcopy(real_board)
        policy = get_action_probs(s, current_player)
        policy = policy / np.sum(policy)
        game_mem.append([board_to_array(real_board, current_player), current_player, policy, None])
        action = np.random.choice(len(policy), p=policy)

        print("policy ", policy, "chosen action", action)
        print_board(real_board)

        next_state, wonBoard = move(real_board, action, current_player, print_on_win=True)

        if len(possible_positions(next_state)) == 0:
            for tup in game_mem:
                tup[3] = 0
            return game_mem

        if wonBoard:
            for tup in game_mem:
                if tup[1] == current_player:
                    tup[3] = 1
                else:
                    tup[3] = -1
            return game_mem

        current_player *= -1



def neural_network():
    input_layer = layers.Input(shape=(9), name="BoardInput")
    dense_0 = layers.Dense(128, activation="relu", name='dense0')(input_layer)
    dense_1 = layers.Dense(512, activation='relu', name='dense1')(dense_0)
    dense_2 = layers.Dense(256, activation='relu', name='dense2')(dense_1)

    pi = layers.Dense(9, activation="softmax", name='pi')(dense_2)
    v = layers.Dense(1, activation="tanh", name='value')(dense_2)

    model = Model(inputs=input_layer, outputs=[pi, v])
    model.compile(loss=['categorical_crossentropy', 'mean_squared_error'], optimizer=Adam(learning_rate))

    model.summary()
    return model


def train_nn(network, game_mem):
    print("Training Network")
    print("lenght of game_mem", len(game_mem))

    state = []
    policy = []
    value = []

    for mem in game_mem:
        state.append(mem[0])
        policy.append(mem[2])
        value.append(mem[3])

    flattened_states = []
    for sub_state in state:
        flattened_states = flattened_states + [flatten(sub_state)]
    state = np.array(flattened_states)
    policy = np.array(policy)
    value = np.array(value)

    history = network.fit(state, [policy, value], batch_size=32, epochs=training_epochs, verbose=1)


def pit(nn, new_nn):
    # function pits the old and new networks. If new network wins 55/100 games or more, then return True
    logger.info("Pitting old and new networks against each other.")
    nn_wins = 0
    new_nn_wins = 0

    for _ in range(n_pit_network):

        s = [[" ", " ", " "],
             [" ", " ", " "],
             [" ", " ", " "]]

        while True:

            policy, v = nn.predict(board_to_array(s, 1).reshape(1, 9))
            valids = np.zeros(9)

            possibleA = possible_positions(s)

            if len(possibleA) == 0:
                break

            np.put(valids, possibleA, 1)
            policy = policy.reshape(9) * valids
            policy = policy / np.sum(policy)
            action = np.argmax(policy)

            next_state, win = move(s, action, 1)
            s = next_state

            if win:
                nn_wins += 1
                break

            # new nn makes move

            policy, v = new_nn.predict(board_to_array(s, -1).reshape(1, 9))
            valids = np.zeros(9)

            possibleA = possible_positions(s)

            if len(possibleA) == 0:
                break

            np.put(valids, possibleA, 1)
            policy = policy.reshape(9) * valids
            policy = policy / np.sum(policy)
            action = np.argmax(policy)

            next_state, win = move(s, action, -1)
            s = next_state

            if win:
                new_nn_wins += 1
                break

    if (new_nn_wins + nn_wins) == 0:
        print("The game was a complete tie")
        now = datetime.utcnow()
        model_path = r'training_dir/tictactoeTie.h5'

        nn.save(model_path)
        return False

    win_percent = float(new_nn_wins) / float(new_nn_wins + nn_wins)
    if win_percent > 0.52:
        logger.info(f"The new network did well - it won a {win_percent} fraction of games.")
        print(win_percent)
        return True
    else:
        print(f"The new network didn't do well enough - it only won a {win_percent} fraction of games.")
        print(new_nn_wins)
        return False



def train():
    global nn
    global Q
    global Nsa
    global Ns
    global W
    global P

    game_mem = []

    for episode in range(train_episodes):
        logger.info(f"Starting training episode {episode}.")

        nn.save('temp.h5')
        old_nn = models.load_model('temp.h5')

        for g in range(playgames_before_training):
            logger.info(f"Starting to play game {g}.")
            game_mem += playgame()

        train_nn(nn, game_mem)
        game_mem = []
        if pit(old_nn, nn):
            del old_nn
            Q = {}
            Nsa = {}
            Ns = {}
            W = {}
            P = {}
        else:
            nn = old_nn
            del old_nn

    now = datetime.utcnow()

    model_path = r'training_dir/tictactoe_new.h5'
    nn.save(model_path)


nn = neural_network()
train()

print("Exiting right now...")