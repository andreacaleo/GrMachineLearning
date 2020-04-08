import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras import models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
import numpy as np
import math
from collections import deque
import os
import time
from datetime import datetime
import h5py
import copy

# Hyperparameters
model_path = "training_dir/tictactoe.h5"


def get_empty_board():
    board = [[" ", " ", " "],
             [" ", " ", " "],
             [" ", " ", " "]]
    return board


def print_board(total_board):
    firstRow = ""
    secondRow = ""
    thirdRow = ""

    # Takes each board, saves the rows in a variable, then prints the variables
    firstRow = firstRow + "|" + " ".join(total_board[0]) + "|"
    secondRow = secondRow + "|" + " ".join(total_board[1]) + "|"
    thirdRow = thirdRow + "|" + " ".join(total_board[2]) + "|"

    print(firstRow)
    print(secondRow)
    print(thirdRow)
    print("---------------------")


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


def move(board, action, player, print_on_win=False):
    if player == 1:
        turn = 'X'
    elif player == -1:
        turn = 'O'
    else:
        raise ("Error in move(): player must be either 1 or -1.")

    position = []
    column = action % 3
    row = action // 3

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


def human_turn(board, turn):
    possible = possible_positions(board)

    print_board(board)
    print("It is " + turn + "'s turn")

    #takes placement of piece as input
    while True:
        try:
            x = int(input("Please enter x coordinate "))
            y = int(input("Please enter y coordinate "))
        except ValueError:
            print("One of those inputs were not valid integers, please try again")
            continue
        if y not in range(3) or x not in range(3):
            print("Integers must be between 0 and 2, please try again")
            continue
        if board[y][x] != " ":
            print("That space has already been taken, please try again")
            continue
        else:
            return y * 3 + x

# ---------------------------------
# Functions for neural network
# --------------------------------


# initializing search tree
Q = {}  # state-action values
Nsa = {}  # number of times certain state-action pair has been visited
Ns = {}   # number of times state has been visited
W = {}  # number of total points collected after taking state action pair
P = {}  # initial predicted probabilities of taking certain actions in state


def letter_to_int(letter, player):
    # based on the letter in a box in the board, replaces 'X' with 1 and 'O' with -1
    if letter == " ":
        return 0
    elif letter == "X":
        return 1 * player
    elif letter == "O":
        return -1 * player

def board_to_array(board, player):
    # takes a board in its normal state, and returns a 1x9 numpy array, changing 'X' = 1 and 'O' = -1

    array = []
    firstline = []
    secondline = []
    thirdline = []

    for item in board[0]:
        firstline.append(letter_to_int(item, player))

    for item in board[1]:
        secondline.append(letter_to_int(item, player))

    for item in board[2]:
        thirdline.append(letter_to_int(item, player))

    array.append(firstline)
    array.append(secondline)
    array.append(thirdline)
    nparray = np.array(array)

    return nparray


nn = load_model(model_path)

def playgame():

    board = get_empty_board()
    global nn

    while True:
        action = human_turn(board, 'X')
        next_board, wonBoard = move(board, action, 1)

        if wonBoard:
            print ("Wow you're really good. You just beat a computer")
            break
        else:
            board = next_board
        
        policy, value = nn.predict(board_to_array(board, -1).reshape(1,9))
        possibleA = possible_positions(board)
        valids = np.zeros(9)
        np.put(valids,possibleA,1)
        policy = policy.reshape(9) * valids
        policy = policy / np.sum(policy)

        action = np.argmax(policy)
        print("action", action)
        print("policy")
        print(policy)
        
        next_board, wonBoard = move(board, action, -1)

        if wonBoard:
            print ("Awww you lost. Better luck next time")
            break
        else:
            board = next_board

playgame()