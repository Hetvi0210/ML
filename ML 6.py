#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np
import random


# In[2]:


class TicTacToe:
    def __init__(self):
        self.board = np.zeros((3, 3))
        self.players = ['X', 'O']
        self.current_player = None
        self.winner = None
        self.game_over = False
    
    def reset(self):
        self.board = np.zeros((3, 3))
        self.current_player = None
        self.winner = None
        self.game_over = False
    
    def available_moves(self):
        moves = []
        for i in range(3):
            for j in range(3):
                if self.board[i][j] == 0:
                    moves.append((i, j))
        return moves
    
    def make_move(self, move):
        if self.board[move[0]][move[1]] != 0:
            return False
        self.board[move[0]][move[1]] = self.players.index(self.current_player) + 1
        self.check_winner()
        self.switch_player()
        return True
    
    def switch_player(self):
        if self.current_player == self.players[0]:
            self.current_player = self.players[1]
        else:
            self.current_player = self.players[0]
    
    def check_winner(self):
        # Check rows
        for i in range(3):
            if self.board[i][0] == self.board[i][1] == self.board[i][2] != 0:
                self.winner = self.players[int(self.board[i][0] - 1)]
                self.game_over = True
        # Check columns
        for j in range(3):
            if self.board[0][j] == self.board[1][j] == self.board[2][j] != 0:
                self.winner = self.players[int(self.board[0][j] - 1)]
                self.game_over = True
        # Check diagonals
        if self.board[0][0] == self.board[1][1] == self.board[2][2] != 0:
            self.winner = self.players[int(self.board[0][0] - 1)]
            self.game_over = True
        if self.board[0][2] == self.board[1][1] == self.board[2][0] != 0:
            self.winner = self.players[int(self.board[0][2] - 1)]
            self.game_over = True
    
    def print_board(self):
        print("-------------")
        for i in range(3):
            print("|", end=' ')
            for j in range(3):
                print(self.players[int(self.board[i][j] - 1)] if self.board[i][j] != 0 else " ", end=' | ')
            print()
            print("-------------")


# In[3]:


game = TicTacToe()
game.current_player = game.players[0]
game.print_board()

while not game.game_over:
    move = input(f"{game.current_player}'s turn. Enter row and column (e.g. 0 0): ")
    move = tuple(map(int, move.split()))
    while move not in game.available_moves():
        move = input("Invalid move. Try again: ")
        move = tuple(map(int, move.split()))
    game.make_move(move)
    game.print_board()

if game.winner:
    print(f"{game.winner} wins!")
else:
    print("It's a tie!")


# In[4]:


import random

class QLearningAgent:
    def __init__(self, alpha, epsilon, discount_factor):
        self.Q = {}
        self.alpha = alpha
        self.epsilon = epsilon
        self.discount_factor = discount_factor

    def get_Q_value(self, state, action):
        if (state, action) not in self.Q:
            self.Q[(state, action)] = 0.0
        return self.Q[(state, action)]

    def choose_action(self, state, available_moves):
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(available_moves)
        else:
            Q_values = [self.get_Q_value(state, action) for action in available_moves]
            max_Q = max(Q_values)
            if Q_values.count(max_Q) > 1:
                best_moves = [i for i in range(len(available_moves)) if Q_values[i] == max_Q]
                i = random.choice(best_moves)
            else:
                i = Q_values.index(max_Q)
            return available_moves[i]

    def update_Q_value(self, state, action, reward, next_state):
        next_Q_values = [self.get_Q_value(next_state, next_action) for next_action in TicTacToe(next_state).available_moves()]
        max_next_Q = max(next_Q_values) if next_Q_values else 0.0
        self.Q[(state, action)] += self.alpha * (reward + self.discount_factor * max_next_Q - self.Q[(state, action)])


# In[5]:


def train(num_episodes, alpha, epsilon, discount_factor):
    agent = QLearningAgent(alpha, epsilon, discount_factor)
    for i in range(num_episodes):
        state = TicTacToe().board
        while not TicTacToe(state).game_over():
            available_moves = TicTacToe(state).available_moves()
            action = agent.choose_action(state, available_moves)
            next_state, reward = TicTacToe(state).make_move(action)
            agent.update_Q_value(state, action, reward, next_state)
            state = next_state
    return agent


# In[6]:


def test(agent, num_games):
    num_wins = 0
    for i in range(num_games):
        state = TicTacToe().board
        while not TicTacToe(state).game_over():
            if TicTacToe(state).player == 1:
                action = agent.choose_action(state, TicTacToe(state).available_moves())
            else:
                action = random.choice(TicTacToe(state).available_moves())
            state, reward = TicTacToe(state).make_move(action)
        if reward == 1:
            num_wins += 1
    return num_wins / num_games * 100


# In[10]:


def train(num_episodes, alpha, epsilon, discount_factor):
    # Create a TicTacToe instance
    game = TicTacToe()

    for i in range(num_episodes):
        state = game.board
        while not game.game_over():
            available_moves = game.available_moves()
            action = agent.choose_action(state, available_moves)
            next_state, reward, done = game.make_move(action)

            # Update your Q-learning agent here
            # You should update the Q-values based on the state, action, next_state, and reward

            state = next_state  # Update the current state

    # Return the trained agent
    return agent

