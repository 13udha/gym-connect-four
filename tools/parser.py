import glob
import importlib
from typing import List

import gym
import numpy as np
import os
import mmap

from tqdm import tqdm

from gym_connect_four import RandomPlayer, SavedPlayer, ConnectFourEnv, Player

env: ConnectFourEnv = gym.make("ConnectFour-v0")


def get_num_lines(file_path):
    fp = open(file_path, "r+")
    buf = mmap.mmap(fp.fileno(), 0)
    lines = 0
    while buf.readline():
        lines += 1
    return lines


def update_board(state, action, side):
    board = state[0].copy()
    for index in list(reversed(range(env.board_shape[0]))):
        if board[index][action] == 0:
            board[index][action] = side
            break
    return board


def is_valid_action(board, action: int) -> bool:
    return board[0][0][action] == 0


def is_win_state(board) -> bool:
    board_shape = [6, 7]
    # Test rows
    for i in range(board_shape[0]):
        for j in range(board_shape[1] - 3):
            value = sum(board[i][j:j + 4])
            if abs(value) == 4:
                return True

    # Test columns on transpose array
    reversed_board = [list(i) for i in zip(*board)]
    for i in range(board_shape[1]):
        for j in range(board_shape[0] - 3):
            value = sum(reversed_board[i][j:j + 4])
            if abs(value) == 4:
                return True

    # Test diagonal
    for i in range(board_shape[0] - 3):
        for j in range(board_shape[1] - 3):
            value = 0
            for k in range(4):
                value += board[i + k][j + k]
                if abs(value) == 4:
                    return True

    reversed_board = np.fliplr(board)
    # Test reverse diagonal
    for i in range(board_shape[0] - 3):
        for j in range(board_shape[1] - 3):
            value = 0
            for k in range(4):
                value += reversed_board[i + k][j + k]
                if abs(value) == 4:
                    return True

    return False


def is_first_player(state):
    if np.sum(state) == 0:
        return 1
    else:
        return -1


class Connect4Moves:
    def __init__(self, boost_win_move: float = 0, is_minmax=False):
        self.board_states = []
        self.actions = []
        self.reward = 0
        self.starting = None
        self.state_list = []
        self.boost_win_move = boost_win_move
        self.is_minmax = is_minmax

    def parse_dump(self, lines: List[str]):
        (board, action, reward) = lines[0].strip().split(',')
        np_board = np.fromstring(board.strip(), dtype=int, sep=' ')
        self.starting = not np_board.any()
        # if not self.starting:
        #     print("starting")

        for line in lines:
            (board, action, reward) = line.strip().split(',')
            state = np.fromstring(board, dtype=int, sep=' ').reshape(1, 6, 7)
            reward = float(reward)
            self.reward = reward
            # if self.reward == 1:
            #     print("match won")

            # if reward != 1:
            #     win_state = False
            #     for i in range(7):
            #         if is_valid_action(state, action):
            #             state_new = update_board(state, action, 1)
            #
            #             for j in range(7):
            #                 if is_valid_action(state_new, j) and is_win_state(update_board(state, j, -1)):
            #                     win_state = True
            #                     break

            if reward == 0 and not self.is_minmax:
                # find if any the other 6 actions would result in a win, is so, reward this move with -1, reward the others with 1
                win_state = False
                for i in range(7):
                    if int(action) == i:
                        continue
                    if is_valid_action(state, i):
                        if is_win_state(update_board(state, i, 1)):
                            self.state_list.append((state, int(i), 2))
                            win_state = True
                if win_state:
                    self.reward = 1
                    # Penalize because it's not a winning move
                    self.state_list.append((state, int(action), -2))
                    break

            self.state_list.append((state, int(action), float(reward)))


def teach(player, moves: Connect4Moves):
    for move in moves.state_list:
        if moves.reward == 1:
            player.teach(move[0], move[1], move[2] + moves.boost_win_move)
        else:
            player.teach(move[0], move[1], move[2])


def is_first_player(state) -> bool:
    return np.sum(np.fromstring(state, dtype=int, sep=' ').reshape(1, 6, 7)) == 0.0


def run(player1, player2, fname, boost_win_move: float = 0):
    # os.chdir(os.path.dirname(__file__))
    # print(os.getcwd())
    is_minmax = ('minmax' in fname)
    with open(fname) as fp:
        match_lines = []
        matches = 0
        for line in tqdm(fp, total=get_num_lines(fname)):
            if line.strip() == 'NEW ROUND':
                if match_lines:
                    moves = Connect4Moves(boost_win_move=boost_win_move, is_minmax=is_minmax)
                    moves.parse_dump(match_lines)
                    if is_first_player(match_lines[0]):
                        teach(player1, moves)
                    else:
                        teach(player2, moves)
                    # rewards[1-reward] += 1
                matches += 1
                match_lines = []
            else:
                match_lines.append(line)

        # Parse last match from file
        if match_lines:
            moves = Connect4Moves()
            moves.parse_dump(match_lines)
            if is_first_player(match_lines[0]):
                teach(player1, moves)
            else:
                teach(player2, moves)


def main():
    # Load models first
    model1 = 'mandark'
    module = importlib.import_module(model1)
    class_ = getattr(module, model1.capitalize())
    player1 = class_(env, name=model1.capitalize())

    # model2 = 'mandark'
    # module = importlib.import_module(model2)
    # class_ = getattr(module, model2.capitalize())
    # player2 = class_(env, name=model2.capitalize())

    player2 = player1

    for f in os.listdir('./parser'):
        if not os.path.isdir('./parser/' + f) and 'csv' in f:
            print(f"Loading {f}")
            if 'minmax' in f:
                print("Boosting neutral moves")
                run(player1, player2, f"./parser/{f}", 0.2)
            else:
                run(player1, player2, f"./parser/{f}", 0.0)
            player1.save_model()
            player2.save_model()

    # files = [f for f in glob.glob("**/*.csv", recursive=True)]
    # for f in files:
    #     print(f"Loading {f}")
    #     if 'minmax' in f:
    #         print("Boosting neutral moves")
    #         run(player1, player2, f"parser/{f}", 0.2)
    #     else:
    #         run(player1, player2, f"parser/{f}", 0.0)
    #
    #     player1.save_model()
    #     player2.save_model()

    # for f in ['parser/minmax_vs_random_depth3.csv', 'parser/minmax_vs_deedee_train_0.csv', 'parser/minmax_vs_cnn_173.csv']:
    #     run(player, f)

    # player1.save_model()
    # player2.save_model()


if __name__ == "__main__":
    main()
