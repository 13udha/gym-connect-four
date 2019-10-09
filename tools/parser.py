import glob
import importlib
from typing import List

import gym
import numpy as np
import os
import mmap

from tqdm import tqdm

from gym_connect_four import RandomPlayer, SavedPlayer, ConnectFourEnv, Player


def get_num_lines(file_path):
    fp = open(file_path, "r+")
    buf = mmap.mmap(fp.fileno(), 0)
    lines = 0
    while buf.readline():
        lines += 1
    return lines


class Connect4Moves:
    def __init__(self, boost_win_move: float = 0):
        self.board_states = []
        self.actions = []
        self.reward = 0
        self.starting = None
        self.state_list = []
        self.boost_win_move = boost_win_move

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
            if reward != -1:
                # TODO: find if from new position there is a winning move, if so, switch reward to -1. if winning move can be prevented generate a 0.5 reward with it
                pass

            self.state_list.append((state, int(action), float(reward)))

            if reward != 1:
                # TODO: find if any the other 6 actions would result in a win
                pass


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
    with open(fname) as fp:
        match_lines = []
        matches = 0
        for line in tqdm(fp, total=get_num_lines(fname)):
            if line.strip() == 'NEW ROUND':
                if match_lines:
                    moves = Connect4Moves(boost_win_move=boost_win_move)
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
    # Load model first
    env: ConnectFourEnv = gym.make("ConnectFour-v0")

    model1 = 'mandarka'
    module = importlib.import_module(model1)
    class_ = getattr(module, model1.capitalize())
    player1 = class_(env, name=model1.capitalize())

    model2 = 'mandarkb'
    module = importlib.import_module(model2)
    class_ = getattr(module, model2.capitalize())
    player2 = class_(env, name=model2.capitalize())

    # player2 = player1

    for f in os.listdir('./parser'):
        if not os.path.isdir('./parser/'+f) and 'csv' in f:
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
