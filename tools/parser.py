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
    def __init__(self):
        self.board_states = []
        self.actions = []
        self.reward = 0
        self.starting = None
        self.state_list = []

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
        player.teach(move[0], move[1], move[2])
        # print(move)


def run(player, fname):
    os.chdir(os.path.dirname(__file__))
    # print(os.getcwd())
    with open(fname) as fp:
        match_lines = []
        matches = 0
        for line in tqdm(fp, total=get_num_lines(fname)):
            if line.strip() == 'NEW ROUND':
                if match_lines:
                    moves = Connect4Moves()
                    moves.parse_dump(match_lines)
                    teach(player, moves)
                    # rewards[1-reward] += 1
                matches += 1
                match_lines = []
            else:
                match_lines.append(line)

        # Parse last match from file
        if match_lines:
            moves = Connect4Moves()
            moves.parse_dump(match_lines)
            teach(player, moves)


def main():
    # Load model first
    env: ConnectFourEnv = gym.make("ConnectFour-v0")
    model = 'mandark'

    module = importlib.import_module(model)
    class_ = getattr(module, model.capitalize())
    player = class_(env, name=model.capitalize())

    files = [f for f in glob.glob("**/*.csv", recursive=True)]
    for f in files:
        run(player, f"parse/{f}")

    # for f in ['parser/minmax_vs_random_depth3.csv', 'parser/minmax_vs_deedee_train_0.csv', 'parser/minmax_vs_cnn_173.csv']:
    #     run(player, f)

    player.save_model()


if __name__ == "__main__":
    main()
