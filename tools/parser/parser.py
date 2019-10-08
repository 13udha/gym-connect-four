from typing import List
import numpy as np

class Connect4Moves:
    def __init__(self):
        self.board_states = []
        self.actions = []
        self.reward = 0
        self.starting = None

    def parse_dump(self, lines: List[str]):
        (board, action, reward) = lines[0].strip().split(',')
        np_board = np.fromstring(board.strip(), dtype=int, sep=' ')
        self.starting = not np_board.any()
        if not self.starting:
            print("starting")

        for line in lines:
            (board, action, reward) = line.strip().split(',')
            self.board_states.append(board)
            self.actions.append(int(action))
            reward = float(reward)
            self.reward = reward
            if self.reward == 1:
                print("match won")


filename = "train_vs_random_10k.csv"

with open(filename) as fp:
    match_lines = []
    line = fp.readline()
    matches = 0
    while line:
        if line.strip() == 'NEW ROUND':
            if match_lines:
                moves = Connect4Moves()
                moves.parse_dump(match_lines)
                # rewards[1-reward] += 1
            matches += 1
            match_lines = []
        else:
            match_lines.append(line)
        line = fp.readline()

print(f"Found {matches} rounds.")
