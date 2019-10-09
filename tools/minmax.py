import gym
from gym_connect_four import RandomPlayer, ConnectFourEnv, Player, SavedPlayer
import numpy as np
import math
import random


class Minmax(Player):
    """ Clone of RandomPlayer for runner.py illustration purpose """

    def __init__(self, env, name='RandomPlayer'):
        super(Minmax, self).__init__(env, name)

        self.LOGGING = True

        self.side = -1
        self.ROW_COUNT = self.env.board_shape[0]
        self.COLUMN_COUNT = self.env.board_shape[1]
        self.PLAYER_PIECE = 1
        self.AI_PIECE = 2
        self.WINDOW_LENGTH = 4
        self.EMPTY = 0
        self.MINMAX_DEPTH = 3
        if self.LOGGING:
            self.train_set_name = f"minmax_train{random.randint(1, 999)}.csv"
            print(f"Training set : {self.train_set_name}")
            self.train_set = open(self.train_set_name, "a+")

    def reset(self, episode: int = 0, side: int = 1) -> None:
        if self.LOGGING:
            self.train_set.write("NEW ROUND\n")

        self.side = side
        if self.side == 1:
            self.PLAYER_PIECE = 2
            self.AI_PIECE = 1
        else:
            self.PLAYER_PIECE = 1
            self.AI_PIECE = 2

    def get_next_action(self, state: np.ndarray) -> int:
        board = self._transform_state(state)
        col, minmax_score = self._minmax(board, self.MINMAX_DEPTH, self.side == 1)  # Probabil daca e starting player sau nu
        if self.env.is_valid_action(col):
            return col
        raise Exception('Unable to determine a valid move!')

    def learn(self, state, action: int, state_next, reward: int, done: bool) -> None:
        if self.LOGGING:
            self.train_set.write(' '.join(map(str, state.reshape(42))) + f",{action},{reward}\n")
            if done:
                self.train_set.flush()

    def _transform_state(self, state):
        board = np.flip(np.copy(state), 0)
        for i in range(self.ROW_COUNT):
            for j in range(self.COLUMN_COUNT):
                if board[i][j] == -1:
                    board[i][j] = 2
        return board

    def _minmax(self, board, depth, maximizingPlayer):
        valid_locations = self._get_valid_locations(board)
        is_terminal = self._is_terminal_node(board)
        if depth == 0 or is_terminal:
            if is_terminal:  # None since we do not know which column produces those
                if self._winning_move(board, self.AI_PIECE):
                    return (None, 999999)
                elif self._winning_move(board, self.PLAYER_PIECE):
                    return (None, -999999)
                else:  # full board
                    return (None, 0)
            else:  # depth == 0
                return (None, self._score_position(board, self.AI_PIECE))

        if maximizingPlayer:
            value = -math.inf  # negative infinity
            column = random.choice(valid_locations)
            for col in valid_locations:
                row = self._get_next_open_row(board, col)
                b_copy = board.copy()  # we would want to copy the location to prevent errors while calling this recursively
                self._drop_piece(b_copy, row, col, self.AI_PIECE)
                new_score = self._minmax(b_copy, depth - 1, not maximizingPlayer)[1]  # calling recursively
                if new_score > value:  # we want to get max points for AI
                    value = new_score
                    column = col
            return column, value

        else:
            value = math.inf  # positive infinity
            column = random.choice(valid_locations)
            for col in valid_locations:
                row = self._get_next_open_row(board, col)
                b_copy = board.copy()
                self._drop_piece(b_copy, row, col, self.PLAYER_PIECE)
                new_score = self._minmax(b_copy, depth - 1, maximizingPlayer)[1]
                if new_score < value:  # we want to reach min points for Player
                    value = new_score
                    column = col
            return column, value

    def _get_valid_locations(self, board):  # to keep track of the valid locations in a list
        valid_locations = []
        for col in range (self.COLUMN_COUNT):
            if self._is_valid_location(board, col):
                valid_locations.append(col)
        return valid_locations

    def _is_terminal_node(self, board):  # if one of the players won, or board is full
        return self._winning_move(board, self.PLAYER_PIECE) or self._winning_move(board, self.AI_PIECE) or len(
            self._get_valid_locations(board)) == 0

    def _winning_move(self, board, piece):
        #check horizontal
        for c in range(self.COLUMN_COUNT-3): #horizontally, it can't start after col_count-3 and have 4 connections
            for r in range(self.ROW_COUNT):
                if board[r][c] == piece and board[r][c+1] == piece and board[r][c+2] == piece and board[r][c+3] == piece:
                    return True
        #check vertical
        for c in range(self.COLUMN_COUNT): #vertically, it can't start below of column_count-3 and have 4 connections
            for r in range(self.ROW_COUNT-3):
                if board[r][c] == piece and board[r+1][c] == piece and board[r+2][c] == piece and board[r+3][c] == piece:
                    return True
        #check positive sloped diagonal
        for c in range(self.COLUMN_COUNT-3):
            for r in range(self.ROW_COUNT-3):
                if board[r][c] == piece and board[r+1][c+1] == piece and board[r+2][c+2] == piece and board[r+3][c+3] == piece:
                    return True
        #check negative sloped diagonal
        for c in range(self.COLUMN_COUNT-3):
            for r in range(3, self.ROW_COUNT): #3rd row until the end. cannot start before 3rd row
                if board[r][c] == piece and board[r-1][c+1] == piece and board[r-2][c+2] == piece and board[r-3][c+3] == piece:
                    return True

    def _get_next_open_row(self, board, col):
        for r in range(self.ROW_COUNT):
            if board[r][col] == 0: #if the current row is empty
                return r

    def _drop_piece(self, board, row, col, piece):
        board[row][col] = piece  # to fill the board with Players's move such as 1 and 2

    def _score_position(self, board, piece):
        score = 0 #as an initial value

        #having the center pieces are important because it allows more opportunity
        center_array = [int(i) for i in list(board[:, self.COLUMN_COUNT//2])] #//2 to have middle
        center_count = center_array.count(piece)
        score += center_count * 3

        #horizontal scoring
        for r in range(self.ROW_COUNT):
            row_array = [int(i) for i in list(board[r,:])] #":" isused to go over every column
            for c in range (self.COLUMN_COUNT-3):
                window = row_array[c:c+self.WINDOW_LENGTH]
                score += self._evaluate_window(window,piece)

        #vertical scoring
        for c in range(self.COLUMN_COUNT):
            col_array = [int(i) for i in list(board[:,c])]
            for r in range(self.ROW_COUNT-3):
                window = col_array[r:r+self.WINDOW_LENGTH]
                score += self._evaluate_window(window,piece)


        #scoring positively sloped diagonal
        for r in range(self.ROW_COUNT-3): #cut off at the top
            for c in range(self.COLUMN_COUNT-3):
                window = [board[r+i][c+i] for i in range(self.WINDOW_LENGTH)] #going diagonal
                score += self._evaluate_window(window,piece)

        #scoring for negative sloped diagonal
        for r in range(self.ROW_COUNT-3):
            for c in range(self.COLUMN_COUNT-3):
                window = [board[r+3-i][c+i] for i in range(self.WINDOW_LENGTH)]
                score += self._evaluate_window(window,piece)

        return score

    def _evaluate_window(self, window, piece):
        score = 0;
        opp_piece = self.PLAYER_PIECE #opponent piece
        if piece == self.PLAYER_PIECE:
            opp_piece = self.AI_PIECE

        if window.count(piece) == 4:
            score += 100
        elif window.count(piece) ==3 and window.count(self.EMPTY) == 1:
            score += 5
        elif window.count(piece) == 2 and window.count(self.EMPTY) == 2:
            score += 2

        if window.count(opp_piece) == 3 and window.count(self.EMPTY) == 1:
            score -= 4 #if opponent has 3 in a row, for preference, block him

        return score

    def _is_valid_location(self, board, col): #to see if there is an empty spot on the top row
        return board[self.ROW_COUNT-1][col] == 0 #then we can drop a piece there
