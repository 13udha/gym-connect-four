import os
from operator import itemgetter

# Comment next line to run on GPU. With this configuration, it looks to run faster on CPU i7-8650U
from keras.engine.saving import load_model

os.environ['CUDA_VISIBLE_D5EVICES'] = '-1'

import random
import warnings

import gym
from keras.layers import Dense, Flatten
from keras.models import Sequential
from keras.optimizers import Adam

from gym_connect_four import ConnectFourEnv, Player, ResultType, SavedPlayer, RandomPlayer

ENV_NAME = "ConnectFour-v0"
TRAIN_EPISODES = 100000

import time
from statistics import mean

from collections import deque
import os
import csv
import numpy as np


class Mandarkcomp(Player):
    def __init__(self, env, name='Mandarkcomp', model_prefix=None):
        super(Mandarkcomp, self).__init__(env, name)

        self.observation_space = env.observation_space.shape
        self.action_space = env.action_space.n

        self.model1 = load_model("Mandarka.h5")
        self.model2 = load_model("Mandarkb.h5")

    def _is_first_player(self, state) -> bool:
        return np.sum(state) == 0

    def get_next_action(self, state: np.ndarray) -> int:
        model = self.model1
        if self._is_first_player(state):
            model = self.model2

        state = np.reshape(state, [1] + list(self.observation_space))
        for _ in range(100):
            q_values = model.predict(state)
            q_values = np.array([[
                x if idx in self.env.available_moves() else -10
                for idx, x in enumerate(q_values[0])
            ]])
            action = np.argmax(q_values[0])
            if self.env.is_valid_action(action):
                return action

        return random.choice(list(self.env.available_moves()))
        #
        # raise Exception(
        #     'Unable to determine a valid move! Maybe invoke at the wrong time?'
        # )

    def save_model(self, model_prefix: str = None):
        pass

    def reset(self, episode: int = 0, side: int = 1) -> None:
        pass

    def teach(self, state: np.ndarray, action: int, reward: float) -> None:
        pass

    def learn(self, state, action, state_next, reward, done) -> None:
        pass


def game(show_boards=False):
    pass


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        game(False)
