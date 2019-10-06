import random
import warnings
from collections import deque
import os

import gym
from keras.engine.saving import load_model

from gym_connect_four import RandomPlayer, ConnectFourEnv, Player, SavedPlayer

import numpy as np
from keras.layers import Dense, Flatten
from keras.models import Sequential
from keras.optimizers import Adam

ENV_NAME = "ConnectFour-v0"
TRAIN_EPISODES = 1000

class DQNSolver:
    """
    Vanilla Multi Layer Perceptron version
    """

    def __init__(self, observation_space, action_space):
        self.GAMMA = 0.95
        self.LEARNING_RATE = 0.35

        self.MEMORY_SIZE = 10000
        self.BATCH_SIZE = 100

        self.EXPLORATION_MIN = 0.01
        self.EXPLORATION_MAX = 1.0
        # self.EXPLORATION_MAX = self.EXPLORATION_MIN
        self.EXPLORATION_DECAY = 0.998

        self.exploration_rate = self.EXPLORATION_MAX

        self.action_space = action_space
        self.memory = deque(maxlen=self.MEMORY_SIZE)

        self.model = Sequential()
        self.model.add(Flatten(input_shape=observation_space))
        self.model.add(Dense(126, activation="relu"))
        self.model.add(Dense(1008, activation="relu"))
        self.model.add(Dense(self.action_space, activation="linear"))
        self.model.compile(loss="mse", optimizer=Adam(lr=self.LEARNING_RATE))

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, available_moves=[]):
        if np.random.rand() < self.exploration_rate:
            return random.randrange(self.action_space)
        q_values = self.model.predict(state)
        q_values = np.array([[x if idx in available_moves else -100 for idx, x in enumerate(q_values[0])]])
        return np.argmax(q_values[0])

    def experience_replay(self):
        if len(self.memory) < self.BATCH_SIZE:
            return
        batch = random.sample(self.memory, self.BATCH_SIZE)
        for state, action, reward, state_next, terminal in batch:
            q_update = reward
            if not terminal:
                q_update = (reward + self.GAMMA * np.amax(self.model.predict(state_next)[0]))
            q_values = self.model.predict(state)
            q_values[0][action] = q_update
            self.model.fit(state, q_values, verbose=0)
        self.exploration_rate *= self.EXPLORATION_DECAY
        self.exploration_rate = max(self.EXPLORATION_MIN, self.exploration_rate)

    def save_model(self, file_prefix: str):
        self.model.save(f"{file_prefix}.h5")

    def load_model(self, model: str):
        self.model = load_model(model)
        self.EXPLORATION_MAX = self.EXPLORATION_MIN


class Deedee(Player):
    def __init__(self, env, name='Deedee'):
        super(Deedee, self).__init__(env, name)

        self.observation_space = env.observation_space.shape
        self.action_space = env.action_space.n

        self.dqn_solver = DQNSolver(self.observation_space, self.action_space)
        if os.path.exists(f"{self.name.capitalize()}.h5"):
            self.dqn_solver.load_model(f"{self.name.capitalize()}.h5")
            self.dqn_solver.exploration_rate = self.dqn_solver.EXPLORATION_MIN

    def get_next_action(self, state: np.ndarray) -> int:
        state = np.reshape(state, [1] + list(self.observation_space))
        for _ in range(100):
            action = self.dqn_solver.act(state, self.env.available_moves())
            if self.env.is_valid_action(action):
                return action
        print("Unable to determine move ?")
        print(self.env.available_moves())
        print(state)
        if len(self.env.available_moves()) > 0:
            action = random.choice(self.env.available_moves())
            print(f"returning random action {action}")
            return action
        raise Exception('Unable to determine a valid move! Maybe invoke at the wrong time?')

    def learn(self, state, action, state_next, reward, done) -> None:
        state = np.reshape(state, [1] + list(self.observation_space))
        state_next = np.reshape(state_next, [1] + list(self.observation_space))

        # reward = reward if not done else -reward
        self.dqn_solver.remember(state, action, reward, state_next, done)

        if not done:
            self.dqn_solver.experience_replay()

    def save_model(self):
        self.dqn_solver.save_model(self.name)


def game():
    env = gym.make(ENV_NAME)

    player = NNPlayer(env, 'NNPlayer')
    opponent = RandomPlayer(env, 'OpponentRandomPlayer')

    total_reward = 0
    wins = 0
    losses = 0
    draws = 0
    run = 0
    while True:
        run += 1
        state = env.reset(opponent=opponent, player_color=1)
        step = 0
        while True:
            step += 1
            # env.render()
            action = player.get_next_action(state)

            state_next, reward, terminal, info = env.step(action)

            player.learn(state, action, reward, state_next, terminal)

            state = state_next

            if terminal:
                total_reward += reward
                print("Run: " + str(run) + ", score: " + str(reward))
                if hasattr(player, 'dqn_solver'):
                    print("exploration: " + str(player.dqn_solver.exploration_rate))
                if reward == 1:
                    wins += 1
                    print(f"winner: {player.name}")
                    print("board state:\n", state)
                    print(f"reward={reward}")
                elif reward == env.LOSS_REWARD:
                    losses += 1
                    print(f"lost to: {env.opponent.name}")
                    print("board state:\n", state)
                    print(f"reward={reward}")
                elif reward == env.DRAW_REWARD:
                    draws += 1
                    print(f"draw after {player.name} move")
                    print("board state:\n", state)
                    print(f"reward={reward}")
                print(f"Wins [{wins}], Draws [{draws}], Losses [{losses}] - Total reward {total_reward}, average reward {total_reward/run}")
                break

        if run == TRAIN_EPISODES:
            if hasattr(player, 'save_model') and callable(player.save_model):
                player.save_model()
            break


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        game()
