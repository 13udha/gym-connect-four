import sys
sys.path.append('..\gym-connect-four')

import random
from colorama import init, Fore , Back , Style
import warnings
from collections import deque
import copy
import matplotlib.pyplot as plt
import pickle

import gym
from gym_connect_four import MinMaxPlayer, LeftiPlayer, RandomPlayer, ConnectFourEnv, Player

import numpy as np
from keras.layers import Dense, Flatten
from keras.models import Sequential
from keras.optimizers import Adam

#init colorama
init()

# from scores.score_logger import ScoreLogger
# TODO bessere Pfade für genaue files
results_path = './results/'
# weights_filename = results_path + 'dqn_weights.h5f'
# plot_filename = results_path + 'reward_plot.png'

ENV_NAME = "ConnectFour-v0"
MAX_RUNS = 1000

GAMMA = 0.95
LEARNING_RATE = 0.001

MEMORY_SIZE = 100000
BATCH_SIZE = 20

EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.01
EXPLORATION_DECAY = 0.995

learn_from_enemy = True
better_reward = True

# Vanilla Multi Layer Perceptron version that starts converging to solution after ~50 runs


class DQNSolver:

    def __init__(self, observation_space, action_space):
        self.exploration_rate = EXPLORATION_MAX

        self.action_space = action_space
        self.memory = deque(maxlen=MEMORY_SIZE)

        self.model = Sequential()
        #self.model.add(Flatten(data_format=observation_space))
        self.model.add(Flatten(input_shape=observation_space))
        self.model.add(Dense(24, activation="relu"))
        self.model.add(Dense(24, activation="relu"))
        self.model.add(Dense(self.action_space, activation="linear"))
        self.model.compile(loss="mse", optimizer=Adam(lr=LEARNING_RATE))

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, available_moves=[]):
        if np.random.rand() < self.exploration_rate:
            return random.randrange(self.action_space)
        q_values = self.model.predict(state)
        q_values = np.array([[x if idx in available_moves else -100 for idx, x in enumerate(q_values[0])]])
        return np.argmax(q_values[0])

    def experience_replay(self):
        if len(self.memory) < BATCH_SIZE:
            return
        batch = random.sample(self.memory, BATCH_SIZE)
        for state, action, reward, state_next, terminal in batch:
            q_update = reward
            if not terminal:
                q_update = (reward + GAMMA * np.amax(self.model.predict(state_next)[0]))
            q_values = self.model.predict(state)
            q_values[0][action] = q_update
            self.model.fit(state, q_values, verbose=0)
        self.exploration_rate *= EXPLORATION_DECAY
        self.exploration_rate = max(EXPLORATION_MIN, self.exploration_rate)


class NNPlayer(Player):
    def __init__(self, env, name='NNPlayer'):
        super(NNPlayer, self).__init__(env, name)

        self.observation_space = env.observation_space.shape
        self.action_space = env.action_space.n
        self.last_move = -1
        self.enemie_move = -1

        self.dqn_solver = DQNSolver(self.observation_space, self.action_space)

    def get_next_action(self, state: np.ndarray) -> int:
        state = np.reshape(state, [1] + list(self.observation_space))
        for _ in range(100):
            action = self.dqn_solver.act(state, self.env.available_moves())
            if self.env.is_valid_action(action):
                return action
        for i in range(env.board_shape[1]):
            if env.is_valid_action(i):
                return i
        print("Can't determine next action")
        print(f"Board is {state}")
        print(f"My last intended action is {action}")
        raise Exception('Unable to determine a valid move! Maybe invoke at the wrong time?')

    def learn(self, state, action, reward, state_next, done) -> None:
        state = np.reshape(state, [1] + list(self.observation_space))
        state_next = np.reshape(state_next, [1] + list(self.observation_space))

        # reward = reward if not done else -reward

        self.dqn_solver.remember(state, action, reward, state_next, done)

        if (learn_from_enemy and self.last_move != -1): # works only if enemie always -1
            reward = reward*(-1)
            pos =self.get_highest_pos(state,self.enemie_move)
            state[0][pos][self.enemie_move] = 0
            nega_state = np.negative(state)
            nega_state_next =  copy.deepcopy(nega_state)
            nega_state_next[0][pos][self.enemie_move] = 1
            nega_pos =self.get_highest_pos(nega_state_next,action)
            nega_state_next[0][nega_pos-1][action] = -1

            state[0][pos][self.enemie_move] = -1 # restore state

            self.dqn_solver.remember(nega_state, self.enemie_move, reward, nega_state_next, False) # Last move fehlt


        self.enemie_move = self.get_emove(state,state_next) #first round fehlt
        self.last_move = action


        if not done:
            self.dqn_solver.experience_replay()

    def get_highest_pos(self, state, row):
        for i in range(0, len(state[0])):
            if (state[0][i][row]!=0):
                return i
        return -1 # row empty

    def get_emove(self, state, state_next):
        last_moves = state_next-state
        for i in range(0, self.env.board_shape[0]):
            if (-1 in last_moves[0][i]):
                return int(np.where(last_moves[0][i]==-1)[0])
        return -1 #TODO

class HumanPlayer(Player):
    def __init__(self, env, name='HumanPlayer'):
        super(HumanPlayer, self).__init__(env, name)


    def get_next_action(self, state: np.ndarray) -> int:
        for _ in range(10):
            action = input("Please enter a number between 1 and 7: ")
            try:
                action = int(action)
                if action >=1 and action <= 7:
                    if self.env.is_valid_action(action-1):
                        return action - 1
                    else:
                        print('That is not a valid action ')
                else:
                    print('That was not a number between 1 and 7 ')
            except ValueError:
                print('Not a number')
        raise Exception('Entered wrong input 10 times')


    def learn(self, state, action, reward, state_next, done) -> None:
        pass



def game(opponent, env, name, paint_board=False):
    # env = gym.make(ENV_NAME)

    player = NNPlayer(env, 'NNPlayer')
    #player = HumanPlayer(env, 'HumanPlayer')
    #opponent = HumanPlayer(env, 'HumanPlayer')
    #opponent = RandomPlayer(env, 'OpponentRandomPlayer')
    #opponent = LeftiPlayer(env, 'LeftiPlayer')
    # opponent = MinMaxPlayer(env, 'MinMaxPlayer',f=0.00001)

    total_reward = 0
    all_rewards = []
    average_rewards = []
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
            oldstate = copy.deepcopy(state)
            action = player.get_next_action(state)
            state_next, reward, terminal, info = env.step(action) # hier die opponent.action holen
            if player.name =='HumanPlayer': #TODO if both human paint for each turn
                paint(state)
            # TODO reward mit steps
            if(better_reward and step>= 4):
                reward = reward/(step-3)

            player.learn(oldstate, action, reward, state_next, terminal)

            state = state_next

            if terminal:
                if paint_board:
                    total_reward += reward
                    all_rewards.append(reward)
                    if player.name=='NNPlayer':
                        print("Run: " + str(run) + ", exploration: " + str(player.dqn_solver.exploration_rate) + ", score: " + str(reward))
                    if reward > 0:
                        wins +=1
                        print(f"winner: {player.name}")
                        paint(state)
                        print(f"reward={reward}")
                    elif reward < 0:
                        losses += 1
                        print(f"lost to: {opponent.name}")
                        paint(state)
                        print(f"reward={reward}")
                    elif reward == env.DRAW_REWARD:
                        draws += 1
                        print(f"draw after {player.name} move")
                        paint(state)
                        print(f"reward={reward}")
                    print(f"Wins [{wins}], Draws [{draws}], Losses [{losses}] - Total reward {total_reward}, average reward {total_reward/run}")
                    average_rewards.append(total_reward/run)
                    # score_logger.add_score(step, run)
                    break
                else:
                    total_reward += reward
                    all_rewards.append(reward)
                    if reward > 0:
                        wins +=1
                    elif reward < 0:
                        losses += 1
                    elif reward == env.DRAW_REWARD:
                        draws += 1
                    average_rewards.append(total_reward/run)
                    break
        lasthundred = all_rewards[-100:]
        # TODO ist nicht mehr nur 1 und -1
        # if lasthundred.count(1) == 100:
        #     player.dqn_solver.model.save_weights(results_path+str(lasthundred.count(1))+'dqn_weights.h5f')
        #     break

        if run >= MAX_RUNS:
            # print(lasthundred.count(1),lasthundred.count(-1),lasthundred.count(0))
            player.dqn_solver.model.save_weights(results_path+name+'lh'+str(lasthundred.count(1))+'dqn_weights.h5f')
            break
    plt.clf()
    pickle.dump( all_rewards, open( results_path+name+'all'+"save.p", "wb" ) )
    plt.plot(all_rewards, 'ro')
    plt.ylabel('Reward')
    plt.xlabel('Episode')
    # plt.show()
    plt.savefig(results_path +name+'all'+ 'reward_plot.png')

    plt.clf()
    pickle.dump( average_rewards, open( results_path+name+'average'+"save.p", "wb" ) )
    plt.plot(average_rewards, 'ro')
    plt.ylabel('Reward')
    plt.xlabel('Episode')
    # plt.show()
    plt.savefig(results_path +name+'average'+ 'reward_plot.png')
    plt.clf()

def paint(board):
    # Render the environment to the screen

    print(Back.BLUE)
    for i in range(len(board)):
        for j in range(len(board[i])):
            if board[i][j]==-1:
                print('\033[31m' + "\u25CF" , end =" ")
            elif board[i][j]==1:
                print('\033[33m' + "\u25CF" , end =" ")
            else:
                print('\033[30m' + "\u25CF" , end =" ")
        print(Style.RESET_ALL + '\x1b[K')
        print(Back.BLUE, end ="")
    print(Style.RESET_ALL+'\x1b[K')


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        env = gym.make(ENV_NAME)
        learn_from_enemy = False
        better_reward = False
        game(RandomPlayer(env, 'OpponentRandomPlayer'),env,'RandomFF')
        better_reward = True
        game(RandomPlayer(env, 'OpponentRandomPlayer'),env,'RandomTF')
        better_reward = False
        game(LeftiPlayer(env, 'LeftiPlayer'),env,'LeftiFF')
        better_reward = True
        game(LeftiPlayer(env, 'LeftiPlayer'),env,'LeftiTF')
        learn_from_enemy= True
        game(RandomPlayer(env, 'OpponentRandomPlayer'),env,'RandomTT')
        game(LeftiPlayer(env, 'LeftiPlayer'),env,'LeftiTT')

        opponents = [
            MinMaxPlayer(env, 'MinMaxPlayer',f=0.0001),
            MinMaxPlayer(env, 'MinMaxPlayer',f=0.001),
            MinMaxPlayer(env, 'MinMaxPlayer',f=0.01),
            MinMaxPlayer(env, 'MinMaxPlayer',f=0.1),
            MinMaxPlayer(env, 'MinMaxPlayer',f=1),
            MinMaxPlayer(env, 'MinMaxPlayer',f=10),
            ]
        for opponent in opponents:
            game(opponent, env,'MM'+str(opponent.f))

