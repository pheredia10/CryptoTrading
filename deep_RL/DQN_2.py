# -*- coding: utf-8 -*-
import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from crypto_env import CryptoEnv,CryptoEnv2
import sys
import matplotlib.pyplot as plt
EPISODES = 50


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()

        model.add(Dense(100, input_dim=self.state_size, activation='relu'))
        # model.add(Dense(100, activation='relu'))
        model.add(Dense(200, activation='sigmoid'))
        # model.add(Dense(200, activation='relu'))
        # model.add(Dense(100, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate),metrics=['mae'])
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
      
        minibatch = random.sample(list(self.memory), batch_size) # list(self.memory)[:batch_size] 
        state_list=[]
        target_f_list=[]
        counter=0
        for state, action, reward, next_state, done in minibatch:
            # print(state)
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            # print('target_f',target_f)
            target_f[0][action] = target
            target_f_list.append(target_f[0])
            state_list.append(state[0])
            
            # print('target_f2',target_f)
        state_list=np.array(state_list)
        target_f_list=np.array(target_f_list)
        # print('state list',state_list)
        # print('state list shape:',state_list.shape)
        # print('target_f list',target_f_list)
        # print('counter',counter)
        self.model.fit(state_list, target_f_list, epochs=1, verbose=1)
        counter+=1
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


if __name__ == "__main__":
    env =CryptoEnv() #gym.make('CartPole-v1')
    state_size =env.dim_state  # env.observation_space.shape[0]
    action_size =env.num_coins*2 #env.action_space.n
    agent = DQNAgent(state_size, action_size)
    # agent.load("./save/cartpole-dqn.h5")
    done = False
    batch_size = 32

    reward_means=[]
    actions=[]
    for e in range(EPISODES):
        state = env.reset()
        state = np.reshape(state, [1,state_size])
        reward_vec=[]
        print('episode',e)
        for time in range(200):
            # env.render()
            # print('time: ',time)
            action = agent.act(state)
            actions.append(action)
            # if time==2:
            #     sys.exit()
            next_state, reward, done, _ = env.step(action)
            # print('reward',reward)
            reward = reward if not done else -10
            reward_vec.append(reward)
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if False:
                print("episode: {}/{}, time: {}, e: {:.2}"
                      .format(e, EPISODES, time, agent.epsilon))
                break
            if len(agent.memory) >= batch_size:
                print('train time:',time)
                agent.replay(batch_size)
                # agent.memory = deque(maxlen=2000)
                # agent.memory.popleft()
        print('mean reward',np.mean(reward_vec))
        reward_means.append(np.mean(reward_vec))
        agent.epsilon
        # plt.plot(env.portfolio_value)
        # plt.show()
plt.figure(0)
plt.plot(reward_means)
plt.figure(1)
plt.plot(actions)
plt.show()


        # if e % 10 == 0:
        #     agent.save("./save/cartpole-dqn.h5")         