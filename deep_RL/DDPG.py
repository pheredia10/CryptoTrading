import numpy as np
import gym

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, Concatenate,LSTM,Reshape
from keras.optimizers import Adam

from rl.agents import DDPGAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess
from crypto_env import CryptoEnv
import sys
import matplotlib.pyplot as plt

# ENV_NAME = 'Pendulum-v0'
gym.undo_logger_setup()


# Get the environment and extract the number of actions.
# env = gym.make(ENV_NAME)
env=CryptoEnv(0,2000)
# np.random.seed(123)
# env.seed(123)
# assert len(env.action_space.shape) == 1
nb_actions = env.num_coins  # env.action_space.shape[0]

# Next, we build a very simple model.
actor = Sequential()
# actor.add(Flatten(input_shape=(1,) + env.observation_space.shape))
actor.add(Flatten(input_shape=(1, env.dim_state)))
# print(Flatten(input_shape=(1, env.dim_state)))


# actor.add(Dense(16))  
# actor.add(Activation('relu'))
# actor.add(Dense(16))
# actor.add(Activation('relu'))
# actor.add(Dense(16))
# actor.add(Dense(nb_actions))
# actor.add(Activation('linear'))

# actor.add(Dense(int(env.dim_state)))  
# actor.add(Activation('relu'))
actor.add(Dense(int(env.dim_state*1.5)))
actor.add(Activation('relu'))
actor.add(Dense(int(env.dim_state*2)))
actor.add(Activation('softmax'))
actor.add(Dense(nb_actions))
actor.add(Activation('linear'))


print(actor.summary())

action_input = Input(shape=(nb_actions,), name='action_input')
# observation_input = Input(shape=(1,) + env.observation_space.shape, name='observation_input')
observation_input = Input(shape=(1, env.dim_state), name='observation_input')
flattened_observation = Flatten()(observation_input)
x = Concatenate()([action_input, flattened_observation])

# x = Dense(32)(x)
# x = Activation('relu')(x)
# x = Dense(32)(x)
# x = Activation('relu')(x)
# x = Dense(32)(x)
# x = Activation('relu')(x)
# x = Dense(1)(x)


# x = Dense(int(env.dim_state*1.5))(x)
# x = Activation('relu')(x)
x = Dense(int(env.dim_state*2))(x)
x = Activation('relu')(x)
x = Dense(int(env.dim_state*2))(x)
x = Activation('softmax')(x)
x = Dense(int(env.dim_state*1.5))(x)
x = Activation('softmax')(x)
x = Dense(1)(x)

# x = LSTM(int(env.dim_state*1.3),)(x)
# x = Activation('relu')(x)
# x = LSTM(int(env.dim_state*1.3))(x)
# x = Activation('relu')(x)
# x = LSTM(int(env.dim_state*1.3))(x)
# x = Activation('relu')(x)
# x = Dense(1)(x)

x = Activation('linear')(x)
critic = Model(inputs=[action_input, observation_input], outputs=x)
print(critic.summary())

# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!
memory = SequentialMemory(limit=100000, window_length=1)
random_process = OrnsteinUhlenbeckProcess(size=nb_actions, theta=.15, mu=0., sigma=.03)
agent = DDPGAgent(nb_actions=nb_actions, actor=actor, critic=critic, critic_action_input=action_input,
                  memory=memory, nb_steps_warmup_critic=100, nb_steps_warmup_actor=100,
                  random_process=random_process, gamma=.8, target_model_update=1e-3)
agent.compile(Adam(lr=.001, clipnorm=1.), metrics=['mae'])

# Okay, now it's time to learn something! We visualize the training here for show, but this
# slows down training quite a lot. You can always safely abort the training prematurely using
# Ctrl + C.
nb_steps= 800*1440  #1*(env.periods-2)# 100*(env.periods-2) #100000+1870#env.periods-2
agent.fit(env,nb_steps, visualize=True, verbose=2, nb_max_episode_steps=1440,log_interval=10)
plt.figure(0)
plt.plot(env.portfolio_value)
plt.figure(1)
noise_over_action_array=np.array(agent.noise_over_action)
noise_over_action_array=np.transpose(noise_over_action_array)
for i in range(nb_actions):
	plt.plot(noise_over_action_array[i,:])
plt.show()
# After training is done, we save the final weights.
agent.save_weights('ddpg_{}_weights5.h5f'.format('Crypto'), overwrite=True)

# Finally, evaluate our algorithm for 5 episodes.
agent.test(env, nb_episodes=5, visualize=True, nb_max_episode_steps=200)