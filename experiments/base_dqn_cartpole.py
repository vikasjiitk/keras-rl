
import numpy as np

import gym
from keras.layers import Dense, Activation, Flatten
from keras.models import Sequential
from keras.optimizers import Adam

from RMGteacher import RMGTeacher
from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import BoltzmannQPolicy

TARGET_ENV_NAME = 'CartPole-target-v0'

# Get the target environment and extract the number of actions.
target_env = gym.make(TARGET_ENV_NAME)
np.random.seed(123)
target_env.seed(123)
target_env_observation_space_shape = target_env.observation_space.shape
nb_actions = target_env.action_space.n


rounds = 10

src_episodes_reward_list = []
target_episodes_reward_list = []

lr = 1e-4

for i in range(rounds):
    print i

    # Next, we build a very simple model.
    model = Sequential()
    model.add(Flatten(input_shape=(1,) + target_env.observation_space.shape))
    model.add(Dense(16))
    model.add(Activation('relu'))
    model.add(Dense(16))
    model.add(Activation('relu'))
    model.add(Dense(16))
    model.add(Activation('relu'))
    model.add(Dense(nb_actions))
    model.add(Activation('linear'))
    print(model.summary())

    # Next, we configure our agent. You can use every built-in Keras optimizer and
    # even the metrics!
    memory = SequentialMemory(limit=50000, window_length=1)
    policy = BoltzmannQPolicy()

    dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10,
                   target_model_update=1e-2, policy=policy)

    dqn.compile(Adam(lr=lr), metrics=['mae'])

    # Next, we configure our teacher
    teacher = RMGTeacher(agent=dqn, target_env=target_env, src_envs=set(), lr=lr)

    # Train the agent through the teacher
    src_episodes_reward, target_episodes_reward = teacher.train()
    src_episodes_reward_list.append(src_episodes_reward)
    target_episodes_reward_list.append(target_episodes_reward)

# save episodes and rewards for plotting
np.savez(
    "/tmp/base_dqn_cartpole"+str(rounds),
    src_episodes_reward_list=src_episodes_reward_list,
    target_episodes_reward_list=target_episodes_reward_list
)

# After training is done, we save the final weights.
# dqn.save_weights('dqn_{}_weights.h5f'.format(TARGET_ENV_NAME), overwrite=True)

# Finally, evaluate our algorithm for 5 episodes.
# dqn.test(target_env, nb_episodes=5, visualize=True)
