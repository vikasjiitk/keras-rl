import numpy as np

import gym
from keras.layers import Dense, Activation, Flatten
from keras.models import Sequential
from keras.optimizers import Adam

from LTMSteacher import LTMSTeacher
from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import BoltzmannQPolicy

from copy import deepcopy

TARGET_ENV_NAME = 'CartPole-target-v0'

SOURCE_ENV_NAMES = ['CartPole-src-v1', 'CartPole-src-v2', 'CartPole-src-v3', 'CartPole-src-v4']

# Get the target environment and extract the number of actions.
target_env = gym.make(TARGET_ENV_NAME)
np.random.seed(123)
target_env.seed(123)
target_env_observation_space_shape = target_env.observation_space.shape
nb_actions = target_env.action_space.n

# Get the source environments and remove the source environments for which the observation space and action space shape
# are not equal
SRC_ENVS_MAP = {}
SRC_ENVS = set()

for env_name in SOURCE_ENV_NAMES:
    env = gym.make(env_name)
    env.seed(123)
    if env.observation_space.shape != target_env_observation_space_shape and env.action_space.n != nb_actions:
        continue
    SRC_ENVS.add(env)
    SRC_ENVS_MAP[env] = env_name


rounds = 10

src_episodes_reward_list = []
target_episodes_reward_list = []

lr = 1e-4

for i in range(rounds):

    # First, we build a very simple model.
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
    src_envs = set()
    source_envs_name_map = {}
    for env in SRC_ENVS:
        copy_env = deepcopy(env)
        src_envs.add(copy_env)
        source_envs_name_map[copy_env] = SRC_ENVS_MAP[env]

    teacher = LTMSTeacher(agent=dqn, target_env=target_env, src_envs=src_envs, lr=lr)

    # Train the agent through the teacher
    src_episodes_reward, target_episodes_reward = teacher.train()

    # print curriculum
    print "Curriculum: ",
    for env in teacher.curriculum:
        print source_envs_name_map[env],
    print ""

    src_episodes_reward_list.append(src_episodes_reward)
    target_episodes_reward_list.append(target_episodes_reward)

# save episodes and rewards for plotting
np.savez(
    "/tmp/ltms_dqn_cartpole"+str(rounds),
    src_episodes_reward_list=src_episodes_reward_list,
    target_episodes_reward_list=target_episodes_reward_list
)

# After training is done, we save the final weights.
# dqn.save_weights('dqn_{}_weights.h5f'.format(TARGET_ENV_NAME), overwrite=True)

# Finally, evaluate our algorithm for 5 episodes.
# dqn.test(target_env, nb_episodes=5, visualize=True)
