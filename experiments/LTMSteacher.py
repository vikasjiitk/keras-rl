import numpy as np

from keras.optimizers import Adam
from rl.agents.dqn import DQNAgent

WEIGHT_FILE_NAME = '/tmp/temp_save_ltms_2.h5py'


class LTMSTeacher:
    """
    LTMS Teacher class
    """

    def __init__(self, agent, target_env, src_envs=set(), lr=1e-3):
        self.agent = agent
        self.target_env = target_env
        self.src_envs = src_envs
        self.lr = lr
        self.curriculum = []
        self.per_task_train_steps = 1000
        self.per_task_transfer_steps = 200
        self.src_task_train_steps = 2000
        self.target_task_train_steps = 6000
        # add extra steps to train target task if source tasks set is empty
        if not len(self.src_envs):
            # we have 4 source tasks in all our experiments
            num_src_tasks = 4
            extra_steps =\
                num_src_tasks*self.src_task_train_steps
            # + (num_src_tasks*(num_src_tasks+1)-2)/2*self.per_task_eval_steps
            self.src_task_train_steps = extra_steps
            self.src_envs.add(self.target_env)

    def train(self):
        src_episodes_reward_lists = self.preprocess()
        src_episodes_reward_lists += self._train_on_src_envs()
        target_episodes_reward_list = self._train_on_target_env()

        return src_episodes_reward_lists, target_episodes_reward_list

    def preprocess(self):
        print "Preprocessing and finding curricula"
        num_envs = len(self.src_envs) + 1
        envs_list = list(self.src_envs)
        envs_list.append(self.target_env)
        transferability_matrix = np.zeros((num_envs-1, num_envs))

        total_reward_list = []
        for src_env_num in range(num_envs-1):
            src_agent = self._clone_agent(self.agent)
            total_reward_list +=\
                self._train_on_task(agent=src_agent, env=envs_list[src_env_num], steps=self.per_task_train_steps)

            for target_env_num in range(num_envs):
                if src_env_num == target_env_num:
                    transferability_matrix[src_env_num][target_env_num] = -1000
                else:
                    transferability_matrix[src_env_num][target_env_num] = 0
                    target_agent = self._clone_agent(src_agent)
                    transfer_reward =\
                        self._train_on_task(
                            agent=target_agent,
                            env=envs_list[target_env_num],
                            steps=self.per_task_transfer_steps
                        )
                    transferability_matrix[src_env_num][target_env_num] += sum(transfer_reward)
                    total_reward_list += transfer_reward

        # index of target task
        last_env_num = num_envs - 1
        while True:
            self.curriculum.append(envs_list[last_env_num])
            transfer_measure = -10001

            for env_num in range(num_envs-1):
                if transfer_measure < transferability_matrix[env_num][last_env_num] and\
                                envs_list[env_num] not in self.curriculum:
                    transfer_measure = transferability_matrix[env_num][last_env_num]
                    next_task = env_num

            if envs_list[next_task] in self.curriculum:
                break
            else:
                last_env_num = next_task

        self.curriculum.reverse()
        # remove the target task environment
        self.curriculum.pop()
        return total_reward_list

    def _train_on_src_envs(self):
        print "Training on source tasks"
        src_episodes_reward_lists = []
        for env in self.curriculum:
            # find task to learn
            self.src_envs.discard(env)
            # learn the task
            episodes_reward_list =\
                self._train_on_task(agent=self.agent, env=env, steps=self.src_task_train_steps)
            # add the reward obtained during learning of the task
            src_episodes_reward_lists += episodes_reward_list
            # total no of steps
        return src_episodes_reward_lists

    def _train_on_target_env(self):
        print "Training on target task"
        return self._train_on_task(agent=self.agent, env=self.target_env, steps=self.target_task_train_steps)

    def _clone_agent(self, agent):
        self.agent.save_weights(WEIGHT_FILE_NAME, overwrite=True)
        cloned_agent =\
            DQNAgent(
                model=agent.model,
                nb_actions=agent.nb_actions,
                memory=agent.memory,
                nb_steps_warmup=agent.nb_steps_warmup,
                target_model_update=agent.target_model_update,
                policy=agent.policy
            )
        cloned_agent.compile(Adam(lr=self.lr), metrics=['mae'])
        cloned_agent.load_weights(WEIGHT_FILE_NAME)
        return cloned_agent

    def _train_on_task(self, agent, env, steps):
        (_, episodes_reward_list) = agent.fit(env, nb_steps=steps, visualize=False, verbose=2)
        return episodes_reward_list
