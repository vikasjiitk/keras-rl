
import random
from keras.optimizers import Adam
from rl.agents.dqn import DQNAgent

WEIGHT_FILE_NAME = '/tmp/temp_save_'


class RMGTeacher:
    """
    RMG Teacher class
    """

    def __init__(self, agent, target_env, src_envs=set(), lr=1e-3, name="base"):
        self.agent = agent
        self.target_env = target_env
        self.src_envs = src_envs
        self.lr = lr
        self.curriculum = []
        self.name = name
        self.src_task_train_steps = 2000
        self.per_task_eval_steps = 200
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
        src_episodes_reward_lists = self._train_on_src_envs()
        target_episodes_reward_list = self._train_on_target_env()

        return src_episodes_reward_lists, target_episodes_reward_list

    def _train_on_src_envs(self):
        print "Training on source tasks"
        src_episodes_reward_lists = []
        while len(self.src_envs):
            cloned_agent = self._clone_agent(agent=self.agent)
            # find task to learn
            episodes_reward_list, env = self._eval_src_envs(cloned_agent=cloned_agent)
            self.src_envs.discard(env)
            # append the task in the curriculum
            self.curriculum.append(env)
            # learn the task
            episodes_reward_list +=\
                self._train_on_task(agent=self.agent, env=env, steps=self.src_task_train_steps)
            # add the reward obtained during learning of the task
            src_episodes_reward_lists += episodes_reward_list
            # total no of steps
        return src_episodes_reward_lists

    def _train_on_target_env(self):
        print "Training on target task"
        return self._train_on_task(agent=self.agent, env=self.target_env, steps=self.target_task_train_steps)

    def _clone_agent(self, agent):
        self.agent.save_weights(WEIGHT_FILE_NAME+self.name+".h5py", overwrite=True)
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
        cloned_agent.load_weights(WEIGHT_FILE_NAME+self.name+".h5py")
        return cloned_agent

    def _eval_src_envs(self, cloned_agent):

        if len(self.src_envs) == 1:
            return [], next(iter(self.src_envs))

        (min_episodes, selected_env) = (self.per_task_eval_steps+1, None)
        total_episode_reward_list = []
        for env in self.src_envs:
            episodes_reward_list = self._train_on_task(agent=cloned_agent, env=env, steps=self.per_task_eval_steps)
            num_episodes = len(episodes_reward_list)
            if num_episodes < min_episodes and num_episodes != 0:
                (min_episodes, selected_env) = (num_episodes, env)
            total_episode_reward_list += episodes_reward_list
        if selected_env is None:
            selected_env = random.sample(self.src_envs, 1)[0]
        return total_episode_reward_list, selected_env

    def _train_on_task(self, agent, env, steps):
        (_, episodes_reward_list) = agent.fit(env, nb_steps=steps, visualize=False, verbose=2)
        return episodes_reward_list
