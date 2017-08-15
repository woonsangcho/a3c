
import numpy as np
from auditor import Auditor
from time import time
import brain
from helper import PartialRollout
from helper import process_rollout
from helper import set_random_seed

class WorkerFactory(object):
    def create_worker(**kwargs):
        algo_name = kwargs['algo_name']

        if (algo_name == 'a3c'):
            return WorkerA3C(**kwargs)

class GeneralWorker(object):
    def __init__(self, **kwargs):
        self.env = kwargs['env']
        self.name = kwargs['worker_name']
        self.initial_learning_rate = kwargs['learning_rate']
        self.algo_name = kwargs['algo_name']

        self.max_master_time_step = kwargs['max_master_time_step']
        self.max_clock_limit = kwargs['max_clock_limit']

        self.episode_reward = 0
        self.episode_length = 0
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_mean_values = []
        self.local_timesteps = 0

        self.auditor = Auditor(**kwargs)

        self.anneal_learning_rate = kwargs['anneal_learning_rate']
        self.start_clock = time()
        self.use_clock = kwargs['anneal_by_clock']
        self.max_clock_limit = kwargs['max_clock_limit']

        self.convs = kwargs['convs']
        self.hiddens = kwargs['hiddens']

        self.device = kwargs['device']


class WorkerA3C(GeneralWorker):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.max_bootstrap_length = kwargs['max_bootstrap_length']
        self.local_brain = brain.A3CFeedForwardNN(**kwargs)
        set_random_seed(kwargs['task_index'] * 89)

        self.last_env_state = self.env.reset()
        self.env.seed(np.random.randint(0, 2 ** 16))


    def work(self, session,
             current_master_timestep,
             ):

        rollout = self._env_sampling(session,
                                     self.max_bootstrap_length,
                                     current_master_timestep
                                    )

        batch = process_rollout(rollout=rollout, gamma=0.99)

        feed_dict = {
            self.local_brain.input: batch.si,
            self.local_brain.actions: batch.actions,
            self.local_brain.advantages: batch.advantages,
            self.local_brain.target_R: batch.target_R,
            self.local_brain.learning_rate: self.initial_learning_rate,
        }

        fetches = [self.local_brain.apply_grads, self.local_brain.policy_loss, self.local_brain.value_loss, self.local_brain.loss]
        fetched = session.run(fetches, feed_dict=feed_dict)

        self.local_timesteps += len(rollout.states)
        self.auditor.recordLosses(fetched[1], fetched[2], fetched[3], current_master_timestep)

        return len(rollout.states)


    def _env_sampling(self, session,
                      max_bootstrap_length,
                      current_master_timestep
                      ):
        while True:
            terminal_end = False
            rollout = PartialRollout()

            for _ in range(max_bootstrap_length):

                fetched = self.local_brain.get_transition(session, self.last_env_state, 1)
                action, value_ = fetched[0], fetched[1]
                state, reward, terminal, info = self.env.step(action.argmax())

                rollout.add(self.last_env_state, action, reward, value_, terminal)
                self.episode_reward += reward
                self.episode_length += 1

                self.last_env_state = state

                timestep_limit = self.env.spec.tags.get('wrapper_config.TimeLimit.max_episode_steps')
                if terminal or self.episode_length >= timestep_limit:
                    terminal_end = True
                    self.last_env_state = self.env.reset()
                    self.env.seed(np.random.randint(0, 2 ** 16))

                    self.auditor.recordStats(
                                        score=self.episode_reward,
                                        length=self.episode_length,
                                        t=current_master_timestep)

                    self.episode_reward = 0
                    self.episode_length = 0
                    break

            if not terminal_end:
                rollout.r = self.local_brain.get_value(session, self.last_env_state)

            return rollout



