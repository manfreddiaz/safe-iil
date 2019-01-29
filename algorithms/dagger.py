import math
import numpy as np
from .iil_learning import InteractiveImitationLearning


class DAgger(InteractiveImitationLearning):

    def __init__(self, env, teacher, learner, horizon, episodes, seed, alpha=0.5):
        InteractiveImitationLearning.__init__(self, env, teacher, learner, horizon, episodes)
        # expert decay
        self.p = alpha
        self.alpha = self.p
        self.seed = seed
        self.sampling_generator = np.random.RandomState(seed)

    def _mix(self):
        control_policy = self.sampling_generator.choice(
            a=[self.teacher, self.learner],
            p=[self.alpha, 1. - self.alpha]
        )

        return control_policy

    def _on_episode_done(self):
        self.alpha = self.p ** self._episode  # decay expert probability after each episode
        InteractiveImitationLearning._on_episode_done(self)


