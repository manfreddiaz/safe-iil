import math
import numpy as np

from .dagger import DAgger
from .iil_learning import InteractiveImitationLearning


class DropoutDAgger(DAgger):
    def __init__(self, env, teacher, learner, threshold, horizon, episodes, seed):
        DAgger.__init__(self, env, teacher, learner, horizon, episodes, seed, alpha=None)
        self.threshold = threshold
        self.learner_uncertainty = [math.inf, math.inf]

    def _mix(self):
        if np.mean(self.learner_uncertainty) > self.threshold:
            return self.teacher
        else:
            return self.learner

    def _on_episode_done(self):
        InteractiveImitationLearning._on_episode_done(self)
