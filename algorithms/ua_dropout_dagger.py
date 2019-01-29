import math

from .dagger import DAgger
from .iil_learning import InteractiveImitationLearning


class DropoutDAgger(DAgger):
    def __init__(self, env, teacher, learner, threshold, horizon, episodes):
        DAgger.__init__(self, env, teacher, learner, horizon, episodes, None)
        self.threshold = threshold
        self.learner_uncertainty = [math.inf, math.inf]

    def _mix(self):
        uncertainty_v, uncertainty_theta = self.learner_uncertainty
        if uncertainty_v > self.threshold or uncertainty_theta > self.threshold:
            return self.teacher
        else:
            return self.learner

    def _on_episode_done(self):
        InteractiveImitationLearning._on_episode_done(self)
