import contextlib
import numpy as np
from .dagger import DAgger


class AggreVaTe(DAgger):

    def __init__(self, env, teacher, learner, explorer, horizon, episodes, seed=None, alpha=0.99):
        DAgger.__init__(self, env, teacher, learner, horizon, episodes, seed, alpha)
        self.break_point = None
        self.explorer = explorer
        self.t = horizon
        self.seed = seed
        self.time_generator = np.random.RandomState(self.seed)

    def train(self, samples=1, debug=False):
        DAgger.train(self, samples, debug)

    def _act(self, observation):
        if self._episode == 0:
            control_policy = self.teacher
        else:
            if self._current_horizon < self.t:
                control_policy = self._mix()
            elif self._current_horizon == self.t:
                control_policy = self.explorer
            else:
                control_policy = self.teacher

        control_action, uncertainty = control_policy.predict(observation, [self._episode, None])

        self._query_expert(control_policy, control_action, uncertainty, observation)

        self.active_policy = control_policy == self.teacher

        return control_action

    def _query_expert(self, control_policy, control_action, uncertainty, observation):
        if control_policy == self.learner:
            self.learner_action = control_action
            self.learner_uncertainty = uncertainty  # it might but it wont
        else:
            self.learner_action, self.learner_uncertainty = self.learner.predict(observation, [self._episode, None])

        if control_policy == self.teacher:
            self.teacher_action = control_action
            self.teacher_uncertainty = uncertainty
        else:
            self.teacher_action, self.teacher_uncertainty = self.teacher.predict(observation, [self._episode, control_action])

        if self.teacher_action is not None:
            if control_policy == self.teacher:  # only aggregate data for t+1 steps
                self._aggregate(observation, self.teacher_action)
                self.teacher_queried = True
        else:
            self.teacher_queried = False

    def _on_sampling_done(self):
        self.t = self.time_generator.randint(0, self._horizon)
