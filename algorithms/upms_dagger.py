from imitation.algorithms import UPMS


class UPMSDAgger(UPMS):

    def _act(self, observation):
        if self._episode == 0:
            control_policy = self.teacher
        else:
            control_policy, _ = self._mix()

        control_action, uncertainty = control_policy.predict(observation, [self._episode, None])

        self._query_expert(control_policy, control_action, uncertainty, observation)

        self._active_policy = control_policy == self.teacher

        return control_action
