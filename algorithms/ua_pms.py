import math

import numpy as np

from .dagger import DAgger


class UPMS(DAgger):

    def __init__(self, env, teacher, learner, explorer, safety_coefficient, horizon, episodes, seed):
        DAgger.__init__(self, env, teacher, learner, horizon, episodes, seed)
        self.explorer = explorer
        self._safety_coefficient = safety_coefficient

    def _normalize_uncertainty(self, uncertainty):
        return np.sum(self.learner_uncertainty) / uncertainty.shape[0]

    def _preferential_coefficient(self, uncertainty):
        return 1. - np.tanh(self._safety_coefficient * uncertainty)

    def _non_preferential_coefficient(self, uncertainty):
        return np.tanh(self._safety_coefficient * uncertainty)

    # Rational Policy Mixing
    def _rpm(self, policy_p, policy_p_uncertainty, policy_q, policy_q_uncertainty):
        alpha_p = self._preferential_coefficient(policy_p_uncertainty)
        alpha_q = self._preferential_coefficient(policy_q_uncertainty)

        # consistency
        normalization = alpha_p + alpha_q
        p_mix, q_mix = alpha_p / normalization, alpha_q / normalization
        # impossibility
        if math.isnan(p_mix) and math.isnan(q_mix):
            return self._on_impossible_selection()
        # rationality
        policy = self.sampling_generator.choice(a=[policy_p, policy_q], p=[p_mix, q_mix])
        if policy == policy_p:
            return policy_p, p_mix

        return policy_q, q_mix

    # Preferential Policy Mixing
    def _ppm(self, policy_p, policy_p_uncertainty, policy_s):
        alpha_p = self._preferential_coefficient(policy_p_uncertainty)
        return self.sampling_generator.choice(a=[policy_p, policy_s], p=[alpha_p, 1. - alpha_p]), alpha_p

    def _mix(self):
        _, teacher_uncertainty = self.teacher.predict(None, [self._episode, 0]) # FIXME: after experiments done
        return self._rpm(self.teacher, teacher_uncertainty, self.learner, self._normalize_uncertainty(self.learner_uncertainty))

    # \mathcal{E}^\prime
    def _mix_exploration(self):
        _, teacher_uncertainty = self.teacher.predict(None, [self._episode, 0])
        return self._ppm(self.teacher, teacher_uncertainty, self.explorer)

    def _act(self, observation):
        if self._episode == 0:
            control_policy = self.teacher
        else:
            pi_i, pi_i_preference = self._mix()
            e_i, e_i_preference = self._mix_exploration()
            control_policy = self.sampling_generator.choice(
                a=[pi_i, e_i],
                p=[pi_i_preference, 1. - pi_i_preference]
            )

        control_action, uncertainty = control_policy.predict(observation, [self._episode, None])

        self._query_expert(control_policy, control_action, uncertainty, observation)

        self.active_policy = control_policy == self.teacher

        return control_action

    def _on_impossible_selection(self):
        return self.teacher, 1.0 # trigger the 'always attentive' algorithmic expert

