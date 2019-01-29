import numpy as np


class Summary:
    def __init__(self, label):
        self.label = label
        self._reward = 0.0
        self._penalties = 0.0
        self._queries = 0
        self._out_bounds = 0
        self._delta_v_l = 0
        self._delta_v_r = 0
        self._no_control = 0


class EpisodeSummary(Summary):
    def __init__(self, label):
        Summary.__init__(self, label)
        self.gain = 1.0
        self.trim = 0.0
        self.radius = 0.0318
        self.k = 27.0
        self.limit = 1.0
        self.wheel_dist = 0.102

    def _convert_to_vel(self, action):
        vel, angle = action

        # Distance between the wheels
        baseline = self.wheel_dist

        # assuming same motor constants k for both motors
        k_r = self.k
        k_l = self.k

        # adjusting k by gain and trim
        k_r_inv = (self.gain + self.trim) / k_r
        k_l_inv = (self.gain - self.trim) / k_l

        omega_r = (vel + 0.5 * angle * baseline) / self.radius
        omega_l = (vel - 0.5 * angle * baseline) / self.radius

        # conversion from motor rotation rate to duty cycle
        u_r = omega_r * k_r_inv
        u_l = omega_l * k_l_inv

        # limiting output to limit, which is 1.0 for the duckiebot
        u_r_limited = max(min(u_r, self.limit), -self.limit)
        u_l_limited = max(min(u_l, self.limit), -self.limit)

        return u_l_limited, u_r_limited

    def process(self, entry):
        state = entry['state']
        metadata = entry['metadata']
        # reward
        reward = state[2]
        self._reward += reward
        # penalties
        if reward < 0.0:
            self._penalties += 1

        if state[3] and reward == -1000:
            self._out_bounds += 1

        if metadata[0]:
            self._queries += 1

        if metadata[1] is not None and not metadata[5]:
            self._no_control += 1

        if metadata[1] is not None:
            # print(state[1], metadata[1])
            velocity_teacher = self._convert_to_vel(state[1])
            velocity_agent = self._convert_to_vel(metadata[1])
            self._delta_v_l += velocity_teacher[0] - velocity_agent[0]
            self._delta_v_r += velocity_teacher[1] - velocity_agent[1]
            # print(self._delta_theta)
        # print(self._delta_v)


class IterationSummary(Summary):
    def __init__(self, label):
        Summary.__init__(self, label)
        self._episodes = []

    def add_episode_summary(self, episode_summary : EpisodeSummary):
        self._episodes.append(episode_summary)

    def reward(self):
        if self._reward == 0.0:
            for episode_summary in self._episodes:
                self._reward += episode_summary._reward

        return self._reward

    def reward_history(self):
        history = []
        for episode_summary in self._episodes:
            history.append(episode_summary._reward)

        return np.array(history)

    def penalties(self):
        if self._penalties == 0.0:
            for episode_summary in self._episodes:
                self._penalties += episode_summary._penalties

        return self._penalties

    def penalties_history(self):
        history = []
        for episode_summary in self._episodes:
            history.append(episode_summary._penalties)

        return np.array(history)

    def queries(self):
        if self._queries == 0:
            for episode_summary in self._episodes:
                self._queries += episode_summary._queries
        return self._queries

    def queries_history(self):
        history = []
        for episode_summary in self._episodes:
            history.append(episode_summary._queries)

        return np.array(history)

    def out_bounds(self):
        if self._out_bounds == 0:
            for episode_summary in self._episodes:
                self._out_bounds += episode_summary._out_bounds
        return self._out_bounds

    def out_bounds_history(self):
        history = []
        for episode_summary in self._episodes:
            history.append(episode_summary._out_bounds)

        return np.array(history)

    def delta_v_l(self):
        if self._delta_v_l == 0:
            for episode_summary in self._episodes:
                self._delta_v_l += episode_summary._delta_v_l
        return self._delta_v_l

    def delta_v_l_history(self):
        history = []
        for episode_summary in self._episodes:
            history.append(episode_summary._delta_v_l)

        return np.array(history)

    def delta_v_r(self):
        if self._delta_v_r == 0:
            for episode_summary in self._episodes:
                self._delta_v_r += episode_summary._delta_v_r
        return self._delta_v_r

    def delta_v_r_history(self):
        history = []
        for episode_summary in self._episodes:
            history.append(episode_summary._delta_v_r)

        return np.array(history)

    def no_control(self):
        if self._no_control == 0:
            for episode_summary in self._episodes:
                self._no_control += episode_summary._no_control

        return self._no_control

    def no_control_history(self):
        history = []
        for episode_summary in self._episodes:
            history.append(episode_summary._no_control)

        return np.array(history)

    def episodes(self):
        return len(self._episodes)