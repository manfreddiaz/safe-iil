# experiments configurations for submissions, define behavior of the environment after each episode
from ._settings import simulation


class Icra2019Behavior:
    def __init__(self, env, at, routine):
        self.env = env
        self.at = at

        self.routine = routine
        self.routine.on_step_done(self)

    def restart(self):
        simulation(self.at, self.env, reset=False)

    def reset(self):
        simulation(self.at, self.env, reset=True)

    def step_done(self, observation, action, reward, done, info):
        pass
        # if done:
        #     self.restart()


class Icra2019TestBehavior:

    def __init__(self, env, starting_positions, routine):
        self.env = env

        self.starting = starting_positions
        self.current_position = 0

        self.routine = routine
        self.routine.on_step_done(self)

    def restart(self):
        self.current_position += 1
        self.current_position %= len(self.starting)
        simulation(self.starting[self.current_position], self.env, reset=False)

    def reset(self):
        simulation(self.starting[self.current_position], self.env, reset=True)

    def step_done(self, observation, action, reward, done, info):
        if done: # if goes out of boundaries...
            self.restart()
