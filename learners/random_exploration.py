import math
import numpy as np


class RandomExploration:

    def __init__(self, env, fake_uncertainty=1):
        self.uncertainty = fake_uncertainty

    def _do_update(self, dt):
        return self.predict(dt)

    def predict(self, observation):
        return np.random.uniform(0, 1, 2)

    def learn(self, observations, actions):
        pass

    def save(self):
        print('I didn\'t learn a thing...')
