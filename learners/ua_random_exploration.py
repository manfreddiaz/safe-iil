import math
import numpy as np


class UARandomExploration:

    def __init__(self, seed, uncertainty=np.inf):
        self.uncertainty = uncertainty
        self.np_random = np.random.RandomState(seed=seed)

    def _do_update(self, dt):
        return self.predict(dt)

    def predict(self, observation, metadata):
        v = self.np_random.uniform(0, 1)
        theta = self.np_random.uniform(0, math.pi)
        return  np.array([v, theta]), self.uncertainty
