import math
from controllers import JoystickController


class UncertaintyAwareHumanController(JoystickController):

    def _do_update(self, dt):
        action = JoystickController._do_update(self, dt)

        if action is None:
            return None, math.inf

        return action, 0.0
