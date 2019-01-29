import math
import numpy as np
from threading import Event
import rospy
import message_filters
import cv_bridge
import cv2
from duckietown_msgs.msg import Twist2DStamped, BoolStamped
from sensor_msgs.msg import Joy

CAR_CMD_TOPIC = '/hero/joy_mapper_node/car_cmd'
JOY_TOPIC = '/hero/joy'
# INTERVENTION_TOPIC = '/hero/anti_instagram_node/click'

RESPONSE_TIMEOUT = 0.01  # 1 second


class UADuckietownPipelinePolicy:
    def __init__(self, env):
        rospy.init_node('duckietown_pipeline_teacher', anonymous=False)
        self.env = env
        self.pipeline_car_cmd_subscriber = rospy.Subscriber(CAR_CMD_TOPIC, Twist2DStamped, self._action_received)
        self.pipeline_joy_subscriber = rospy.Subscriber(JOY_TOPIC, Joy, self._uncertainty_received)

        self._last_cmd = [0.0, 0.0]
        self._uncertainty = np.inf

    def predict(self, observation, metadata):
        if metadata[0] == 0 or metadata[1] is None: # first iteration or asking for input
            while self._uncertainty == np.inf or round(self._last_cmd[0], 2) == 0.0:
                print '\r give me input!',
        if self._uncertainty == np.inf or round(self._last_cmd[0], 2) == 0.0:
            return None, np.inf

        return self._last_cmd, self._uncertainty

    def _action_received(self, car_cmd):
        self._last_cmd[0] = car_cmd.v
        self._last_cmd[1] = car_cmd.omega
        print(self._last_cmd)

    def _uncertainty_received(self, joy):
        if joy.buttons[5] == 1:
            self._uncertainty = 0.0
        else:
            self._uncertainty = np.inf
        # print(self._uncertainty)