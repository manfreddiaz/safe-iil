import math

POSITION_THRESHOLD = 0.6  # 1/4 of lane width
ANGLE_THRESHOLD = math.radians(30)  # 30 degrees in rad
REF_VELOCITY = 0.4

K_P = 10
K_D = 10


class UAPurePursuitPolicy:
    def __init__(self, env,
                 ref_velocity=REF_VELOCITY,
                 position_threshold=POSITION_THRESHOLD,
                 angle_threshold=ANGLE_THRESHOLD):
        self.env = env
        self.ref_velocity = ref_velocity
        self.position_threshold = position_threshold
        self.angle_threshold = angle_threshold

    def predict(self, observation, metadata):
        lane_pose = self.env.get_lane_pos(self.env.cur_pos, self.env.cur_angle)
        distance_to_road_center = lane_pose.dist
        angle_from_straight_in_rads = lane_pose.angle_rad

        steering = K_P * distance_to_road_center + K_D * angle_from_straight_in_rads

        action = [self.ref_velocity, steering]

        if abs(distance_to_road_center) > self.position_threshold\
                or abs(angle_from_straight_in_rads) > self.angle_threshold:
            return action, 0.0
        else:
            if metadata[0] == 0:
                return action, 0.0
            if metadata[1] is None:
                return action, 0.0

        return None, math.inf
