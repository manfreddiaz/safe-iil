from math import cos, sin

import numpy as np
import tensorflow as tf
from learning_iil.learners.parametrizations.tf.tf_online_learner import TensorflowOnlineLearner

tf.set_random_seed(1234)


class GoalOrientedTensorflowOnlineLearner(TensorflowOnlineLearner):
    def __init__(self, checkpoint_file=None):
        TensorflowOnlineLearner.__init__(self, checkpoint_file)
        self.goal_tensor = None

    def predict(self, state, goal, horizon=1):
        action = self.tf_session.run([self.policy_model], feed_dict={
            self.state_tensor: [state],
            self.goal_tensor: [goal]
        })
        action = np.squeeze(action)
        return action

    def execute(self, action):
        self.theta = action
        self.x = self.x + self.v * cos(self.theta)
        self.y = self.y + self.v * sin(self.theta)

    def explore(self, state, horizon=1):
        pass

    def learn(self, state, action, goal):
        summary, step, _, learning_loss = self.tf_session.run(
            [self.summary_merge, self.global_step, self.optimization_algorithm, self.loss_function],
            feed_dict={
                self.state_tensor: [state],
                self.action_tensor: action,
                self.goal_tensor: [goal]
            }
        )
        self.summary_writer.add_episode_summary(summary, step)
        self.last_loss = learning_loss
        return learning_loss

    def architecture(self):
        raise NotImplementedError()

    def get_optimizer(self, loss):
        raise NotImplementedError()

    def _state_action_tensors(self, input_shape=(None, 1), output_shape=(1,), goal_shape=(1, 2)):
        if len(input_shape) == 3:
            input_shape = (1, input_shape[0], input_shape[1], input_shape[2])
        if len(output_shape) == 0:
            output_shape = (1,)
        with tf.name_scope('input'):
            self.state_tensor = tf.placeholder(dtype=tf.float32, shape=input_shape, name='state')
            tf.summary.image('state', self.state_tensor, 1)
            self.action_tensor = tf.placeholder(dtype=tf.float32, shape=output_shape, name='action')
            tf.summary.scalar('action', tf.reshape(self.action_tensor, shape=[]))
            self.goal_tensor = tf.placeholder(dtype=tf.float32, shape=(1,), name='goal')
            tf.summary.scalar('goal', tf.reshape(self.goal_tensor, shape=[]))
