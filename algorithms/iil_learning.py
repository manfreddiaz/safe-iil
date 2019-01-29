

class InteractiveImitationLearning:
    def __init__(self, env, teacher, learner, horizon, episodes):

        self.env = env
        self.teacher = teacher
        self.learner = learner

        # from IIL
        self._horizon = horizon
        self._episodes = episodes

        # data
        self._observations = []
        self._expert_actions = []

        # statistics
        self.learner_action = None
        self.learner_uncertainty = None

        self.teacher_queried = True
        self.teacher_uncertainty = None
        self.teacher_action = None

        # internal count
        self._current_horizon = 0
        self._episode = 0

        # event listeners
        self._step_done_listeners = []
        self._optimization_done_listener = []
        self._episode_done_listeners = []
        self._process_done_listeners = []

    def train(self, samples=1, debug=False):
        self._debug = bool(debug)
        for episode in range(self._episodes):
            self._episode = episode
            self._sampling(samples)
            self._optimize() # episodic learning
            self._on_episode_done()
        self._on_process_done()

    def _sampling(self, samples):
        for sample in range(samples):  # number of T-step trajectories
            observation = self.env.reset()
            for horizon in range(self._horizon):
                self._current_horizon = horizon
                action = self._act(observation)
                next_observation, reward, done, info = self.env.step(action)
                if self._debug:
                    self.env.render()
                self._on_step_done(observation, action, reward, done, info)
                observation = next_observation
        self._on_sampling_done()

    # execute current control policy
    def _act(self, observation):
        if self._episode == 0:  # initial policy equals expert's
            control_policy = self.teacher
        else:
            control_policy = self._mix()

        control_action, uncertainty = control_policy.predict(observation, [self._episode, None])

        self._query_expert(control_policy, control_action, uncertainty, observation)

        return control_action

    def _query_expert(self, control_policy, control_action, uncertainty, observation):
        if control_policy == self.learner:
            self.learner_action = control_action
            self.learner_uncertainty = uncertainty  # it might but it wont
        else:
            self.learner_action, self.learner_uncertainty = self.learner.predict(observation, [self._episode, None])

        if control_policy == self.teacher:
            self.teacher_action = control_action
            self.teacher_uncertainty = uncertainty
        else:
            self.teacher_action, self.teacher_uncertainty = self.teacher.predict(observation, [self._episode, control_action])

        if self.teacher_action is not None:
            self._aggregate(observation, self.teacher_action)
            self.teacher_queried = True
        else:
            self.teacher_queried = False

    def _mix(self):
        raise NotImplementedError()

    def _aggregate(self, observation, action):
        self._observations.append(observation)
        self._expert_actions.append(action)

    def _optimize(self):
        loss = self.learner.optimize(self._observations, self._expert_actions, self._episode)
        self.learner.save()
        self._on_optimization_done(loss)

    # TRAINING EVENTS

    # triggered after an episode of learning is done
    def on_episode_done(self, listener):
        self._episode_done_listeners.append(listener)

    def _on_episode_done(self):
        for listener in self._episode_done_listeners:
            listener.episode_done(self._episode)

    # triggered when the learning step after an episode is done
    def on_optimization_done(self, listener):
        self._optimization_done_listener.append(listener)

    def _on_optimization_done(self, loss):
        for listener in self._optimization_done_listener:
            listener.optimization_done(loss)

    def on_process_done(self, listener):
        self._process_done_listeners.append(listener)

    # triggered when the whole training is done
    def _on_process_done(self):
        for listener in self._process_done_listeners:
            listener.process_done()

    # triggered when one step of sampling is done
    def _on_sampling_done(self):
        pass

    def on_step_done(self, listener):
        self._step_done_listeners.append(listener)

    def _on_step_done(self, observation, action, reward, done, info):
        for listener in self._step_done_listeners:
            listener.step_done(observation, action, reward, done, info)

