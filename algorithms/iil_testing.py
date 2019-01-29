from tqdm import tqdm

class InteractiveImitationTesting:
    def __init__(self, env, teacher, learner, horizon, episodes):

        self.environment = env
        self.learner = learner
        self.teacher = teacher

        # from IIL
        self._horizon = horizon
        self._episodes = episodes

        # statistics
        self.teacher_queried = True
        self.teacher_action = None
        self.active_policy = True  # if teacher is active
        self.learner_uncertainty = None
        self.teacher_uncertainty = None
        self.learner_action = None

        # internal count
        self._current_horizon = 0
        self._current_episode = 0

        # event listeners
        self._step_done_listeners = []
        self._episode_done_listeners = []
        self._process_done_listeners = []

    def test(self, debug=False):
        self._debug = debug
        observation = self.environment.render_obs()
        for episode in tqdm(range(self._episodes)):
            self._current_episode = episode
            for t in range(self._horizon):
                self._current_horizon = t
                action = self._act(observation)
                next_observation, reward, done, info = self.environment.step(action)
                if self._debug:
                    self.environment.render()
                self._on_step_done(observation, action, reward, done, info)
                observation = next_observation
            self._on_episode_done()
        self._on_process_done()

    # execute learner control policy
    def _act(self, observation):
        control_action = self.learner.predict(observation, [self._current_episode, None])

        if isinstance(control_action, tuple):
            control_action, self.active_uncertainty = control_action  # if we have uncertainty as input, we do not record it

        self._query_expert(self.learner, control_action, observation)

        return control_action

    def _query_expert(self, control_policy, control_action, observation):

        expert_action = self.teacher.predict(observation, [self._current_episode, control_action])

        if isinstance(expert_action, tuple):
            self.teacher_action, self.teacher_uncertainty = expert_action  # if we have uncertainty as input, we do not record it

        if expert_action is not None:
            self.teacher_queried = True
        else:
            self.teacher_queried = False

    # TRAINING EVENTS

    # triggered after an episode of learning is done
    def on_episode_done(self, listener):
        self._episode_done_listeners.append(listener)

    def _on_episode_done(self):
        for listener in self._episode_done_listeners:
            listener.episode_done(self._current_episode)

    def on_process_done(self, listener):
        self._process_done_listeners.append(listener)

    # triggered when the whole training is done
    def _on_process_done(self):
        for listener in self._process_done_listeners:
            listener.process_done()

    def on_step_done(self, listener):
        self._step_done_listeners.append(listener)

    def _on_step_done(self, observation, action, reward, done, info):
        for listener in self._step_done_listeners:
            listener.step_done(observation, action, reward, done, info)

