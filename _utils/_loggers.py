from concurrent.futures import ThreadPoolExecutor

import os
import pickle

# Stats for RSS 2019 submission


class Logger:
    def __init__(self, routine, log_file):
        self.routine = routine

        self._log_file = open(log_file, 'wb')
        self._multithreaded_recording = ThreadPoolExecutor(4)
        self._recording = []
        self._configure()

    def _configure(self):
        self.routine.on_step_done(self)
        self.routine.on_episode_done(self)
        self.routine.on_process_done(self)

    def step_done(self, observation, action, reward, done, info):
        self._recording.append({
            'state': [
                (self.routine.env.cur_pos, self.routine.env.cur_angle),
                action,
                reward,
                done,
                info
            ],
            'metadata': [
                self.routine.teacher_queried,  # 0
                self.routine.teacher_action,  # 1
                self.routine.teacher_uncertainty,  # 2
                self.routine.learner_action, # 3
                self.routine.learner_uncertainty  # 4
            ]
        })

    def episode_done(self, episode):
        self._multithreaded_recording.submit(self._commit)

    def _commit(self):
        pickle.dump(self._recording, self._log_file)
        self._log_file.flush()
        self._recording.clear()

    def process_done(self):
        self._log_file.close()
        os.chmod(self._log_file.name, 0o444)  # make file read-only after finishing
        self._multithreaded_recording.shutdown()
