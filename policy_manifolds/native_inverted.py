"""InvertedPendulum-v5 physics without Gymnasium's optional rendering imports.

Uses the installed Gymnasium XML and real MuJoCo integration. Reset noise,
frame skip, observations, reward and termination follow inverted_pendulum_v5.py.
No OpenGL context or video encoder is needed for the browser's canvas renderer.
"""
from pathlib import Path


class HeadlessInvertedPendulum:
    def __init__(self):
        import gymnasium
        import mujoco
        import numpy as np
        self._mujoco, self._np = mujoco, np
        self.fullpath = str(Path(gymnasium.__file__).parent/'envs/mujoco/assets/inverted_pendulum.xml')
        self.model = mujoco.MjModel.from_xml_path(self.fullpath)
        if (self.model.nq, self.model.nv, self.model.nu) != (2, 2, 1):
            raise ValueError('Expected the two-joint, one-actuator InvertedPendulum XML.')
        self.data = mujoco.MjData(self.model)
        self.frame_skip = 2
        self.dt = self.frame_skip * self.model.opt.timestep
        self.init_qpos, self.init_qvel = self.data.qpos.copy(), self.data.qvel.copy()
        self.np_random = None
        self.steps = 0

    @property
    def unwrapped(self):
        return self

    def _observation(self):
        return self._np.concatenate([self.data.qpos, self.data.qvel]).ravel()

    def set_state(self, qpos, qvel):
        self.data.qpos[:] = qpos
        self.data.qvel[:] = qvel
        self._mujoco.mj_forward(self.model, self.data)

    def reset(self, *, seed=None):
        from gymnasium.utils import seeding
        if seed is not None or self.np_random is None:
            self.np_random, _ = seeding.np_random(seed)
        self._mujoco.mj_resetData(self.model, self.data)
        self.set_state(self.init_qpos + self.np_random.uniform(-.01, .01, 2),
                       self.init_qvel + self.np_random.uniform(-.01, .01, 2))
        self.steps = 0
        return self._observation(), {}

    def step(self, action):
        control = self._np.asarray(action, dtype=float)
        if control.shape != (1,) or not self._np.isfinite(control).all():
            raise ValueError('Expected one finite cart control.')
        self.data.ctrl[:] = self._np.clip(control, -3., 3.)
        self._mujoco.mj_step(self.model, self.data, nstep=self.frame_skip)
        self._mujoco.mj_rnePostConstraint(self.model, self.data)
        observation = self._observation()
        terminated = bool(not self._np.isfinite(observation).all() or abs(observation[1]) > .2)
        reward = int(not terminated)
        self.steps += 1
        return observation, reward, terminated, self.steps >= 1000, {'reward_survive': reward}

    def close(self):
        self.data = None
        self.model = None
