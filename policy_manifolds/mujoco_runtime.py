"""Lazy, headless MuJoCo physics for the browser's InvertedPendulum adapter."""
import hashlib
import importlib
import json
import math
from pathlib import Path
import shlex
import sys
import threading
from .geometry3d import scene3d


class RuntimeUnavailable(RuntimeError):
    pass


def state_vector(values):
    if not isinstance(values, list) or len(values) != 4 or any(type(x) not in (float, int) or not math.isfinite(x) or abs(x) > 1e6 for x in values):
        raise ValueError('Expected four finite state values.')
    return values


def browser_actor(data):
    from .evaluation import DiscreteActor
    if data.get('environment') != 'InvertedPendulum-v5' or data.get('actionType') != 'continuous' or data.get('torqueLimit') != 3:
        raise ValueError('Expected an InvertedPendulum actor with [-3, 3] controls.')
    raw = {'format': 'inkan-research-actor-v1', 'environment': 'InvertedPendulum-v5',
           'action_mapping': 'tanh_then_box_rescale', 'action_low': [-3.], 'action_high': [3.],
           'observation_scale': data.get('observationScale'),
           'observation_normalization': data.get('observationNormalization'), 'head': data['head'],
           'layers': [{'coefficients': layer['coefficients'], 'base_weights': layer['baseWeights'],
                       'grid_start': layer['gridStart'], 'inv_h': layer['invH']} for layer in data['layers']]}
    return DiscreteActor(raw, 'InvertedPendulum-v5')


class InvertedPendulumRuntime:
    """Stateless physics calls; separate clients cannot modify each other's state."""
    def __init__(self, model_folder):
        self.folder = Path(model_folder)
        self._env = None
        self.lock = threading.Lock()

    def _ready(self):
        if self._env is not None:
            return
        try:
            # A user may install the optional package while this server is running.
            importlib.invalidate_caches()
            import mujoco
            from .native_inverted import HeadlessInvertedPendulum
            env = HeadlessInvertedPendulum()
        except ImportError as error:
            command = shlex.join([sys.executable, '-m', 'pip', 'install', 'gymnasium[mujoco]==1.1.1', 'mujoco==3.2.7'])
            raise RuntimeUnavailable(f'MuJoCo dependencies are missing from the Python running this app. Run this command in your terminal:\n{command}\nThen click Retry connection.') from error
        try:
            path = self.folder/'environment.json'
            expected = json.loads(path.read_text()).get('mujoco_xml_sha256') if path.exists() else None
            actual = hashlib.sha256(Path(env.unwrapped.fullpath).read_bytes()).hexdigest()
            if expected and expected != actual:
                raise RuntimeUnavailable('The installed MuJoCo environment XML differs from the trained model. Install the pinned requirements-mujoco.txt.')
            self._mujoco = mujoco
            self._env = env
        except BaseException:
            env.close()
            raise

    def status(self):
        with self.lock:
            self._ready()
            return {'available': True, 'environment': 'InvertedPendulum-v5',
                    'physics': 'MuJoCo', 'version': self._mujoco.__version__, 'dt': self._env.unwrapped.dt}

    def _set_state(self, state):
        import numpy as np
        env = self._env.unwrapped
        self._mujoco.mj_resetData(env.model, env.data)
        env.steps = 0
        env.set_state(np.asarray(state[:2]), np.asarray(state[2:]))

    def step(self, payload):
        items = payload.get('runs')
        if not isinstance(items, list) or not 1 <= len(items) <= 2:
            raise ValueError('Expected one or two live/ghost runs.')
        for item in items:
            if item is None:
                continue
            state_vector(item['state'])
            if type(item['action']) not in (float, int) or not math.isfinite(item['action']):
                raise ValueError('Expected a finite control signal.')
        with self.lock:
            self._ready()
            output = []
            for item in items:
                if item is None:
                    output.append(None); continue
                self._set_state(item['state'])
                state, reward, terminated, _, _ = self._env.unwrapped.step([max(-3., min(3., item['action']))])
                output.append({'state': state.tolist(), 'reward': reward, 'terminated': bool(terminated),
                               'scene': {'threeD': scene3d(self._env.model, self._env.data, 'invertedpendulum')}})
            return {'runs': output}

    def scene(self, payload):
        """Describe initial/restarted poses without taking a simulation step."""
        starts = payload.get('states')
        if not isinstance(starts, list) or not 1 <= len(starts) <= 2:
            raise ValueError('Expected one or two initial states.')
        for state in starts:
            state_vector(state)
        with self.lock:
            self._ready()
            scenes = []
            for state in starts:
                self._set_state(state)
                scenes.append({'threeD': scene3d(self._env.model, self._env.data, 'invertedpendulum')})
            return {'scenes': scenes}

    def compare(self, payload):
        starts = payload.get('starts')
        if not isinstance(starts, list) or not 1 <= len(starts) <= 20:
            raise ValueError('Compare one to twenty paired initial states.')
        for start in starts:
            state_vector(start)
        actors = [browser_actor(payload[name]) for name in ['edited', 'original']]
        results = [[], []]
        with self.lock:
            self._ready()
            for start in starts:
                for actor, rows in zip(actors, results):
                    self._set_state(start)
                    state = start
                    total = 0.
                    for step in range(1, 1001):
                        action = actor.predict(state)[0]
                        state, reward, terminated, _, _ = self._env.unwrapped.step(action)
                        total += reward
                        if terminated:
                            break
                    rows.append({'reward': total, 'steps': step, 'success': not terminated})
        return {'edited': results[0], 'original': results[1]}

    def close(self):
        with self.lock:
            if self._env is not None:
                self._env.close(); self._env = None
