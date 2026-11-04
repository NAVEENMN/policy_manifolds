"""Lazy Gymnasium sessions for editable MuJoCo and Box2D policies.

Keep live and original simulators independent, preserving full physics state
(including contacts and RNG) between requests. Browser inference supplies each
next action; comparisons use the same exported actor and frozen preprocessing.
"""
import hashlib
import importlib
import json
import math
from pathlib import Path
import secrets
import shlex
import sys
import threading
import time

from .paths import SIMULATIONS
from .mujoco_runtime import RuntimeUnavailable
from .geometry3d import scene3d


def browser_actor(data, environment):
    from .evaluation import DiscreteActor
    if data.get('environment') != environment:
        raise ValueError('Actor belongs to a different environment.')
    continuous = data.get('actionType') == 'continuous'
    raw = {'format': 'inkan-research-actor-v1', 'environment': environment,
           'action_mapping': 'tanh_then_box_rescale' if continuous else 'argmax',
           'action_low': data.get('actionLow'), 'action_high': data.get('actionHigh'),
           'observation_scale': data.get('observationScale'),
           'observation_normalization': data.get('observationNormalization'),
           'head': data['head'],
           'layers': [{'coefficients': l['coefficients'], 'base_weights': l['baseWeights'],
                       'grid_start': l['gridStart'], 'inv_h': l['invH']} for l in data['layers']]}
    return DiscreteActor(raw, environment)


def scene_geometry(env, key):
    """Project real simulator geometry in the browser; no OpenGL/video renderer."""
    base = env.unwrapped
    if key == 'lunarlander':
        shapes = []
        for body in base.world.bodies:
            for fixture in body.fixtures:
                if not hasattr(fixture.shape, 'vertices'):
                    continue
                vertices = [list(body.transform * vertex) for vertex in fixture.shape.vertices]
                if len(vertices) >= 2:
                    shapes.append({'kind': 'polygon', 'points': vertices, 'static': body.type == 0})
        return {'kind': 'box2d', 'shapes': shapes, 'center': [10., 6.5], 'span': 21.}
    model, data = base.model, base.data
    import numpy as np
    axes = [0, 1] if key in ('ant', 'swimmer', 'reacher') else [0, 2]
    shapes = []
    for i in range(model.ngeom):
        kind = int(model.geom_type[i])
        if kind == 0:  # The floor is drawn as a grid, not an infinite polygon.
            continue
        position = data.geom_xpos[i]
        size = model.geom_size[i]
        rotation = data.geom_xmat[i].reshape(3, 3)
        static = int(model.geom_bodyid[i]) == 0
        if kind in (3, 5):
            direction = rotation[:, 2] * size[1]
            shapes.append({'kind': 'capsule', 'a': (position-direction)[axes].tolist(),
                           'b': (position+direction)[axes].tolist(), 'radius': float(size[0]), 'static': static})
        elif kind == 6:
            corners = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]) * size
            projected = (corners @ rotation.T + position)[:, axes]
            shapes.append({'kind': 'box', 'center': position[axes].tolist(),
                           'radius': float(np.max(np.linalg.norm(projected-position[axes], axis=1))), 'static': static})
        else:
            shapes.append({'kind': 'circle', 'center': position[axes].tolist(),
                           'radius': float(max(size[0], .015)), 'static': static})
    center = data.xpos[1, axes].copy()
    span = 6.
    if key == 'reacher': center = np.array([0., 0.]); span = .65
    elif key == 'swimmer': span = 7.
    elif key == 'inverteddoublependulum': center = np.array([data.qpos[0], .65]); span = 3.
    elif axes == [0, 2]: center[1] = .9
    return {'kind': 'mujoco', 'view': 'top' if axes == [0, 1] else 'side',
            'threeD': scene3d(model, data, key),
            'shapes': shapes, 'center': center.tolist(), 'span': span}


class SimulationRuntime:
    def __init__(self, models):
        self.models = Path(models)
        self.lock = threading.RLock()
        self.sessions = {}

    def _definition(self, key):
        if key not in SIMULATIONS:
            raise ValueError('Unknown simulation environment.')
        return SIMULATIONS[key]

    def _make(self, key):
        environment, inputs, actions, algorithm, limit, target = self._definition(key)
        try:
            importlib.invalidate_caches()
            import gymnasium as gym
            env = gym.make(environment)
        except Exception as error:
            if not isinstance(error, ImportError) and type(error).__name__ != 'DependencyNotInstalled':
                raise
            packages = ['swig', 'gymnasium[box2d]==1.1.1'] if key == 'lunarlander' else ['gymnasium[mujoco]==1.1.1', 'mujoco==3.2.7', 'imageio']
            command = shlex.join([sys.executable, '-m', 'pip', 'install', *packages])
            if key == 'lunarlander':
                command = shlex.join([sys.executable, '-m', 'pip', 'install', 'swig==4.*']) + '\n' + shlex.join([sys.executable, '-m', 'pip', 'install', 'gymnasium[box2d]==1.1.1'])
            raise RuntimeUnavailable(f'Install the simulation dependencies in the Python running this app:\n{command}\nThen click Retry connection.') from error
        try:
            if env.observation_space.shape != (inputs,):
                raise RuntimeUnavailable(f'{environment} observation shape differs from the saved model ({inputs} inputs).')
            path = self.models / environment / 'environment.json'
            if path.exists() and key != 'lunarlander':
                expected = json.loads(path.read_text()).get('mujoco_xml_sha256')
                actual = hashlib.sha256(Path(env.unwrapped.fullpath).read_bytes()).hexdigest()
                if expected and expected != actual:
                    raise RuntimeUnavailable('Simulator XML differs from training. Install the pinned requirements-mujoco.txt.')
            return env
        except BaseException:
            env.close()
            raise

    def status(self, key):
        with self.lock:
            env = self._make(key)
            try:
                return {'available': True, 'environment': SIMULATIONS[key][0],
                        'physics': 'Box2D' if key == 'lunarlander' else 'MuJoCo'}
            finally:
                env.close()

    def _drop(self, token):
        item = self.sessions.pop(token, None)
        if item:
            for env in item['envs']: env.close()

    def _cleanup(self):
        now = time.monotonic()
        for token, item in list(self.sessions.items()):
            if now - item['used'] > 900: self._drop(token)
        while len(self.sessions) >= 8:
            self._drop(min(self.sessions, key=lambda token: self.sessions[token]['used']))

    @staticmethod
    def _seed(value):
        if type(value) != int or not 0 <= value <= 4294967295:
            raise ValueError('Expected an unsigned 32-bit reset seed.')
        return value

    def _packet(self, env, key, observation, reward=0., terminated=False, truncated=False):
        return {'state': observation.tolist(), 'reward': float(reward), 'terminated': bool(terminated),
                'truncated': bool(truncated), 'scene': scene_geometry(env, key)}

    def reset(self, key, payload):
        self._definition(key)
        seed = self._seed(payload.get('seed'))
        with self.lock:
            self._cleanup()
            envs = []
            try:
                envs.append(self._make(key)); envs.append(self._make(key))
                packets = [self._packet(env, key, env.reset(seed=seed)[0]) for env in envs]
            except BaseException:
                for env in envs: env.close()
                raise
            # A new token makes late replies distinguishable after resets/switches.
            self._drop(payload.get('session'))
            token = secrets.token_urlsafe(24)
            self.sessions[token] = {'key': key, 'envs': envs, 'done': [False, False],
                                    'sequence': 0, 'used': time.monotonic()}
            return {'session': token, 'sequence': 0, 'runs': packets}

    def step(self, key, payload):
        self._definition(key)
        with self.lock:
            session = self.sessions.get(payload.get('session'))
            if not session or session['key'] != key:
                raise ValueError('Simulation session expired. Restart the episode.')
            if type(payload.get('sequence')) != int or payload['sequence'] != session['sequence']:
                raise ValueError('Stale simulation step. Restart the episode.')
            actions = payload.get('actions')
            if not isinstance(actions, list) or len(actions) != 2:
                raise ValueError('Expected live and original actions.')
            import numpy as np
            checked = []
            # Validate the whole request before advancing either simulator.
            for action, env, done in zip(actions, session['envs'], session['done']):
                if action is None:
                    if not done: raise ValueError('An active run requires an action.')
                    checked.append(None)
                elif done:
                    raise ValueError('An ended run cannot be stepped.')
                elif key == 'lunarlander':
                    if type(action) != int or not 0 <= action < 4: raise ValueError('Expected a LunarLander action from 0 to 3.')
                    checked.append(action)
                else:
                    if not isinstance(action, list) or len(action) != SIMULATIONS[key][2] or any(type(x) not in (int, float) or not math.isfinite(x) for x in action):
                        raise ValueError('Control vector does not match the saved action space.')
                    checked.append(np.clip(np.asarray(action, dtype=np.float32), env.action_space.low, env.action_space.high))
            packets = []
            for i, (action, env) in enumerate(zip(checked, session['envs'])):
                if action is None: packets.append(None); continue
                obs, reward, terminated, truncated, _ = env.step(action)
                session['done'][i] = bool(terminated or truncated)
                packets.append(self._packet(env, key, obs, reward, terminated, truncated))
            session['sequence'] += 1; session['used'] = time.monotonic()
            return {'session': payload['session'], 'sequence': session['sequence'], 'runs': packets}

    def release(self, key, payload):
        with self.lock:
            session = self.sessions.get(payload.get('session'))
            if session and session['key'] == key: self._drop(payload['session'])
        return {'released': True}

    def compare(self, key, payload):
        environment, _, _, _, limit, target = self._definition(key)
        seeds = payload.get('seeds')
        if not isinstance(seeds, list) or not 1 <= len(seeds) <= 20:
            raise ValueError('Compare one to twenty paired reset seeds.')
        for seed in seeds: self._seed(seed)
        actors = [browser_actor(payload[name], environment) for name in ('edited', 'original')]
        results = [[], []]
        # Separate environments leave live-session physics and RNG untouched.
        env = self._make(key)
        try:
            for seed in seeds:
                for actor, rows in zip(actors, results):
                    observation, _ = env.reset(seed=seed); total = 0.
                    for step in range(1, limit+1):
                        action, _ = actor.predict(observation)
                        observation, reward, terminated, truncated, _ = env.step(action)
                        total += float(reward)
                        if terminated or truncated: break
                    rows.append({'reward': total, 'steps': step, 'success': total >= target})
        finally:
            env.close()
        return {'edited': results[0], 'original': results[1]}

    def close(self):
        with self.lock:
            for token in list(self.sessions): self._drop(token)
