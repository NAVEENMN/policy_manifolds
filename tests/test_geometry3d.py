"""Run: python -m unittest discover -s tests -p test_geometry3d.py -v"""
import json
from pathlib import Path
import sys
import unittest
import numpy as np
import gymnasium as gym
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from policy_manifolds.geometry3d import scene3d
from policy_manifolds.simulation_runtime import scene_geometry
from policy_manifolds.paths import SIMULATIONS
from policy_manifolds.mujoco_runtime import InvertedPendulumRuntime


class TestGeometry3D(unittest.TestCase):
    def test_visualization_preserves_physics(self):
        for key,(name,*_) in SIMULATIONS.items():
            if key=='lunarlander':continue
            with self.subTest(environment=name):
                env=gym.make(name)
                reference=gym.make(name)
                try:
                    env.reset(seed=42);reference.reset(seed=42)
                    for step in range(8):
                        data=env.unwrapped.data
                        fields=['qpos','qvel','qacc','qacc_warmstart','act','ctrl','qfrc_applied','xfrc_applied']
                        before={field:getattr(data,field).copy() for field in fields}
                        scene=scene_geometry(env,key)['threeD']
                        self.assertEqual(scene['unsupported_types'],[])
                        json.dumps(scene,allow_nan=False)
                        for field in fields:np.testing.assert_array_equal(before[field],getattr(data,field))
                        for geom in scene['geoms']:
                            i=geom['id'];np.testing.assert_array_equal(geom['position'],data.geom_xpos[i])
                            np.testing.assert_array_equal(geom['rotation'],data.geom_xmat[i])
                            np.testing.assert_array_equal(geom['size'],env.unwrapped.model.geom_size[i])
                        action=np.full(env.action_space.shape,.03*np.sin(step))
                        rendered=env.step(action);untouched=reference.step(action)
                        np.testing.assert_array_equal(rendered[0],untouched[0])
                        self.assertEqual(rendered[1:4],untouched[1:4])
                        if rendered[2] or rendered[3]:break
                finally:env.close();reference.close()

    def test_inverted_initial_scene_does_not_step(self):
        runtime=InvertedPendulumRuntime(ROOT/'models/InvertedPendulum-v5')
        try:
            state=[0.,.01,0.,0.]
            scenes=runtime.scene({'states':[state,state]})
            self.assertEqual(len(scenes['scenes']),2)
            self.assertEqual(runtime._env.data.time,0.)
            np.testing.assert_array_equal(runtime._env.data.qpos,state[:2])
            result=runtime.step({'runs':[{'state':state,'action':0.}]})['runs'][0]
            from policy_manifolds.native_inverted import HeadlessInvertedPendulum
            reference=HeadlessInvertedPendulum()
            try:
                reference.reset(seed=42);reference.set_state(np.array(state[:2]),np.array(state[2:]))
                expected=reference.step([0.])
                np.testing.assert_array_equal(result['state'],expected[0])
                self.assertEqual(result['reward'],expected[1]);self.assertEqual(result['terminated'],expected[2])
            finally:reference.close()
        finally:runtime.close()


if __name__=='__main__':unittest.main()
