"""Verify selected artifacts and browser/native inference. Run with --native locally.

Native checks need the local binary checkpoints, Node.js, Torch, InKAN, SB3 and
Gymnasium MuJoCo. The default check needs only Python and Node.js.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def node(source, data=None):
    out = subprocess.run(['node', '--input-type=module', '-e', source], cwd=ROOT,
                         input=json.dumps(data), text=True, capture_output=True, check=True)
    return json.loads(out.stdout)


def main(native=False):
    from policy_manifolds.exporting import prepare_web
    selection = json.loads((ROOT / 'models/width2_selection.json').read_text())
    selected = selection['models']
    old = {p.name: p.read_bytes() for p in (ROOT / 'web_dist').glob('*.json')}
    ready = prepare_web(preserve_bundled=True)
    assert len(ready) == 14, ready
    assert old == {p.name: p.read_bytes() for p in (ROOT / 'web_dist').glob('*.json')}, 'Startup changed bundled exports'
    report = {'selected_models': [], 'bundled_models_preserved': len(ready)}
    for row in selected:
        folder = ROOT / 'models' / row['environment']
        actor = json.loads((folder / 'actor.json').read_text())
        assert row['demo'] == actor['demo']
        assert row['demo']['score'] == json.loads((folder / 'result.json').read_text())['validation_mean_return']
        assert (folder / 'learning_curve.csv').read_bytes() == (ROOT / 'web_dist/results' / (row['environment'].split('-v')[0].lower() + '-learning-curve.csv')).read_bytes()
        if (folder / 'model.zip').exists():
            assert hashlib.sha256((folder / 'model.zip').read_bytes()).hexdigest() == actor['checkpoint_sha256']
        report['selected_models'].append({'environment': row['environment'], 'architecture': row['architecture'],
                                         'parameters': row['actor_parameters'], 'score': row['demo']['score']})
    report['browser'] = node('''
import fs from 'node:fs';
import assert from 'node:assert/strict';
import {environments} from './web_dist/environments.mjs';
import {infer, clone, basis} from './web_dist/engine.mjs';
const results=[];
for(const [id,env] of Object.entries(environments)) {
 const model=env.decode(JSON.parse(fs.readFileSync('web_dist/'+env.file)));
 const obs=model.observationNormalization?.mean ?? model.architecture.slice(0,1).flatMap(n=>Array(n).fill(0));
 const before=infer(model,obs); assert(before.logits.every(Number.isFinite), id);
 const edited=clone(model); const layer=edited.layers[0];
 const b=basis(layer,before.activations[0][0]); const k=b.indexOf(Math.max(...b));
 layer.coefficients[0][0][k]+=1;
 const after=infer(edited,obs);
 assert(after.logits.some((x,i)=>Math.abs(x-before.logits[i])>1e-9),id+' edit must affect actor');
 results.push({id,editable:true,architecture:model.architecture,parameters:model.actorParameters});
}
console.log(JSON.stringify(results));
''')
    if native:
        import pickle
        import numpy as np
        import torch
        from inkan import KANLayer
        import gymnasium as gym
        torch.set_num_threads(1)
        cases, expected, details = [], [], []
        for row in selected:
            env_name = row['environment']; folder = ROOT / 'models' / env_name
            actor = json.loads((folder / 'actor.json').read_text())
            # These are the user's own saved VecNormalize snapshots.
            with (folder / 'normalization.pkl').open('rb') as f:
                norm = pickle.load(f)
            n = actor['observation_normalization']
            np.testing.assert_array_equal(n['mean'], norm.obs_rms.mean)
            np.testing.assert_array_equal(n['var'], norm.obs_rms.var)
            assert n['clip'] == norm.clip_obs and n['epsilon'] == norm.epsilon
            with zipfile.ZipFile(folder / 'model.zip') as z:
                state = torch.load(io.BytesIO(z.read('policy.pth')), map_location='cpu', weights_only=True)
            layers = []
            for index, raw in enumerate(actor['layers']):
                layer = KANLayer(raw['inputs'], raw['outputs'], grid_size=raw['n_bases'] - 3, compile_basis=False)
                prefix = f'actor.latent_pi.{index}.'
                layer.load_state_dict({k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)})
                np.testing.assert_array_equal(layer.spline_weight.detach().numpy(), raw['coefficients'])
                np.testing.assert_array_equal(layer.base_weight.detach().numpy(), raw['base_weights'])
                assert layer.grid_starts[0].item() == raw['grid_start'] and float(layer.inv_h) == raw['inv_h']
                layers.append(layer)
            np.testing.assert_array_equal(state['actor.mu.weight'], actor['head']['weight'])
            np.testing.assert_array_equal(state['actor.mu.bias'], actor['head']['bias'])
            np.testing.assert_array_equal(state['actor.features_extractor.scale'], actor['observation_scale'])
            def predict(obs):
                with torch.no_grad():
                    x = torch.tensor(norm.normalize_obs(np.asarray(obs)), dtype=torch.float32)
                    x = x / state['actor.features_extractor.scale']
                    for layer in layers:
                        x = layer(x.unsqueeze(0)).squeeze(0)
                    logits = torch.nn.functional.linear(x, state['actor.mu.weight'], state['actor.mu.bias'])
                    low, high = np.asarray(actor['action_low']), np.asarray(actor['action_high'])
                    return low + (torch.tanh(logits).numpy() + 1) * .5 * (high - low)
            env = gym.make(env_name)
            try:
                xml = hashlib.sha256(Path(env.unwrapped.fullpath).read_bytes()).hexdigest()
                assert xml == actor['environment_metadata']['mujoco_xml_sha256']
                obs, _ = env.reset(seed=123456)
                total, steps = 0., 0
                while True:
                    action = predict(obs)
                    if steps % 10 == 0:
                        cases.append({'id': env_name.split('-v')[0].lower(), 'observation': obs.tolist()})
                        expected.append(action)
                    obs, reward, terminated, truncated, _ = env.step(action)
                    total += reward; steps += 1
                    if terminated or truncated:
                        break
                details.append({'environment': env_name, 'normalization_matches': True, 'checkpoint_tensors_match': True,
                                'simulator_xml_matches': True, 'smoke_seed': 123456, 'smoke_return': total, 'steps': steps})
            finally:
                env.close()
        actual = node('''
import fs from 'node:fs';
import {environments} from './web_dist/environments.mjs';
import {infer} from './web_dist/engine.mjs';
const models=new Map();
const output=JSON.parse(fs.readFileSync(0,'utf8')).map(({id,observation})=>{
 if(!models.has(id)){const e=environments[id];models.set(id,e.decode(JSON.parse(fs.readFileSync('web_dist/'+e.file))));}
 return infer(models.get(id),observation).action;
});
console.log(JSON.stringify(output));
''', cases)
        max_error = 0.
        for a, b in zip(actual, expected):
            np.testing.assert_allclose(a, b, atol=5e-5, rtol=5e-5)
            max_error = max(max_error, float(np.max(np.abs(np.array(a) - b))))
        report.update(native=details, parity_observations=len(cases), max_action_error=max_error)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    report = main(args.native)
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
