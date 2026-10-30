"""Local Policy Manifolds lab. Run: venv/bin/python app.py"""
import argparse
import shutil
import sys
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'policy-manifolds'))
from policy_manifolds.server import AppHandler
from policy_manifolds.mujoco_runtime import InvertedPendulumRuntime
from policy_manifolds.simulation_runtime import SimulationRuntime


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=7860)
    parser.add_argument('--export', action='store_true', help='Refresh browser actors from saved checkpoints')
    parser.add_argument('--export-only', action='store_true')
    parser.add_argument('--models-dir', type=Path, default=ROOT / 'models',
                        help='Model collection with environment subfolders (flat result folders also supported)')
    parser.add_argument('--acrobot-model', type=Path, help='Acrobot actor JSON or raw-observation InKAN PPO ZIP')
    parser.add_argument('--mountaincar-model', type=Path, help='MountainCar actor JSON or raw-observation InKAN PPO ZIP')
    args = parser.parse_args()
    static = ROOT / 'policy-manifolds/web/dist'
    models = ROOT / 'models'
    cartpole_actor = models / 'CartPole-v1/actor.json'
    pendulum_actor = models / 'Pendulum-v1/actor.json'
    if args.export or args.export_only or not cartpole_actor.exists():
        from export_policy import export_policy
        export_policy(output=cartpole_actor)
    if args.export or args.export_only or not pendulum_actor.exists():
        from export_pendulum import export
        export(models / 'Pendulum-v1/sac_kan_pendulum_smallest.zip', pendulum_actor)
    exports = [(cartpole_actor, static / 'model.json'), (pendulum_actor, static / 'pendulum.json')]
    inverted_actor = args.models_dir / 'InvertedPendulum-v5/actor.json'
    if inverted_actor.exists():
        exports.append((inverted_actor, static / 'invertedpendulum.json'))
    for key, environment in [('acrobot', 'Acrobot-v1'), ('mountaincar', 'MountainCar-v0')]:
        explicit = getattr(args, key + '_model')
        output = models / environment / 'actor.json'
        candidates = [args.models_dir / environment / 'actor.json',
                      args.models_dir / environment / f'ppo_kan_{key}_best.zip',
                      args.models_dir / environment / 'model.zip',
                      args.models_dir / environment / f'{environment}_best_kan.zip',
                      args.models_dir / f'{environment}_actor.json',
                      args.models_dir / f'{environment}_best_kan.zip',
                      output, models / environment / f'{environment}_best_kan.zip',
                      ROOT / 'results' / f'{environment}_best_kan.zip',
                      ROOT / 'kan_search_results' / f'{environment}_best_kan.zip']
        source = explicit or next((p for p in candidates if p.exists()), None)
        if source and source.resolve() != output.resolve() and (explicit or args.export or args.export_only or not output.exists()
                       or source.stat().st_mtime > output.stat().st_mtime):
            from export_classic import export_classic
            try:
                actor = export_classic(source, output, environment)
            except (OSError, ValueError, KeyError, IndexError, TypeError) as exc:
                parser.error(f'{environment}: {exc}')
            print(f"Exported {environment}: {' → '.join(map(str, actor['architecture']))}", flush=True)
        elif not output.exists():
            print(f'{environment}: model not found; its panel will offer Load actor JSON.', flush=True)
        if output.exists():
            exports.append((output, static / f'{key}.json'))
    # The server exposes only these lightweight browser exports, not training checkpoints.
    from policy_manifolds.exporting import prepare_web
    prepare_web(args.models_dir, static)
    for source, destination in exports:
        if not destination.exists() or source.read_bytes() != destination.read_bytes():
            shutil.copy2(source, destination)
    if args.export_only:
        return
    server = ThreadingHTTPServer(('127.0.0.1', args.port), partial(AppHandler, directory=str(static)))
    server.mujoco_runtime = InvertedPendulumRuntime(args.models_dir / 'InvertedPendulum-v5')
    server.simulation_runtime = SimulationRuntime(args.models_dir)
    print(f'Policy Manifolds: http://127.0.0.1:{args.port}', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.mujoco_runtime.close()
        server.simulation_runtime.close()
        server.server_close()


if __name__ == '__main__':
    main()
