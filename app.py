"""Policy Manifolds: run the bundled policy editor with optional simulator APIs."""
import argparse
from functools import partial
from http.server import ThreadingHTTPServer
from pathlib import Path

from policy_manifolds.paths import MODELS, WEB
from policy_manifolds.exporting import prepare_web
from policy_manifolds.server import AppHandler
from policy_manifolds.mujoco_runtime import InvertedPendulumRuntime
from policy_manifolds.simulation_runtime import SimulationRuntime


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=7860)
    parser.add_argument('--models-dir', type=Path, default=MODELS)
    parser.add_argument('--export', action='store_true', help='Refresh actors from local checkpoints (requires original training workspace and dependencies)')
    parser.add_argument('--export-only', action='store_true', help='Prepare browser exports and exit')
    args = parser.parse_args()
    # Git bundles actor JSONs; large training checkpoints are optional.
    prepare_web(args.models_dir, WEB, refresh=args.export,
                preserve_bundled=args.models_dir.resolve() == MODELS.resolve())
    if args.export_only:
        return
    server = ThreadingHTTPServer(('127.0.0.1', args.port), partial(AppHandler, directory=str(WEB)))
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
