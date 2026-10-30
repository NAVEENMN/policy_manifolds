"""Static app server with optional local MuJoCo endpoints."""
import json
from http.server import SimpleHTTPRequestHandler
from urllib.parse import urlsplit

from .mujoco_runtime import RuntimeUnavailable


class AppHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        super().end_headers()

    def json_response(self, code, data):
        body = json.dumps(data, allow_nan=False).encode()
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def api(self, operation, payload=None):
        try:
            result = operation() if payload is None else operation(payload)
            self.json_response(200, result)
        except RuntimeUnavailable as error:
            self.json_response(503, {'available': False, 'error': str(error)})
        except (ValueError, KeyError, TypeError, IndexError, OverflowError) as error:
            self.json_response(400, {'error': str(error)})
        except Exception as error:
            self.log_error('MuJoCo request failed: %s', error)
            self.json_response(500, {'error': f'MuJoCo simulation failed: {type(error).__name__}: {str(error)[:500]}'})

    def do_GET(self):
        parts = urlsplit(self.path).path.strip('/').split('/')
        if len(parts) == 4 and parts[:2] == ['api', 'simulation'] and parts[3] == 'status':
            self.api(lambda: self.server.simulation_runtime.status(parts[2]))
        elif urlsplit(self.path).path == '/api/invertedpendulum/status':
            self.api(self.server.mujoco_runtime.status)
        else:
            super().do_GET()

    def do_POST(self):
        operation = {'/api/invertedpendulum/step': self.server.mujoco_runtime.step,
                     '/api/invertedpendulum/compare': self.server.mujoco_runtime.compare}.get(urlsplit(self.path).path)
        parts = urlsplit(self.path).path.strip('/').split('/')
        if len(parts) == 4 and parts[:2] == ['api', 'simulation'] and parts[3] in ('reset', 'step', 'compare', 'release'):
            operation = lambda data: getattr(self.server.simulation_runtime, parts[3])(parts[2], data)
        if operation is None:
            self.json_response(404, {'error': 'Unknown endpoint.'}); return
        origin = self.headers.get('Origin')
        if origin and urlsplit(origin).netloc != self.headers.get('Host'):
            self.json_response(403, {'error': 'Use the app from this local server.'}); return
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if not 0 < length <= 1048576:
                raise ValueError('Request must contain at most 1 MiB of JSON.')
            payload = json.loads(self.rfile.read(length))
            if not isinstance(payload, dict):
                raise ValueError('Expected a JSON object.')
        except (ValueError, UnicodeError) as error:
            self.json_response(400, {'error': str(error)}); return
        self.api(operation, payload)
