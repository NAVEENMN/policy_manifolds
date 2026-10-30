// Physics lives in the local app server. Policies and editable splines stay in the browser.
export async function mujocoRequest(operation, payload) {
  return simulationRequest('invertedpendulum', operation, payload, false);
}

export async function simulationRequest(environment, operation, payload, session = true) {
  if (globalThis.location?.protocol === 'file:')
    throw new Error('Run python app.py and open its localhost URL for simulation. Offline files support curve inspection only.');
  const route = session ? `/api/simulation/${environment}/${operation}` : `/api/invertedpendulum/${operation}`;
  const response = await fetch(route, payload === undefined ? {} : {
    method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(payload),
  });
  if (!response.headers.get('content-type')?.includes('application/json'))
    throw new Error('Restart the updated app.py server to enable this environment’s simulation.');
  const data = await response.json();
  if (!response.ok) {
    if (data.error === 'MuJoCo simulation failed. Check the app server terminal.')
      throw new Error('Restart the updated app.py server to load the MuJoCo simulator fix, then click Retry connection.');
    throw new Error(data.error || `Simulation request failed (${response.status}).`);
  }
  return data;
}
