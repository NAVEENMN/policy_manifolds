// Load only requested actors. Cached policies never run in the background.
export function createPolicyLoader(definitions, read) {
  const policies = new Map(), pending = new Map(), errors = new Map();
  function load(id) {
    if (!Object.prototype.hasOwnProperty.call(definitions, id)) return Promise.resolve(null);
    if (policies.has(id)) return Promise.resolve(policies.get(id));
    if (pending.has(id)) return pending.get(id);
    errors.delete(id);
    const request = Promise.resolve().then(() => read(definitions[id])).then(raw => {
      // A manually imported actor takes precedence over a request still in flight.
      if (policies.has(id)) return policies.get(id);
      if (!raw) return null;
      const policy = definitions[id].decode(raw);
      policies.set(id, policy);
      return policy;
    }).catch(error => {
      if (!policies.has(id)) errors.set(id, error.message);
      return policies.get(id) ?? null;
    }).finally(() => pending.delete(id));
    pending.set(id, request);
    return request;
  }
  return {policies, pending, errors, load};
}
