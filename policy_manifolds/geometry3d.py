"""Read-only MuJoCo geometry for the browser viewer. Never advances physics."""

KINDS = {0: 'plane', 2: 'sphere', 3: 'capsule', 4: 'ellipsoid', 5: 'cylinder', 6: 'box'}
CAMERAS = {
    'humanoid': (1.05, 4.4, .38), 'halfcheetah': (.55, 3.8, .35),
    'hopper': (.85, 3.5, .32), 'walker2d': (.85, 3.9, .32),
    'ant': (.35, 4.6, .65), 'swimmer': (0., 5.6, .95),
    'reacher': (0., .8, 1.05), 'invertedpendulum': (.6, 3.4, .3),
    'inverteddoublependulum': (.9, 4.2, .3),
}


def scene3d(model, data, key):
    geoms, unsupported = [], []
    for i in range(model.ngeom):
        # Match MuJoCo's default visible geometry groups.
        if model.geom_group[i] > 2 or model.geom_rgba[i, 3] <= 0:
            continue
        kind = KINDS.get(int(model.geom_type[i]))
        if kind is None:
            unsupported.append(int(model.geom_type[i]))
            continue
        geoms.append({'id': i, 'kind': kind,
                      'position': data.geom_xpos[i].tolist(),
                      'rotation': data.geom_xmat[i].tolist(),
                      'size': model.geom_size[i].tolist(),
                      'color': model.geom_rgba[i].tolist(),
                      'static': bool(model.geom_bodyid[i] == 0)})
    height, distance, elevation = CAMERAS[key]
    target = [float(data.xpos[1, 0]), float(data.xpos[1, 1]), height]
    if key == 'reacher':
        target = [0., 0., 0.]
    return {'format': 'mujoco-scene-3d-v1', 'environment': key,
            'geoms': geoms, 'target': target, 'distance': distance,
            'elevation': elevation, 'unsupported_types': sorted(set(unsupported))}
