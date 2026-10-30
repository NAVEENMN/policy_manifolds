// A canvas projection of geometry returned by the actual simulator.
export function drawSimulationScene(ctx, w, h, live, baseline, ghost) {
  const scene = live?.scene;
  ctx.fillStyle = '#18221c'; ctx.fillRect(0, 0, w, h);
  if (!scene) {
    ctx.fillStyle = '#93ad9c'; ctx.font = '13px system-ui'; ctx.textAlign = 'center';
    ctx.fillText('Connect the simulator to start this environment', w / 2, h / 2);
    return;
  }
  const moving = scene.shapes.filter(shape => !shape.static);
  const verticalReach = Math.max(.1, ...moving.flatMap(shape => {
    const points = shape.points || (shape.a ? [shape.a, shape.b] : [shape.center]);
    return points.map(point => Math.abs(point[1] - scene.center[1]) + (shape.radius || 0));
  }));
  const scale = scene.kind === 'mujoco'
    ? Math.min((w - 48) / scene.span, (h - 50) / (2 * verticalReach))
    : (w - 48) / scene.span;
  const x = value => w / 2 + (value - scene.center[0]) * scale;
  const y = value => h * .52 - (value - scene.center[1]) * scale;
  const spacing = scene.span < 1 ? .1 : 1;
  ctx.strokeStyle = '#2a3b30'; ctx.lineWidth = .7;
  const left = scene.center[0] - scene.span / 2, bottom = scene.center[1] - h / scale;
  for (let v = Math.floor(left / spacing) * spacing; v < left + scene.span; v += spacing) {
    ctx.beginPath(); ctx.moveTo(x(v), 0); ctx.lineTo(x(v), h); ctx.stroke();
  }
  for (let v = Math.floor(bottom / spacing) * spacing; v < bottom + 2 * h / scale; v += spacing) {
    ctx.beginPath(); ctx.moveTo(0, y(v)); ctx.lineTo(w, y(v)); ctx.stroke();
  }
  function draw(data, original) {
    if (!data) return;
    ctx.save(); ctx.globalAlpha = original ? .4 : 1;
    for (const shape of data.shapes) {
      if (original && shape.static) continue;
      ctx.fillStyle = shape.static ? '#425a49' : original ? '#e2b08d' : '#b7dec9';
      ctx.strokeStyle = ctx.fillStyle;
      ctx.lineWidth = original ? 1.5 : 1;
      if (shape.kind === 'capsule') {
        ctx.lineWidth = Math.max(2, 2 * shape.radius * scale); ctx.lineCap = 'round';
        ctx.beginPath(); ctx.moveTo(x(shape.a[0]), y(shape.a[1])); ctx.lineTo(x(shape.b[0]), y(shape.b[1])); ctx.stroke();
      } else if (shape.kind === 'polygon') {
        ctx.beginPath(); shape.points.forEach(([px, py], i) => i ? ctx.lineTo(x(px), y(py)) : ctx.moveTo(x(px), y(py)));
        ctx.closePath(); original ? ctx.stroke() : ctx.fill();
      } else {
        ctx.beginPath(); ctx.arc(x(shape.center[0]), y(shape.center[1]), Math.max(2, shape.radius * scale), 0, Math.PI * 2);
        original ? ctx.stroke() : ctx.fill();
      }
    }
    ctx.restore();
  }
  if (ghost) draw(baseline?.scene, true);
  draw(scene, false);
  ctx.fillStyle = '#8ba393'; ctx.font = '10px system-ui'; ctx.textAlign = 'left';
  ctx.fillText(scene.kind === 'box2d' ? 'Box2D · live geometry' : `MuJoCo · ${scene.view} view`, 16, 24);
}
