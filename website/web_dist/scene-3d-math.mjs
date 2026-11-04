// MuJoCo and this viewer both use right-handed, Z-up world coordinates.
export const clamp = (x, lo, hi) => Math.max(lo, Math.min(hi, x));
export const add = (a, b) => a.map((x, i) => x + b[i]);
export const sub = (a, b) => a.map((x, i) => x - b[i]);
export const dot = (a, b) => a.reduce((s, x, i) => s + x * b[i], 0);
export const cross = (a, b) => [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]];
export const unit = a => { const n = Math.hypot(...a) || 1; return a.map(x => x / n); };
export const IDENTITY = [1,0,0, 0,1,0, 0,0,1];
export function multiply(a, b) {
  const c = new Float32Array(16);
  for (let col=0; col<4; col++) for (let row=0; row<4; row++)
    for (let k=0; k<4; k++) c[col*4+row] += a[k*4+row]*b[col*4+k];
  return c;
}
export function transform(m, p) {
  const v = [...p, 1];
  return [0,1,2,3].map(r => v.reduce((s,x,c) => s+m[c*4+r]*x,0));
}
export function modelMatrix(position, rotation, scale) {
  const m = new Float32Array(16);
  for (let c=0;c<3;c++) for(let r=0;r<3;r++) m[c*4+r]=rotation[r*3+c]*scale[c];
  m.set(position,12);m[15]=1;return m;
}
export function lookAt(eye, target) {
  const z=unit(sub(eye,target)), x=unit(cross([0,0,1],z)), y=cross(z,x);
  return new Float32Array([x[0],y[0],z[0],0,x[1],y[1],z[1],0,x[2],y[2],z[2],0,-dot(x,eye),-dot(y,eye),-dot(z,eye),1]);
}
export function perspective(fov, aspect, near, far) {
  const f=1/Math.tan(fov/2), nf=1/(near-far);
  return new Float32Array([f/aspect,0,0,0,0,f,0,0,0,0,(far+near)*nf,-1,0,0,2*far*near*nf,0]);
}
export function ortho(size, near, far) {
  return new Float32Array([1/size,0,0,0,0,1/size,0,0,0,0,-2/(far-near),0,0,0,-(far+near)/(far-near),1]);
}
export class OrbitCamera {
  constructor(scene) { this.reset(scene); }
  reset(scene) {
    this.yaw=-Math.PI/3;this.pitch=scene.elevation;this.zoom=1;this.distance=scene.distance;
    this.offset=[0,0,0];this.anchor=scene.target.slice();this.follow=true;
  }
  track(scene) { if(this.follow)this.anchor=scene.target.slice(); }
  orbit(dx,dy) { this.yaw-=dx*.008;this.pitch=clamp(this.pitch+dy*.008,.08,1.53); }
  dolly(delta) { this.zoom=clamp(this.zoom*Math.exp(delta*.001),.25,4); }
  pan(dx,dy,height) {
    const right=[-Math.sin(this.yaw),Math.cos(this.yaw),0], up=[-Math.cos(this.yaw)*Math.sin(this.pitch),-Math.sin(this.yaw)*Math.sin(this.pitch),Math.cos(this.pitch)];
    const s=this.distance*this.zoom/Math.max(height,1);
    this.offset=this.offset.map((v,i)=>v+(-dx*right[i]+dy*up[i])*s);
  }
  preset(name,scene) {
    this.offset=[0,0,0];this.zoom=1;this.anchor=scene.target.slice();
    if(name==='top'){this.yaw=-Math.PI/2;this.pitch=1.53;}
    else if(name==='side'){this.yaw=-Math.PI/2;this.pitch=.08;}
    else {this.yaw=-Math.PI/3;this.pitch=scene.elevation;}
  }
  get target(){return add(this.anchor,this.offset);}
  get eye(){const d=this.distance*this.zoom,c=Math.cos(this.pitch);return add(this.target,[d*c*Math.cos(this.yaw),d*c*Math.sin(this.yaw),d*Math.sin(this.pitch)]);}
}
// A capsule uses a true cylinder and hemispheres of exactly MuJoCo's radius.
// Sphere intersections are internal; their visible union is the native capsule.
export function instances(scene, ghost=false) {
  return scene.geoms.flatMap(g=>{
    if(ghost && g.static)return [];
    const color=ghost?[.89,.61,.37,.27]:g.kind==='plane'?[.105,.15,.12,1]:[
      .45*g.color[0]+.55*.65,.45*g.color[1]+.55*.82,.45*g.color[2]+.55*.72,g.color[3]];
    const make=(kind,scale,position=g.position)=>({kind,position,rotation:g.rotation,scale,color,static:g.static});
    const r=g.size[0],h=g.size[1];
    if(g.kind==='capsule'){
      const dz=[g.rotation[2]*h,g.rotation[5]*h,g.rotation[8]*h];
      return [make('cylinder',[r,r,h]),make('sphere',[r,r,r],add(g.position,dz)),make('sphere',[r,r,r],sub(g.position,dz))];
    }
    if(g.kind==='sphere')return [make('sphere',[r,r,r])];
    if(g.kind==='ellipsoid')return [make('sphere',g.size)];
    if(g.kind==='cylinder')return [make('cylinder',[r,r,h])];
    if(g.kind==='plane') {
      // Plane has infinite extent in MuJoCo. Keep its visual patch near the camera.
      return [make('plane',[Math.max(30,scene.distance*10),Math.max(30,scene.distance*10),1],
        [scene.target[0],scene.target[1],g.position[2]])];
    }
    return [make('box',g.size)];
  });
}
export function meshData(kind, segments=28, rings=18) {
  const p=[],n=[];
  const tri=(a,b,c,na,nb,nc)=>{p.push(...a,...b,...c);n.push(...na,...nb,...nc);};
  if(kind==='sphere') {
    const point=(r,s)=>{const t=r*Math.PI/rings,f=s*2*Math.PI/segments;return [Math.sin(t)*Math.cos(f),Math.sin(t)*Math.sin(f),Math.cos(t)];};
    for(let r=0;r<rings;r++)for(let s=0;s<segments;s++) {
      const a=point(r,s),b=point(r+1,s),c=point(r+1,s+1),d=point(r,s+1);
      tri(a,b,c,a,b,c);tri(a,c,d,a,c,d);
    }
  } else if(kind==='cylinder') {
    for(let i=0;i<segments;i++) {
      const a=i*2*Math.PI/segments,b=(i+1)*2*Math.PI/segments;
      const na=[Math.cos(a),Math.sin(a),0],nb=[Math.cos(b),Math.sin(b),0];
      const lo=[na[0],na[1],-1],hi=[na[0],na[1],1],lo2=[nb[0],nb[1],-1],hi2=[nb[0],nb[1],1];
      tri(lo,lo2,hi2,na,nb,nb);tri(lo,hi2,hi,na,nb,na);
      tri([0,0,1],hi,hi2,[0,0,1],[0,0,1],[0,0,1]);
      tri([0,0,-1],lo2,lo,[0,0,-1],[0,0,-1],[0,0,-1]);
    }
  } else {
    const faces=kind==='plane'?[[[1,0,0],[0,1,0],[0,0,1],0]]:[
      [[0,1,0],[0,0,1],[1,0,0],1],[[0,0,1],[0,1,0],[-1,0,0],1],
      [[0,0,1],[1,0,0],[0,1,0],1],[[1,0,0],[0,0,1],[0,-1,0],1],
      [[1,0,0],[0,1,0],[0,0,1],1],[[0,1,0],[1,0,0],[0,0,-1],1]];
    for(const [u,v,normal,offset] of faces){
      const point=(x,y)=>normal.map((z,i)=>z*offset+u[i]*x+v[i]*y);
      const a=point(-1,-1),b=point(1,-1),c=point(1,1),d=point(-1,1);
      tri(a,b,c,normal,normal,normal);tri(a,c,d,normal,normal,normal);
    }
  }
  return {positions:new Float32Array(p),normals:new Float32Array(n)};
}
