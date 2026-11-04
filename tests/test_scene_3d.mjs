import assert from 'node:assert/strict';
import fs from 'node:fs';
import {meshData,modelMatrix,transform,OrbitCamera,lookAt,perspective,multiply,instances,cross,sub,dot,IDENTITY} from '../web_dist/scene-3d-math.mjs';
import {Simulation3D} from '../web_dist/simulation-3d.mjs';

for(const kind of ['sphere','cylinder','box','plane']){
 const {positions:p,normals:n}=meshData(kind);
 assert.equal(p.length,n.length);assert(p.every(Number.isFinite));assert(n.every(Number.isFinite));
 for(let i=0;i<p.length;i+=9){
  const a=[...p.slice(i,i+3)],b=[...p.slice(i+3,i+6)],c=[...p.slice(i+6,i+9)],normal=[...n.slice(i,i+3)];
  const geometric=cross(sub(b,a),sub(c,a));
  if(Math.hypot(...geometric)>1e-8)assert(dot(geometric,normal)>0,kind+' triangle winding must face outwards');
 }
}
assert.deepEqual(transform(modelMatrix([3,4,5],[0,-1,0,1,0,0,0,0,1],[2,3,4]),[1,0,0]),[3,6,5,1]);
const source={environment:'test',target:[0,0,1],distance:4,elevation:.4,geoms:[
 {kind:'capsule',position:[0,0,1],rotation:[0,0,1,0,1,0,-1,0,0],size:[.2,.5,0],color:[1,1,1,1],static:false},
 {kind:'plane',position:[0,0,0],rotation:IDENTITY,size:[0,0,1],color:[1,1,1,1],static:true}]};
const before=JSON.stringify(source),parts=instances(source);
assert.deepEqual(parts[1].position,[.5,0,1]);assert.deepEqual(parts[2].position,[-.5,0,1]);
assert.equal(instances(source,true).length,3);assert.equal(JSON.stringify(source),before);
const camera=new OrbitCamera(source);camera.orbit(100000,100000);camera.dolly(100000);
assert(camera.pitch<Math.PI/2&&camera.zoom===4);camera.reset(source);
camera.follow=false;camera.track({...source,target:[100,0,0]});assert.deepEqual(camera.target,[0,0,1]);
camera.follow=true;camera.track({...source,target:[1,2,1]});assert.deepEqual(camera.target,[1,2,1]);
for(const preset of ['orbit','side','top']){
 camera.preset(preset,source);
 const clip=transform(multiply(perspective(Math.PI/4,1.5,.01,100),lookAt(camera.eye,camera.target)),camera.target);
 assert(clip.every(Number.isFinite)&&clip[3]>0);assert(Math.abs(clip[0]/clip[3])<1e-6&&Math.abs(clip[1]/clip[3])<1e-6);
}
const unavailable=new Simulation3D({getContext:()=>null});assert.equal(unavailable.available,false);assert.equal(unavailable.render(source,null,false),false);
if(process.argv[2]){
 let count=0;
 for(const file of fs.readdirSync(process.argv[2]).filter(f=>f.endsWith('.json'))){
  const scene=JSON.parse(fs.readFileSync(process.argv[2]+'/'+file));const original=JSON.stringify(scene);
  const camera=new OrbitCamera(scene),vp=multiply(perspective(Math.PI/4,1.5,.001,1000),lookAt(camera.eye,camera.target));
  for(const item of instances(scene)){
   assert(item.position.every(Number.isFinite)&&item.scale.every(Number.isFinite));
   const m=modelMatrix(item.position,item.rotation,item.scale);assert([...m].every(Number.isFinite));
   if(!item.static){const p=transform(vp,item.position);assert(p[3]>0,file+' agent behind camera');}
  }
  assert.equal(JSON.stringify(scene),original);count++;
 }
 assert.equal(count,9);
}
console.log('Mesh winding, transforms, capsule geometry, camera controls, immutable scene handling and WebGL fallback passed.');
