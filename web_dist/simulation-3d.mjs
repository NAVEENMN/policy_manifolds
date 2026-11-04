// Dependency-free WebGL 2 viewer. Rendering never calls physics or policy inference.
import {OrbitCamera,instances,meshData,multiply,modelMatrix,lookAt,perspective,ortho,add,sub,dot,unit} from './scene-3d-math.mjs';

const vertex=`#version 300 es
in vec3 position;
in vec3 normal;
uniform mat4 model, viewProjection, lightProjection;
out vec3 world, surfaceNormal;
out vec4 shadowPosition;
void main(){
 vec4 p=model*vec4(position,1.0);world=p.xyz;
 surfaceNormal=transpose(inverse(mat3(model)))*normal;
 shadowPosition=lightProjection*p;gl_Position=viewProjection*p;
}`;
const fragment=`#version 300 es
precision highp float;
in vec3 world,surfaceNormal;
in vec4 shadowPosition;
uniform vec4 color;
uniform vec3 eye,lightDirection;
uniform bool floorSurface,ghost;
uniform float gridSize,fogDistance;
uniform sampler2D shadowMap;
out vec4 outColor;
float shadow(vec3 n){
 vec3 p=shadowPosition.xyz/shadowPosition.w*.5+.5;
 if(p.x<0.0||p.x>1.0||p.y<0.0||p.y>1.0||p.z>1.0)return 1.0;
 float bias=max(.0005,.002*(1.0-max(dot(n,lightDirection),0.0))),lit=0.0;
 for(int x=-1;x<=1;x++)for(int y=-1;y<=1;y++){
  float depth=texture(shadowMap,p.xy+vec2(x,y)/1024.0).r;
  lit+=p.z-bias>depth?.32:1.0;
 }
 return lit/9.0;
}
void main(){
 vec3 n=normalize(surfaceNormal),v=normalize(eye-world);
 vec3 base=color.rgb;
 if(floorSurface){
  vec2 coord=world.xy/gridSize;
  vec2 grid=abs(fract(coord-.5)-.5)/max(fwidth(coord),vec2(.0001));
  float line=1.0-min(min(grid.x,grid.y),1.0);
  float checker=mod(floor(coord.x)+floor(coord.y),2.0);
  base+=vec3(.014)*checker+vec3(.032,.047,.037)*line;
 }
 float diffuse=max(dot(n,lightDirection),0.0),visibility=shadow(n);
 vec3 h=normalize(v+lightDirection);
 float specular=pow(max(dot(n,h),0.0),floorSurface?16.0:48.0)*(floorSurface?.06:.28);
 float rim=pow(1.0-max(dot(n,v),0.0),3.0)*.12;
 vec3 shaded=base*(.34+.72*diffuse*visibility)+vec3(specular*visibility)+vec3(.18,.27,.23)*rim;
 if(ghost)shaded=mix(shaded,color.rgb,.55);
 float fog=1.0-exp(-pow(length(eye-world)/fogDistance,2.0));
 shaded=mix(shaded,vec3(.061,.089,.073),clamp(fog,0.0,.94));
 outColor=vec4(pow(max(shaded,vec3(0)),vec3(1.0/1.3)),color.a);
}`;
const depthVertex=`#version 300 es
in vec3 position;uniform mat4 model,lightProjection;
void main(){gl_Position=lightProjection*model*vec4(position,1.0);}`;
const depthFragment=`#version 300 es
precision highp float;
void main(){}`;

function shader(gl,type,source){
 const s=gl.createShader(type);gl.shaderSource(s,source);gl.compileShader(s);
 if(!gl.getShaderParameter(s,gl.COMPILE_STATUS)){const message=gl.getShaderInfoLog(s);gl.deleteShader(s);throw new Error(message);}
 return s;
}
function program(gl,vs,fs){
 const p=gl.createProgram(),v=shader(gl,gl.VERTEX_SHADER,vs),f=shader(gl,gl.FRAGMENT_SHADER,fs);
 gl.attachShader(p,v);gl.attachShader(p,f);gl.bindAttribLocation(p,0,'position');gl.bindAttribLocation(p,1,'normal');gl.linkProgram(p);
 gl.deleteShader(v);gl.deleteShader(f);
 if(!gl.getProgramParameter(p,gl.LINK_STATUS))throw new Error(gl.getProgramInfoLog(p));
 return p;
}

export class Simulation3D {
 constructor(canvas,onChange=()=>{}){
  this.canvas=canvas;this.onChange=onChange;this.cameras=new Map();this.available=false;this.error='';
  this.abort=new AbortController();this.pointers=new Map();
  try{this.gl=canvas.getContext('webgl2',{antialias:true,alpha:false,powerPreference:'low-power'});if(!this.gl)throw new Error('WebGL 2 is unavailable.');this.initGL();}
  catch(error){this.error=error.message;return;}
  const listen=(type,fn,options={})=>canvas.addEventListener(type,fn,{...options,signal:this.abort.signal});
  listen('webglcontextlost',e=>{e.preventDefault();this.available=false;this.error='3D graphics context paused.';this.onChange();});
  listen('webglcontextrestored',()=>{try{this.initGL();this.error='';}catch(error){this.error=error.message;}this.onChange();});
  listen('pointerdown',e=>{if(!this.camera)return;canvas.focus({preventScroll:true});canvas.setPointerCapture(e.pointerId);this.pointers.set(e.pointerId,{x:e.clientX,y:e.clientY});this.onChange();});
  listen('pointermove',e=>{
   if(!this.camera||!this.pointers.has(e.pointerId))return;
   const old=this.pointers.get(e.pointerId),next={x:e.clientX,y:e.clientY};
   if(this.pointers.size===2){const other=[...this.pointers.entries()].find(([id])=>id!==e.pointerId)[1];
    const before=Math.hypot(old.x-other.x,old.y-other.y),after=Math.hypot(next.x-other.x,next.y-other.y);
    if(before>0&&after>0)this.camera.dolly(Math.log(before/after)*1000);
   }else if(e.buttons===2||e.shiftKey)this.camera.pan(next.x-old.x,next.y-old.y,canvas.clientHeight);
   else this.camera.orbit(next.x-old.x,next.y-old.y);
   this.pointers.set(e.pointerId,next);this.onChange();
  });
  for(const type of ['pointerup','pointercancel','lostpointercapture'])listen(type,e=>this.pointers.delete(e.pointerId));
  listen('wheel',e=>{if(!this.camera)return;e.preventDefault();const delta=e.deltaY*(e.deltaMode===1?16:e.deltaMode===2?canvas.clientHeight:1);this.camera.dolly(delta);this.onChange();},{passive:false});
  listen('contextmenu',e=>e.preventDefault());
  listen('keydown',e=>{
   if(!this.camera)return;
   const moves={ArrowLeft:[-10,0],ArrowRight:[10,0],ArrowUp:[0,-10],ArrowDown:[0,10]};
   if(moves[e.key]){e.preventDefault();this.camera.orbit(...moves[e.key]);}
   else if(['+','=','-','_'].includes(e.key)){e.preventDefault();this.camera.dolly(['-','_'].includes(e.key)?90:-90);}
   else if(e.key==='Home'){e.preventDefault();this.reset();}
   else return;this.onChange();
  });
 }
 initGL(){
  const gl=this.gl;this.main=program(gl,vertex,fragment);this.depth=program(gl,depthVertex,depthFragment);
  this.uniforms=new Map();this.meshes=new Map();
  for(const kind of ['sphere','cylinder','box','plane']){
   const data=meshData(kind),vao=gl.createVertexArray(),buffers=[];gl.bindVertexArray(vao);
   for(const [location,values] of [[0,data.positions],[1,data.normals]]){
    const buffer=gl.createBuffer();buffers.push(buffer);gl.bindBuffer(gl.ARRAY_BUFFER,buffer);gl.bufferData(gl.ARRAY_BUFFER,values,gl.STATIC_DRAW);
    gl.enableVertexAttribArray(location);gl.vertexAttribPointer(location,3,gl.FLOAT,false,0,0);
   }
   this.meshes.set(kind,{vao,buffers,count:data.positions.length/3});
  }
  this.shadow=gl.createTexture();gl.bindTexture(gl.TEXTURE_2D,this.shadow);
  gl.texImage2D(gl.TEXTURE_2D,0,gl.DEPTH_COMPONENT24,1024,1024,0,gl.DEPTH_COMPONENT,gl.UNSIGNED_INT,null);
  for(const key of [gl.TEXTURE_MIN_FILTER,gl.TEXTURE_MAG_FILTER])gl.texParameteri(gl.TEXTURE_2D,key,gl.NEAREST);
  for(const key of [gl.TEXTURE_WRAP_S,gl.TEXTURE_WRAP_T])gl.texParameteri(gl.TEXTURE_2D,key,gl.CLAMP_TO_EDGE);
  this.framebuffer=gl.createFramebuffer();gl.bindFramebuffer(gl.FRAMEBUFFER,this.framebuffer);
  gl.framebufferTexture2D(gl.FRAMEBUFFER,gl.DEPTH_ATTACHMENT,gl.TEXTURE_2D,this.shadow,0);gl.drawBuffers([gl.NONE]);gl.readBuffer(gl.NONE);
  if(gl.checkFramebufferStatus(gl.FRAMEBUFFER)!==gl.FRAMEBUFFER_COMPLETE)throw new Error('Shadow buffer is unavailable.');
  gl.bindFramebuffer(gl.FRAMEBUFFER,null);gl.enable(gl.DEPTH_TEST);gl.enable(gl.CULL_FACE);gl.cullFace(gl.BACK);this.available=true;
 }
 location(p,name){const key=(p===this.main?'main:':'depth:')+name;if(!this.uniforms.has(key))this.uniforms.set(key,this.gl.getUniformLocation(p,name));return this.uniforms.get(key);}
 matrix(p,name,value){this.gl.uniformMatrix4fv(this.location(p,name),false,value);}
 activate(scene){
  this.scene=scene;
  if(!this.cameras.has(scene.environment))this.cameras.set(scene.environment,new OrbitCamera(scene));
  this.camera=this.cameras.get(scene.environment);this.camera.track(scene);
 }
 reset(){if(this.camera&&this.scene)this.camera.reset(this.scene);this.onChange();}
 preset(name){if(this.camera&&this.scene)this.camera.preset(name,this.scene);this.onChange();}
 setFollow(value){if(this.camera)this.camera.follow=value;this.onChange();}
 drawObject(object,p){
  if(object.scale.some(x=>x<=0))return;
  const gl=this.gl,mesh=this.meshes.get(object.kind);if(!mesh)return;
  this.matrix(p,'model',modelMatrix(object.position,object.rotation,object.scale));
  if(p===this.main){gl.uniform4fv(this.location(p,'color'),object.color);gl.uniform1i(this.location(p,'floorSurface'),object.kind==='plane');}
  gl.bindVertexArray(mesh.vao);gl.drawArrays(gl.TRIANGLES,0,mesh.count);
 }
 render(scene,original,showGhost){
  if(!this.available||scene?.format!=='mujoco-scene-3d-v1'||scene.unsupported_types?.length)return false;
  this.activate(scene);
  const gl=this.gl,canvas=this.canvas,dpr=Math.min(globalThis.devicePixelRatio||1,2);
  const width=Math.max(1,Math.round(canvas.clientWidth*dpr)),height=Math.max(1,Math.round(canvas.clientHeight*dpr));
  if(canvas.width!==width||canvas.height!==height){canvas.width=width;canvas.height=height;}
  const camera=this.camera,target=camera.target,eye=camera.eye,d=scene.distance;
  const viewProjection=multiply(perspective(Math.PI/4,width/height,d*.005,d*100),lookAt(eye,target));
  const lightEye=add(target,[-d*2,-d*3,d*5]),lightProjection=multiply(ortho(d*1.5,d*.1,d*15),lookAt(lightEye,target));
  const objects=instances(scene),ghosts=showGhost&&original?.environment===scene.environment?instances(original,true):[];
  // Only the live agent casts a shadow; the original is a translucent reference.
  gl.disable(gl.BLEND);gl.depthMask(true);gl.bindFramebuffer(gl.FRAMEBUFFER,this.framebuffer);gl.viewport(0,0,1024,1024);
  gl.clear(gl.DEPTH_BUFFER_BIT);gl.useProgram(this.depth);this.matrix(this.depth,'lightProjection',lightProjection);
  gl.enable(gl.POLYGON_OFFSET_FILL);gl.polygonOffset(2,4);
  for(const object of objects)if(object.kind!=='plane')this.drawObject(object,this.depth);
  gl.disable(gl.POLYGON_OFFSET_FILL);gl.bindFramebuffer(gl.FRAMEBUFFER,null);gl.viewport(0,0,width,height);
  gl.clearColor(.10,.145,.118,1);gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);gl.useProgram(this.main);
  this.matrix(this.main,'viewProjection',viewProjection);this.matrix(this.main,'lightProjection',lightProjection);
  gl.uniform3fv(this.location(this.main,'eye'),eye);gl.uniform3fv(this.location(this.main,'lightDirection'),unit(sub(lightEye,target)));
  gl.uniform1f(this.location(this.main,'gridSize'),d<1?.1:1);gl.uniform1f(this.location(this.main,'fogDistance'),d*6);
  gl.activeTexture(gl.TEXTURE0);gl.bindTexture(gl.TEXTURE_2D,this.shadow);gl.uniform1i(this.location(this.main,'shadowMap'),0);
  gl.uniform1i(this.location(this.main,'ghost'),false);
  for(const object of objects)this.drawObject(object,this.main);
  if(ghosts.length){
   gl.enable(gl.BLEND);gl.blendFunc(gl.SRC_ALPHA,gl.ONE_MINUS_SRC_ALPHA);gl.depthMask(false);gl.uniform1i(this.location(this.main,'ghost'),true);
   // Draw through the live actor so an overlapping original remains visible.
   gl.depthFunc(gl.LEQUAL);
   ghosts.sort((a,b)=>dot(sub(b.position,eye),sub(b.position,eye))-dot(sub(a.position,eye),sub(a.position,eye)));
   for(const object of ghosts)this.drawObject(object,this.main);
   gl.depthFunc(gl.LESS);gl.depthMask(true);gl.disable(gl.BLEND);
  }
  gl.bindVertexArray(null);return true;
 }
 dispose(){
  this.abort.abort();if(!this.gl)return;const gl=this.gl;
  for(const mesh of this.meshes?.values()||[]){gl.deleteVertexArray(mesh.vao);for(const b of mesh.buffers)gl.deleteBuffer(b);}
  gl.deleteProgram(this.main);gl.deleteProgram(this.depth);gl.deleteTexture(this.shadow);gl.deleteFramebuffer(this.framebuffer);this.available=false;
 }
}
