// LAB geometry belongs under the same rotated world group as the vehicle meshes.
// SOURCE forward is local +X. All extension rails use the mesh's full 3D pose.
// A ground-flattened strip cannot bound the perspective image of a tall vehicle.
export const HEADING_STYLES = [
 {car:0,role:'prediction',label:'车0预测',color:'#69baff',dashed:false},
 {car:0,role:'reference',label:'车0真值',color:'#55f3d3',dashed:true},
 {car:1,role:'prediction',label:'车1预测',color:'#ffb669',dashed:false},
 {car:1,role:'reference',label:'车1真值',color:'#f58de5',dashed:true},
];
const finite=(a,n)=>Array.isArray(a)&&a.length>=n&&a.slice(0,n).every(Number.isFinite);
const validPose=(record,origin)=>record&&record.pose_status!=='unknown'&&finite(record.pose,3)&&finite(record.quaternion_xyzw,4)&&finite(origin,3)&&Math.hypot(...record.quaternion_xyzw)>1e-8&&Math.abs(Math.hypot(...record.quaternion_xyzw)-1)<=.02;
function rotate(v,q){
 const [x,y,z,w]=q,n=Math.hypot(...q),a=x/n,b=y/n,c=z/n,d=w/n;
 const tx=2*(b*v[2]-c*v[1]),ty=2*(c*v[0]-a*v[2]),tz=2*(a*v[1]-b*v[0]);
 return [v[0]+d*tx+b*tz-c*ty,v[1]+d*ty+c*tx-a*tz,v[2]+d*tz+a*ty-b*tx];
}
export function headingLab(record,origin=[0,0,0]) {
 if(!validPose(record,origin))return null;
 const q=record.quaternion_xyzw,n=Math.hypot(...q);
 if(n<1e-8||Math.abs(n-1)>.02)return null;
 const [x,y,z,w]=q.map(v=>v/n);
 const forward=[1-2*(y*y+z*z),2*(x*y+w*z),2*(x*z-w*y)];
 const horizontal=Math.hypot(forward[0],forward[1]);
 if(horizontal<1e-6)return null;
 return {center:record.pose.slice(0,3).map((v,i)=>v-origin[i]),direction:[forward[0]/horizontal,forward[1]/horizontal,0],bodyForward:forward};
}

export function headingStrip(record,vertices,origin=[0,0,0]) {
 const heading=headingLab(record,origin);if(!heading)return null;
 const [dx,dy]=heading.direction,normal=[-dy,dx,0];
 let left=-Infinity,right=Infinity,bottom=Infinity;
 for(const vertex of vertices){const p=rotate(vertex,record.quaternion_xyzw),support=p[0]*normal[0]+p[1]*normal[1];left=Math.max(left,support);right=Math.min(right,support);bottom=Math.min(bottom,p[2]);}
 return {...heading,normal,left,right,width:left-right,bottom};
}

export function bodyDirectionEnvelope(record,vertices,origin=[0,0,0]) {
 if(!validPose(record,origin))return null;
 const min=[Infinity,Infinity,Infinity],max=[-Infinity,-Infinity,-Infinity];
 for(const vertex of vertices)for(let i=0;i<3;i++){min[i]=Math.min(min[i],vertex[i]);max[i]=Math.max(max[i],vertex[i]);}
 const center=record.pose.slice(0,3).map((v,i)=>v-origin[i]);
 const point=(x,y,z)=>rotate([x,y,z],record.quaternion_xyzw).map((v,i)=>v+center[i]);
 return {center,min,max,point,forward:rotate([1,0,0],record.quaternion_xyzw)};
}

export function setupHeadingLines({THREE,world,origin,geometry,legendHost,halfLength=500}) {
 if(!Number.isFinite(halfLength)||halfLength<10)throw new Error('heading halfLength must be at least 10 metres');
 const vertices=geometry?.vertices;
 if(!Array.isArray(vertices)||!vertices.length||!vertices.every(v=>finite(v,3)))throw new Error('heading boundaries require finite SOURCE geometry vertices');
 const group=new THREE.Group();group.name='vehicle-heading-lines';world.add(group);
 let currentFrame=null,currentVersion=null,currentOptions={};
 const entries=HEADING_STYLES.map(style=>{
  const material=new (style.dashed?THREE.LineDashedMaterial:THREE.LineBasicMaterial)({color:style.color,depthTest:false,depthWrite:false,transparent:true,opacity:.95,...(style.dashed?{dashSize:.6,gapSize:.4}:{} )});
  const makeLine=(mat,order)=>{const line=new THREE.Line(new THREE.BufferGeometry(),mat);line.renderOrder=order;line.frustumCulled=false;line.visible=false;group.add(line);return line;};
  const boundaries=[makeLine(material,24),makeLine(material,24)];
  const upperMaterial=material.clone();upperMaterial.opacity=.45;
  const upperBoundaries=[makeLine(upperMaterial,24),makeLine(upperMaterial,24)];
  const sections=[makeLine(upperMaterial,24),makeLine(upperMaterial,24)];
  const arrowMaterial=new THREE.LineBasicMaterial({color:style.color,depthTest:false,depthWrite:false});
  const arrows=[makeLine(arrowMaterial,25),makeLine(arrowMaterial,25)];
  const center=makeLine(material.clone(),23);center.material.opacity=.35;
  const entry={...style,boundaries,upperBoundaries,sections,arrows,center,enabled:true,checkbox:null,label:null};
  if(legendHost){
   const label=legendHost.ownerDocument.createElement('label');label.style.cssText='display:inline-flex;gap:5px;align-items:center;margin-right:12px';
   const checkbox=legendHost.ownerDocument.createElement('input');checkbox.type='checkbox';checkbox.checked=true;
   const swatch=legendHost.ownerDocument.createElement('i');swatch.style.cssText=`display:inline-block;width:26px;border-top:3px ${style.dashed?'dashed':'solid'} ${style.color}`;
   const text=legendHost.ownerDocument.createTextNode(style.label);label.append(checkbox,swatch,text);legendHost.append(label);
   checkbox.onchange=()=>{entry.enabled=checkbox.checked;if(currentFrame)update(currentFrame,currentVersion,currentOptions);};entry.checkbox=checkbox;entry.label=label;
  }
  return entry;
 });
 let centerToggle=null;
 if(legendHost){
  const label=legendHost.ownerDocument.createElement('label');centerToggle=legendHost.ownerDocument.createElement('input');centerToggle.type='checkbox';label.append(centerToggle,' 辅助中心线');legendHost.append(label);
  centerToggle.onchange=()=>{if(currentFrame)update(currentFrame,currentVersion,{...currentOptions,showCenter:centerToggle.checked});};
  const note=legendHost.ownerDocument.createElement('small');note.style.display='block';note.textContent=`方向框与车身使用同一三维姿态：左右下边缘加较淡的上边缘，前后截框贴住当前简化车身体积；所有长线沿车头方向双向各延长${halfLength}米。箭头指车头，实线为预测、虚线为真值。上下边界一起表示立体空间，不能只用地面两条线框住有高度的车。该框用于核对当前朝向与体积，不是未来轨迹、驾驶意图或碰撞判据；未知姿态不画。`;legendHost.append(note);
 }
 function setPoints(line,points){line.geometry.setFromPoints(points.map(p=>new THREE.Vector3(...p)));line.geometry.computeBoundingSphere();}
 function update(frame,version,{showPrediction=true,showReference=true,showCenter=centerToggle?.checked??false}={}) {
  currentFrame=frame;currentVersion=version;currentOptions={showPrediction,showReference,showCenter};
  if(centerToggle)centerToggle.checked=showCenter;
  for(const entry of entries){
   // Hide before lookup so a missing car/pose never leaves an older frame's line.
   for(const line of [...entry.boundaries,...entry.upperBoundaries,...entry.sections,...entry.arrows,entry.center])line.visible=false;
   const car=frame?.cars?.find(c=>c.car===entry.car);
   const record=entry.role==='reference'?car?.reference:car?.versions?.[version];
   const heading=bodyDirectionEnvelope(record,vertices,origin);
   const enabled=entry.enabled&&(entry.role==='reference'?showReference:showPrediction);
   if(entry.label){entry.label.title=`f${frame?.frame??'—'} · ${entry.label.textContent} · ${heading?'短箭头指向车头':'当前姿态未知，隐藏'}`;entry.label.style.opacity=heading?'1':'.5';}
   if(!enabled||!heading)continue;
   const {min,max,point:at}=heading;
   for(const [index,offset] of [max[1],min[1]].entries()){
    const line=entry.boundaries[index],arrow=entry.arrows[index];
    setPoints(line,[at(-halfLength,offset,min[2]),at(halfLength,offset,min[2])]);
    if(entry.dashed)line.computeLineDistances();
    const top=entry.upperBoundaries[index];
    setPoints(top,[at(-halfLength,offset,max[2]),at(halfLength,offset,max[2])]);
    if(entry.dashed)top.computeLineDistances();top.visible=true;
    setPoints(arrow,[at(max[0]+.6,offset+.25,min[2]),at(max[0]+1.2,offset,min[2]),at(max[0]+.6,offset-.25,min[2])]);
    line.userData.headingLab=heading;line.userData.frame=frame.frame;line.userData.version=version;line.userData.side=index===0?'left':'right';
    line.visible=true;arrow.visible=true;
   }
   for(const [index,x] of [min[0],max[0]].entries()){
    const section=entry.sections[index];setPoints(section,[at(x,min[1],min[2]),at(x,max[1],min[2]),at(x,max[1],max[2]),at(x,min[1],max[2]),at(x,min[1],min[2])]);
    if(entry.dashed)section.computeLineDistances();section.visible=true;
   }
   setPoints(entry.center,[at(-halfLength,0,0),at(halfLength,0,0)]);if(entry.dashed)entry.center.computeLineDistances();entry.center.visible=showCenter;
  }
 }
 return {update,group,entries};
}
