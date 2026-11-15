// Display-only convex hull: keep source points and triangles, orient faces outward.
// SciPy ConvexHull simplices do not promise consistent vertex winding.
export function buildVehicleGeometry(THREE,source){
 const vertices=source.vertices;
 const center=[0,0,0];
 for(const point of vertices)for(let axis=0;axis<3;axis++)center[axis]+=point[axis]/vertices.length;
 let reversed=0;
 const indices=source.faces.flatMap(face=>{
  const [ia,ib,ic]=face,[a,b,c]=face.map(index=>vertices[index]);
  const u=b.map((value,axis)=>value-a[axis]),v=c.map((value,axis)=>value-a[axis]);
  const normal=[u[1]*v[2]-u[2]*v[1],u[2]*v[0]-u[0]*v[2],u[0]*v[1]-u[1]*v[0]];
  const outward=normal.reduce((sum,value,axis)=>sum+value*((a[axis]+b[axis]+c[axis])/3-center[axis]),0);
  if(outward<0){reversed++;return [ia,ic,ib];}
  return [ia,ib,ic];
 });
 const result=new THREE.BufferGeometry();
 result.setAttribute('position',new THREE.Float32BufferAttribute(vertices.flat(),3));
 result.setIndex(indices);result.computeVertexNormals();
 result.userData={displayOnly:true,reversedFaces:reversed};
 return result;
}
