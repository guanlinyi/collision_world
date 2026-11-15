"""Fixed observed anchors matched to freshly certified SOURCE contour segments."""
from collections import Counter,defaultdict
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from geometry_pool import ProjectedUnion
from tangent_kernel import certify
from exterior_matching import admission
REQUIRE_FULL_POOL=False


def source_graph(pool):
    segments=pool['segment_ids'];node_by_key={};node_ids=[]
    for endpoints in segments:
        pair=[]
        for sample in endpoints:
            key=pool['unique_point_keys'][sample];node_by_key.setdefault(key,len(node_by_key));pair.append(node_by_key[key])
        node_ids.append(pair)
    nodes=np.array(node_ids,int);length=np.linalg.norm(pool['xy'][segments[:,1]]-pool['xy'][segments[:,0]],axis=1)
    links={}
    for pair,l in zip(nodes,length):
        key=tuple(sorted(pair.tolist()));links[key]=min(links.get(key,np.inf),float(l))
    row,col,weight=[],[],[];adj=defaultdict(list)
    for (a,b),l in links.items():
        row.extend([a,b]);col.extend([b,a]);weight.extend([l,l]);adj[a].append((b,l));adj[b].append((a,l))
    matrix=csr_matrix((weight,(row,col)),shape=(len(node_by_key),len(node_by_key)))
    distances=shortest_path(matrix,directed=False)
    # Cumulative coordinates are descriptive; pairwise shortest arc distance also handles wraparound.
    coordinate=np.full(len(node_by_key),np.nan);components=np.full(len(node_by_key),-1,int);perimeters={};branched=[]
    unseen=set(range(len(node_by_key)))
    while unseen:
        stack=[next(iter(unseen))];component=[]
        while stack:
            node=stack.pop()
            if node not in unseen:continue
            unseen.remove(node);component.append(node);stack.extend(other for other,l in adj[node] if other in unseen)
        cid=len(perimeters);components[component]=cid
        perimeters[cid]=sum(l for (a,b),l in links.items() if a in component)
        if any(len(adj[node])>2 for node in component):branched.append(cid);continue
        ends=[node for node in component if len(adj[node])==1];start=min(ends if ends else component)
        current,previous,distance=start,None,0.
        while np.isnan(coordinate[current]):
            coordinate[current]=distance
            choices=[(other,l) for other,l in adj[current] if other!=previous and np.isnan(coordinate[other])]
            if not choices:break
            other,l=choices[0];previous,current=current,other;distance+=l
    return dict(nodes=nodes,length=length,distances=distances,component=components,coordinate=coordinate,
        component_length=perimeters,branched_components=branched)


def point_arc_distance(graph,left,alpha,right,beta):
    if left==right:return abs(alpha-beta)*graph['length'][left]
    a,b=graph['nodes'][left];c,d=graph['nodes'][right]
    la,lb=graph['length'][left],graph['length'][right]
    return min(alpha*la+graph['distances'][a,c]+beta*lb,
        alpha*la+graph['distances'][a,d]+(1-beta)*lb,
        (1-alpha)*la+graph['distances'][b,c]+beta*lb,
        (1-alpha)*la+graph['distances'][b,d]+(1-beta)*lb)


def refresh(builder,pool,anchors,problem,source_foot_allowed=None):
    segments=pool['segment_ids'];xy=pool['xy'][segments];vector=xy[:,1]-xy[:,0]
    normals=pool['normal'][segments].sum(1);normals/=np.linalg.norm(normals,axis=1)[:,None]
    graph=source_graph(pool)
    pose=pool['pose'];cam=builder.module.camera(problem.row)
    if 'projected' not in pool:
        cp=(builder.triangles@builder.module.rot(pose).T+pose[:3]-cam[1])@np.linalg.inv(cam[0].T)
        uv=cp[:,:,:2]/cp[:,:,2,None]*cam[2]+cam[3]
        pool['projected']=ProjectedUnion(uv,cp[:,:,2],(problem.row['image_height'],problem.row['image_width']))
    associations=[];counts=Counter()
    for anchor in anchors:
        point=np.asarray(anchor['observed_xy']);normal=np.asarray(anchor['normal'])
        alpha=np.sum((point-xy[:,0])*vector,axis=1)/np.sum(vector*vector,axis=1)
        feet=xy[:,0]+alpha[:,None]*vector;distance=np.linalg.norm(point-feet,axis=1)
        eligible=np.where((alpha>0)&(alpha<1)&(distance<=5)&(normals@normal>=np.cos(np.pi/6)))[0]
        guard_removed=0
        if source_foot_allowed is not None and len(eligible):
            candidate_ends=pool['local'][segments[eligible]]
            _,candidate_z=builder.module.project(candidate_ends.reshape(-1,3),pose,problem.row)
            candidate_z=candidate_z.reshape(-1,2);s=alpha[eligible]
            t=s*candidate_z[:,0]/((1-s)*candidate_z[:,1]+s*candidate_z[:,0])
            candidate_local=candidate_ends[:,0]+t[:,None]*(candidate_ends[:,1]-candidate_ends[:,0])
            candidate_q,_=builder.module.project(candidate_local,pose,problem.row)
            allowed=source_foot_allowed(candidate_q);guard_removed=int((~allowed).sum());eligible=eligible[allowed]
        eligible=eligible[np.argsort(distance[eligible],kind='stable')]
        reason='source_foot_mutual_occlusion_unknown' if guard_removed and not len(eligible) else 'no_source_segment_meets_fixed_gates';match=None
        if len(eligible):
            best=int(eligible[0]);component=int(graph['component'][graph['nodes'][best,0]])
            if component in graph['branched_components']:reason='unknown_branched_source_arc'
            else:
                competitor=None;separation=None
                for other in eligible[1:]:
                    path=point_arc_distance(graph,best,float(alpha[best]),int(other),float(alpha[other]))
                    if path>=10:competitor=int(other);separation=float(path);break
                if competitor is not None and distance[competitor]-distance[best]<2:reason='competing_source_arc_margin_under_2px'
                else:
                    left,right=segments[best];ends=pool['local'][[left,right]]
                    _,z=builder.module.project(ends,pose,problem.row)
                    s=float(alpha[best]);t=s*z[0]/((1-s)*z[1]+s*z[0])
                    local=ends[0]+t*(ends[1]-ends[0]);edge_id=int(pool['edge_ids'][left])
                    edge_t=float(pool['t'][left]+t*(pool['t'][right]-pool['t'][left]))
                    q,depth=builder.module.project(local[None],pose,problem.row)
                    result,proof=certify(pool['projected'],q[0],float(depth[0]),builder.edges[edge_id],edge_t,builder.faces,exact_incident_depth=True)
                    reason='source_foot_certificate_'+result
                    if result=='accepted' and source_foot_allowed is not None and not bool(source_foot_allowed(q)[0]):
                        reason='source_foot_mutual_occlusion_unknown'
                    elif result=='accepted':
                        match=dict(anchor_id=anchor['id'],local=local,source_edge_id=edge_id,source_t=edge_t,
                            source_segment_id=best,source_segment_sample_ids=[int(left),int(right)],source_graph_component=component,
                            source_foot_image_fraction=s,source_arc_coordinate_at_left=float(graph['coordinate'][graph['nodes'][best,0]]),
                            source_component_length=graph['component_length'][component],distance=float(distance[best]),
                            competitor_segment_id=competitor,competitor_arc_separation=separation,
                            competitor_distance=float(distance[competitor]) if competitor is not None else None,
                            certificate=proof,status='matched')
        if match is None:match=dict(anchor_id=anchor['id'],status='missing',reason=reason)
        counts[match['status'] if match['status']=='matched' else match['reason']]+=1;associations.append(match)
    matched=[a for a in associations if a['status']=='matched'];matched_ids=[a['anchor_id'] for a in matched]
    support=float(sum(anchors[index]['weight'] for index in matched_ids))
    active=[dict(anchors[a['anchor_id']],local=a['local']) for a in matched]
    weights=np.array([a['weight'] for a in active])
    gates=admission(active,builder,problem,pose,weights) if active else dict(passed=False,reasons=['no_refreshed_associations'])
    if support<.8:gates['passed']=False;gates['reasons'].append('initial_weight_support_under_80percent')
    if REQUIRE_FULL_POOL and not pool['admitted']:gates['passed']=False;gates['reasons'].append('SOURCE_pool_geometry_not_admitted')
    return dict(associations=associations,support_weight=support,admission=gates,counts=dict(counts),
        lost_anchor_ids=[a['anchor_id'] for a in associations if a['status']=='missing'],
        source_graph=dict(nodes=len(graph['coordinate']),components=len(graph['component_length']),branched_components=graph['branched_components']))

from exact_union_index import get_index as get_exact_union_index
_original_source_graph=source_graph
def source_graph(pool):
    if '_unchanged_sample_graph' not in pool:pool['_unchanged_sample_graph']=_original_source_graph(pool)
    return pool['_unchanged_sample_graph']
def refresh(builder, pool, anchors, problem, source_foot_allowed=None):
    segments = pool['segment_ids']
    xy = pool['xy'][segments]
    vector = xy[:, 1] - xy[:, 0]
    normals = pool['normal'][segments].sum(1)
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    graph = source_graph(pool)
    union_index = get_exact_union_index(builder, pool, problem)
    pose = pool['pose']
    cam = builder.module.camera(problem.row)
    if 'projected' not in pool:
        cp = (builder.triangles @ builder.module.rot(pose).T + pose[:3] - cam[1]) @ np.linalg.inv(cam[0].T)
        uv = cp[:, :, :2] / cp[:, :, 2, None] * cam[2] + cam[3]
        pool['projected'] = ProjectedUnion(uv, cp[:, :, 2], (problem.row['image_height'], problem.row['image_width']))
    associations = []
    counts = Counter()
    for anchor in anchors:
        point = np.asarray(anchor['observed_xy'])
        normal = np.asarray(anchor['normal'])
        alpha = np.sum((point - xy[:, 0]) * vector, axis=1) / np.sum(vector * vector, axis=1)
        alpha = np.clip(alpha, 0, 1)
        feet = xy[:, 0] + alpha[:, None] * vector
        distance = np.linalg.norm(point - feet, axis=1)
        eligible = np.where((distance <= 5) & (normals @ normal >= np.cos(np.pi / 6)))[0]
        guard_removed = 0
        if source_foot_allowed is not None and len(eligible):
            candidate_ends = pool['local'][segments[eligible]]
            _, candidate_z = builder.module.project(candidate_ends.reshape(-1, 3), pose, problem.row)
            candidate_z = candidate_z.reshape(-1, 2)
            s = alpha[eligible]
            t = s * candidate_z[:, 0] / ((1 - s) * candidate_z[:, 1] + s * candidate_z[:, 0])
            candidate_local = candidate_ends[:, 0] + t[:, None] * (candidate_ends[:, 1] - candidate_ends[:, 0])
            candidate_q, _ = builder.module.project(candidate_local, pose, problem.row)
            allowed = source_foot_allowed(candidate_q)
            guard_removed = int((~allowed).sum())
            eligible = eligible[allowed]
        eligible = eligible[np.argsort(distance[eligible], kind='stable')]
        reason = 'source_foot_mutual_occlusion_unknown' if guard_removed and (not len(eligible)) else 'no_source_segment_meets_fixed_gates'
        match = None
        if len(eligible):
            best = int(eligible[0])
            component = int(graph['component'][graph['nodes'][best, 0]])
            best_binding = union_index.locate(best, float(alpha[best]))
            if not best_binding['valid']:
                reason = 'unknown_projected_union_boundary_location'
            else:
                competitor = None
                separation = None
                ambiguous_relation = False
                for other in eligible[1:]:
                    path = union_index.arc_distance(best, float(alpha[best]), int(other), float(alpha[other]))
                    if np.isnan(path):
                        ambiguous_relation = True
                        break
                    if path >= 10:
                        competitor = int(other)
                        separation = float(path)
                        ambiguous_relation = False
                        break
                if ambiguous_relation:
                    reason = 'unknown_projected_union_competitor_location'
                elif competitor is not None and distance[competitor] - distance[best] < 2:
                    reason = 'competing_source_arc_margin_under_2px'
                else:
                    left, right = segments[best]
                    ends = pool['local'][[left, right]]
                    _, z = builder.module.project(ends, pose, problem.row)
                    s = float(alpha[best])
                    t = s * z[0] / ((1 - s) * z[1] + s * z[0])
                    local = ends[0] + t * (ends[1] - ends[0])
                    edge_id = int(pool['edge_ids'][left])
                    edge_t = float(pool['t'][left] + t * (pool['t'][right] - pool['t'][left]))
                    q, depth = builder.module.project(local[None], pose, problem.row)
                    result, proof = certify(pool['projected'], q[0], float(depth[0]), builder.edges[edge_id], edge_t, builder.faces, exact_incident_depth=True)
                    reason = 'source_foot_certificate_' + result
                    if result == 'accepted' and source_foot_allowed is not None and (not bool(source_foot_allowed(q)[0])):
                        reason = 'source_foot_mutual_occlusion_unknown'
                    elif result == 'accepted':
                        match = dict(anchor_id=anchor['id'], local=local, source_edge_id=edge_id, source_t=edge_t, source_segment_id=best, source_segment_sample_ids=[int(left), int(right)], source_graph_component=component, source_foot_image_fraction=s, source_arc_coordinate_at_left=float(graph['coordinate'][graph['nodes'][best, 0]]) if np.isfinite(graph['coordinate'][graph['nodes'][best, 0]]) else None, source_component_length=graph['component_length'][component], distance=float(distance[best]), competitor_segment_id=competitor, competitor_arc_separation=float(separation) if separation is not None and np.isfinite(separation) else None, competitor_distance=float(distance[competitor]) if competitor is not None else None, certificate=proof, status='matched', old_sample_graph_coordinate_unknown=not np.isfinite(graph['coordinate'][graph['nodes'][best, 0]]), source_projected_union_binding=best_binding, competitor_projected_union_binding=union_index.locate(competitor, float(alpha[competitor])) if competitor is not None else None, competitor_arc_separation_is_infinite=bool(separation is not None and np.isinf(separation)))
        if match is None:
            match = dict(anchor_id=anchor['id'], status='missing', reason=reason)
        counts[match['status'] if match['status'] == 'matched' else match['reason']] += 1
        associations.append(match)
    matched = [a for a in associations if a['status'] == 'matched']
    matched_ids = [a['anchor_id'] for a in matched]
    support = float(sum((anchors[index]['weight'] for index in matched_ids)))
    active = [dict(anchors[a['anchor_id']], local=a['local']) for a in matched]
    weights = np.array([a['weight'] for a in active])
    gates = admission(active, builder, problem, pose, weights) if active else dict(passed=False, reasons=['no_refreshed_associations'])
    if support < 0.8:
        gates['passed'] = False
        gates['reasons'].append('initial_weight_support_under_80percent')
    if REQUIRE_FULL_POOL and (not pool['admitted']):
        gates['passed'] = False
        gates['reasons'].append('SOURCE_pool_geometry_not_admitted')
    return dict(associations=associations, support_weight=support, admission=gates, counts=dict(counts), lost_anchor_ids=[a['anchor_id'] for a in associations if a['status'] == 'missing'], source_graph=dict(nodes=len(graph['coordinate']), components=len(graph['component_length']), branched_components=graph['branched_components']), sample_graph_is_descriptive_only=True, complete_projected_union=union_index.summary())
