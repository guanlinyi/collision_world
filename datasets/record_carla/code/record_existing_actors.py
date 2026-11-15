"""Record numeric actor states in an existing CARLA simulation without altering it.

This records ground-truth metadata, not RGB model inputs or collision labels.
It never loads a map, spawns an actor, changes settings, or controls a vehicle.
"""
import argparse
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host',default='127.0.0.1')
    parser.add_argument('--port',type=int,default=2000)
    parser.add_argument('--actor-ids',type=int,nargs='+',required=True)
    parser.add_argument('--frames',type=int,default=80)
    parser.add_argument('--output',type=Path,required=True,help='New output_json directory')
    args=parser.parse_args()
    if args.frames<1:parser.error('--frames must be positive')
    import carla
    client=carla.Client(args.host,args.port);client.set_timeout(10)
    world=client.get_world()
    if any(world.get_actor(i) is None for i in args.actor_ids):raise ValueError('Actor ID missing')
    root=Path(__file__).resolve().parents[1]
    output=args.output.resolve()
    if not output.is_relative_to(root):raise ValueError('Output must stay inside datasets/record_carla')
    output.mkdir(parents=True,exist_ok=False)
    records=[]
    for _ in range(args.frames):
        snapshot=world.wait_for_tick(10)
        actors=[]
        for actor_id in args.actor_ids:
            actor=world.get_actor(actor_id);state=snapshot.find(actor_id)
            if actor is None or state is None:
                actors.append(dict(id=actor_id,status='missing'));continue
            transform=state.get_transform();velocity=state.get_velocity();box=actor.bounding_box
            actors.append(dict(id=actor_id,type_id=actor.type_id,status='present',
                world_matrix=transform.get_matrix(),velocity_xyz_mps=[velocity.x,velocity.y,velocity.z],
                bbox_center_local_m=[box.location.x,box.location.y,box.location.z],
                bbox_half_extent_xyz_m=[box.extent.x,box.extent.y,box.extent.z]))
        records.append(dict(frame=snapshot.frame,time_s=snapshot.timestamp.elapsed_seconds,actors=actors))
    with (output/'actor_states.json').open('x',encoding='utf-8') as f:
        json.dump(dict(schema='carla_actor_record_v1',carla_client_version=client.get_client_version(),
            carla_server_version=client.get_server_version(),map=world.get_map().name,
            scope='CARLA numeric ground truth only. No RGB, detector, masks, learned proposals or accident labels.',
            records=records),f,indent=2,allow_nan=False)
    print(output/'actor_states.json')


if __name__=='__main__':main()
