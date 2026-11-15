"""Recompute historical position/velocity metrics and check real HTTP resources."""
import importlib.util
import json
import math
from pathlib import Path
import threading
from urllib.request import urlopen

ROOT=Path(__file__).resolve().parents[2]


def read(path):
    return json.loads(path.read_text(encoding='utf-8'),parse_constant=lambda v: (_ for _ in ()).throw(ValueError(v)))


def main():
    report={'scope':'Numeric replay and HTTP acceptance; no GPU/browser rendering or new CARLA simulation.', 'cases':{}}
    for case in ['11','07']:
        data=read(ROOT/f'datasets/carla_replay/input_json/case{case}.json')
        assert len(data['frames'])==80
        position_max=velocity_max=0.
        red=[]; fp=fn=0
        for i,frame in enumerate(data['frames']):
            assert frame['frame']==i and len(frame['cars'])==2
            for car in frame['cars']:
                p=car['prediction']['pose']; r=car['reference']['pose']; m=car['prediction']['metrics']
                err=math.dist(p[:3],r[:3]);position_max=max(position_max,abs(err-m['center_error_m']))
                if i:
                    prev=data['frames'][i-1]['cars'][car['car']]
                    dt=frame['time_s']-data['frames'][i-1]['time_s']
                    pe=prev['prediction']['pose'];re=prev['reference']['pose']
                    velocity=math.sqrt(sum(((p[k]-pe[k])-(r[k]-re[k]))**2 for k in range(3)))/dt
                    stored=m.get('velocity_error_mps',m.get('velocity_vector_error_mps'))
                    assert stored is not None
                    velocity_max=max(velocity_max,abs(velocity-stored))
                    if velocity>.3:red.append([i,car['car']])
            pair=frame['pair'];pred=pair['prediction']['contact'];truth=pair['reference']['contact']
            fp+=int(pred and not truth);fn+=int(truth and not pred)
        assert position_max<1e-6 and velocity_max<1e-5,(position_max,velocity_max)
        report['cases'][case]=dict(frames=80,poses=160,maximum_position_metric_difference_m=position_max,
            maximum_velocity_metric_difference_mps=velocity_max,velocity_red_frames=red,
            stored_geometric_contact_false_positive_frames=fp,stored_geometric_contact_false_negative_frames=fn)
    spec=importlib.util.spec_from_file_location('replay_server',Path(__file__).with_name('serve.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    server=module.ThreadingHTTPServer(('127.0.0.1',0),module.partial(module.Handler,directory=str(ROOT)))
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    try:
        urls=['/','/inspect_predictions/web/style.css','/inspect_predictions/web/app.js',
              '/datasets/carla_replay/input_json/case11.json','/datasets/carla_replay/input_json/case07.json']
        report['http']={}
        for url in urls:
            with urlopen(f'http://127.0.0.1:{server.server_port}{url}') as response:
                content=response.read();assert response.status==200 and content
                report['http'][url]={'status':response.status,'bytes':len(content)}
    finally:server.shutdown();server.server_close();thread.join()
    print(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False))


if __name__=='__main__':main()
