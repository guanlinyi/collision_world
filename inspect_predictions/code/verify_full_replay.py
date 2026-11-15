"""Check the original full replay through real HTTP, including all PNG resources."""
import hashlib
import importlib.util
import json
from pathlib import Path
import threading
from urllib.request import urlopen

ROOT=Path(__file__).resolve().parents[2]


def main():
    spec=importlib.util.spec_from_file_location('full_replay',Path(__file__).with_name('rich_serve.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    case=ROOT/'datasets/carla_replay/case11'
    server=module.make_server(case,0,case/'input_json/image_map.json')
    # Avoid thousands of routine request lines in acceptance output.
    server.RequestHandlerClass.log_message=lambda *args:None
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    report={'scope':'Real HTTP and PNG signature/hash checks, no GPU/browser screenshot acceptance.','cases':{}}
    try:
        def fetch(url):
            with urlopen(f'http://127.0.0.1:{server.server_port}'+url) as r:
                content=r.read();assert r.status==200;return content
        for key in ['11','07']:
            prefix=f'/cases/{key}'
            assert b'<html' in fetch(prefix+'/').lower()
            data=json.loads(fetch(prefix+'/data/case.json'))
            poses=json.loads(fetch(prefix+'/data/pose-experiments.json'))
            pairs=json.loads(fetch(prefix+'/data/pair-geometry.json'))
            camera=json.loads(fetch(prefix+'/data/camera.json'))
            truth=json.loads(fetch(prefix+'/data/case-truth.json'))
            assert len(data['frames'])==80 and len(camera['records'])==160 and len(truth['records'])==80
            method='contiguous_exact_contour_bt8' if key=='11' else 'case07_far_exterior_bt8'
            latest=next(m for m in poses['pose_experiments'] if m['id']==method)
            assert len(latest['records'])==160 and len(pairs['records'])==80
            urls=set()
            for frame in data['frames']:
                for car in frame['cars']:
                    urls.add(car['image_url'])
                    for part in car['observation']['parts']:
                        urls.update([part['mask_url'],part['boundary_url']])
            for url in urls:
                content=fetch(url);assert content.startswith(b'\x89PNG\r\n\x1a\n'),url
            mapping=module.read(ROOT/f'datasets/carla_replay/case{key}/input_json/image_map.json')
            for url,local in mapping.items():
                assert hashlib.sha256(fetch(prefix+url)).digest()==hashlib.sha256((ROOT/local).read_bytes()).digest()
            report['cases'][key]={'frames':80,'latest_method':method,'poses':160,'rgb_urls':160,
                'unique_rgb_files':len(set(mapping.values())),'png_urls_checked':len(urls),
                'all_rgb_http_hashes_match_packaged_files':True}
        for file in (ROOT/'inspect_predictions/web_rich').rglob('*'):
            if file.is_file():assert fetch('/web/'+file.relative_to(ROOT/'inspect_predictions/web_rich').as_posix())
        print(json.dumps(report,ensure_ascii=False,indent=2))
    finally:server.shutdown();server.server_close();thread.join()


if __name__=='__main__':main()
