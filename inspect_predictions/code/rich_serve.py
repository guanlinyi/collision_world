"""Foreground loopback-only viewer for an explicitly exported case; no inference or arbitrary-file API."""
import argparse,functools,json,mimetypes,shutil,os,socket
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit,unquote
PURPOSE=Path(__file__).resolve().parents[1];PROJECT=PURPOSE.parent;WEB=PURPOSE/'web_rich'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))

def required_camera_path(bundle, config=None):
    """Resolve one project-relative, required calibration; never infer or search disks."""
    config = config or read(PURPOSE / 'config/replay_resources.json')
    relative = Path(config['recorded_camera'])
    if relative.is_absolute():
        raise ValueError('Recorded camera path must be project-relative')
    camera_path = (PROJECT / relative).resolve()
    if not camera_path.is_relative_to(PROJECT.resolve()):
        raise ValueError('Recorded camera path escapes project')
    if not camera_path.is_file():
        raise FileNotFoundError('Required original camera parameters missing: ' + str(camera_path))
    document = read(camera_path)
    if config['case'] != bundle['case'] or document.get('case') != bundle['case'] or document.get('schema') != 'cwm_recorded_camera_v1':
        raise ValueError('Recorded camera case/schema differs from replay')
    expected = {(frame, car) for frame in range(bundle['frame_count']) for car in range(bundle['car_count'])}
    records = document['records']
    if len(records) != len(expected) or {(row['frame'], row['car']) for row in records} != expected:
        raise ValueError('Recorded camera requires the full unique frame/car set')
    for row in records:
        if [row['image_width'], row['image_height']] != bundle['image_dimensions'] or abs(row['time_s'] - bundle['frames'][row['frame']]['time_s']) > 1e-5:
            raise ValueError('Recorded camera time/resolution differs from replay')
    return camera_path


def optional_structure_path(bundle):
    """Resolve the single optional config resource, never a client supplied path."""
    config = read(PURPOSE / 'config/replay_resources.json')
    relative_text = config.get('structure_comparison')
    if not relative_text:
        return None, 'Structure comparison is not configured'
    relative = Path(relative_text)
    if relative.is_absolute():
        raise ValueError('Structure comparison path must be project-relative')
    target = (PROJECT / relative).resolve()
    if not target.is_relative_to(PROJECT.resolve()):
        raise ValueError('Structure comparison path escapes project')
    if not target.is_file():
        return None, 'Configured structure comparison file is not ready'
    return target, None


def pose_experiment_bytes(bundle, config=None):
    """A live config pointer selects immutable experiment evidence inside project."""
    if config is None:
        config = read(PURPOSE / 'config/replay_resources.json')
    relative_text = config.get('pose_experiments')
    if not relative_text:
        return b'{"pose_experiments":[]}'
    relative = Path(relative_text)
    target = (PROJECT / relative).resolve()
    if relative.is_absolute() or not target.is_relative_to(PROJECT.resolve()):
        raise ValueError('Pose experiment resource must stay inside project')
    document = read(target)
    if document.get('case') != bundle['case'] or document.get('schema') != 'cwm_pose_experiments_v1':
        raise ValueError('Pose experiment case/schema differs from replay')
    return json.dumps(document, ensure_ascii=False).encode('utf-8')


def pair_geometry_path(bundle, config=None):
    """Optional frozen full-3D scoring; no arbitrary resource paths or computation."""
    config = config or read(PURPOSE / 'config/replay_resources.json')
    value = config.get('pair_geometry')
    if not value:
        return None
    relative = Path(value)
    target = (PROJECT / relative).resolve()
    if relative.is_absolute() or not target.is_relative_to(PROJECT.resolve()):
        raise ValueError('Pair geometry resource must stay inside project')
    document = read(target)
    if document.get('schema') != 'cwm_pair_geometry_v1' or document.get('case') != bundle['case']:
        raise ValueError('Pair geometry case/schema differs from replay')
    records = document.get('records', [])
    if len(records) != bundle['frame_count'] or {row.get('frame') for row in records} != set(range(bundle['frame_count'])):
        raise ValueError('Pair geometry requires the full unique frame set')
    return target


def case_truth_path(bundle, config=None):
    """Expose one configured, audited original-record explanation; never infer it."""
    config = config or read(PURPOSE / 'config/replay_resources.json')
    value = config.get('case_truth')
    if not value:
        return None
    relative = Path(value)
    target = (PROJECT / relative).resolve()
    if relative.is_absolute() or not target.is_relative_to(PROJECT.resolve()):
        raise ValueError('Recorded truth resource must stay inside project')
    document = read(target)
    records = document.get('records', [])
    if document.get('schema') != 'cwm_recorded_case_truth_v1' or document.get('case') != bundle['case']:
        raise ValueError('Recorded truth case/schema differs from replay')
    if len(records) != bundle['frame_count'] or {r.get('frame') for r in records} != set(range(bundle['frame_count'])):
        raise ValueError('Recorded truth requires the full unique frame set')
    if any(abs(r['time_s']-bundle['frames'][r['frame']]['time_s']) > 1e-8 for r in records):
        raise ValueError('Recorded truth times differ from replay')
    return target


class LoopbackHTTPServer(ThreadingHTTPServer):
    """On Windows prevent multiple viewers sharing the same listening port."""
    allow_reuse_address = os.name != 'nt'

    def server_bind(self):
        if os.name == 'nt':
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()

def make_server(case_dir,port=8878,image_map=None):
 case=Path(case_dir).resolve()
 if not case.is_relative_to(PROJECT.resolve()):raise ValueError('Case directory must be inside this project')
 data=case/'input_json';bundle=read(data/'case_bundle.json')
 if bundle['case']!='joint_fresh_11' or bundle['frame_count']!=80 or bundle['car_count']!=2:raise ValueError('Only the frozen 80-frame two-car negative case is allowed')
 camera_path=required_camera_path(bundle)
 structure_path,structure_reason=optional_structure_path(bundle)
 structure_missing=json.dumps({"schema":"cwm_structure_comparison_v1","available":False,"case":bundle["case"],"reason":structure_reason},ensure_ascii=False).encode("utf-8")
 routes={'/data/camera.json':camera_path,'/':WEB/'index.html','/index.html':WEB/'index.html','/data/case.json':data/'case_bundle.json','/data/geometry.json':data/'asset_convex_hull.json'}
 pair_path=pair_geometry_path(bundle)
 if pair_path is not None:routes['/data/pair-geometry.json']=pair_path
 truth_path=case_truth_path(bundle)
 if truth_path is not None:routes['/data/case-truth.json']=truth_path
 if structure_path is not None:routes['/data/structure.json']=structure_path
 for f in WEB.rglob('*'):
  if f.is_file():routes['/web/'+f.relative_to(WEB).as_posix()]=f
 for frame in bundle['frames']:
  for car in frame['cars']:
   for part in car['observation']['parts']:
    for key in ['mask_url','boundary_url']:
     name=Path(part[key]).name;target=(case/'input_png/masks'/name).resolve()
     if not target.is_relative_to((case/'input_png/masks').resolve()):raise ValueError('Mask outside export')
     routes[part[key]]=target
 if image_map is None:
  allow=read(data/'source_allowlist.json')
 else:
  mapping_file=Path(image_map).resolve()
  if not mapping_file.is_relative_to(PROJECT.resolve()):raise ValueError('Portable image map must stay inside this project')
  allow={}
  for url,relative in read(mapping_file).items():
   if Path(relative).is_absolute():raise ValueError('Portable image paths must be project-relative')
   target=(PROJECT/relative).resolve()
   if not target.is_relative_to(PROJECT.resolve()):raise ValueError('Portable image path escapes project')
   allow[url]=str(target)
 expected={car['image_url'] for frame in bundle['frames'] for car in frame['cars']}
 if set(allow)!=expected:raise ValueError('Image allowlist differs from the frozen case')
 routes.update({url:Path(path).resolve() for url,path in allow.items()})
 missing=[str(p) for p in routes.values() if not p.is_file()]
 if missing:raise FileNotFoundError('Missing required local dependencies/data: '+str(missing[:4]))
 # Each URL owns its case. No mutable current-case state or Referer routing.
 payloads={};pose_routes={'/data/pose-experiments.json':(bundle,None)}
 def scoped_bundle(document,prefix):
  result=json.loads(json.dumps(document))
  result['geometry_url']=prefix+'/data/geometry.json'
  for frame in result['frames']:
   for car in frame['cars']:
    car['image_url']=prefix+car['image_url']
    for part in car['observation']['parts']:
     for key in ['mask_url','boundary_url']:part[key]=prefix+part[key]
  return json.dumps(result,ensure_ascii=False).encode('utf-8')
 for url,target in list(routes.items()):
  if not url.startswith('/web/'):
   routes['/cases/11'+url]=target
 payloads['/cases/11/data/case.json']=scoped_bundle(bundle,'/cases/11')
 routes['/cases/11/']=WEB/'index.html'
 pose_routes['/cases/11/data/pose-experiments.json']=(bundle,None)
 def project_resource(value):
  relative=Path(value);target=(PROJECT/relative).resolve()
  if relative.is_absolute() or not target.is_relative_to(PROJECT.resolve()):raise ValueError('Case resource must stay inside project')
  if not target.is_file():raise FileNotFoundError(target)
  return target
 extra_config=PURPOSE/'config/case07_resources.json'
 if extra_config.exists():
  config=read(extra_config)
  if config.get('case')!='joint_fresh_07':raise ValueError('Only explicit case07 resource config is allowed')
  exported=read(project_resource(config['case_bundle']))
  if exported['case']!='joint_fresh_07' or exported['frame_count']!=80 or exported['car_count']!=2:raise ValueError('Case07 must have frozen 80-frame two-car coverage')
  camera7=required_camera_path(exported,config);prefix='/cases/07'
  payloads[prefix+'/data/case.json']=scoped_bundle(exported,prefix)
  payloads[prefix+'/data/structure.json']=json.dumps(dict(schema='cwm_structure_comparison_v1',case=exported['case'],available=False,reason='No reviewed case07 road/partition resource; case11 resources are not reused'),ensure_ascii=False).encode('utf-8')
  routes.update({prefix+'/':WEB/'index.html',prefix+'/index.html':WEB/'index.html',prefix+'/data/camera.json':camera7,prefix+'/data/geometry.json':project_resource(config['geometry'])})
  pair7=pair_geometry_path(exported,config)
  if pair7 is not None:routes[prefix+'/data/pair-geometry.json']=pair7
  truth7=case_truth_path(exported,config)
  if truth7 is not None:routes[prefix+'/data/case-truth.json']=truth7
  image_mapping=read(project_resource(config['image_map']))
  expected7={car['image_url'] for frame in exported['frames'] for car in frame['cars']}
  if set(image_mapping)!=expected7:raise ValueError('Case07 image allowlist differs from bundle')
  for url,relative in image_mapping.items():
   if not url.startswith('/source-images/') or '..' in url:raise ValueError('Invalid exported image URL')
   routes[prefix+url]=project_resource(relative)
  mask_dir=(PROJECT/config['mask_directory']).resolve()
  if Path(config['mask_directory']).is_absolute() or not mask_dir.is_relative_to(PROJECT.resolve()):raise ValueError('Case07 mask directory escapes project')
  for frame in exported['frames']:
   for car in frame['cars']:
    for part in car['observation']['parts']:
     for key in ['mask_url','boundary_url']:
      url=part[key]
      if not url.startswith('/masks/') or Path(url).name!=url.split('/')[-1]:raise ValueError('Invalid mask URL')
      target=(mask_dir/Path(url).name).resolve()
      if not target.is_relative_to(mask_dir) or not target.is_file():raise ValueError('Missing or invalid case07 mask')
      routes[prefix+url]=target
  pose_routes[prefix+'/data/pose-experiments.json']=(exported,config)
 class Handler(BaseHTTPRequestHandler):
  def log_message(self,fmt,*args):print('%s %s'%(self.address_string(),fmt%args),flush=True)
  def do_HEAD(self):self.respond(False)
  def do_GET(self):self.respond(True)
  def respond(self,body):
   req=urlsplit(self.path);path=unquote(req.path)
   if req.query or '\\' in path or ':' in path or '..' in path.split('/'):
    self.send_error(400,'Only fixed case resource URLs are allowed');return
   if path=='/data/structure.json' and structure_path is None:
    self.send_response(200);self.send_header('Content-Type','application/json; charset=utf-8');self.send_header('Content-Length',str(len(structure_missing)));self.send_header('Cache-Control','no-cache');self.send_header('X-Content-Type-Options','nosniff');self.end_headers()
    if body:self.wfile.write(structure_missing)
    return
   if path in payloads:
    payload=payloads[path]
    self.send_response(200);self.send_header('Content-Type','application/json; charset=utf-8');self.send_header('Content-Length',str(len(payload)));self.send_header('Cache-Control','no-cache');self.send_header('X-Content-Type-Options','nosniff');self.end_headers()
    if body:self.wfile.write(payload)
    return
   if path in pose_routes:
    selected_bundle,selected_config=pose_routes[path]
    try:payload=pose_experiment_bytes(selected_bundle,selected_config)
    except (ValueError,OSError) as error:self.send_error(500,str(error));return
    self.send_response(200);self.send_header('Content-Type','application/json; charset=utf-8');self.send_header('Content-Length',str(len(payload)));self.send_header('Cache-Control','no-cache');self.send_header('X-Content-Type-Options','nosniff');self.end_headers()
    if body:self.wfile.write(payload)
    return
   file=routes.get(path)
   if file is None:self.send_error(404,'Resource is not in the case allowlist');return
   mime={'.js':'text/javascript; charset=utf-8','.json':'application/json; charset=utf-8','.css':'text/css; charset=utf-8','.html':'text/html; charset=utf-8','.ttf':'font/ttf'}.get(file.suffix,mimetypes.guess_type(file.name)[0] or 'application/octet-stream')
   self.send_response(200);self.send_header('Content-Type',mime);self.send_header('Content-Length',str(file.stat().st_size));self.send_header('Cache-Control','no-cache');self.send_header('X-Content-Type-Options','nosniff');self.send_header('Referrer-Policy','no-referrer');self.send_header('Content-Security-Policy',"default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; font-src 'self'; connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'");self.end_headers()
   if body:
    try:
     with file.open('rb') as f:shutil.copyfileobj(f,self.wfile)
    except (BrokenPipeError,ConnectionResetError):pass
 server=LoopbackHTTPServer(('127.0.0.1',port),Handler);server.daemon_threads=True;return server

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--case-dir',type=Path,required=True);ap.add_argument('--port',type=int,default=8878);ap.add_argument('--image-map',type=Path,help='Optional project-relative image mapping for portable replay; omit to keep original source mapping');args=ap.parse_args()
 try:server=make_server(args.case_dir,args.port,args.image_map)
 except OSError as e:raise SystemExit('Cannot start local viewer (missing resource or occupied port): '+str(e))
 print('Collision World Model | cached replay only | negative event',flush=True);print(f'Viewer PID {os.getpid()} | source {Path(__file__).resolve()}',flush=True);print(f'Open manually: http://127.0.0.1:{server.server_port}/',flush=True);print('Foreground service. Close this console or press Ctrl+C to stop. No inference task is running.',flush=True)
 try:server.serve_forever(poll_interval=.2)
 except KeyboardInterrupt:pass
 finally:server.server_close();print('Local viewer stopped.',flush=True)
if __name__=='__main__':main()

