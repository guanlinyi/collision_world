"""Start the complete RGB/mask/3D replay, without models or external disks."""
import argparse
import importlib.util
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port',type=int,default=8883)
    args=parser.parse_args()
    root=Path(__file__).resolve().parent
    spec=importlib.util.spec_from_file_location('full_replay',root/'inspect_predictions/code/rich_serve.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    case=root/'datasets/carla_replay/case11'
    server=module.make_server(case,args.port,case/'input_json/image_map.json')
    print(f'Open http://127.0.0.1:{server.server_port}/cases/11/ or /cases/07/ ; Ctrl+C stops.',flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.server_close()


if __name__=='__main__':main()
