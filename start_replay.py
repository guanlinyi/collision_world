"""Start the standard-library numerical replay viewer."""
import runpy
from pathlib import Path
runpy.run_path(str(Path(__file__).resolve().parent/'inspect_predictions/code/serve.py'),run_name='__main__')
