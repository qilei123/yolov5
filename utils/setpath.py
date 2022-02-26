from pathlib import Path
import sys,os

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0].parent  # YOLOv5 root directory
print(ROOT)
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative