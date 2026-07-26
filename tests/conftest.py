import sys
from pathlib import Path

# Tests run against the source tree without requiring an install.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
