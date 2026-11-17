"""Independent offline A1 anchor dataset build."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from data_pipeline.expert_collection.a1_dataset import main
if __name__ == "__main__":
    raise SystemExit(main())
